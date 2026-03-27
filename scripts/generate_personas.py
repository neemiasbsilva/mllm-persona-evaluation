"""CLI entry point: generate_personas.py

Reads a demographics CSV and runs two passes in parallel:

1. Baseline pass (instant) — converts the CSV directly to
   ``outputs/personas_baseline.jsonl`` with no LLM calls.  Each record contains
   only the demographic fields; this is the Condition A flat-demographics dataset
   used by the downstream annotation experiment.

2. Synthetic interview pass (slow) — drives the full LangGraph pipeline
   (Nodes A → B → C → D → E) and writes richly annotated records including the
   Isabella Q&A interview transcript and expert reflections to
   ``outputs/personas_synthetic_interview.jsonl``.

Both passes run concurrently so the baseline file is ready the moment the script
starts, while the LLM phase runs in the background.

Usage examples
--------------
# Full run with defaults from .env:
uv run python scripts/generate_personas.py --csv data/baseline_distribution.csv

# Smoke-test with 5 personas and 2 concurrent workers:
uv run python scripts/generate_personas.py \\
    --csv data/baseline_distribution.csv \\
    --limit 5 \\
    --max-concurrent 2

# Override model and output dir at runtime:
uv run python scripts/generate_personas.py \\
    --csv data/baseline_distribution.csv \\
    --output-dir /tmp/persona_test \\
    --text-model llama3.1:8b
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the src/ package is on the path when running the script directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persona_generator.config import Settings
from persona_generator.logging_config import configure_logging, get_logger
from persona_generator.pipeline import run_parallel_batch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MLLM personas: baseline JSONL + LangGraph synthetic interviews.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to the demographics seed CSV (e.g. data/baseline_distribution.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSONL outputs, logs, and SQLite checkpoints. "
        "Overrides OUTPUT_DIR env var.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum simultaneous persona graph invocations. "
        "Overrides MAX_CONCURRENT env var.",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default=None,
        help="Ollama model for Nodes B/C. Overrides TEXT_MODEL env var.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed. Overrides RANDOM_SEED env var.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows (useful for smoke-testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Build a Settings instance, applying CLI overrides where provided
    overrides: dict = {}
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.max_concurrent is not None:
        overrides["max_concurrent"] = args.max_concurrent
    if args.text_model is not None:
        overrides["text_model"] = args.text_model
    if args.seed is not None:
        overrides["random_seed"] = args.seed

    cfg = Settings(**overrides)
    cfg.ensure_output_dirs()

    # Initialise logging AFTER output dirs exist so the file handler can open
    configure_logging(log_file=cfg.log_file)
    log = get_logger(__name__)

    log.info(
        "cli_start",
        csv=str(args.csv),
        output_dir=str(cfg.output_dir),
        text_model=cfg.text_model,
        max_concurrent=cfg.max_concurrent,
        max_retries=cfg.max_retries,
        random_seed=cfg.random_seed,
        limit=args.limit,
    )

    if not args.csv.exists():
        log.error("csv_not_found", path=str(args.csv))
        sys.exit(1)

    result = asyncio.run(
        run_parallel_batch(
            demographics_csv=args.csv,
            cfg=cfg,
            limit=args.limit,
        )
    )

    baseline = result["baseline"]
    synthetic = result["synthetic"]

    # ── Final summary printed to stdout ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("PERSONA GENERATION COMPLETE")
    print("=" * 60)

    print("\n  [Baseline — demographic fields only]")
    print(f"    Records written : {baseline['total']}")
    print(f"    Output          : {cfg.personas_baseline_jsonl}")

    print("\n  [Synthetic interviews — full LangGraph pipeline]")
    print(f"    Total processed : {synthetic['total']}")
    print(f"    Successful      : {synthetic['success']}")
    print(f"    Failed/discarded: {synthetic['failed']}")
    print(f"    Avg retry count : {synthetic['avg_retry']:.3f}")
    print(f"    Total duration  : {synthetic['duration_s']:.1f}s")
    print(f"    Output          : {cfg.personas_synthetic_interview_jsonl}")

    print("\n  [Shared artifacts]")
    print(f"    Failed JSONL    : {cfg.failed_jsonl}")
    print(f"    Run log         : {cfg.log_file}")
    print(f"    Checkpoints     : {cfg.checkpoint_db}")
    print("=" * 60)

    # Exit with error code if any synthetic personas failed (useful for CI pipelines)
    if synthetic["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
