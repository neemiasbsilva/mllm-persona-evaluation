"""CLI entry point for the no-persona annotation experiment.

Annotates images using qwen3-vl:8b without any persona or demographic
conditioning, for both think=True (extended reasoning) and think=False
(direct response) modes.

Outputs are written to NEW files and never touch the existing baseline results:
  outputs/annotations_no_persona_think.jsonl
  outputs/annotations_no_persona_no_think.jsonl
  outputs/annotation_failures_no_persona.jsonl

IMPORTANT — same image set as the baseline
------------------------------------------
Pass --baseline-jsonl so the no-persona run covers exactly the same
50 images that were used in the baseline persona experiment:

  uv run python scripts/run_annotation.py \\
      --condition baseline --n-personas 1200 --n-images 50

  uv run python scripts/run_annotation_no_persona.py \\
      --think --baseline-jsonl outputs/annotations_baseline.jsonl

  uv run python scripts/run_annotation_no_persona.py \\
      --no-think --baseline-jsonl outputs/annotations_baseline.jsonl

Usage examples
--------------
# Recommended: match the exact 50 images from the baseline experiment
uv run python scripts/run_annotation_no_persona.py \\
    --think --baseline-jsonl outputs/annotations_baseline.jsonl

uv run python scripts/run_annotation_no_persona.py \\
    --no-think --baseline-jsonl outputs/annotations_baseline.jsonl

# Both conditions in sequence
uv run python scripts/run_annotation_no_persona.py \\
    --both --baseline-jsonl outputs/annotations_baseline.jsonl

# Smoke-test: 3 images, no baseline constraint
uv run python scripts/run_annotation_no_persona.py \\
    --think --n-images 3 --limit 3

# Override Ollama concurrency (requires OLLAMA_NUM_PARALLEL set on the server)
uv run python scripts/run_annotation_no_persona.py \\
    --think --baseline-jsonl outputs/annotations_baseline.jsonl --max-concurrent 2
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from annotator.config import AnnotatorSettings
from annotator.pipeline import run_no_persona_annotation_batch
from persona_generator.logging_config import configure_logging, get_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run no-persona image annotation with qwen3-vl:8b.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    think_group = parser.add_mutually_exclusive_group(required=True)
    think_group.add_argument(
        "--think",
        dest="think",
        action="store_true",
        help="Enable qwen3-vl:8b extended chain-of-thought reasoning (think=True).",
    )
    think_group.add_argument(
        "--no-think",
        dest="think",
        action="store_false",
        help="Disable reasoning prefix — direct JSON response (think=False).",
    )
    think_group.add_argument(
        "--both",
        action="store_true",
        default=False,
        help="Run think=True then think=False sequentially.",
    )

    parser.add_argument(
        "--baseline-jsonl",
        type=Path,
        default=None,
        help=(
            "Path to an existing annotations_baseline.jsonl. "
            "When provided, the no-persona run uses exactly the same image IDs "
            "as the baseline experiment (recommended for paper comparisons). "
            "Overrides --n-images."
        ),
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=50,
        help=(
            "Number of unique images to sample (stratified by sentiment × in_out_door). "
            "Ignored when --baseline-jsonl is provided."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total images — useful for smoke-testing.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Override ANNOTATION_MAX_CONCURRENT env var.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for JSONL files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override NO_PERSONA_MODEL env var (default: qwen3-vl:8b).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override RANDOM_SEED for stratified image sampling.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of independent annotation passes over the image set (default: 1). "
            "Each pass uses a distinct run ID (np_r01, np_r02, …) so its annotation "
            "IDs never collide with previous passes. Results are *appended* to the "
            "same output JSONL — the file is never overwritten. Re-running is safe: "
            "already-completed pass/image pairs are skipped automatically."
        ),
    )
    return parser.parse_args()


def _print_banner(cfg: AnnotatorSettings, think: bool) -> None:
    n = cfg.annotation_max_concurrent
    think_label = "think=True  (extended reasoning)" if think else "think=False (direct response)"
    print()
    print("=" * 62)
    print("  NO-PERSONA ANNOTATION — GPU TUNING")
    print("=" * 62)
    print(f"  Model      : {cfg.no_persona_model}")
    print(f"  Think mode : {think_label}")
    print(f"  export OLLAMA_NUM_PARALLEL={n:<6}  # must match --max-concurrent")
    print( "  export OLLAMA_FLASH_ATTENTION=1    # reduces VRAM per slot")
    print( "  export OLLAMA_KV_CACHE_TYPE=q8_0  # halves K/V cache VRAM")
    print()
    print(f"  Active settings:")
    print(f"    --max-concurrent : {n}")
    print(f"    num_ctx          : {cfg.num_ctx}")
    print(f"    num_predict      : {cfg.annotation_num_predict}")
    print(f"    output_dir       : {cfg.output_dir}")
    print("=" * 62)
    print()


def _run_one(think: bool, args: argparse.Namespace, cfg: AnnotatorSettings) -> dict:
    _print_banner(cfg, think)
    return asyncio.run(
        run_no_persona_annotation_batch(
            think=think,
            cfg=cfg,
            n_images=args.n_images,
            limit=args.limit,
            baseline_jsonl=args.baseline_jsonl,
            n_runs=args.n_runs,
        )
    )


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  COMPLETE — {summary['condition'].upper()}")
    print("=" * 60)
    print(f"  Runs             : {summary.get('n_runs', 1)}")
    print(f"  Images processed : {summary['total']}")
    print(f"  Successful       : {summary['success']}")
    print(f"  Failed           : {summary['failed']}")
    print(f"  Duration         : {summary['duration_s']:.1f}s")
    print(f"  Output           : {summary['output']}")
    print("=" * 60)


def main() -> None:
    args = _parse_args()

    overrides: dict = {}
    if args.max_concurrent is not None:
        overrides["annotation_max_concurrent"] = args.max_concurrent
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    if args.model is not None:
        overrides["no_persona_model"] = args.model

    cfg = AnnotatorSettings(**overrides)
    cfg.ensure_output_dirs()

    configure_logging(log_file=cfg.log_file)

    modes: list[bool]
    if args.both:
        modes = [True, False]
    else:
        modes = [bool(args.think)]

    failed_total = 0
    for think in modes:
        summary = _run_one(think, args, cfg)
        _print_summary(summary)
        failed_total += summary["failed"]

    if failed_total > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
