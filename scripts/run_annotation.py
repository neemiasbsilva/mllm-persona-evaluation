"""CLI entry point for the image annotation pipeline (Phase 2).

Runs the baseline condition over a stratified sample of 1,200 personas × 50 images,
writing results to outputs/annotations_baseline.jsonl.

Pre-computed results are already available in outputs/annotations_baseline.jsonl
(59,708 successful annotations, 292 failures, 0.49% failure rate).

Usage examples
--------------
# Smoke-test: 3 personas x 5 images
uv run python scripts/run_annotation.py \\
    --condition baseline \\
    --n-personas 3 --n-images 5 \\
    --limit 10

# Full paper replication: 1,200 personas x 50 images
uv run python scripts/run_annotation.py --condition baseline \\
    --n-personas 1200 --n-images 50

# 20 GB GPU (2 parallel slots — requires OLLAMA_NUM_PARALLEL=2)
uv run python scripts/run_annotation.py \\
    --condition baseline \\
    --max-concurrent 2

# 95 GB GPU (6 parallel slots — requires OLLAMA_NUM_PARALLEL=6)
uv run python scripts/run_annotation.py \\
    --condition baseline \\
    --max-concurrent 6
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from annotator.config import AnnotatorSettings
from annotator.pipeline import run_annotation_batch
from annotator.state import CONDITIONS
from persona_generator.logging_config import configure_logging, get_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MLLM image sentiment annotation (Phase 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--condition",
        choices=list(CONDITIONS),
        required=True,
        help="Experimental condition. Only 'baseline' (flat demographic tags) is supported.",
    )
    parser.add_argument(
        "--n-personas",
        type=int,
        default=200,
        help="Number of personas to sample (stratified by political_spectrum × economic_status).",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=500,
        help="Number of unique images to sample (stratified by sentiment × in_out_door).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total (persona × image) triples — useful for smoke-testing.",
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
        "--personas-jsonl",
        type=Path,
        default=None,
        help="Override path to personas.jsonl (default: outputs/personas.jsonl).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override RANDOM_SEED for stratified sampling.",
    )
    return parser.parse_args()


def _print_gpu_tuning_banner(cfg: "AnnotatorSettings") -> None:
    """Print the Ollama server env vars required to match this run's concurrency."""
    n = cfg.annotation_max_concurrent
    print()
    print("=" * 62)
    print("  GPU TUNING — set these BEFORE starting `ollama serve`")
    print("=" * 62)
    print(f"  export OLLAMA_NUM_PARALLEL={n:<6}  # must match --max-concurrent")
    print( "  export OLLAMA_FLASH_ATTENTION=1    # reduces VRAM per slot")
    print( "  export OLLAMA_KV_CACHE_TYPE=q8_0  # halves K/V cache VRAM")
    print()
    print(f"  Then: ollama serve  (then run this script)")
    print()
    print(f"  Active settings:")
    print(f"    --max-concurrent : {n}")
    print(f"    num_ctx          : {cfg.num_ctx}")
    print(f"    num_predict      : {cfg.annotation_num_predict}")
    print(f"    vision_model     : {cfg.vision_model}")
    print("=" * 62)
    print()


def main() -> None:
    args = _parse_args()

    overrides: dict = {}
    if args.max_concurrent is not None:
        overrides["annotation_max_concurrent"] = args.max_concurrent
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["random_seed"] = args.seed

    cfg = AnnotatorSettings(**overrides)
    cfg.ensure_output_dirs()

    # Override personas_jsonl path if provided
    if args.personas_jsonl is not None:
        # Monkey-patch: pydantic-settings properties aren't overridable via __init__
        cfg.__dict__["_personas_jsonl_override"] = args.personas_jsonl

    _print_gpu_tuning_banner(cfg)

    configure_logging(log_file=cfg.log_file)
    log = get_logger(__name__)

    log.info(
        "cli_annotation_start",
        condition=args.condition,
        n_personas=args.n_personas,
        n_images=args.n_images,
        limit=args.limit,
        max_concurrent=cfg.annotation_max_concurrent,
        num_ctx=cfg.num_ctx,
        vision_model=cfg.vision_model,
        output_dir=str(cfg.output_dir),
    )

    summary = asyncio.run(
        run_annotation_batch(
            condition=args.condition,
            cfg=cfg,
            n_personas=args.n_personas,
            n_images=args.n_images,
            limit=args.limit,
        )
    )

    print("\n" + "=" * 60)
    print(f"ANNOTATION COMPLETE — condition: {args.condition.upper()}")
    print("=" * 60)
    print(f"  Total triples processed : {summary['total']}")
    print(f"  Successful annotations  : {summary['success']}")
    print(f"  Failed / discarded      : {summary['failed']}")
    print(f"  Total duration          : {summary['duration_s']:.1f}s")
    print(f"\n  Output file : {summary['output']}")
    print(f"  Failures    : {cfg.annotation_failures_jsonl}")
    print(f"  Run log     : {cfg.log_file}")
    print("=" * 60)

    if summary["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
