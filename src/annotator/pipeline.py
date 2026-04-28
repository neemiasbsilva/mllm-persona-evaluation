"""Async batch orchestrator for the image annotation pipeline.

Outer loop: images × personas
Execution: semaphore-bounded concurrency (ANNOTATION_MAX_CONCURRENT slots).
           Set ANNOTATION_MAX_CONCURRENT = OLLAMA_NUM_PARALLEL to saturate
           all GPU inference slots without overwhelming the server queue.
           When ANNOTATION_MAX_CONCURRENT=1 the behaviour is fully sequential.
Outputs: append-safe JSONL (annotations_baseline.jsonl) + annotation_failures.jsonl

Stratified sampling
-------------------
``run_annotation_batch`` accepts ``n_personas`` and ``n_images`` limits.
Personas are stratified by ``political_spectrum × economic_status``;
images are stratified by ``sentiment × in_out_door``.

Persona source
--------------
- ``baseline`` → ``outputs/personas_baseline.jsonl``  (flat demographic tags)
"""

import asyncio
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from annotator.config import AnnotatorSettings, annotator_settings as default_cfg
from annotator.graph import build_annotation_graph
from annotator.state import AnnotationState, CONDITIONS

# Dummy persona record used for no-persona conditions (no demographic conditioning).
_NO_PERSONA_RECORD = {
    "persona_id":       "no_persona",
    "raw_demographics": {},
}
from persona_generator.logging_config import get_logger

log = get_logger(__name__)


# ── Sampling helpers ──────────────────────────────────────────────────────────

def _stratified_sample_personas(
    personas: list[dict],
    n: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Stratify personas by political_spectrum × economic_status, sample n total."""
    if n >= len(personas):
        return personas

    groups: dict[str, list] = defaultdict(list)
    for p in personas:
        demo = p["raw_demographics"]
        key = f"{demo.get('political_spectrum', '?')}|{demo.get('economic_status', '?')}"
        groups[key].append(p)

    selected: list[dict] = []
    per_group = max(1, n // len(groups))
    for group in groups.values():
        take = min(per_group, len(group))
        indices = rng.choice(len(group), size=take, replace=False)
        selected.extend(group[i] for i in indices)

    # Top up or trim to exactly n
    rng.shuffle(selected)  # type: ignore[arg-type]
    return selected[:n]


def _stratified_sample_images(
    images: list[dict],
    n: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Stratify images by sentiment × in_out_door, sample n unique images."""
    # Deduplicate images (dataset has 5 annotations per image; keep one per id)
    seen: set[str] = set()
    unique: list[dict] = []
    for img in images:
        if img["id"] not in seen:
            seen.add(img["id"])
            unique.append(img)

    if n >= len(unique):
        return unique

    groups: dict[str, list] = defaultdict(list)
    for img in unique:
        key = f"{img.get('sentiment', '?')}|{img.get('in_out_door', '?')}"
        groups[key].append(img)

    selected: list[dict] = []
    per_group = max(1, n // len(groups))
    for group in groups.values():
        take = min(per_group, len(group))
        indices = rng.choice(len(group), size=take, replace=False)
        selected.extend(group[i] for i in indices)

    rng.shuffle(selected)  # type: ignore[arg-type]
    return selected[:n]


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_baseline_personas(personas_jsonl: Path) -> list[dict]:
    """Load persona records from ``personas_baseline.jsonl``.

    Baseline records have flat top-level demographic fields.  This function
    normalises each record by wrapping those fields into a ``raw_demographics``
    sub-dict so the downstream assembler is condition-agnostic.
    """
    personas = []
    with open(personas_jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            # Normalise: wrap flat fields into raw_demographics
            p["raw_demographics"] = {
                "gender":             p.get("gender", "Unknown"),
                "economic_status":    p.get("economic_status", "Unknown"),
                "political_spectrum": p.get("political_spectrum", "Unknown"),
                "personality":        p.get("personality", "Unknown"),
            }
            personas.append(p)
    return personas


def load_images(dataset_json: Path) -> list[dict]:
    """Flatten all image records from the perceptsent dataset.json tasks list."""
    with open(dataset_json, encoding="utf-8") as fh:
        data = json.load(fh)
    images = [img for task in data["tasks"] for img in task["images"]]
    return images


# ── Resume helper ─────────────────────────────────────────────────────────────

def _load_done_annotation_ids(jsonl_path: Path) -> set[str]:
    """Return the set of annotation_ids already written to *jsonl_path*.

    Used by ``run_annotation_batch`` to skip already-completed triples on
    restart, avoiding redundant llava calls.
    """
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                aid = record.get("annotation_id")
                if aid:
                    done.add(str(aid))
            except json.JSONDecodeError:
                pass
    return done


# ── Per-triple runner ─────────────────────────────────────────────────────────

def _build_annotation_id(persona_id: str, image_id: str, condition: str) -> str:
    # Use first 8 chars of the UUID for a compact but unique prefix
    prefix = str(persona_id)[:8]
    return f"p{prefix}_img_{image_id}_{condition}"


def _serialize_result(state: AnnotationState, persona: dict) -> dict:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "annotation_id":          _build_annotation_id(
                                      state["persona_id"],
                                      state["image_id"],
                                      state["condition"],
                                  ),
        "persona_id":             state["persona_id"],
        "image_id":               state["image_id"],
        "condition":              state["condition"],
        "raw_demographics":       persona["raw_demographics"],
        "predicted_sentiment":    state["predicted_sentiment"],
        "predicted_perceptions":  state["predicted_perceptions"],
        "caption":                state["caption"],
        "justification":          state["justification"],
        "parse_retries":          state["parse_retries"],
        "timestamp_utc":          ts,
    }


async def _preload_image_cache(
    images: list[dict],
    image_dir: Path,
) -> dict[str, str]:
    """Base64-encode every unique image once, concurrently, before annotation.

    Eliminates N-fold redundant disk I/O for N personas × M images: each
    image is read and encoded once, then shared across all personas.

    Args:
        images:    List of unique image dicts (each must have an ``"id"`` key).
        image_dir: Directory containing ``{image_id}.jpg`` files.

    Returns:
        Mapping of ``image_id → base64-encoded JPEG string``.
    """
    import base64

    async def _load_one(image_id: str) -> tuple[str, str]:
        path = image_dir / f"{image_id}.jpg"
        raw = await asyncio.to_thread(path.read_bytes)
        return image_id, base64.b64encode(raw).decode("utf-8")

    pairs = await asyncio.gather(*(_load_one(img["id"]) for img in images))
    return dict(pairs)


async def _annotate_one(
    graph,
    persona: dict,
    image: dict,
    condition: str,
    success_fh,
    failed_fh,
    image_cache: dict[str, str] | None = None,
) -> bool:
    """Run one (persona × image × condition) triple through the annotation graph."""
    persona_id = persona["persona_id"]
    image_id   = image["id"]

    initial_state = AnnotationState(
        persona_id=persona_id,
        image_id=image_id,
        condition=condition,
        persona_record=persona,
        image_b64=image_cache.get(image_id, "") if image_cache else "",
        system_prompt="",
        predicted_sentiment="",
        predicted_perceptions=[],
        caption="",
        justification="",
        raw_response="",
        parse_retries=0,
    )

    try:
        final_state: AnnotationState = await graph.ainvoke(initial_state)
    except Exception as exc:
        log.error(
            "annotation_triple_error",
            persona_id=persona_id,
            image_id=image_id,
            condition=condition,
            error=str(exc),
        )
        record = {
            "annotation_id": _build_annotation_id(persona_id, image_id, condition),
            "persona_id": persona_id,
            "image_id": image_id,
            "condition": condition,
            "error": str(exc),
        }
        failed_fh.write(json.dumps(record) + "\n")
        failed_fh.flush()
        return False

    success = bool(final_state.get("predicted_sentiment"))
    record = _serialize_result(final_state, persona)

    if success:
        success_fh.write(json.dumps(record) + "\n")
        success_fh.flush()
    else:
        record["error"] = "parse_retries_exhausted"
        failed_fh.write(json.dumps(record) + "\n")
        failed_fh.flush()

    return success


# ── Public entry point ────────────────────────────────────────────────────────

async def run_annotation_batch(
    condition: str,
    cfg: AnnotatorSettings | None = None,
    n_personas: int = 200,
    n_images: int = 500,
    limit: int | None = None,
) -> dict:
    """Annotate ``n_personas × n_images`` for a single condition.

    Args:
        condition:  'baseline' | 'full_persona'
        cfg:        Settings override (defaults to module singleton).
        n_personas: Number of personas to sample (stratified).
        n_images:   Number of unique images to sample (stratified).
        limit:      If set, cap total triples (for smoke-testing).

    Returns:
        Summary dict: {condition, total, success, failed, duration_s}
    """
    if cfg is None:
        cfg = default_cfg

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got '{condition}'")

    cfg.ensure_output_dirs()

    rng = np.random.default_rng(cfg.random_seed)

    # ── Load personas ─────────────────────────────────────────────────────────
    personas_path = cfg.personas_baseline_jsonl
    log.info("loading_personas", condition=condition, path=str(personas_path))
    personas = load_baseline_personas(personas_path)

    log.info("loaded_personas", count=len(personas))

    all_images = load_images(cfg.dataset_json)
    log.info("loaded_images", total_annotations=len(all_images))

    personas = _stratified_sample_personas(personas, n_personas, rng)
    images   = _stratified_sample_images(all_images, n_images, rng)
    log.info(
        "sampled",
        condition=condition,
        n_personas=len(personas),
        n_images=len(images),
    )

    # ── Output files ──────────────────────────────────────────────────────────
    out_path = cfg.annotations_baseline_jsonl
    failed_path = cfg.annotation_failures_jsonl

    # ── Build triples ─────────────────────────────────────────────────────────
    triples = [(p, img) for p in personas for img in images]
    if limit is not None:
        triples = triples[:limit]

    # Skip triples already written to the output JSONL (resume support)
    done_ids = _load_done_annotation_ids(out_path)
    if done_ids:
        before = len(triples)
        triples = [
            (p, img) for p, img in triples
            if _build_annotation_id(p["persona_id"], img["id"], condition) not in done_ids
        ]
        log.info(
            "annotation_batch_resume",
            condition=condition,
            skipped=before - len(triples),
            remaining=len(triples),
        )

    total = len(triples)
    log.info("annotation_batch_start", condition=condition, total=total)

    graph = build_annotation_graph()

    # Pre-load all selected images once to eliminate redundant per-triple disk I/O
    log.info("image_cache_preload_start", n_images=len(images))
    image_cache = await _preload_image_cache(images, cfg.image_dir)
    log.info("image_cache_preload_complete", n_images=len(image_cache))

    concurrency = cfg.annotation_max_concurrent
    log.info(
        "annotation_concurrency",
        condition=condition,
        max_concurrent=concurrency,
        num_ctx=cfg.num_ctx,
    )

    t0 = time.monotonic()
    success_count = failed_count = 0

    try:
        from tqdm import tqdm as tqdm_sync

        # Semaphore keeps exactly `concurrency` calls in-flight at any moment,
        # matching OLLAMA_NUM_PARALLEL on the server side.  When concurrency=1
        # this is identical to the old sequential behaviour.
        sem = asyncio.Semaphore(concurrency)

        async def _bounded_annotate(p, img):
            async with sem:
                return await _annotate_one(
                    graph, p, img, condition, success_fh, failed_fh, image_cache
                )

        with (
            open(out_path,    "a", encoding="utf-8") as success_fh,
            open(failed_path, "a", encoding="utf-8") as failed_fh,
        ):
            with tqdm_sync(total=total, desc=f"Annotating [{condition}] ×{concurrency}", unit="triple") as pbar:
                tasks = [asyncio.create_task(_bounded_annotate(p, img)) for p, img in triples]
                for future in asyncio.as_completed(tasks):
                    result = await future
                    if result:
                        success_count += 1
                    else:
                        failed_count += 1
                    pbar.update(1)

    finally:
        duration_s = round(time.monotonic() - t0, 1)

    summary = {
        "condition":  condition,
        "total":      total,
        "success":    success_count,
        "failed":     failed_count,
        "duration_s": duration_s,
        "output":     str(out_path),
    }
    log.info("annotation_batch_complete", **summary)
    return summary


# ── No-persona batch runner ───────────────────────────────────────────────────

def _load_image_ids_from_baseline(baseline_jsonl: Path) -> set[str]:
    """Return the set of image IDs present in an existing annotations JSONL file.

    Used to ensure the no-persona run covers the exact same images as the
    baseline experiment rather than drawing a new random sample.
    """
    ids: set[str] = set()
    if not baseline_jsonl.exists():
        return ids
    with open(baseline_jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                img_id = rec.get("image_id")
                if img_id:
                    ids.add(str(img_id))
            except json.JSONDecodeError:
                pass
    return ids


async def run_no_persona_annotation_batch(
    think: bool,
    cfg: AnnotatorSettings | None = None,
    n_images: int = 500,
    limit: int | None = None,
    baseline_jsonl: Path | None = None,
) -> dict:
    """Annotate images with qwen3-vl:8b without any persona conditioning.

    One annotation is produced per image (no persona loop).  Results are
    written to separate JSONL files so existing baseline results are never
    modified:
      - think=True  → ``outputs/annotations_no_persona_think.jsonl``
      - think=False → ``outputs/annotations_no_persona_no_think.jsonl``
    Failures → ``outputs/annotation_failures_no_persona.jsonl``

    Args:
        think:          When True, qwen3-vl:8b uses extended chain-of-thought.
        cfg:            Settings override; falls back to module singleton.
        n_images:       Number of unique images to sample (stratified).
                        Ignored when *baseline_jsonl* is provided.
        limit:          Cap total images — useful for smoke-testing.
        baseline_jsonl: When provided, restricts the image set to the exact
                        image IDs present in that JSONL file (e.g. the 50
                        images used in the baseline persona experiment).
                        Takes priority over *n_images*.

    Returns:
        Summary dict: {condition, total, success, failed, duration_s, output}
    """
    if cfg is None:
        cfg = default_cfg

    condition = "no_persona_think" if think else "no_persona_no_think"

    cfg.ensure_output_dirs()
    rng = np.random.default_rng(cfg.random_seed)

    all_images = load_images(cfg.dataset_json)

    if baseline_jsonl is not None:
        baseline_ids = _load_image_ids_from_baseline(baseline_jsonl)
        if not baseline_ids:
            raise ValueError(
                f"No image IDs found in baseline JSONL: {baseline_jsonl}. "
                "Ensure the baseline annotation has been run first."
            )
        # Deduplicate and filter to baseline image IDs only
        seen: set[str] = set()
        images = []
        for img in all_images:
            if img["id"] in baseline_ids and img["id"] not in seen:
                seen.add(img["id"])
                images.append(img)
        log.info(
            "no_persona_image_filter_from_baseline",
            baseline_jsonl=str(baseline_jsonl),
            baseline_ids=len(baseline_ids),
            matched_images=len(images),
        )
    else:
        images = _stratified_sample_images(all_images, n_images, rng)

    if limit is not None:
        images = images[:limit]

    log.info(
        "no_persona_batch_start",
        condition=condition,
        n_images=len(images),
        model=cfg.no_persona_model,
        think=think,
    )

    # Route to condition-specific output file — never touch existing baseline JSONL.
    out_path    = (cfg.annotations_no_persona_think_jsonl if think
                   else cfg.annotations_no_persona_no_think_jsonl)
    failed_path = cfg.annotation_failures_no_persona_jsonl

    # Resume: skip images already annotated in a previous run.
    done_ids = _load_done_annotation_ids(out_path)
    if done_ids:
        before = len(images)
        images = [img for img in images
                  if _build_annotation_id("no_persona", img["id"], condition) not in done_ids]
        log.info(
            "no_persona_batch_resume",
            condition=condition,
            skipped=before - len(images),
            remaining=len(images),
        )

    total = len(images)

    # Build one annotation graph with the no-persona model and think setting.
    graph = build_annotation_graph(model=cfg.no_persona_model, think=think, cfg=cfg)

    log.info("image_cache_preload_start", n_images=len(images))
    image_cache = await _preload_image_cache(images, cfg.image_dir)
    log.info("image_cache_preload_complete", n_images=len(image_cache))

    concurrency = cfg.annotation_max_concurrent
    t0 = time.monotonic()
    success_count = failed_count = 0

    try:
        from tqdm import tqdm as tqdm_sync

        sem = asyncio.Semaphore(concurrency)

        async def _bounded_annotate(img):
            async with sem:
                return await _annotate_one(
                    graph,
                    _NO_PERSONA_RECORD,
                    img,
                    condition,
                    success_fh,
                    failed_fh,
                    image_cache,
                )

        with (
            open(out_path,    "a", encoding="utf-8") as success_fh,
            open(failed_path, "a", encoding="utf-8") as failed_fh,
        ):
            with tqdm_sync(
                total=total,
                desc=f"Annotating [{condition}] ×{concurrency}",
                unit="image",
            ) as pbar:
                tasks = [asyncio.create_task(_bounded_annotate(img)) for img in images]
                for future in asyncio.as_completed(tasks):
                    result = await future
                    if result:
                        success_count += 1
                    else:
                        failed_count += 1
                    pbar.update(1)

    finally:
        duration_s = round(time.monotonic() - t0, 1)

    summary = {
        "condition":  condition,
        "total":      total,
        "success":    success_count,
        "failed":     failed_count,
        "duration_s": duration_s,
        "output":     str(out_path),
    }
    log.info("no_persona_batch_complete", **summary)
    return summary
