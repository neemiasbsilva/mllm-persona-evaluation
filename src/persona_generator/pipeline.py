"""Async batch orchestrator for persona generation.

Exposes two public coroutines:

* ``run_synthetic_batch`` — drives the full LangGraph pipeline (Nodes A–E) over
  every CSV row, writing results to ``outputs/personas_synthetic_interview.jsonl``.
  This is the slow path: each persona requires two LLM calls (Node B + Node C).

* ``generate_baseline_jsonl`` — reads the CSV and writes a lightweight JSONL record
  containing only the demographic fields for each persona.  No LLM calls — completes
  in seconds.

* ``run_parallel_batch`` — runs both of the above simultaneously using
  ``asyncio.gather`` so the baseline file is ready while LLM generation runs.

Key design decisions
--------------------
- asyncio.Semaphore caps simultaneous Ollama calls to ``settings.max_concurrent``
  so the local GPU server is never overwhelmed.
- SQLite checkpointing (via LangGraph) means each persona's graph state is
  persisted after every super-step.  If the process crashes at persona 750,
  restarting will skip already-completed threads and resume interrupted ones.
- Completed personas are appended atomically to ``outputs/personas_synthetic_interview.jsonl``.
- Personas that exhaust all retries are written to ``outputs/failed_personas.jsonl``
  so they can be inspected and re-seeded offline.
- tqdm provides a real-time progress bar with a running success/failure count.
"""

import asyncio
import json
import time
from pathlib import Path

import pandas as pd
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from tqdm.asyncio import tqdm

from persona_generator.config import Settings, settings as default_settings
from persona_generator.graph import build_persona_graph
from persona_generator.logging_config import get_logger
from persona_generator.state import PersonaState

log = get_logger(__name__)


def _build_initial_state(row: dict, persona_id: str) -> PersonaState:
    """Construct the initial PersonaState for a single demographics row."""
    return PersonaState(
        persona_id=persona_id,
        raw_demographics={
            "gender": str(row["gender"]),
            "economic_status": str(row["economic_status"]),
            "political_spectrum": str(row["political_spectrum"]),
            "personality": str(row["personality"]),
        },
        synthetic_interview="",
        expert_reflections=[],
        validation_status=False,
        retry_count=0,
        evaluation_feedback="",
        final_system_prompt="",
    )


def _serialize_state(state: PersonaState) -> dict:
    """Convert a final PersonaState to a JSON-serialisable dict for JSONL output."""
    return {
        "persona_id": state["persona_id"],
        "raw_demographics": state["raw_demographics"],
        "synthetic_interview": state["synthetic_interview"],
        "expert_reflections": state["expert_reflections"],
        "validation_status": state["validation_status"],
        "retry_count": state["retry_count"],
        "final_system_prompt": state["final_system_prompt"],
    }


async def _generate_one(
    compiled_graph,
    initial_state: PersonaState,
    semaphore: asyncio.Semaphore,
    success_fh,
    failed_fh,
    cfg: Settings,
) -> bool:
    """Run the graph for a single persona and write the result to the appropriate file.

    Args:
        compiled_graph: The compiled LangGraph StateGraph.
        initial_state:  Pre-populated PersonaState for this persona.
        semaphore:      Concurrency limiter.
        success_fh:     Open file handle for personas_synthetic_interview.jsonl.
        failed_fh:      Open file handle for failed_personas.jsonl.
        cfg:            Settings instance.

    Returns:
        True if the persona was successfully generated, False if discarded.
    """
    persona_id = initial_state["persona_id"]
    thread_config = {"configurable": {"thread_id": persona_id}}

    async with semaphore:
        t0 = time.monotonic()
        try:
            final_state: PersonaState = await compiled_graph.ainvoke(
                initial_state, config=thread_config
            )
        except Exception as exc:
            log.error(
                "pipeline_persona_error",
                persona_id=persona_id,
                error=str(exc),
                exc_info=True,
            )
            record = {
                "persona_id": persona_id,
                "raw_demographics": initial_state["raw_demographics"],
                "error": str(exc),
            }
            failed_fh.write(json.dumps(record) + "\n")
            failed_fh.flush()
            return False

        duration_ms = int((time.monotonic() - t0) * 1000)
        success = bool(final_state.get("final_system_prompt"))

        if success:
            record = _serialize_state(final_state)
            success_fh.write(json.dumps(record) + "\n")
            success_fh.flush()
            log.info(
                "pipeline_persona_success",
                persona_id=persona_id,
                retry_count=final_state["retry_count"],
                duration_ms=duration_ms,
            )
        else:
            # Retries exhausted — final_system_prompt is empty
            record = {
                **_serialize_state(final_state),
                "discarded_reason": "retry_budget_exhausted",
            }
            failed_fh.write(json.dumps(record) + "\n")
            failed_fh.flush()
            log.warning(
                "pipeline_persona_discarded",
                persona_id=persona_id,
                retry_count=final_state["retry_count"],
                duration_ms=duration_ms,
            )

        return success


async def run_synthetic_batch(
    demographics_csv: Path,
    cfg: Settings | None = None,
    limit: int | None = None,
) -> dict:
    """Run the full LangGraph pipeline (Nodes A–E) for every row in the CSV.

    Generates rich Isabella Q&A interview transcripts and expert reflections,
    writing to ``outputs/personas_synthetic_interview.jsonl``.

    Args:
        demographics_csv: Path to the CSV produced by ``data/generate_demographics.py``.
        cfg:              Settings override.  Defaults to the module singleton.
        limit:            If set, only process the first ``limit`` rows (smoke-testing).

    Returns:
        Summary dict: {"total", "success", "failed", "avg_retry", "duration_s"}
    """
    if cfg is None:
        cfg = default_settings

    cfg.ensure_output_dirs()

    df = pd.read_csv(demographics_csv)
    if limit is not None:
        df = df.head(limit)

    # Skip personas already written to the success JSONL (resume support)
    done_ids = _load_done_persona_ids(cfg.personas_synthetic_interview_jsonl)
    if done_ids:
        before = len(df)
        df = df[~df["persona_id"].astype(str).isin(done_ids)]
        log.info(
            "pipeline_synthetic_batch_resume",
            skipped=before - len(df),
            remaining=len(df),
        )

    total = len(df)
    log.info("pipeline_synthetic_batch_start", total=total, csv=str(demographics_csv))

    semaphore = asyncio.Semaphore(cfg.max_concurrent)

    async with AsyncSqliteSaver.from_conn_string(str(cfg.checkpoint_db)) as checkpointer:
        compiled_graph = build_persona_graph(checkpointer=checkpointer)

        t_batch_start = time.monotonic()
        success_count = 0
        failed_count = 0

        with (
            open(cfg.personas_synthetic_interview_jsonl, "a", encoding="utf-8") as success_fh,
            open(cfg.failed_jsonl, "a", encoding="utf-8") as failed_fh,
        ):
            tasks = []
            for _, row in df.iterrows():
                persona_id = str(row["persona_id"])
                initial_state = _build_initial_state(row.to_dict(), persona_id)
                tasks.append(
                    _generate_one(
                        compiled_graph,
                        initial_state,
                        semaphore,
                        success_fh,
                        failed_fh,
                        cfg,
                    )
                )

            results = await tqdm.gather(
                *tasks,
                desc="Synthetic interviews",
                total=total,
                unit="persona",
            )

        for r in results:
            if r:
                success_count += 1
            else:
                failed_count += 1

    batch_duration_s = time.monotonic() - t_batch_start
    avg_retry = _compute_avg_retry(cfg.personas_synthetic_interview_jsonl)

    summary = {
        "total": total,
        "success": success_count,
        "failed": failed_count,
        "avg_retry": round(avg_retry, 3),
        "duration_s": round(batch_duration_s, 1),
    }

    log.info("pipeline_synthetic_batch_complete", **summary)
    return summary


def generate_baseline_jsonl(
    demographics_csv: Path,
    cfg: Settings | None = None,
    limit: int | None = None,
) -> dict:
    """Convert the demographics CSV directly to a JSONL file — no LLM calls.

    Each output record contains only the demographic fields from the CSV, serving
    as the Condition A (baseline flat demographics) data for the annotation experiment.

    Args:
        demographics_csv: Path to the CSV produced by ``data/generate_demographics.py``.
        cfg:              Settings override.  Defaults to the module singleton.
        limit:            If set, only process the first ``limit`` rows.

    Returns:
        Summary dict: {"total", "output_path"}
    """
    if cfg is None:
        cfg = default_settings

    cfg.ensure_output_dirs()

    df = pd.read_csv(demographics_csv)
    if limit is not None:
        df = df.head(limit)

    total = len(df)
    log.info("pipeline_baseline_start", total=total, csv=str(demographics_csv))

    with open(cfg.personas_baseline_jsonl, "w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            record = {
                "persona_id": str(row["persona_id"]),
                "profile_id": str(row["profile_id"]),
                "gender": str(row["gender"]),
                "economic_status": str(row["economic_status"]),
                "political_spectrum": str(row["political_spectrum"]),
                "personality": str(row["personality"]),
            }
            fh.write(json.dumps(record) + "\n")

    log.info(
        "pipeline_baseline_complete",
        total=total,
        output=str(cfg.personas_baseline_jsonl),
    )
    return {"total": total, "output_path": str(cfg.personas_baseline_jsonl)}


async def run_parallel_batch(
    demographics_csv: Path,
    cfg: Settings | None = None,
    limit: int | None = None,
) -> dict:
    """Run baseline JSONL generation and LLM interview synthesis simultaneously.

    The baseline pass (CSV → JSONL, no LLM) completes near-instantly.
    The synthetic pass (LangGraph A→B→C→D→E, two LLM calls per persona) runs
    concurrently so wall-clock time is dominated only by the LLM phase.

    Args:
        demographics_csv: Path to the demographics CSV.
        cfg:              Settings override.
        limit:            Row limit for smoke-testing.

    Returns:
        dict with keys ``"baseline"`` and ``"synthetic"``, each a summary dict.
    """
    if cfg is None:
        cfg = default_settings

    baseline_summary, synthetic_summary = await asyncio.gather(
        # Wrap the sync baseline generator in a thread so it doesn't block the loop
        asyncio.to_thread(generate_baseline_jsonl, demographics_csv, cfg, limit),
        run_synthetic_batch(demographics_csv, cfg, limit),
    )

    return {"baseline": baseline_summary, "synthetic": synthetic_summary}


def _load_done_persona_ids(jsonl_path: Path) -> set[str]:
    """Return the set of persona_ids already written to *jsonl_path*.

    Used by ``run_synthetic_batch`` to skip already-completed personas on
    restart, avoiding redundant LLM calls.
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
                pid = record.get("persona_id")
                if pid:
                    done.add(str(pid))
            except json.JSONDecodeError:
                pass
    return done


def _compute_avg_retry(jsonl_path: Path) -> float:
    """Read the personas JSONL and compute the mean retry_count."""
    if not jsonl_path.exists():
        return 0.0
    retry_counts = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                retry_counts.append(record.get("retry_count", 0))
            except json.JSONDecodeError:
                pass
    return sum(retry_counts) / len(retry_counts) if retry_counts else 0.0
