"""AnnotatorSettings: extends the persona-generator Settings with Phase 2 paths."""

from pathlib import Path

from pydantic_settings import SettingsConfigDict

from persona_generator.config import Settings


class AnnotatorSettings(Settings):
    """All settings for the image annotation pipeline.

    Inherits persona generator settings (text_model, vision_model,
    ollama_base_url, max_concurrent, random_seed, output_dir, etc.)
    and adds annotation-specific fields.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Dataset paths ─────────────────────────────────────────────────────────
    image_dir: Path = Path("/mnt/raid5/neemias/agent-annotator/data/raw_images")
    """Directory containing the 5,000 JPEG images, named {image_id}.jpg."""

    dataset_json: Path = Path("data/perceptsent-raw/dataset.json")
    """perceptsent dataset with human ground-truth annotations (tasks list)."""

    perceptions_vocab: Path = Path("data/unique_perceptions.json")
    """JSON array of 593 closed-vocabulary perception labels."""

    # ── Annotation pipeline control ───────────────────────────────────────────
    max_parse_retries: int = 3
    """Max JSON parse attempts in the vision annotator node before discarding."""

    annotation_max_concurrent: int = 1
    """Number of (persona × image) triples annotated in parallel.

    Must match the OLLAMA_NUM_PARALLEL value set on the Ollama server so the
    pipeline keeps that many GPU inference slots busy at all times.

    Tuning guide  (VRAM budget = total_VRAM − model_weight_size)
    ────────────────────────────────────────────────────────────────
    qwen3-vl:32b Q4_K_M weights ≈ 20 GB

      20 GB GPU  → budget ≈  0 GB raw; with OLLAMA_FLASH_ATTENTION=1 +
                              OLLAMA_KV_CACHE_TYPE=q8_0 free ~2 GB → set 2
      40 GB GPU  → budget ≈ 20 GB                                  → set 4
      95 GB GPU  → budget ≈ 75 GB                                  → set 6–8

    Required Ollama server env vars (set before `ollama serve`):
      OLLAMA_NUM_PARALLEL      — must equal this value
      OLLAMA_FLASH_ATTENTION=1 — reduces VRAM per context slot
      OLLAMA_KV_CACHE_TYPE=q8_0— halves K/V cache VRAM vs f16 default

    Set via ANNOTATION_MAX_CONCURRENT env var."""

    num_ctx: int = 4096
    """Context window (tokens) for each vision model call.

    Smaller values use less VRAM per parallel slot, directly enabling higher
    OLLAMA_NUM_PARALLEL.  Annotation JSON responses are ~200 tokens, so
    2048–4096 is sufficient.  Set via NUM_CTX env var."""

    llm_timeout_s: int = 360
    """Hard timeout (seconds) for a single vision model call.
    Prevents hangs when a small MLLM stalls mid-generation.
    Set via LLM_TIMEOUT_S env var."""

    annotation_num_predict: int = -1
    """Max tokens the vision model may generate per annotation call.
    Capped at 512 — annotation JSON responses are short (~200 tokens).
    Frees the inference slot sooner, improving parallel throughput.
    Set via ANNOTATION_NUM_PREDICT env var."""

    # ── Output paths (override parent properties for annotation outputs) ──────
    @property
    def annotations_baseline_jsonl(self) -> Path:
        return self.output_dir / "annotations_baseline.jsonl"

    @property
    def annotations_full_persona_jsonl(self) -> Path:
        return self.output_dir / "annotations_full_persona.jsonl"

    @property
    def annotation_failures_jsonl(self) -> Path:
        return self.output_dir / "annotation_failures.jsonl"

    def ensure_output_dirs(self) -> None:
        """Create all output directories."""
        super().ensure_output_dirs()
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Module-level singleton
annotator_settings = AnnotatorSettings()
