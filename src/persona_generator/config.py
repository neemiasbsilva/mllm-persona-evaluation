"""Application-wide settings loaded from environment variables / .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunable parameters for the persona generation pipeline.

    Values are read from environment variables (case-insensitive) or a .env
    file in the working directory.  Defaults are sensible for a single-machine
    research run against a local Ollama instance.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM models ────────────────────────────────────────────────────────────
    text_model: str = "llama3.3:70b"
    """Ollama model used for Nodes B (backstory) and C (expert reflection)."""

    # vision_model: str = "qwen3-vl:32b"
    vision_model: str = "qwen3-vl:8b"
    """Ollama model reference for the downstream vision annotation task (Node E output)."""

    ollama_base_url: str = "http://localhost:11434"
    """Base URL of the local Ollama server."""

    # ── Generation parameters ─────────────────────────────────────────────────
    temperature: float = 0.7
    """Sampling temperature for LLM calls.  Higher = more creative backstories."""

    random_seed: int = 42
    """Global seed for numpy (demographics sampler) and Ollama num_seed option."""

    # ── Pipeline control ──────────────────────────────────────────────────────
    max_concurrent: int = 5
    """Maximum number of persona graph invocations running simultaneously.
    Keep low (3-10) for a single local Ollama instance."""

    max_retries: int = 3
    """Maximum Node B re-invocations per persona before the seed is discarded."""

    # ── Paths ─────────────────────────────────────────────────────────────────
    output_dir: Path = Path("outputs")
    """Root directory for all runtime artifacts (JSONL, logs, checkpoints)."""

    @property
    def personas_baseline_jsonl(self) -> Path:
        """JSONL file containing one record per persona with demographic fields only.
        Generated instantly from the CSV with no LLM calls."""
        return self.output_dir / "personas_baseline.jsonl"

    @property
    def personas_synthetic_interview_jsonl(self) -> Path:
        """JSONL file containing one record per persona including the full
        Isabella Q&A interview transcript and compiled system prompt."""
        return self.output_dir / "personas_synthetic_interview.jsonl"

    @property
    def failed_jsonl(self) -> Path:
        return self.output_dir / "failed_personas.jsonl"

    @property
    def log_file(self) -> Path:
        return self.output_dir / "run.log"

    @property
    def checkpoint_db(self) -> Path:
        return self.output_dir / "checkpoints" / "personas.db"

    def ensure_output_dirs(self) -> None:
        """Create output directories if they don't already exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)


# Module-level singleton — import this everywhere instead of instantiating Settings()
settings = Settings()
