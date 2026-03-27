"""Structured logging setup using structlog.

Dual output:
  - Console: human-readable colored output for interactive monitoring
  - File:    newline-delimited JSON for post-run bias and failure analysis

Import ``get_logger`` everywhere inside this package.
"""

import logging
import sys
from pathlib import Path

import structlog


def configure_logging(log_file: Path | None = None) -> None:
    """Configure structlog with console + optional JSON file handlers.

    Call once at application startup (in the CLI entry point) before any
    logging occurs.  Subsequent calls to ``get_logger()`` will use this config.

    Args:
        log_file: Path to write JSON-structured log events.  If None, only
                  console output is configured.
    """
    # ── stdlib logging integration ────────────────────────────────────────────
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=handlers,
    )

    # ── Shared processors applied to every log event ──────────────────────────
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # ── Console renderer (pretty, human-readable) ─────────────────────────────
    console_renderer = structlog.dev.ConsoleRenderer(colors=True)

    # ── File renderer (JSON) ──────────────────────────────────────────────────
    # When a file handler is attached we route JSON output there via stdlib.
    if log_file is not None:
        file_processor = structlog.processors.JSONRenderer()
        # Use a custom wrapper that routes JSON to file, pretty to stderr
        structlog.configure(
            processors=shared_processors
            + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Console handler: pretty
        console_handler = handlers[0]
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=console_renderer,
                foreign_pre_chain=shared_processors,
            )
        )

        # File handler: JSON
        file_handler = handlers[1]  # type: ignore[assignment]
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=file_processor,
                foreign_pre_chain=shared_processors,
            )
        )
    else:
        structlog.configure(
            processors=shared_processors
            + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        console_handler = handlers[0]
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=console_renderer,
                foreign_pre_chain=shared_processors,
            )
        )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for the given module name."""
    return structlog.get_logger(name)
