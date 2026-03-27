"""Node: image_loader.

Reads a JPEG from the local image directory and returns a base64-encoded
data URI string ready to be embedded in a llava:34b multimodal prompt.
"""

import base64
import time
from pathlib import Path

from annotator.config import annotator_settings
from annotator.state import AnnotationState
from persona_generator.logging_config import get_logger

log = get_logger(__name__)


def image_loader(state: AnnotationState) -> AnnotationState:
    """Read {IMAGE_DIR}/{image_id}.jpg and base64-encode it.

    Args:
        state: Must have ``image_id`` set.

    Returns:
        Updated state with ``image_b64`` populated.

    Raises:
        FileNotFoundError: If the image file does not exist at the expected path.
    """
    image_id = state["image_id"]

    # Fast path: image was pre-loaded by the batch orchestrator's image cache
    if state.get("image_b64"):
        return state

    image_path = Path(annotator_settings.image_dir) / f"{image_id}.jpg"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}. "
            f"Check that IMAGE_DIR ({annotator_settings.image_dir}) is correct."
        )

    t0 = time.monotonic()
    raw_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(raw_bytes).decode("utf-8")
    duration_ms = int((time.monotonic() - t0) * 1000)

    log.debug(
        "image_loader_complete",
        persona_id=state["persona_id"],
        image_id=image_id,
        size_kb=round(len(raw_bytes) / 1024, 1),
        duration_ms=duration_ms,
    )

    return {**state, "image_b64": image_b64}
