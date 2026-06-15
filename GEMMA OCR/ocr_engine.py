"""Local OCR engine backed by the Gemma-3 vision model via Ollama.

Everything runs on your machine: the image is sent to a locally running
Ollama server, no data leaves the device.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import ollama
from PIL import Image

DEFAULT_MODEL = "gemma3:4b"

# Prompt tuned to return clean, structured Markdown rather than chatty prose.
STRUCTURED_PROMPT = """You are an expert OCR and document-understanding engine.

Analyze the provided image and extract ALL readable text.

Rules:
- Return the result as clean, well-structured Markdown.
- Preserve the document's logical structure: use headings (#, ##), lists,
  and Markdown tables where the layout implies them.
- Keep numbers, dates, currencies, emails and codes exactly as written.
- Do NOT add commentary, explanations, or any text that is not in the image.
- If a region is unreadable, write [illegible] in its place.
- If the image contains no text, reply with exactly: (no text found)
"""


@dataclass
class OCRResult:
    """Outcome of an OCR run."""

    text: str
    model: str


def list_local_models() -> list[str]:
    """Return the names of models currently pulled into the local Ollama."""
    try:
        response = ollama.list()
    except Exception:  # noqa: BLE001 - surfaced to the UI layer
        return []

    models: list[str] = []
    for item in response.get("models", []):
        name = item.get("model") or item.get("name")
        if name:
            models.append(name)
    return models


def is_ollama_available() -> bool:
    """Check whether a local Ollama server is reachable."""
    try:
        ollama.list()
        return True
    except Exception:  # noqa: BLE001
        return False


def _image_to_png_bytes(image: Image.Image) -> bytes:
    """Normalize any PIL image into PNG bytes Ollama can ingest."""
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def extract_text(
    image: Image.Image,
    model: str = DEFAULT_MODEL,
    prompt: str = STRUCTURED_PROMPT,
) -> OCRResult:
    """Run structured OCR on a PIL image and return Markdown text.

    Raises:
        RuntimeError: if the Ollama call fails (server down, model missing, ...).
    """
    png_bytes = _image_to_png_bytes(image)
    encoded = base64.b64encode(png_bytes).decode("utf-8")

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encoded],
                }
            ],
            options={"temperature": 0.1},
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(str(exc)) from exc

    content = response.get("message", {}).get("content", "").strip()
    return OCRResult(text=content, model=model)
