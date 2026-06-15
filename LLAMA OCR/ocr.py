"""Core OCR library: turn an image into structured Markdown, 100% locally.

Uses a vision-capable Llama model served by a local Ollama instance. No data
ever leaves your machine.

The public surface is small:

    cfg = Config()
    markdown = image_to_markdown("receipt.png", cfg)

Everything is driven by `Config`, which reads sensible defaults from environment
variables / a `.env` file.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path

import requests

try:  # Optional: load a local .env if python-dotenv is installed.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience, not required.
    pass


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
# A faithful, OCR-focused prompt: get *everything* on the page back as clean
# Markdown, with no commentary from the model.
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise OCR engine. Convert the provided image into clean, "
    "well-structured Markdown.\n\n"
    "Rules:\n"
    "- Transcribe ALL visible text: headings, paragraphs, lists, footers, "
    "captions, page numbers, and any small print.\n"
    "- Preserve the reading order and structure of the document.\n"
    "- Use real Markdown: '#' headings, '-' bullet lists, '1.' numbered lists, "
    "and GitHub-flavored '|' tables for tabular data.\n"
    "- Keep numbers, dates, currency, and symbols exactly as written.\n"
    "- For images/figures, write a short '![alt text](image)' description.\n"
    "- Do NOT add explanations, summaries, or commentary of your own.\n"
    "- Output ONLY the Markdown for the page content. No code fences around the "
    "whole answer."
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Config:
    """Runtime configuration for the local OCR pipeline."""

    ollama_host: str = field(
        default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434")
    )
    model: str = field(
        default_factory=lambda: _env("LLAMA_OCR_MODEL", "llama3.2-vision")
    )
    system_prompt: str = field(
        default_factory=lambda: _env("LLAMA_OCR_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    )
    # Vision models can be slow to load on the first call; give them room.
    timeout: int = field(default_factory=lambda: int(_env("LLAMA_OCR_TIMEOUT", "300")))
    # 0.0 keeps the transcription faithful (low creativity).
    temperature: float = field(
        default_factory=lambda: float(_env("LLAMA_OCR_TEMPERATURE", "0.0"))
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _encode_image(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def clean_markdown(text: str) -> str:
    """Strip an outer ```markdown ... ``` fence the model sometimes adds."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop the opening fence (``` or ```markdown).
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # Drop the closing fence.
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def image_to_markdown(
    image_path: str | Path,
    cfg: Config | None = None,
    *,
    prompt: str | None = None,
) -> str:
    """Transcribe an image to Markdown using a local Ollama vision model.

    Args:
        image_path: Path to the image file.
        cfg: Optional configuration. A default `Config()` is used if omitted.
        prompt: Optional extra user instruction (e.g. "only the table").

    Returns:
        The transcribed content as a Markdown string.
    """
    cfg = cfg or Config()
    image_b64 = _encode_image(image_path)

    user_text = prompt or "Convert this page to Markdown."

    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_text, "images": [image_b64]},
        ],
        "stream": False,
        "options": {"temperature": cfg.temperature},
    }

    url = f"{cfg.ollama_host.rstrip('/')}/api/chat"

    try:
        resp = requests.post(url, json=payload, timeout=cfg.timeout)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {cfg.ollama_host}. Is it running? "
            "Start it with `ollama serve`."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Ollama timed out after {cfg.timeout}s. The model may still be "
            "loading; try again, or raise LLAMA_OCR_TIMEOUT."
        ) from exc

    if resp.status_code == 404:
        raise RuntimeError(
            f"Ollama does not have the model '{cfg.model}'. Pull it with:\n"
            f"    ollama pull {cfg.model}"
        )
    resp.raise_for_status()

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return clean_markdown(content)


def list_models(cfg: Config | None = None) -> list[str]:
    """Return the vision-capable model tags installed in the local Ollama."""
    cfg = cfg or Config()
    url = f"{cfg.ollama_host.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return []
    return [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
