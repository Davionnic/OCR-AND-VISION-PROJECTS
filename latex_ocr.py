"""LaTeX OCR with Llama 3.2 Vision.

Convert an image of a mathematical equation into LaTeX code using a
vision-capable Llama model.

Two backends are supported:

1. ``ollama``  -> runs Llama 3.2 Vision locally through Ollama (default, free).
2. ``openai``  -> any OpenAI-compatible vision API (Groq, Together, OpenRouter,
                  OpenAI, ...). Handy if your machine cannot run the model.

Configuration is read from environment variables (see ``.env.example``) but can
also be passed directly to :func:`image_to_latex`.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path

import requests

# Load variables from a local .env file if python-dotenv is installed.
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


# The instruction we send to the model. Kept strict so the model returns code
# we can drop straight into a LaTeX document instead of a chatty explanation.
PROMPT = (
    "You are an expert OCR engine specialised in mathematical notation. "
    "Look at the image and transcribe every equation it contains into valid "
    "LaTeX math code.\n"
    "Rules:\n"
    "1. Output ONLY the LaTeX code, nothing else.\n"
    "2. Do NOT wrap the code in markdown fences (no ```), and do NOT add "
    "explanations.\n"
    "3. Do NOT include surrounding $ or $$ delimiters.\n"
    "4. Preserve fractions, exponents, subscripts, Greek letters, integrals, "
    "sums and matrices accurately.\n"
    "If the image contains no equation, output an empty line."
)


@dataclass
class Config:
    """Runtime configuration for an OCR request."""

    backend: str = os.getenv("LATEX_OCR_BACKEND", "ollama")
    # Ollama settings
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
    # OpenAI-compatible settings
    api_base: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    api_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Shared
    timeout: int = int(os.getenv("LATEX_OCR_TIMEOUT", "180"))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _read_image_b64(image_path: str | Path) -> str:
    """Return the base64-encoded contents of ``image_path``."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _guess_mime(image_path: str | Path) -> str:
    ext = Path(image_path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/png")


def clean_latex(text: str) -> str:
    """Strip markdown fences and stray math delimiters from model output."""
    text = text.strip()

    # Remove ```latex ... ``` or ``` ... ``` fences.
    fence = re.match(r"^```[a-zA-Z]*\n?(.*?)\n?```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Remove a single pair of surrounding $$ or $ delimiters.
    for delim in ("$$", "$", r"\[", r"\("):
        closing = {"$$": "$$", "$": "$", r"\[": r"\]", r"\(": r"\)"}[delim]
        if text.startswith(delim) and text.endswith(closing) and len(text) > len(delim) + len(closing):
            text = text[len(delim): -len(closing)].strip()
            break

    return text.strip()


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _ocr_with_ollama(image_b64: str, cfg: Config) -> str:
    url = f"{cfg.ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": cfg.ollama_model,
        "messages": [{"role": "user", "content": PROMPT, "images": [image_b64]}],
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(url, json=payload, timeout=cfg.timeout)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {cfg.ollama_host}. "
            "Is it installed and running? Start it with `ollama serve` and pull "
            f"the model with `ollama pull {cfg.ollama_model}`."
        ) from exc

    if resp.status_code == 404:
        raise RuntimeError(
            f"Ollama does not have the model '{cfg.ollama_model}'. "
            f"Pull it first: `ollama pull {cfg.ollama_model}`."
        )
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


def _ocr_with_openai(image_b64: str, mime: str, cfg: Config) -> str:
    if not cfg.api_key:
        raise RuntimeError(
            "No API key set. Export OPENAI_API_KEY (and optionally "
            "OPENAI_BASE_URL / OPENAI_MODEL) to use the cloud backend."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'openai' package is required for the cloud backend. "
            "Install it with `pip install openai`."
        ) from exc

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.api_base)
    resp = client.chat.completions.create(
        model=cfg.api_model,
        temperature=0,
        timeout=cfg.timeout,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                ],
            }
        ],
    )
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def image_to_latex(image_path: str | Path, cfg: Config | None = None) -> str:
    """Convert an equation image to LaTeX code.

    Args:
        image_path: Path to a PNG/JPG/etc. image of an equation.
        cfg: Optional :class:`Config`. Defaults to values from the environment.

    Returns:
        Cleaned LaTeX code as a string.
    """
    cfg = cfg or Config()
    image_b64 = _read_image_b64(image_path)

    if cfg.backend == "ollama":
        raw = _ocr_with_ollama(image_b64, cfg)
    elif cfg.backend == "openai":
        raw = _ocr_with_openai(image_b64, _guess_mime(image_path), cfg)
    else:
        raise ValueError(
            f"Unknown backend '{cfg.backend}'. Use 'ollama' or 'openai'."
        )

    return clean_latex(raw)
