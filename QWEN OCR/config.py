"""Central configuration for the Qwen 2.5 VL OCR app."""

from __future__ import annotations

import os

# --- Model -----------------------------------------------------------------
# The Hugging Face repo id for the local model weights. Override with the
# QWEN_OCR_MODEL_ID environment variable if you want a different size
# (e.g. "Qwen/Qwen2.5-VL-3B-Instruct").
MODEL_ID: str = os.getenv("QWEN_OCR_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")

# torch dtype used when loading the model. "auto" lets transformers pick.
TORCH_DTYPE: str = os.getenv("QWEN_OCR_DTYPE", "auto")

# Load the model in 4-bit (requires bitsandbytes + CUDA). Saves a lot of VRAM.
LOAD_IN_4BIT: bool = os.getenv("QWEN_OCR_4BIT", "0") == "1"

# Use FlashAttention-2 if installed (CUDA only). Falls back automatically.
USE_FLASH_ATTENTION: bool = os.getenv("QWEN_OCR_FLASH_ATTN", "0") == "1"

# --- Image preprocessing ----------------------------------------------------
# Bounds (in pixels, as multiples of 28) for the processor's smart-resize.
# Larger MAX_PIXELS => more detail but more VRAM/time.
MIN_PIXELS: int = int(os.getenv("QWEN_OCR_MIN_PIXELS", 256 * 28 * 28))
MAX_PIXELS: int = int(os.getenv("QWEN_OCR_MAX_PIXELS", 1280 * 28 * 28))

# --- Generation -------------------------------------------------------------
MAX_NEW_TOKENS: int = int(os.getenv("QWEN_OCR_MAX_NEW_TOKENS", 2048))

# --- Prompts ----------------------------------------------------------------
FULL_PAGE_OCR_PROMPT: str = (
    "Read all the text in this image and output it exactly as it appears, "
    "preserving the reading order, line breaks and layout as closely as "
    "possible. Do not add explanations or commentary."
)

# Text spotting asks the model for line-level boxes in a strict JSON schema so
# we can parse and draw them reliably.
TEXT_SPOTTING_PROMPT: str = (
    "Spotting all the text in the image with line-level granularity. "
    "Return ONLY a JSON array, where each element is an object with the keys "
    '"bbox_2d" (a list of four integers [x1, y1, x2, y2] in pixel coordinates) '
    'and "text_content" (the recognized text for that box). '
    "Do not include any text outside of the JSON array."
)
