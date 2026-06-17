"""Loading and caching of the Qwen 2.5 VL model and processor.

The model is heavy to load, so we cache it as a Streamlit resource. Importing
this module outside of Streamlit also works (the cache decorator degrades to a
plain function) so it can be reused from scripts/tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import config


def _cache_resource(func):
    """Use st.cache_resource when running under Streamlit, else passthrough."""
    try:
        import streamlit as st

        return st.cache_resource(show_spinner=False)(func)
    except Exception:  # pragma: no cover - non-streamlit usage
        return func


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    device: str
    model_id: str


def _resolve_dtype(torch, dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    return getattr(torch, dtype_str)


@_cache_resource
def load_model(
    model_id: str = config.MODEL_ID,
    dtype_str: str = config.TORCH_DTYPE,
    load_in_4bit: bool = config.LOAD_IN_4BIT,
    use_flash_attention: bool = config.USE_FLASH_ATTENTION,
    min_pixels: int = config.MIN_PIXELS,
    max_pixels: int = config.MAX_PIXELS,
) -> LoadedModel:
    """Load the Qwen2.5-VL model + processor.

    Returns a ``LoadedModel`` holding both objects plus some metadata.
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"

    model_kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_dtype(torch, dtype_str),
        "device_map": "auto" if has_cuda else None,
    }

    if use_flash_attention and has_cuda:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if load_in_4bit and has_cuda:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, **model_kwargs
    )

    if not has_cuda:
        model = model.to(device)
    model.eval()

    # The processor governs how images are resized into vision tokens.
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    return LoadedModel(
        model=model, processor=processor, device=device, model_id=model_id
    )
