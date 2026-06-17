"""Inference helpers for full-page OCR and text spotting.

Qwen2.5-VL emits bounding-box coordinates in the pixel space of the image
*after* the processor's smart-resize. To draw boxes on the original image we
rescale them using the actual resized dimensions (derived from
``image_grid_thw``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from PIL import Image

import config
from model_loader import LoadedModel


@dataclass
class TextBox:
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in ORIGINAL image pixels
    text: str


def _build_messages(image: Image.Image, prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _run(loaded: LoadedModel, image: Image.Image, prompt: str):
    """Run a single chat turn. Returns (generated_text, inputs)."""
    import torch
    from qwen_vl_utils import process_vision_info

    processor = loaded.processor
    model = loaded.model

    messages = _build_messages(image, prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False
        )

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text, inputs


def full_page_ocr(loaded: LoadedModel, image: Image.Image) -> str:
    """Return all text recognized in the image as a single string."""
    text, _ = _run(loaded, image, config.FULL_PAGE_OCR_PROMPT)
    return text.strip()


def _resized_dims(inputs) -> tuple[int, int]:
    """Compute (resized_width, resized_height) the model actually saw."""
    grid = inputs["image_grid_thw"][0].tolist()  # [t, h, w] in patches
    _, h, w = grid
    # Each patch is 14px; grid is already in patch units.
    return w * 14, h * 14


def _extract_json_array(text: str):
    """Pull the first JSON array out of the model output, tolerating fences."""
    # Strip ```json ... ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else text

    start = candidate.find("[")
    end = candidate.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = candidate[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def text_spotting(
    loaded: LoadedModel, image: Image.Image
) -> tuple[list[TextBox], str]:
    """Detect text with bounding boxes.

    Returns ``(boxes, raw_output)``. Boxes are rescaled to the ORIGINAL image
    coordinate space so they can be drawn directly on the uploaded image.
    """
    raw, inputs = _run(loaded, image, config.TEXT_SPOTTING_PROMPT)

    parsed = _extract_json_array(raw)
    if not parsed:
        return [], raw

    resized_w, resized_h = _resized_dims(inputs)
    orig_w, orig_h = image.size
    sx = orig_w / resized_w if resized_w else 1.0
    sy = orig_h / resized_h if resized_h else 1.0

    boxes: list[TextBox] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_2d") or item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue

        # Rescale from resized space to original pixels and clamp.
        x1, x2 = sorted((x1 * sx, x2 * sx))
        y1, y2 = sorted((y1 * sy, y2 * sy))
        x1 = max(0, min(orig_w, x1))
        x2 = max(0, min(orig_w, x2))
        y1 = max(0, min(orig_h, y1))
        y2 = max(0, min(orig_h, y2))

        text_content = str(
            item.get("text_content") or item.get("text") or ""
        ).strip()
        boxes.append(
            TextBox(bbox=(int(x1), int(y1), int(x2), int(y2)), text=text_content)
        )

    return boxes, raw
