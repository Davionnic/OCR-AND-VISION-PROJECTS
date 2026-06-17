"""Drawing utilities for visualizing detected text boxes."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from inference import TextBox

# A small palette so adjacent boxes are easier to tell apart.
_PALETTE = [
    (239, 68, 68),
    (34, 197, 94),
    (59, 130, 246),
    (234, 179, 8),
    (168, 85, 247),
    (14, 165, 233),
    (244, 114, 182),
    (249, 115, 22),
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_boxes(
    image: Image.Image,
    boxes: list[TextBox],
    show_labels: bool = True,
    show_index: bool = True,
) -> Image.Image:
    """Return a copy of ``image`` with bounding boxes (and optional labels)."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    line_width = max(2, round(min(canvas.size) / 400))
    font = _load_font(max(12, round(min(canvas.size) / 55)))

    for i, box in enumerate(boxes):
        color = _PALETTE[i % len(_PALETTE)]
        x1, y1, x2, y2 = box.bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = None
        if show_index and show_labels and box.text:
            label = f"{i + 1}. {box.text}"
        elif show_index:
            label = str(i + 1)
        elif show_labels and box.text:
            label = box.text

        if not label:
            continue

        # Truncate long labels so they don't dominate the image.
        if len(label) > 40:
            label = label[:39] + "\u2026"

        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = y1 - th - 4
        if ly < 0:  # not enough room above the box -> draw inside
            ly = y1 + 2
        draw.rectangle([x1, ly, x1 + tw + 6, ly + th + 4], fill=color)
        draw.text((x1 + 3, ly + 2), label, fill=(255, 255, 255), font=font)

    return canvas
