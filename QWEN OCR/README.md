# Qwen 2.5 VL OCR

> **[WORK IN PROGRESS]**

A Streamlit-based application for Optical Character Recognition using the
**Qwen 2.5 VL (Vision-Language)** model, running **locally** via
🤗 Transformers.

## Overview

This application provides a user-friendly interface for extracting and
analyzing text from images using the Qwen 2.5 VL model. It offers two primary
modes of operation:

- **Full Page OCR** — Extract all text content from an uploaded image.
- **Text Spotting** — Detect and visualize text with bounding boxes in the image.

## How it works

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI: upload, mode switching, results & downloads. |
| `config.py` | Central settings (model id, dtype, pixel bounds, prompts). |
| `model_loader.py` | Loads & caches the Qwen2.5-VL model + processor. |
| `inference.py` | Full-page OCR and text-spotting (with bbox rescaling). |
| `drawing.py` | Draws bounding boxes / labels onto the image. |

Qwen2.5-VL returns bounding boxes in the coordinate space of the image *after*
the processor's smart-resize. `inference.py` rescales those coordinates back to
the original image using the `image_grid_thw` grid before drawing.

## Requirements

- Python 3.10+
- A CUDA GPU is strongly recommended for the 7B model. CPU works but is slow.
  - ~16 GB VRAM for fp16, or enable 4-bit quantization to fit in ~8 GB.

## Installation

```bash
# (recommended) create a virtual environment first
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **PyTorch + CUDA:** if you have a GPU, install the matching CUDA build of
> torch from <https://pytorch.org/get-started/locally/> before/along with the
> rest of the requirements.

## Usage

```bash
streamlit run app.py
```

Then:

1. (Optional) Click **Load / warm up model** in the sidebar. The first run
   downloads the weights (~16 GB for the 7B model) from Hugging Face.
2. Upload an image.
3. Pick a mode (**Full Page OCR** or **Text Spotting**).
4. Click **Run OCR**.

## Configuration

All settings can be overridden with environment variables (see `config.py`):

| Variable | Default | Description |
| --- | --- | --- |
| `QWEN_OCR_MODEL_ID` | `Qwen/Qwen2.5-VL-7B-Instruct` | HF model repo. Use `...-3B-...` for less VRAM. |
| `QWEN_OCR_DTYPE` | `auto` | torch dtype (`auto`, `float16`, `bfloat16`). |
| `QWEN_OCR_4BIT` | `0` | Set `1` for 4-bit quantization (needs `bitsandbytes` + CUDA). |
| `QWEN_OCR_FLASH_ATTN` | `0` | Set `1` to use FlashAttention-2 (needs `flash-attn` + CUDA). |
| `QWEN_OCR_MAX_PIXELS` | `1003520` | Max pixels for smart-resize (detail vs. VRAM). |
| `QWEN_OCR_MAX_NEW_TOKENS` | `2048` | Generation length cap. |

Example (PowerShell):

```powershell
$env:QWEN_OCR_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
$env:QWEN_OCR_4BIT = "1"
streamlit run app.py
```

## Notes & limitations

- Text-spotting accuracy depends on image resolution; raise `QWEN_OCR_MAX_PIXELS`
  for dense documents (at the cost of more VRAM/time).
- This is a work in progress — batching, PDF input, and multi-page support are
  planned.
