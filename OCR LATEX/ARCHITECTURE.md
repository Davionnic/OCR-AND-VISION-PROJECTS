# LaTeX OCR Project — Architecture & Code Guide

This project turns **photos or screenshots of math equations** into **LaTeX code** using a **vision AI model** (a model that can "see" images and understand them).

---

## Table of Contents

1. [Big Picture Architecture](#1-big-picture-architecture)
2. [Project Files](#2-project-files)
3. [Libraries & Terms Glossary](#3-libraries--terms-glossary)
4. [Core Engine: latex_ocr.py](#4-core-engine-latex_ocrpy)
5. [Web UI: app.py](#5-web-ui-apppy)
6. [CLI: cli.py](#6-cli-clipy)
7. [Sample Generator: make_sample.py](#7-sample-generator-make_samplepy)
8. [Configuration](#8-configuration)
9. [End-to-End Flow](#9-end-to-end-flow)

---

## 1. Big Picture Architecture

Think of the project as **three doors into one engine**:

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACES                          │
├──────────────┬──────────────────┬───────────────────────────┤
│   app.py     │     cli.py       │   Your own Python code    │
│  (Web UI)    │  (Terminal)      │   import latex_ocr        │
└──────┬───────┴────────┬─────────┴─────────────┬─────────────┘
       │                │                       │
       └────────────────┼───────────────────────┘
                        ▼
              ┌─────────────────────┐
              │    latex_ocr.py     │  ← CORE ENGINE
              │  image_to_latex()   │
              └─────────┬───────────┘
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
┌─────────────────┐         ┌─────────────────────┐
│  Ollama backend │         │  OpenAI-compatible  │
│  (local, free)  │         │  backend (cloud API)│
│  qwen2.5vl:3b   │         │  Groq, OpenAI, etc. │
└─────────────────┘         └─────────────────────┘
```

**What happens in plain English:**

1. You give it an **image** of an equation.
2. The image is **encoded** so the AI can read it.
3. A **strict prompt** tells the model: "output only LaTeX, no chat."
4. The model returns text; the code **cleans** it (removes markdown fences, `$` signs, etc.).
5. You get clean LaTeX like `\frac{-b \pm \sqrt{b^2-4ac}}{2a}`.

---

## 2. Project Files

| File | Role |
|------|------|
| `latex_ocr.py` | **Brain** — OCR logic, API calls, config |
| `app.py` | **Web app** — drag-and-drop UI (Streamlit) |
| `cli.py` | **Terminal tool** — convert one image from command line |
| `make_sample.py` | **Test helper** — creates sample equation PNGs |
| `requirements.txt` | Python packages to install |
| `.env.example` | Template for API keys and settings |

---

## 3. Libraries & Terms Glossary

### Python standard library (built-in, no pip install)

| Term | What it is |
|------|------------|
| `from __future__ import annotations` | Lets you write modern type hints (e.g. `str \| Path`) on older Python versions |
| `base64` | Encodes binary image data as text so HTTP APIs can send it |
| `os` | Reads **environment variables** (`OPENAI_API_KEY`, etc.) |
| `re` | **Regular expressions** — pattern matching on text (e.g. strip markdown fences) |
| `dataclass` | A decorator that auto-generates a simple **settings class** (`Config`) |
| `pathlib.Path` | Modern way to work with file paths (`path.read_bytes()`, `path.suffix`) |
| `tempfile` | Creates temporary files (web upload → save to disk briefly) |
| `argparse` | Parses command-line flags like `--backend ollama` |
| `sys` | System interface; here used to print errors to stderr and exit with code `1` |

### Third-party libraries (pip install -r requirements.txt)

| Library | Used in | Purpose |
|---------|---------|---------|
| **requests** | `latex_ocr.py` | HTTP POST to Ollama's `/api/chat` |
| **openai** | `latex_ocr.py` | Official client for OpenAI-style vision APIs |
| **python-dotenv** | `latex_ocr.py` | Loads `.env` file into environment variables |
| **streamlit** (`st`) | `app.py` | Builds the web UI in pure Python |
| **matplotlib** | `make_sample.py` | Renders LaTeX strings into PNG images |

### Domain terms

| Term | Meaning |
|------|---------|
| **LaTeX** | Typesetting language for math: `\frac{a}{b}`, `\int`, `\sum` |
| **OCR** | Optical Character Recognition — reading text/symbols from images |
| **Vision model** | AI that accepts **images + text** (not just text) |
| **Backend** | Which service runs the model: local Ollama vs cloud API |
| **Base64** | Text encoding of binary data; APIs often want `data:image/png;base64,...` |
| **MIME type** | File type label: `image/png`, `image/jpeg` |
| **Prompt** | Instructions sent to the model (`PROMPT` constant) |
| **Temperature** | `0` = deterministic, less creative (good for OCR) |
| **Streamlit** | Framework where each script re-runs top-to-bottom on interaction |

---

## 4. Core Engine: latex_ocr.py

This is the most important file. Everything else calls `image_to_latex()`.

### Module docstring (lines 1–14)

```python
"""LaTeX OCR with Llama 3.2 Vision.
...
"""
```

A **docstring** — documentation string at the top of a file or function. Not executed; describes what the module does.

### Imports (lines 16–24)

```python
from __future__ import annotations
```

Enables postponed evaluation of type hints (Python 3.7+ style).

```python
import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path

import requests
```

- `base64` — encode image bytes for API payloads
- `os.getenv(...)` — read config from environment
- `re` — regex for cleaning model output
- `@dataclass` — defines `Config` with less boilerplate
- `Path` — file paths as objects
- `requests` — HTTP client for Ollama

### Optional .env loading (lines 26–32)

```python
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
```

**Try/except** — if `python-dotenv` is missing or `.env` fails, the app still runs (env vars can be set manually).

`load_dotenv()` reads a `.env` file and sets variables like `OPENAI_API_KEY`.

### The prompt (lines 37–49)

```python
PROMPT = (
    "You are an expert OCR engine specialised in mathematical notation. "
    ...
)
```

A **constant** (ALL_CAPS name) — the exact instructions sent to the vision model.

Rules enforced:

1. Output **only** LaTeX
2. No markdown code fences
3. No `$` or `$$` wrappers
4. Empty line if no equation

This keeps output usable without manual cleanup.

### Config dataclass (lines 52–65)

```python
@dataclass
class Config:
    backend: str = os.getenv("LATEX_OCR_BACKEND", "ollama")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ...
```

| Field | Default | Meaning |
|-------|---------|---------|
| `backend` | `"ollama"` | `"ollama"` or `"openai"` |
| `ollama_host` | `http://localhost:11434` | Where Ollama runs locally |
| `ollama_model` | `qwen2.5vl:3b` | Local vision model name |
| `api_base` | OpenAI URL | Base URL for cloud API |
| `api_key` | from env | Secret key for cloud |
| `api_model` | `gpt-4o-mini` | Cloud model name |
| `timeout` | `180` | Max seconds to wait for AI |

`os.getenv("NAME", "default")` — read env var, or use default if missing.

### Helper: _read_image_b64 (lines 71–76)

```python
def _read_image_b64(image_path: str | Path) -> str:
```

- `def` — defines a function
- `_` prefix — **private** convention (internal helper, not public API)
- `str | Path` — accepts string path or `Path` object
- `-> str` — returns a string

```python
path = Path(image_path)
if not path.is_file():
    raise FileNotFoundError(f"Image not found: {path}")
return base64.b64encode(path.read_bytes()).decode("utf-8")
```

1. Convert to `Path`
2. Check file exists; **raise** error if not
3. `read_bytes()` — raw image bytes
4. `b64encode(...).decode("utf-8")` — base64 text for JSON/API

### Helper: _guess_mime (lines 79–88)

```python
def _guess_mime(image_path: str | Path) -> str:
    ext = Path(image_path).suffix.lower()
    return { ... }.get(ext, "image/png")
```

- `.suffix` — file extension (e.g. `.png`)
- **Dictionary** maps extension → MIME type
- `.get(ext, "image/png")` — default to PNG if unknown

Cloud APIs need MIME type in `data:image/png;base64,...`.

### Helper: clean_latex (lines 91–107)

```python
def clean_latex(text: str) -> str:
    text = text.strip()
```

`strip()` removes leading/trailing whitespace.

```python
fence = re.match(r"^```[a-zA-Z]*\n?(.*?)\n?```$", text, flags=re.DOTALL)
if fence:
    text = fence.group(1).strip()
```

**Regex** removes wrapping markdown code fences around the LaTeX.

- `re.match` — match from start of string
- `(.*?)` — capture group (the LaTeX inside)
- `re.DOTALL` — `.` matches newlines

```python
for delim in ("$$", "$", r"\[", r"\("):
    closing = {"$$": "$$", ...}[delim]
    if text.startswith(delim) and text.endswith(closing):
        text = text[len(delim): -len(closing)].strip()
        break
```

Strips outer math delimiters models sometimes add.

### Backend: _ocr_with_ollama (lines 113–137)

```python
def _ocr_with_ollama(image_b64: str, cfg: Config) -> str:
    url = f"{cfg.ollama_host.rstrip('/')}/api/chat"
```

- **f-string** — embed variables in strings
- `rstrip('/')` — remove trailing slash so URL is clean

```python
payload = {
    "model": cfg.ollama_model,
    "messages": [{"role": "user", "content": PROMPT, "images": [image_b64]}],
    "stream": False,
    "options": {"temperature": 0},
}
```

**JSON payload** for Ollama chat API:

- `messages` — chat format (user message + image)
- `stream: False` — wait for full response
- `temperature: 0` — consistent OCR output

```python
resp = requests.post(url, json=payload, timeout=cfg.timeout)
```

HTTP POST with JSON body.

```python
except requests.exceptions.ConnectionError as exc:
    raise RuntimeError(...) from exc
```

Friendly error if Ollama isn't running.

```python
if resp.status_code == 404:
    raise RuntimeError(...)  # model not pulled
resp.raise_for_status()
data = resp.json()
return data.get("message", {}).get("content", "")
```

- `404` — model missing → tell user to `ollama pull`
- `raise_for_status()` — error on other HTTP failures
- `.get("message", {})` — safe nested dict access (no KeyError)

### Backend: _ocr_with_openai (lines 140–172)

```python
if not cfg.api_key:
    raise RuntimeError("No API key set...")
```

Cloud backend requires a key.

```python
from openai import OpenAI
```

**Lazy import** — only loads `openai` when cloud backend is used.

```python
client = OpenAI(api_key=cfg.api_key, base_url=cfg.api_base)
```

Works with OpenAI, Groq, Together, OpenRouter (same API shape).

```python
resp = client.chat.completions.create(
    model=cfg.api_model,
    temperature=0,
    timeout=cfg.timeout,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ],
    }],
)
return resp.choices[0].message.content or ""
```

OpenAI vision format: text + image URL (here a **data URL** with embedded base64).

### Public API: image_to_latex (lines 178–200)

```python
def image_to_latex(image_path: str | Path, cfg: Config | None = None) -> str:
```

**Main entry point** — imported by `app.py`, `cli.py`, and your code.

```python
cfg = cfg or Config()
image_b64 = _read_image_b64(image_path)

if cfg.backend == "ollama":
    raw = _ocr_with_ollama(image_b64, cfg)
elif cfg.backend == "openai":
    raw = _ocr_with_openai(image_b64, _guess_mime(image_path), cfg)
else:
    raise ValueError(...)

return clean_latex(raw)
```

1. Use passed `Config` or create default
2. Encode image
3. Route to correct backend
4. Clean and return LaTeX string

---

## 5. Web UI: app.py

Streamlit app: upload image → convert → preview + download.

### Setup (lines 7–19)

```python
from latex_ocr import Config, image_to_latex
import streamlit as st

st.set_page_config(page_title="LaTeX OCR with Llama", page_icon="∑", layout="wide")
st.title("∑  LaTeX OCR with Llama 3.2 Vision")
st.caption("Upload a picture...")
```

- Imports core function and config
- `st.set_page_config` — browser tab title, icon, wide layout
- `st.title` / `st.caption` — heading and subtitle

### Sidebar settings (lines 24–52)

```python
with st.sidebar:
    st.header("Settings")
    backend = st.radio("Backend", options=["ollama", "openai"], ...)
```

`with st.sidebar:` — widgets inside appear in the left panel.

```python
cfg = Config()
cfg.backend = backend
```

Create config, then override from UI.

```python
if backend == "ollama":
    cfg.ollama_host = st.text_input("Ollama host", value=cfg.ollama_host)
    ...
else:
    cfg.api_base = st.text_input("API base URL", ...)
    key_input = st.text_input("API key", type="password", ...)
    if key_input:
        cfg.api_key = key_input
```

- `st.text_input` — text box; `type="password"` hides API key
- Only overwrites `api_key` if user typed something (else env var still works)

### Main area (lines 58–106)

```python
uploaded = st.file_uploader("Equation image", type=["png", "jpg", ...])
col_img, col_out = st.columns(2)
```

- `file_uploader` — drag-and-drop file
- `st.columns(2)` — two-column layout (input | output)

```python
if uploaded is not None:
    with col_img:
        st.image(uploaded, use_container_width=True)
```

Show uploaded image in left column.

```python
if st.button("Convert to LaTeX", type="primary"):
```

Button click triggers conversion (`type="primary"` = highlighted).

```python
suffix = Path(uploaded.name).suffix or ".png"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getvalue())
    tmp_path = tmp.name
```

Streamlit holds upload **in memory**; `image_to_latex` needs a **file path**:

1. Get extension from filename
2. Write bytes to a temp file
3. `delete=False` — file stays until we delete it

```python
with st.spinner("Asking Llama..."):
    try:
        latex = image_to_latex(tmp_path, cfg)
    except Exception as exc:
        st.error(str(exc))
        latex = None
    finally:
        Path(tmp_path).unlink(missing_ok=True)
```

- `st.spinner` — loading animation
- `try/except` — show errors in UI
- `finally` — always delete temp file (`unlink`, `missing_ok=True` = no error if gone)

```python
if latex is not None:
    if latex.strip():
        st.latex(latex)           # rendered math preview
        st.code(latex, language="latex")  # copyable code block
        st.download_button(...)   # save as .tex file
    else:
        st.warning("The model did not find an equation...")
```

- `st.latex` — renders LaTeX in the page
- `st.code` — monospace code block
- `st.download_button` — browser download

```python
else:
    st.info("Upload an image to get started...")
```

Shown when nothing uploaded yet.

---

## 6. CLI: cli.py

Terminal interface for scripts and automation.

```python
def main(argv: list[str] | None = None) -> int:
```

- `argv` — optional argument list (useful for testing)
- Returns `int` exit code: `0` = success, `1` = error

```python
parser = argparse.ArgumentParser(description="...")
parser.add_argument("image", help="Path to the equation image...")
parser.add_argument("--backend", choices=["ollama", "openai"], ...)
parser.add_argument("--model", help="Override the model name.")
parser.add_argument("--host", help="Ollama host URL...")
parser.add_argument("-o", "--output", help="Write LaTeX to file...")
args = parser.parse_args(argv)
```

**argparse** defines CLI:

- Positional: `image` path
- Optional: `--backend`, `--model`, `--host`, `-o` output file

```python
cfg = Config()
if args.backend:
    cfg.backend = args.backend
...
```

Merge CLI flags into `Config`.

```python
try:
    latex = image_to_latex(args.image, cfg)
except (RuntimeError, FileNotFoundError, ValueError) as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return 1
```

Catch expected errors; print to **stderr**; exit `1`.

```python
if args.output:
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(latex + "\n")
else:
    print(latex)
return 0
```

Write file or print to stdout.

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

Only runs `main()` when you execute `python cli.py` (not when imported).

---

## 7. Sample Generator: make_sample.py

Creates test images so you don't need real photos.

```python
SAMPLES = [
    ("quadratic.png", r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"),
    ...
]
```

List of `(filename, latex_string)` pairs. `r"..."` = **raw string** (backslashes literal).

```python
def render(latex: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(6, 1.6))
    fig.text(0.5, 0.5, f"${latex}$", fontsize=28, ha="center", va="center")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
```

- `matplotlib` figure with equation centered
- `f"${latex}$"` — mathtext needs `$...$`
- `savefig` — PNG at 200 DPI
- `plt.close(fig)` — free memory

```python
out_dir = Path(__file__).parent / "examples"
out_dir.mkdir(exist_ok=True)
```

- `__file__` — path to this script
- `.parent / "examples"` — `examples/` next to script
- `mkdir(exist_ok=True)` — create folder if missing

---

## 8. Configuration

Settings come from (in order of override):

1. **Defaults** in `Config` class
2. **Environment variables** / `.env` file
3. **UI** (`app.py` sidebar) or **CLI flags** (`cli.py`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LATEX_OCR_BACKEND` | `ollama` | Local vs cloud |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen2.5vl:3b` | Local model |
| `OPENAI_BASE_URL` | OpenAI v1 URL | API endpoint |
| `OPENAI_MODEL` | `gpt-4o-mini` | Cloud model |
| `OPENAI_API_KEY` | empty | API secret |
| `LATEX_OCR_TIMEOUT` | `180` | Request timeout (seconds) |

---

## 9. End-to-End Flow

### Web app flow

```
User uploads PNG
       ↓
app.py saves to temp file
       ↓
image_to_latex(path, cfg)
       ↓
_read_image_b64()  →  base64 string
       ↓
_ocr_with_ollama() OR _ocr_with_openai()
       ↓
Vision model reads image + PROMPT
       ↓
Raw text response
       ↓
clean_latex()  →  strip fences and $ delimiters
       ↓
app.py shows st.latex() preview + st.code() + download
```

### Data flow

```
Image file → base64 encode → backend choice
  → ollama: POST /api/chat
  → openai: chat.completions.create
  → raw model text → clean_latex → LaTeX string
```

---

## Quick Reference: Who Calls What

| You run | It calls | Result |
|---------|----------|--------|
| `streamlit run app.py` | `image_to_latex()` | Web UI with preview |
| `python cli.py img.png` | `image_to_latex()` | LaTeX in terminal |
| `python make_sample.py` | matplotlib only | PNGs in `examples/` |
| `from latex_ocr import image_to_latex` | Ollama or OpenAI | LaTeX string in your code |

---

## Mental Model (one sentence)

**Upload an equation image → encode it → ask a vision AI "write this as LaTeX only" → clean the answer → use the LaTeX in documents or the web preview.**
