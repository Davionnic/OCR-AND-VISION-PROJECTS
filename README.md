# LaTeX OCR with Llama 3.2 Vision

Convert images of mathematical equations into ready-to-use **LaTeX code** using
a vision-capable Llama model. Snap a photo of a formula (or screenshot one from
a PDF) and get back code like:

```latex
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
```

It ships with three ways to use it:

| Interface | File | What it's for |
|-----------|------|----------------|
| Web app   | `app.py` | Drag-and-drop UI with a rendered preview |
| CLI       | `cli.py` | Convert a file from the terminal |
| Library   | `latex_ocr.py` | `image_to_latex(...)` for your own code |

There are **two backends** so it works on any machine:

- **`ollama`** (default) — runs a vision model **locally and free**.
- **`openai`** — any OpenAI-compatible vision API (OpenAI, Groq, Together,
  OpenRouter). Great if your computer can't run the model locally.

> **Heads up about Llama 3.2 Vision:** recent Ollama versions (0.30.x) dropped
> support for the `mllama` architecture, so `llama3.2-vision` currently fails to
> load with `unknown model architecture: 'mllama'`
> (see [ollama#16490](https://github.com/ollama/ollama/issues/16490)). The
> default local model is therefore **`qwen2.5vl:3b`**, which is excellent at OCR,
> runs well on CPU, and works with current Ollama. Use `qwen2.5vl:7b` for higher
> accuracy if you have the RAM/GPU. To use real Llama 3.2 Vision, roll Ollama
> back to a version that supports it (~0.24.0) and set
> `OLLAMA_MODEL=llama3.2-vision`.

---

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Pick a backend

### Option A — Local with Ollama (recommended, free)

1. Install Ollama from <https://ollama.com/download>.
2. Pull the vision model (≈ 6 GB download, needs ~8 GB RAM/VRAM):

   ```bash
   ollama pull qwen2.5vl:3b
   ```

   > **Low on C: disk space?** Ollama stores models on `C:` by default. To put
   > them on another drive, set `OLLAMA_MODELS` before starting Ollama, e.g.
   > `setx OLLAMA_MODELS "D:\OllamaModels"`, then restart Ollama.

3. Make sure Ollama is running (`ollama serve`, or it starts automatically on
   most installs). That's it — the defaults already point at it.

### Option B — Cloud API (no local GPU needed)

Copy `.env.example` to `.env` and fill in your provider. Example for Groq
(fast and has a free tier):

```env
LATEX_OCR_BACKEND=openai
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.2-90b-vision-preview
OPENAI_API_KEY=your_key_here
```

## 3. Make a test image (optional)

No equation image handy? Generate a few:

```bash
python make_sample.py
```

This writes sample PNGs into `examples/`.

## 4. Run it

### Web app

```bash
streamlit run app.py
```

Then open the URL it prints, choose your backend in the sidebar, upload an
image, and click **Convert to LaTeX**. You get a live rendered preview plus the
copyable code.

### Command line

```bash
python cli.py examples/quadratic.png
```

Useful flags:

```bash
python cli.py photo.jpg --backend ollama --model llama3.2-vision
python cli.py photo.jpg --backend openai --model gpt-4o-mini -o equation.tex
```

### As a library

```python
from latex_ocr import image_to_latex, Config

cfg = Config(backend="ollama", ollama_model="llama3.2-vision")
print(image_to_latex("examples/euler.png", cfg))
```

---

## How it works

1. The image is base64-encoded and sent to the model with a strict prompt that
   asks for **LaTeX code only**.
2. `latex_ocr.py` calls either the Ollama `/api/chat` endpoint or an
   OpenAI-compatible `chat/completions` endpoint.
3. The response is cleaned (`clean_latex`) to strip markdown fences and stray
   `$`/`$$` delimiters so the output drops straight into a document.

## Configuration reference

All settings can come from environment variables / `.env` (see
`.env.example`) or be set on a `Config` object:

| Variable | Default | Meaning |
|----------|---------|---------|
| `LATEX_OCR_BACKEND` | `ollama` | `ollama` or `openai` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5vl:3b` | Local model name |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Cloud API base URL |
| `OPENAI_MODEL` | `gpt-4o-mini` | Cloud model name |
| `OPENAI_API_KEY` | _(empty)_ | Cloud API key |
| `LATEX_OCR_TIMEOUT` | `180` | Request timeout (seconds) |

## Troubleshooting

- **"Could not reach Ollama"** — start it with `ollama serve` and confirm
  `http://localhost:11434` is reachable.
- **"Ollama does not have the model"** — run `ollama pull qwen2.5vl:3b`.
- **"unknown model architecture: 'mllama'"** — this is `llama3.2-vision` being
  unsupported in recent Ollama. Use `qwen2.5vl` (default) or roll Ollama back.
- **First request is slow** — the model loads into memory on the first call;
  later calls are faster.
- **Wrong/garbled LaTeX** — use a clearer, higher-resolution, well-cropped
  image. Vision OCR struggles with blurry or skewed photos.

## Project layout

```
latex_ocr.py     Core library (backends, prompt, output cleaning)
cli.py           Command-line interface
app.py           Streamlit web UI
make_sample.py   Generates sample equation images
requirements.txt Dependencies
.env.example     Configuration template
```
