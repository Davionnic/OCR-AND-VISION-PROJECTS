# 🦙 Llama OCR

**100% local OCR** powered by **Llama 3.2 Vision** and **Streamlit**. Drop in an
image (receipt, document, screenshot, handwriting, a table) and get back clean,
structured **Markdown** — headings, lists, and GitHub-flavored tables included.

Everything runs on your own machine through [Ollama](https://ollama.com). No API
keys, no cloud, nothing leaves your computer.

Three ways to use it:

| Interface | File | What it's for |
|-----------|------|----------------|
| Web app   | `app.py` | Drag-and-drop UI with rendered + raw Markdown |
| CLI       | `cli.py` | Convert a file from the terminal |
| Library   | `ocr.py` | `image_to_markdown(...)` for your own code |

---

## 1. Install Python deps

```bash
pip install -r requirements.txt
```

## 2. Set up Ollama (store models on the D drive)

1. Install Ollama from <https://ollama.com/download>.
2. **Put models on D:** (more space than C:). From PowerShell in this folder:

   ```powershell
   ./setup_d_drive.ps1
   ```

   This sets `OLLAMA_MODELS=D:\OllamaModels` for your user and creates the
   folder. Pass a custom path with `./setup_d_drive.ps1 -ModelsPath "D:\AI\Ollama"`.

   > Manual equivalent: `setx OLLAMA_MODELS "D:\OllamaModels"`.

3. **Restart Ollama** (quit it from the system tray, or stop `ollama serve`) so
   it picks up the new model location.
4. **Pull the vision model** (downloads to D: now, ≈ 6–8 GB):

   ```bash
   ollama pull llama3.2-vision
   ```

   > **If `llama3.2-vision` fails to load** (`unknown model architecture:
   > 'mllama'` on recent Ollama 0.30.x — see
   > [ollama#16490](https://github.com/ollama/ollama/issues/16490)), use a
   > currently-supported vision model instead:
   > `ollama pull qwen2.5vl:3b` and set `LLAMA_OCR_MODEL=qwen2.5vl:3b`.

## 3. Run it

### Web app

```bash
streamlit run app.py
```

Open the URL it prints, pick your model in the sidebar, upload an image, and
click **Extract text**. You get a rendered preview, the raw Markdown, and a
**Download .md** button.

### Command line

```bash
python cli.py receipt.jpg
python cli.py page.png --model llama3.2-vision -o page.md
python cli.py invoice.png --prompt "only extract the totals table"
```

### As a library

```python
from ocr import image_to_markdown, Config

cfg = Config(model="llama3.2-vision")
print(image_to_markdown("receipt.png", cfg))
```

---

## How it works

1. The image is base64-encoded and sent to the local Ollama `/api/chat`
   endpoint with a strict OCR system prompt that asks for **Markdown only**.
2. `ocr.py` calls the vision model (`llama3.2-vision` by default) at
   `temperature=0.0` for a faithful transcription.
3. The response is cleaned (`clean_markdown`) to strip any wrapping code fence,
   so the output drops straight into a `.md` file.

## Configuration reference

Set these via environment variables or a `.env` file (copy `.env.example`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `LLAMA_OCR_MODEL` | `llama3.2-vision` | Vision model tag |
| `LLAMA_OCR_TIMEOUT` | `300` | Request timeout (seconds) |
| `LLAMA_OCR_TEMPERATURE` | `0.0` | Sampling temperature |
| `LLAMA_OCR_SYSTEM_PROMPT` | _(built-in)_ | Override the OCR prompt |
| `OLLAMA_MODELS` | _(unset)_ | **Read by Ollama**, not this app — where models are stored (set to a D: path) |

## Troubleshooting

- **"Could not reach Ollama"** — start it with `ollama serve` and confirm
  `http://localhost:11434` is reachable.
- **"Ollama does not have the model"** — run `ollama pull llama3.2-vision`.
- **"unknown model architecture: 'mllama'"** — `llama3.2-vision` is unsupported
  on recent Ollama. Use `qwen2.5vl:3b` and set `LLAMA_OCR_MODEL=qwen2.5vl:3b`.
- **Models still landing on C:** — you must restart Ollama *after* setting
  `OLLAMA_MODELS`; the variable is only read at server startup.
- **First request is slow** — the model loads into memory on the first call;
  later calls are much faster.

## Project layout

```
ocr.py            Core library (Ollama call, prompt, Markdown cleanup)
cli.py            Command-line interface
app.py            Streamlit web UI
setup_d_drive.ps1 Point Ollama's model storage at the D drive
requirements.txt  Python dependencies
.env.example      Configuration template
```
