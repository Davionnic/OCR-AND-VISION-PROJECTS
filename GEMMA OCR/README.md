# 🔎 Gemma-3 OCR

Local, **private** OCR with **structured text extraction**, powered by Google's
[Gemma-3](https://ai.google.dev/gemma) vision model running on
[Ollama](https://ollama.com). Upload an image and get back clean, structured
Markdown — headings, lists, and tables are preserved. Nothing ever leaves your
machine.

![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Ollama](https://img.shields.io/badge/Runtime-Ollama-000000)
![Gemma-3](https://img.shields.io/badge/Model-Gemma--3-4285F4)

## ✨ Features

- **100% local & private** — runs against a local Ollama server, no API keys.
- **Structured output** — returns Markdown with headings, lists and tables.
- **Live preview** — rendered Markdown + raw Markdown side by side.
- **One-click download** — save the extracted text as a `.md` file.
- **Model picker** — switch between `gemma3:4b`, `12b`, `27b`, etc.
- **Editable prompt** — tune the extraction instructions from the sidebar.

## 📋 Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running — https://ollama.com/download

## 🚀 Setup

### 1. Install and start Ollama, then pull the model

```bash
# In one terminal (keeps the local server running)
ollama serve

# In another terminal — pull a Gemma-3 vision model
ollama pull gemma3:4b
```

> Larger models = better accuracy on tricky documents:
> `ollama pull gemma3:12b` or `ollama pull gemma3:27b`.

### 2. Install Python dependencies

```bash
# Optional but recommended: a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Upload an image and click
**Extract text**.

## 🗂️ Project structure

```
.
├── app.py            # Streamlit UI
├── ocr_engine.py     # Ollama / Gemma-3 OCR logic
├── requirements.txt  # Python dependencies
└── README.md
```

## 🧠 How it works

1. The uploaded image is normalized to PNG and base64-encoded.
2. It's sent to the local Ollama server together with a structured-OCR prompt
   (`STRUCTURED_PROMPT` in `ocr_engine.py`) at a low temperature for
   deterministic output.
3. Gemma-3 reads the image and returns Markdown, which the UI renders and lets
   you download.

## 🛠️ Troubleshooting

| Problem | Fix |
| --- | --- |
| "Ollama not reachable" | Run `ollama serve` and reload the page. |
| Model not in the dropdown | Run `ollama pull gemma3:4b`. |
| Slow on first run | The model is loading into memory; subsequent runs are faster. |
| Poor accuracy | Try a larger model (`gemma3:12b` / `27b`) or a higher-resolution image. |

## 📄 License

MIT — do what you like.
