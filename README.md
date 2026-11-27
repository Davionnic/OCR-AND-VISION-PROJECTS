# OCR and Vision Projects

A collection of computer-vision and OCR tools. Each project lives in its own folder with its own dependencies and documentation.

## Projects

| Project | Description |
|---------|-------------|
| [OCR LATEX](OCR%20LATEX/README.md) | Convert images of mathematical equations into LaTeX using vision models |
| [LLAMA OCR](LLAMA%20OCR/README.md) | 100% local image-to-Markdown OCR with Llama 3.2 Vision and Streamlit |
| [GEMMA OCR](GEMMA%20OCR/README.md) | 100% local, private structured OCR with Google's Gemma-3 vision model and Streamlit |
| [QWEN OCR](QWEN%20OCR/README.md) | Local OCR and text spotting with Qwen 2.5 VL via Hugging Face Transformers |

## How it works

### OCR LATEX

```mermaid
flowchart TD
  image[Equation image] --> encode[Encode as base64]
  encode --> backend{Backend}
  backend -->|ollama| ollama["Ollama qwen2.5vl:3b"]
  backend -->|openai| openai["OpenAI-compatible API, default gpt-4o-mini"]
  ollama --> clean[clean_latex]
  openai --> clean
  clean --> latex[LaTeX math code]
```

### LLAMA OCR

```mermaid
flowchart TD
  image[Image] --> encode[Encode as base64]
  encode --> ollama["Ollama llama3.2-vision"]
  ollama --> clean[clean_markdown]
  clean --> md[Structured Markdown]
  md --> ui[Streamlit or CLI]
```

### GEMMA OCR

```mermaid
flowchart TD
  image[Image] --> png[Normalize to PNG]
  png --> ollama["Ollama gemma3:4b chat with the image"]
  ollama --> md[Structured Markdown]
  md --> ui[Streamlit]
```

### QWEN OCR

```mermaid
flowchart TD
  image[Image] --> model["Qwen2.5-VL-7B-Instruct via Transformers"]
  model --> mode{Mode}
  mode -->|"Full Page OCR"| text[Extracted page text]
  mode -->|"Text Spotting"| json["JSON boxes and text"]
  json --> draw[Draw bounding boxes]
  text --> ui[Streamlit]
  draw --> ui
```

## Repository layout

```
OCR-AND-VISION-PROJECTS/
├── OCR LATEX/          # LaTeX OCR from equation images
├── LLAMA OCR/          # 100% local image-to-Markdown OCR (Llama 3.2 Vision)
├── GEMMA OCR/          # 100% local structured OCR (Gemma-3 Vision)
├── QWEN OCR/           # Local OCR & text spotting (Qwen 2.5 VL)
└── README.md           # This file
```

Each subproject has its own `README.md`, `requirements.txt`, and `.gitignore`. Install and run from inside the project folder you care about.
