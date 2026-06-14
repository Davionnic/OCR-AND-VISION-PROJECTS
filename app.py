"""Streamlit web UI for LaTeX OCR with Llama 3.2 Vision.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from latex_ocr import Config, image_to_latex

st.set_page_config(page_title="LaTeX OCR with Llama", page_icon="∑", layout="wide")

st.title("∑  LaTeX OCR with Llama 3.2 Vision")
st.caption("Upload a picture of an equation and get clean LaTeX code back.")

# --------------------------------------------------------------------------- #
# Sidebar: backend configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    backend = st.radio(
        "Backend",
        options=["ollama", "openai"],
        help="`ollama` runs locally and is free. `openai` uses any "
        "OpenAI-compatible vision API (Groq, Together, OpenRouter, OpenAI).",
    )

    cfg = Config()
    cfg.backend = backend

    if backend == "ollama":
        cfg.ollama_host = st.text_input("Ollama host", value=cfg.ollama_host)
        cfg.ollama_model = st.text_input("Model", value=cfg.ollama_model)
        st.info(
            "Make sure Ollama is running and the model is pulled:\n\n"
            f"`ollama pull {cfg.ollama_model}`",
            icon="💡",
        )
    else:
        cfg.api_base = st.text_input("API base URL", value=cfg.api_base)
        cfg.api_model = st.text_input("Model", value=cfg.api_model)
        key_input = st.text_input(
            "API key", value=cfg.api_key, type="password",
            help="Leave blank to use the OPENAI_API_KEY environment variable.",
        )
        if key_input:
            cfg.api_key = key_input


# --------------------------------------------------------------------------- #
# Main: upload + convert
# --------------------------------------------------------------------------- #
uploaded = st.file_uploader(
    "Equation image",
    type=["png", "jpg", "jpeg", "webp", "bmp", "gif"],
)

col_img, col_out = st.columns(2)

if uploaded is not None:
    with col_img:
        st.subheader("Input")
        st.image(uploaded, use_container_width=True)

    if st.button("Convert to LaTeX", type="primary"):
        suffix = Path(uploaded.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        with col_out:
            st.subheader("Output")
            with st.spinner("Asking Llama to read the equation..."):
                try:
                    latex = image_to_latex(tmp_path, cfg)
                except Exception as exc:  # noqa: BLE001 - surface any error to UI
                    st.error(str(exc))
                    latex = None
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

            if latex is not None:
                if latex.strip():
                    st.markdown("**Rendered preview**")
                    try:
                        st.latex(latex)
                    except Exception:
                        st.warning("Could not render this LaTeX, but here is the code:")
                    st.markdown("**LaTeX code**")
                    st.code(latex, language="latex")
                    st.download_button(
                        "Download .tex",
                        data=latex + "\n",
                        file_name="equation.tex",
                        mime="text/plain",
                    )
                else:
                    st.warning("The model did not find an equation in the image.")
else:
    st.info("Upload an image to get started, or generate one with "
            "`python make_sample.py`.")
