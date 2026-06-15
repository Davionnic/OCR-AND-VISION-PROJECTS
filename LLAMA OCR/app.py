"""Streamlit web UI for Llama OCR - 100% local OCR with Llama 3.2 Vision.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from ocr import Config, image_to_markdown, list_models

st.set_page_config(page_title="Llama OCR", page_icon="🦙", layout="wide")

st.title("🦙 Llama OCR")
st.caption("100% local OCR with Llama 3.2 Vision and Streamlit. Nothing leaves your machine.")

# --------------------------------------------------------------------------- #
# Sidebar: configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")

    cfg = Config()
    cfg.ollama_host = st.text_input("Ollama host", value=cfg.ollama_host)

    installed = list_models(cfg)
    if installed:
        default_idx = installed.index(cfg.model) if cfg.model in installed else 0
        cfg.model = st.selectbox("Model", options=installed, index=default_idx)
    else:
        cfg.model = st.text_input("Model", value=cfg.model)
        st.warning(
            "Couldn't list local models. Make sure Ollama is running and the "
            f"vision model is pulled:\n\n`ollama pull {cfg.model}`",
            icon="⚠️",
        )

    with st.expander("Advanced"):
        cfg.temperature = st.slider("Temperature", 0.0, 1.0, cfg.temperature, 0.05)
        cfg.timeout = st.number_input(
            "Timeout (s)", min_value=30, max_value=1800, value=cfg.timeout, step=30
        )
        cfg.system_prompt = st.text_area(
            "System prompt", value=cfg.system_prompt, height=200
        )

    st.divider()
    st.markdown(
        "**Model storage on D:**\n\n"
        "Run `setup_d_drive.ps1` once so Ollama keeps models on the D drive "
        "(`OLLAMA_MODELS`), then `ollama pull llama3.2-vision`."
    )


# --------------------------------------------------------------------------- #
# Main: upload + convert
# --------------------------------------------------------------------------- #
uploaded = st.file_uploader(
    "Image to read",
    type=["png", "jpg", "jpeg", "webp", "bmp", "gif", "tiff"],
)

extra_prompt = st.text_input(
    "Optional instruction",
    placeholder="e.g. 'only extract the table' or leave blank for the whole page",
)

col_img, col_out = st.columns(2)

if uploaded is not None:
    with col_img:
        st.subheader("Input")
        st.image(uploaded, use_container_width=True)

    if st.button("Extract text", type="primary"):
        suffix = Path(uploaded.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        with col_out:
            st.subheader("Output")
            with st.spinner(f"Reading with {cfg.model}... (first run loads the model)"):
                try:
                    markdown = image_to_markdown(
                        tmp_path, cfg, prompt=extra_prompt or None
                    )
                except Exception as exc:  # noqa: BLE001 - surface any error to UI
                    st.error(str(exc))
                    markdown = None
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

            if markdown is not None:
                if markdown.strip():
                    tab_render, tab_code = st.tabs(["Rendered", "Markdown"])
                    with tab_render:
                        st.markdown(markdown)
                    with tab_code:
                        st.code(markdown, language="markdown")
                    st.download_button(
                        "Download .md",
                        data=markdown + "\n",
                        file_name=f"{Path(uploaded.name).stem}.md",
                        mime="text/markdown",
                    )
                else:
                    st.warning("The model returned no text for this image.")
else:
    st.info("Upload an image to get started.")
