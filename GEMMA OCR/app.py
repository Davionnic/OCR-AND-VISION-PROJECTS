"""Gemma-3 OCR — a local, private OCR app with structured text extraction.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from PIL import Image

import ocr_engine

st.set_page_config(
    page_title="Gemma-3 OCR",
    page_icon="🔎",
    layout="wide",
)


def init_state() -> None:
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("result_name", None)


def render_header() -> None:
    left, right = st.columns([0.85, 0.15])
    with left:
        st.title("🔎 Gemma-3 OCR")
        st.caption(
            "Local, private OCR with structured Markdown extraction — "
            "powered by Gemma-3 running on Ollama. No data leaves your machine."
        )
    with right:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.result = None
            st.session_state.result_name = None
            st.rerun()
    st.divider()


def render_sidebar() -> tuple[str, str]:
    st.sidebar.header("⚙️ Settings")

    online = ocr_engine.is_ollama_available()
    if online:
        st.sidebar.success("Ollama is running")
    else:
        st.sidebar.error("Ollama not reachable")
        st.sidebar.markdown(
            "Start it first:\n\n"
            "```bash\n"
            "ollama serve\n"
            "ollama pull gemma3:4b\n"
            "```"
        )

    local_models = ocr_engine.list_local_models()
    vision_candidates = [m for m in local_models if "gemma3" in m.lower()]
    options = vision_candidates or local_models or [ocr_engine.DEFAULT_MODEL]

    default_index = 0
    if ocr_engine.DEFAULT_MODEL in options:
        default_index = options.index(ocr_engine.DEFAULT_MODEL)

    model = st.sidebar.selectbox("Model", options, index=default_index)

    st.sidebar.markdown("**Extraction prompt**")
    prompt = st.sidebar.text_area(
        "prompt",
        value=ocr_engine.STRUCTURED_PROMPT,
        height=260,
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Tip: `gemma3:4b` is a good balance of speed and accuracy. "
        "Use `gemma3:12b` or `gemma3:27b` for harder documents."
    )
    return model, prompt


def main() -> None:
    init_state()
    render_header()
    model, prompt = render_sidebar()

    upload_col, result_col = st.columns(2, gap="large")

    with upload_col:
        st.subheader("1 · Upload image")
        uploaded = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
            label_visibility="collapsed",
        )

        image: Image.Image | None = None
        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, caption=uploaded.name, use_container_width=True)

        run = st.button(
            "✨ Extract text",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )

    if run and image is not None:
        with st.spinner(f"Reading the image with {model}…"):
            try:
                result = ocr_engine.extract_text(image, model=model, prompt=prompt)
                st.session_state.result = result.text
                st.session_state.result_name = uploaded.name
            except RuntimeError as exc:
                st.session_state.result = None
                st.error(f"OCR failed: {exc}")

    with result_col:
        st.subheader("2 · Structured result")
        result_text = st.session_state.get("result")
        if result_text:
            tab_render, tab_raw = st.tabs(["📄 Rendered", "✏️ Markdown"])
            with tab_render:
                st.markdown(result_text)
            with tab_raw:
                st.code(result_text, language="markdown")

            base = (st.session_state.result_name or "ocr").rsplit(".", 1)[0]
            st.download_button(
                "⬇️ Download Markdown",
                data=result_text,
                file_name=f"{base}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.info("Upload an image and click **Extract text** to see results here.")


if __name__ == "__main__":
    main()
