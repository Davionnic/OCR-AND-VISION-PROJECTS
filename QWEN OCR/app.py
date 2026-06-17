"""Qwen 2.5 VL OCR — Streamlit app.

Two modes:
  * Full Page OCR   — extract all text from an image.
  * Text Spotting   — detect text and draw bounding boxes.
"""

from __future__ import annotations

import io
import time

import streamlit as st
from PIL import Image

import config
from drawing import draw_boxes

st.set_page_config(
    page_title="Qwen 2.5 VL OCR",
    page_icon="\U0001F4C4",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _get_model():
    # Imported lazily so the UI renders even before heavy deps load.
    from model_loader import load_model

    return load_model()


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def sidebar() -> dict:
    with st.sidebar:
        st.header("Settings")
        st.caption(f"Model: `{config.MODEL_ID}`")

        mode = st.radio(
            "Mode",
            ["Full Page OCR", "Text Spotting"],
            help=(
                "Full Page OCR extracts all text. Text Spotting also returns "
                "bounding boxes drawn on the image."
            ),
        )

        st.divider()
        opts = {"mode": mode}
        if mode == "Text Spotting":
            opts["show_labels"] = st.checkbox("Show recognized text labels", True)
            opts["show_index"] = st.checkbox("Show box numbers", True)

        st.divider()
        if st.button("Load / warm up model", use_container_width=True):
            with st.spinner("Loading Qwen 2.5 VL weights… (first run downloads them)"):
                _get_model()
            st.success("Model ready.")

        return opts


def run_full_page_ocr(loaded, image: Image.Image) -> None:
    from inference import full_page_ocr

    start = time.time()
    with st.spinner("Reading text from the image…"):
        text = full_page_ocr(loaded, image)
    elapsed = time.time() - start

    st.success(f"Done in {elapsed:.1f}s")
    st.subheader("Extracted text")
    st.text_area("Result", value=text, height=400, label_visibility="collapsed")
    st.download_button(
        "Download .txt",
        data=text.encode("utf-8"),
        file_name="ocr_result.txt",
        mime="text/plain",
    )
    with st.expander("Markdown preview"):
        st.markdown(text)


def run_text_spotting(loaded, image: Image.Image, opts: dict) -> None:
    from inference import text_spotting

    start = time.time()
    with st.spinner("Detecting and locating text…"):
        boxes, raw = text_spotting(loaded, image)
    elapsed = time.time() - start

    if not boxes:
        st.warning("No text boxes could be parsed from the model output.")
        with st.expander("Raw model output"):
            st.code(raw)
        return

    annotated = draw_boxes(
        image,
        boxes,
        show_labels=opts.get("show_labels", True),
        show_index=opts.get("show_index", True),
    )

    st.success(f"Found {len(boxes)} text region(s) in {elapsed:.1f}s")
    st.image(annotated, caption="Detected text", use_container_width=True)
    st.download_button(
        "Download annotated image",
        data=_image_to_png_bytes(annotated),
        file_name="text_spotting.png",
        mime="image/png",
    )

    st.subheader("Detected text regions")
    st.dataframe(
        [
            {
                "#": i + 1,
                "text": b.text,
                "x1": b.bbox[0],
                "y1": b.bbox[1],
                "x2": b.bbox[2],
                "y2": b.bbox[3],
            }
            for i, b in enumerate(boxes)
        ],
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Raw model output"):
        st.code(raw)


def main() -> None:
    st.title("\U0001F4C4 Qwen 2.5 VL OCR")
    st.caption(
        "Extract and locate text in images using the Qwen 2.5 Vision-Language model."
    )

    opts = sidebar()

    uploaded = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "webp", "tiff"],
    )

    if uploaded is None:
        st.info("Upload an image to get started, or warm up the model from the sidebar.")
        return

    image = Image.open(uploaded).convert("RGB")

    left, right = st.columns(2)
    with left:
        st.subheader("Input image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Results")
        if st.button("Run OCR", type="primary", use_container_width=True):
            try:
                loaded = _get_model()
            except Exception as exc:  # surface load errors nicely
                st.error(f"Failed to load the model: {exc}")
                return

            if opts["mode"] == "Full Page OCR":
                run_full_page_ocr(loaded, image)
            else:
                run_text_spotting(loaded, image, opts)
        else:
            st.info("Click **Run OCR** to process the image.")


if __name__ == "__main__":
    main()
