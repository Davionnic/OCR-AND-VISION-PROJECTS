"""Command-line interface for Llama OCR.

Examples:
    python cli.py receipt.jpg
    python cli.py page.png --model llama3.2-vision -o page.md
    python cli.py invoice.png --prompt "only extract the totals table"
"""

from __future__ import annotations

import argparse
import sys

from ocr import Config, image_to_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local image-to-Markdown OCR with Llama 3.2 Vision (Ollama)."
    )
    parser.add_argument("image", help="Path to the image file to read.")
    parser.add_argument("--model", help="Ollama vision model tag to use.")
    parser.add_argument("--host", help="Ollama host URL.")
    parser.add_argument("--prompt", help="Extra instruction for the model.")
    parser.add_argument(
        "-o", "--output", help="Write Markdown to this file instead of stdout."
    )
    args = parser.parse_args(argv)

    cfg = Config()
    if args.model:
        cfg.model = args.model
    if args.host:
        cfg.ollama_host = args.host

    try:
        markdown = image_to_markdown(args.image, cfg, prompt=args.prompt)
    except Exception as exc:  # noqa: BLE001 - friendly CLI errors
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(markdown + "\n")
        print(f"Wrote {args.output}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
