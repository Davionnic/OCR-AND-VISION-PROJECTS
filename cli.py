"""Command-line interface for LaTeX OCR.

Examples:
    python cli.py examples/equation.png
    python cli.py equation.png --backend ollama --model qwen2.5vl:3b
    python cli.py equation.png --backend openai --model gpt-4o-mini -o out.tex
"""

from __future__ import annotations

import argparse
import sys

from latex_ocr import Config, image_to_latex


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert an equation image to LaTeX code using Llama 3.2 Vision."
    )
    parser.add_argument("image", help="Path to the equation image (png/jpg/...).")
    parser.add_argument(
        "--backend",
        choices=["ollama", "openai"],
        help="Which backend to use (default: env LATEX_OCR_BACKEND or 'ollama').",
    )
    parser.add_argument("--model", help="Override the model name.")
    parser.add_argument("--host", help="Ollama host URL (ollama backend only).")
    parser.add_argument(
        "-o", "--output", help="Write the LaTeX to this file instead of stdout."
    )
    args = parser.parse_args(argv)

    cfg = Config()
    if args.backend:
        cfg.backend = args.backend
    if args.host:
        cfg.ollama_host = args.host
    if args.model:
        if cfg.backend == "ollama":
            cfg.ollama_model = args.model
        else:
            cfg.api_model = args.model

    try:
        latex = image_to_latex(args.image, cfg)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(latex + "\n")
        print(f"Wrote LaTeX to {args.output}")
    else:
        print(latex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
