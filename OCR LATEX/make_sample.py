"""Generate sample equation images so you can test the OCR without hunting
for pictures. Uses matplotlib's mathtext renderer.

Run with:
    python make_sample.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# (filename, latex source) pairs rendered into examples/.
SAMPLES = [
    ("quadratic.png", r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"),
    ("gaussian.png", r"\int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi}"),
    ("euler.png", r"e^{i\pi} + 1 = 0"),
    ("sum.png", r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}"),
]


def render(latex: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(6, 1.6))
    fig.text(0.5, 0.5, f"${latex}$", fontsize=28, ha="center", va="center")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).parent / "examples"
    out_dir.mkdir(exist_ok=True)
    for name, latex in SAMPLES:
        path = out_dir / name
        render(latex, path)
        print(f"Wrote {path}  ({latex})")


if __name__ == "__main__":
    main()
