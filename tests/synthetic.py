"""Synthetic charts with known ground-truth values for integration tests."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def test_font(  # noqa: PT028 - helper, not a test
size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """DejaVu Sans from matplotlib's bundled fonts, so tests render identically on Windows, Linux and macOS."""
    from pathlib import Path

    import matplotlib

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


def _to_image(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def line_chart(
    series: dict[str, np.ndarray] | None = None,
    x: np.ndarray | None = None,
    title: str = "Line Voltage (V)",
    ylim: tuple[float, float] = (200, 260),
    log: bool = False,
    reverse: bool = False,
    size: tuple[float, float] = (8, 5),
) -> tuple[Image.Image, dict[str, np.ndarray], np.ndarray]:
    x = np.arange(0, 60) if x is None else x
    if series is None:
        series = {"Phase A": 230 + 10 * np.sin(x / 8.0), "Phase B": 220 + 0.4 * x}
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=size)
    for (name, y), c in zip(series.items(), colors, strict=False):
        ax.plot(x, y, color=c, linewidth=2.2, label=name)
    ax.set_title(title, fontsize=16)
    if log:
        from matplotlib.ticker import FuncFormatter

        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(*ylim)
    if reverse:
        ax.invert_yaxis()
    ax.tick_params(labelsize=13)
    ax.grid(True, color="#dddddd")
    if len(series) > 1:
        ax.legend(fontsize=13, loc="upper left")
    fig.tight_layout()
    return _to_image(fig), series, x


def bar_chart(values: list[float] | None = None, title: str = "Feeder Load (kW)") -> tuple[Image.Image, list[float]]:
    values = values or [12.0, 30.0, 22.0, 41.0, 18.0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(values)), values, color="#1f77b4", width=0.6)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([f"F{i + 1}" for i in range(len(values))], fontsize=13)
    ax.set_ylim(0, 50)
    ax.tick_params(labelsize=13)
    ax.set_title(title, fontsize=16)
    fig.tight_layout()
    return _to_image(fig), values


def kpi_tiles(items: list[tuple[str, str, str]] | None = None) -> Image.Image:
    items = items or [("Voltage", "231.4", "V"), ("Frequency", "49.98", "Hz"), ("Power Factor", "0.93", "")]
    img = Image.new("RGB", (900, 220), "white")
    d = ImageDraw.Draw(img)
    try:
        font = test_font(30)
        small = test_font(22)
    except OSError:  # pragma: no cover
        font = small = ImageFont.load_default()
    for i, (label, value, unit) in enumerate(items):
        x = 30 + i * 290
        d.rectangle([x - 10, 40, x + 260, 180], outline="#999999", width=2)
        d.text((x, 60), label, fill="black", font=small)
        d.text((x, 105), f"{value} {unit}".strip(), fill="black", font=font)
    return img
