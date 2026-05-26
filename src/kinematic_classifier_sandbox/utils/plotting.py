from __future__ import annotations

from io import BytesIO
from pathlib import Path

from matplotlib.figure import Figure

from .runtime import configure_matplotlib_environment


def prepare_matplotlib():
    configure_matplotlib_environment()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def write_plot(fig: Figure, path: Path, *, dpi: int = 160, close: bool = True) -> None:
    plt = prepare_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, format="png", dpi=dpi, bbox_inches="tight")
    finally:
        if close:
            plt.close(fig)


def figure_to_png_bytes(fig: Figure, *, dpi: int = 160, close: bool = True) -> bytes:
    plt = prepare_matplotlib()
    buffer = BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        if close:
            plt.close(fig)


def _figure_to_png(fig: Figure) -> bytes:
    return figure_to_png_bytes(fig)
