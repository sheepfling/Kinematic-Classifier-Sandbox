from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol, overload

from matplotlib.figure import Figure

from .runtime import configure_matplotlib_environment


class MatplotlibAxesCollection(Protocol):
    def __iter__(self) -> Iterator["MatplotlibAxes"]: ...

    def __getitem__(self, index: int) -> "MatplotlibAxes": ...

    def __len__(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def flat(self) -> Iterator["MatplotlibAxes"]: ...

    def ravel(self) -> Any: ...

    def flatten(self) -> Any: ...


class MatplotlibAxes(Protocol):
    def plot(self, *args: Any, **kwargs: Any) -> Any: ...

    def scatter(self, *args: Any, **kwargs: Any) -> Any: ...

    def bar(self, *args: Any, **kwargs: Any) -> Any: ...

    def barh(self, *args: Any, **kwargs: Any) -> Any: ...

    def imshow(self, *args: Any, **kwargs: Any) -> Any: ...

    def fill_between(self, *args: Any, **kwargs: Any) -> Any: ...

    def grid(self, *args: Any, **kwargs: Any) -> Any: ...

    def legend(self, *args: Any, **kwargs: Any) -> Any: ...

    def text(self, *args: Any, **kwargs: Any) -> Any: ...

    def annotate(self, *args: Any, **kwargs: Any) -> Any: ...

    def axhline(self, *args: Any, **kwargs: Any) -> Any: ...

    def axvline(self, *args: Any, **kwargs: Any) -> Any: ...

    def axis(self, *args: Any, **kwargs: Any) -> Any: ...

    def set_title(self, *args: Any, **kwargs: Any) -> Any: ...

    def set_xlabel(self, *args: Any, **kwargs: Any) -> Any: ...

    def set_ylabel(self, *args: Any, **kwargs: Any) -> Any: ...

    def set_xlim(self, *args: Any, **kwargs: Any) -> Any: ...

    def set_ylim(self, *args: Any, **kwargs: Any) -> Any: ...

    def set_xticks(self, *args: Any, **kwargs: Any) -> Any: ...

    def set_yticks(self, *args: Any, **kwargs: Any) -> Any: ...

    def set_xticklabels(self, *args: Any, **kwargs: Any) -> Any: ...

    def set_yticklabels(self, *args: Any, **kwargs: Any) -> Any: ...

    def set_visible(self, *args: Any, **kwargs: Any) -> Any: ...

    def set_axisbelow(self, *args: Any, **kwargs: Any) -> Any: ...

    def tick_params(self, *args: Any, **kwargs: Any) -> Any: ...

    def twiny(self) -> "MatplotlibAxes": ...

    def twinx(self) -> "MatplotlibAxes": ...

    def add_patch(self, *args: Any, **kwargs: Any) -> Any: ...

    def table(self, *args: Any, **kwargs: Any) -> Any: ...

    @property
    def transAxes(self) -> Any: ...


class MatplotlibPyplot(Protocol):
    @overload
    def subplots(self, nrows: Literal[1] = 1, ncols: Literal[1] = 1, *args: Any, **kwargs: Any) -> tuple[Figure, MatplotlibAxes]: ...

    @overload
    def subplots(self, nrows: int, ncols: int = 1, *args: Any, **kwargs: Any) -> tuple[Figure, MatplotlibAxesCollection]: ...

    def figure(self, *args: Any, **kwargs: Any) -> Figure: ...

    def close(self, fig: Figure | None = None) -> None: ...

    def __getattr__(self, name: str) -> Any: ...


configure_matplotlib_environment()
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as _pyplot

plt: Any = _pyplot


def prepare_matplotlib() -> Any:
    return plt


def write_plot(fig: Figure, path: Path, *, dpi: int = 160, close: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, format="png", dpi=dpi, bbox_inches="tight")
    finally:
        if close:
            plt.close(fig)


def figure_to_png_bytes(fig: Figure, *, dpi: int = 160, close: bool = True) -> bytes:
    buffer = BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        if close:
            plt.close(fig)


def _figure_to_png(fig: Figure) -> bytes:
    return figure_to_png_bytes(fig)


def render_labeled_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    *,
    title: str,
    cmap: str = "Blues",
    figsize: tuple[float, float] = (8.5, 6.8),
    aspect: str = "auto",
    value_format: str = ".2f",
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, format(value, value_format), ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label is not None:
        colorbar.set_label(colorbar_label)
    fig.tight_layout()
    return fig
