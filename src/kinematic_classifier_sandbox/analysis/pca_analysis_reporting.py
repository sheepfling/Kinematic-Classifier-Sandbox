from __future__ import annotations

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ..utils.plotting import plt
from .pca_analysis_contracts import PcaAnalysisResult


def render_pca_scatter(result: PcaAnalysisResult):
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    class_names = sorted({row["true_class"] for row in result.coordinates})
    colors = {
        name: color
        for name, color in zip(
            class_names,
            ("#2563eb", "#16a34a", "#7c3aed", "#d97706", "#db2777", "#0f766e", "#dc2626"),
        )
    }
    for class_name in class_names:
        rows = [row for row in result.coordinates if row["true_class"] == class_name]
        ax.scatter(
            [float(row.get("pc1", 0.0)) for row in rows],
            [float(row.get("pc2", 0.0)) for row in rows],
            label=class_name,
            color=colors[class_name],
            alpha=0.8,
            s=28,
        )
    ax.set_title("PCA Feature Scatter", loc="left", fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def render_pca_variance(result: PcaAnalysisResult):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    indices = [component.component_index for component in result.components]
    ratios = [component.explained_variance_ratio for component in result.components]
    cumulative = []
    running = 0.0
    for ratio in ratios:
        running += ratio
        cumulative.append(running)
    ax.bar(indices, ratios, color="#2563eb", alpha=0.85, label="explained")
    ax.plot(indices, cumulative, color="#dc2626", linewidth=2.0, marker="o", label="cumulative")
    ax.set_title("Explained Variance", loc="left", fontweight="bold")
    ax.set_xlabel("component")
    ax.set_ylabel("variance ratio")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.2, axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def render_pca_loadings(result: PcaAnalysisResult):
    component_labels = [f"PC{component.component_index}" for component in result.components]
    matrix = [
        [component.loadings[feature_name] for feature_name in result.feature_names]
        for component in result.components
    ]
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("PCA Loadings", loc="left", fontweight="bold")
    ax.set_xticks(range(len(result.feature_names)))
    ax.set_xticklabels(result.feature_names, rotation=35, ha="right")
    ax.set_yticks(range(len(component_labels)))
    ax.set_yticklabels(component_labels)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def render_pca_analysis_report(result: PcaAnalysisResult) -> str:
    report = MarkdownDocument("PCA Feature Analysis")
    report.paragraph("This PCA is a diagnostic over the engineered feature matrix, not a replacement classifier.")
    report.bullet_list(
        [
            f"Feature set: {result.feature_set_name}",
            f"Active features: {', '.join(result.feature_names)}",
        ]
    )
    report.heading("Explained Variance", level=2)
    report.table(
        ["component", "explained_variance", "explained_variance_ratio"],
        [
            (
                f"PC{component.component_index}",
                f"{component.explained_variance:.4f}",
                f"{component.explained_variance_ratio:.4f}",
            )
            for component in result.components
        ],
    )

    report.heading("Dominant Loadings", level=2)
    for component in result.components[:3]:
        report.heading(f"PC{component.component_index}", level=3)
        top_loadings = sorted(component.loadings.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
        report.bullet_list([f"`{feature_name}`: {value:.4f}" for feature_name, value in top_loadings])

    report.heading("Notes", level=2)
    report.bullet_list(
        [
            "PCA is run on standardized engineered features.",
            "Class-colored scatter plots help check whether major variance directions align with class identity.",
            "Strong loadings identify which feature groups dominate the first principal directions.",
        ]
    )
    return report.text()
