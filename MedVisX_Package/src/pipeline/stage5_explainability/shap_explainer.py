"""
SHAP Waterfall Plot for Hypothesis Scoring Explainability
Shows which clinical entity contributed how much to the final score.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


def generate_shap_plot(shap_values: dict, top_disease: str,
                        top_score: float) -> Image.Image:
    """
    Generate a SHAP-style waterfall plot showing entity contributions.
    Returns PIL Image of the plot.
    """
    if not shap_values:
        # Return informative blank image
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No SHAP attribution data available\n"
                "(No matching entities found for top hypothesis)",
                ha='center', va='center', fontsize=12, color='gray',
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    items = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in items]
    values = [i[1] for i in items]

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.6)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)

    ax.set_xlabel("Contribution to Hypothesis Score", fontsize=11)
    ax.set_title(
        f"SHAP Feature Attribution \u2192 {top_disease.title()}\n"
        f"Final Score: {top_score:.2f}",
        fontsize=12, fontweight="bold"
    )
    ax.axvline(0, color="black", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"+{val:.3f}" if val >= 0 else f"{val:.3f}",
            va="center", fontsize=9
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()
