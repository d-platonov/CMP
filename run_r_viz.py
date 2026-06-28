import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np


ALPHA_COLORS = {0.5: "#c0392b", 1.0: "#e67e22", 1.5: "#1a6faf"}

BG = "#ffffff"
PANEL = "#f8f9fa"
GRID = "#e9ecef"
BORDER = "#ced4da"
TEXT = "#212529"
MUTED = "#495057"
CURVE = "#1a6faf"
REF1 = "#c0392b"
VLINE = "#e67e22"


def theta(alpha):
    return (2.0 - alpha) / alpha


def r(eps, alpha, sigma):
    exponent = theta(alpha)
    return ((1.0 - 2.0 * sigma) / (1.0 - 2.0 * sigma + eps * exponent)) * (1.0 - eps) ** (-exponent)


def main() -> None:
    alphas = [0.5, 1.0, 1.5]
    sigmas = [0.05, 0.25, 0.45]

    warnings.filterwarnings("ignore")
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "text.color": TEXT,
            "font.family": "monospace",
            "axes.grid": True,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
        }
    )

    nrows, ncols = len(sigmas), len(alphas)
    fig = plt.figure(figsize=(14, 10))
    outer = gridspec.GridSpec(
        nrows + 1,
        ncols,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
        top=0.91,
        bottom=0.07,
        left=0.08,
        right=0.97,
        height_ratios=[0.001] + [1] * nrows,
    )

    for column in range(ncols):
        fig.add_subplot(outer[0, column]).set_visible(False)

    axes = np.empty((nrows, ncols), dtype=object)
    for row, sigma in enumerate(sigmas):
        for column, alpha in enumerate(alphas):
            ax = fig.add_subplot(outer[row + 1, column])
            axes[row, column] = ax

            eps_star = sigma * alpha
            eps = np.linspace(0.0, 2.0 * eps_star, 1000)
            values = r(eps, alpha, sigma)

            ax.fill_between(eps, 1.0, values, where=(values >= 1.0), color=CURVE, alpha=0.08, zorder=0)
            ax.fill_between(eps, 1.0, values, where=(values < 1.0), color=REF1, alpha=0.06, zorder=0)

            color = ALPHA_COLORS[alpha]
            ax.plot(eps, values, color=color, linewidth=2.2, zorder=3)
            ax.axhline(1.0, color=REF1, linestyle="--", linewidth=1.2, zorder=2)
            ax.axvline(eps_star, color=VLINE, linestyle=":", linewidth=1.4, zorder=2)

            blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(
                eps_star,
                -0.02,
                rf"$\varepsilon^*$={eps_star:.3f}",
                transform=blended,
                color=VLINE,
                fontsize=8,
                ha="left",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": VLINE,
                    "linewidth": 0.6,
                    "alpha": 0.85,
                },
            )

            ax.set_title(
                rf"α={alpha}  σ={sigma}  θ={theta(alpha):.3f}",
                fontsize=8.5,
                color=TEXT,
                pad=6,
                loc="left",
            )
            ax.set_xlabel(r"ε", fontsize=10, labelpad=2)
            ax.set_ylabel(r"R(ε)", fontsize=10, labelpad=4)
            ax.tick_params(labelsize=8)
            ax.spines["left"].set_color(color)
            ax.spines["left"].set_linewidth(2)
            ax.spines["bottom"].set_color(BORDER)

    for column, alpha in enumerate(alphas):
        position = axes[0, column].get_position()
        fig.text(
            position.x0 + position.width / 2.0,
            position.y1 + 0.012,
            rf"α = {alpha}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=CURVE,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": PANEL,
                "edgecolor": CURVE,
                "linewidth": 0.8,
                "alpha": 0.8,
            },
        )

    plt.show()


if __name__ == "__main__":
    main()
