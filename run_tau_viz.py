"""Plot positive alpha-stable tau curves without importing this repository."""

from math import gamma, pi, sin

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    alpha = 1.5
    expected_jumps = 1000
    t_final = 1.0
    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Levy-density constant for a symmetric alpha-stable process.
    m_alpha = gamma(alpha) * sin(pi * alpha / 2.0) / pi

    t_grid = np.linspace(0.0, t_final, 1200)
    colors = plt.get_cmap("cividis")(np.linspace(0.3, 0.9, len(eps_values)))
    fig, ax = plt.subplots(figsize=(10, 6))

    for eps, color in zip(eps_values, colors):
        # DC step size chosen to give the requested expected number of jumps.
        h = (2.0 * t_final ** (1.0 - eps) / (expected_jumps * (1.0 - eps))) ** (1.0 / eps)

        # Positive alpha-stable tau(s) = (M_alpha * s)^(1 / alpha),
        # evaluated at s = (t * h)^eps.
        tau_values = (m_alpha * (t_grid * h) ** eps) ** (1.0 / alpha)

        ax.plot(
            t_grid,
            tau_values,
            color=color,
            linewidth=2.2,
            label=rf"$\varepsilon={eps:.1f},\ h={h:.2e}$",
        )

    ax.set(
        xlabel=r"$t$",
        ylabel=r"$\tau((t h)^{\varepsilon})$",
        title=rf"$\tau((t h)^{{\varepsilon}})$ for $\alpha$-Stable DC, $\alpha={alpha}$",
        xlim=(0.0, t_final),
    )
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
