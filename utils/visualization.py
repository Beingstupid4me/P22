"""
Visualisation utilities for benchmark results and episode timelines.

All functions gracefully skip if matplotlib is not installed.
"""

from typing import Dict, List, Optional
from pathlib import Path

import numpy as np


def plot_benchmark_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Bar chart comparing agents across key metrics.

    Args:
        results: Output from ``BenchmarkRunner.compare_agents()``.
        save_path: Optional file path to save the figure.
        show: Whether to call ``plt.show()``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping visualisation.")
        return

    agents = list(results.keys())
    metrics = [
        "avg_throughput_ratio",
        "avg_reward",
        "stability_index",
        "avg_utilization_std",
    ]
    labels = [
        "Throughput Ratio",
        "Avg Reward",
        "Stability Index",
        "Utilisation Std",
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(agents)))

    for ax, metric, label in zip(axes, metrics, labels):
        means = [results[a].get(metric, {}).get("mean", 0) for a in agents]
        stds = [results[a].get(metric, {}).get("std", 0) for a in agents]
        ax.bar(agents, means, yerr=stds, capsize=5, color=colors)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_episode_timeline(
    episode_metrics: List[Dict],
    title: str = "Episode Timeline",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot per-step metrics for a single episode.

    Args:
        episode_metrics: List of per-step metric dicts
                         (as stored in ``env._episode_metrics``).
        title: Figure title.
        save_path: Optional file path.
        show: Whether to display interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping visualisation.")
        return

    steps = range(len(episode_metrics))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    # Throughput ratio
    axes[0, 0].plot(
        steps, [m["throughput_ratio"] for m in episode_metrics], "b-"
    )
    axes[0, 0].set_title("Throughput Ratio")
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].set_xlabel("Step")

    # Max utilisation
    axes[0, 1].plot(
        steps, [m["max_utilization"] for m in episode_metrics], "r-"
    )
    axes[0, 1].axhline(y=0.8, color="orange", linestyle="--", label="80 % threshold")
    axes[0, 1].set_title("Max Link Utilisation")
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("Step")

    # Utilisation std (load balance)
    axes[1, 0].plot(
        steps, [m["utilization_std"] for m in episode_metrics], "g-"
    )
    axes[1, 0].set_title("Load Balance (Utilisation Std)")
    axes[1, 0].set_xlabel("Step")

    # Reward
    axes[1, 1].plot(
        steps, [m["reward"] for m in episode_metrics], color="purple"
    )
    axes[1, 1].set_title("Per-Step Reward")
    axes[1, 1].set_xlabel("Step")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
