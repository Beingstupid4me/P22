"""
Evaluation metrics for network routing benchmarking.

Required metrics (aligned with research standards):
    1. Tail Flow Completion Time (p99 FCT)
    2. Link Utilisation Balance (Jain's fairness index)
    3. Throughput Improvement over ECMP
    4. Convergence / Adaptation Speed
    + Aggregate reward, stability index
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics collected over a single benchmark episode.

    Tracks per-step data for computing tail FCT, utilisation balance,
    throughput improvement, and convergence speed.
    """

    # Per-step raw data
    throughput_ratios: List[float] = field(default_factory=list)
    max_utilizations: List[float] = field(default_factory=list)
    utilization_stds: List[float] = field(default_factory=list)
    link_utilizations_per_step: List[np.ndarray] = field(default_factory=list)
    queue_depths: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    total_throughputs: List[float] = field(default_factory=list)
    total_demands: List[float] = field(default_factory=list)
    dropped_traffic: List[float] = field(default_factory=list)

    # FCT tracking
    elephant_fcts: List[int] = field(default_factory=list)
    elephant_slowdowns: List[float] = field(default_factory=list)

    # TCP health tracking
    retransmissions_per_step: List[int] = field(default_factory=list)
    drops_mb_per_step: List[float] = field(default_factory=list)
    ecn_marks_per_step: List[int] = field(default_factory=list)
    cwnd_fractions: List[float] = field(default_factory=list)
    goodput_per_step: List[float] = field(default_factory=list)

    # Fabric realism / impairment tracking
    capacity_ratios_per_step: List[float] = field(default_factory=list)
    active_link_events_per_step: List[int] = field(default_factory=list)

    # Stability
    weight_changes: int = 0

    # ECMP baseline reference (set externally for relative metrics)
    ecmp_avg_throughput: Optional[float] = None

    # ── Recording ───────────────────────────────────────

    def record(self, info: Dict, reward: float):
        """Log metrics from a single step."""
        demand = info.get("total_demand", 1e-8)
        throughput = info.get("total_throughput", 0.0)
        self.throughput_ratios.append(throughput / max(demand, 1e-8))
        self.max_utilizations.append(info.get("max_utilization", 0.0))
        self.utilization_stds.append(info.get("utilization_std", 0.0))
        self.rewards.append(reward)
        self.total_throughputs.append(throughput)
        self.total_demands.append(demand)
        self.dropped_traffic.append(info.get("dropped_traffic", 0.0))

        # Store full link utilisation vector if provided
        if "link_utilizations" in info:
            self.link_utilizations_per_step.append(
                np.array(info["link_utilizations"])
            )

        # FCT data from this step
        for fct in info.get("elephant_fcts_this_step", []):
            self.elephant_fcts.append(fct)
        for sd in info.get("elephant_slowdowns", []):
            self.elephant_slowdowns.append(sd)

        # TCP health data
        self.retransmissions_per_step.append(info.get("total_retransmissions", 0))
        self.drops_mb_per_step.append(info.get("total_drops_mb", 0.0))
        self.ecn_marks_per_step.append(info.get("total_ecn_marks", 0))
        self.cwnd_fractions.append(info.get("avg_cwnd_fraction", 1.0))
        self.goodput_per_step.append(info.get("total_goodput", 0.0))
        self.capacity_ratios_per_step.append(info.get("avg_capacity_ratio", 1.0))
        self.active_link_events_per_step.append(info.get("active_link_events", 0))

    # ── 1. Tail FCT (p99) ──────────────────────────────

    @property
    def tail_fct_p99(self) -> float:
        """99th percentile elephant flow completion time."""
        if not self.elephant_fcts:
            return 0.0
        return float(np.percentile(self.elephant_fcts, 99))

    @property
    def tail_fct_p95(self) -> float:
        """95th percentile elephant FCT."""
        if not self.elephant_fcts:
            return 0.0
        return float(np.percentile(self.elephant_fcts, 95))

    @property
    def mean_fct(self) -> float:
        """Mean elephant FCT."""
        if not self.elephant_fcts:
            return 0.0
        return float(np.mean(self.elephant_fcts))

    @property
    def median_fct(self) -> float:
        """Median elephant FCT."""
        if not self.elephant_fcts:
            return 0.0
        return float(np.median(self.elephant_fcts))

    @property
    def mean_slowdown(self) -> float:
        """Mean FCT slowdown (actual / ideal). 1.0 = perfect."""
        if not self.elephant_slowdowns:
            return 1.0
        return float(np.mean(self.elephant_slowdowns))

    # ── 2. Link Utilisation Balance ─────────────────────

    @property
    def jains_fairness(self) -> float:
        """Jain's fairness index on link utilisation (closer to 1 = better).

        J = (Σ x_i)² / (n × Σ x_i²)
        """
        if not self.link_utilizations_per_step:
            # Fallback: use utilization_stds
            if not self.utilization_stds:
                return 1.0
            # Approximate: lower std → higher fairness
            avg_std = float(np.mean(self.utilization_stds))
            return max(0.0, 1.0 - avg_std)

        # Average across all steps
        fairness_values = []
        for utils in self.link_utilizations_per_step:
            if len(utils) == 0 or np.sum(utils) < 1e-8:
                fairness_values.append(1.0)
                continue
            n = len(utils)
            numerator = np.sum(utils) ** 2
            denominator = n * np.sum(utils ** 2)
            fairness_values.append(numerator / max(denominator, 1e-8))
        return float(np.mean(fairness_values))

    @property
    def avg_utilization_std(self) -> float:
        """Average standard deviation of link utilisation."""
        return float(np.mean(self.utilization_stds)) if self.utilization_stds else 0.0

    @property
    def link_balance_ratio(self) -> float:
        """min(util) / max(util) averaged over steps. 1.0 = perfect balance."""
        if not self.link_utilizations_per_step:
            return 0.0
        ratios = []
        for utils in self.link_utilizations_per_step:
            if utils.max() < 1e-8:
                ratios.append(1.0)
            else:
                ratios.append(float(utils.min() / utils.max()))
        return float(np.mean(ratios))

    # ── 3. Throughput Improvement over ECMP ─────────────

    @property
    def avg_throughput(self) -> float:
        """Average total throughput (Mbps)."""
        return float(np.mean(self.total_throughputs)) if self.total_throughputs else 0.0

    @property
    def avg_throughput_ratio(self) -> float:
        """Average throughput / demand ratio."""
        return float(np.mean(self.throughput_ratios)) if self.throughput_ratios else 0.0

    @property
    def throughput_improvement_over_ecmp(self) -> float:
        """Percentage improvement in throughput over ECMP baseline.

        Returns 0.0 if ECMP baseline not set. Positive = better than ECMP.
        """
        if self.ecmp_avg_throughput is None or self.ecmp_avg_throughput < 1e-8:
            return 0.0
        return (self.avg_throughput - self.ecmp_avg_throughput) / self.ecmp_avg_throughput * 100.0

    @property
    def total_dropped(self) -> float:
        """Total dropped traffic across episode."""
        return float(np.sum(self.dropped_traffic))

    # ── 4. Convergence / Adaptation Speed ───────────────

    @property
    def convergence_step(self) -> int:
        """Step at which reward stabilises (within 5% of final mean).

        Uses a sliding window to detect when the agent reaches
        steady-state performance. Lower = faster convergence.
        """
        if len(self.rewards) < 20:
            return 0

        rewards = np.array(self.rewards)
        # Final 20% mean as "converged" reference
        final_window = max(1, len(rewards) // 5)
        final_mean = np.mean(rewards[-final_window:])

        if abs(final_mean) < 1e-8:
            return 0

        threshold = abs(final_mean) * 0.05  # 5% band

        window = max(5, len(rewards) // 20)
        for i in range(len(rewards) - window):
            rolling_mean = np.mean(rewards[i:i + window])
            if abs(rolling_mean - final_mean) < threshold:
                return i
        return len(rewards)

    @property
    def adaptation_speed(self) -> float:
        """Fraction of episode spent at converged performance. Higher = better."""
        total = len(self.rewards)
        if total == 0:
            return 0.0
        conv = self.convergence_step
        return (total - conv) / total

    # ── Legacy / aggregate ──────────────────────────────

    @property
    def avg_max_utilization(self) -> float:
        return float(np.mean(self.max_utilizations)) if self.max_utilizations else 0.0

    @property
    def total_reward(self) -> float:
        return float(np.sum(self.rewards))

    @property
    def avg_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def stability_index(self) -> float:
        """Control action churn per step.

        In V2/V3 flowlet routing this is a descriptive metric only.
        Lower means fewer split-ratio updates, but higher values are
        not automatically bad because flowlet steering is expected to
        react dynamically.
        """
        total_steps = len(self.rewards)
        return self.weight_changes / max(total_steps, 1)

    # ── TCP health metrics ──────────────────────────────

    @property
    def final_retransmissions(self) -> int:
        """Total retransmission events at end of episode."""
        return self.retransmissions_per_step[-1] if self.retransmissions_per_step else 0

    @property
    def final_drops_mb(self) -> float:
        """Total dropped data at end of episode."""
        return self.drops_mb_per_step[-1] if self.drops_mb_per_step else 0.0

    @property
    def final_ecn_marks(self) -> int:
        """Total ECN marks at end of episode."""
        return self.ecn_marks_per_step[-1] if self.ecn_marks_per_step else 0

    @property
    def avg_cwnd_fraction(self) -> float:
        """Average cwnd as fraction of line rate. 1.0 = no throttling."""
        return float(np.mean(self.cwnd_fractions)) if self.cwnd_fractions else 1.0

    @property
    def avg_goodput(self) -> float:
        """Average goodput (useful throughput excl. retransmits)."""
        return float(np.mean(self.goodput_per_step)) if self.goodput_per_step else 0.0

    @property
    def goodput_ratio(self) -> float:
        """Goodput / total throughput. 1.0 = no retransmission overhead."""
        if self.avg_throughput < 1e-8:
            return 1.0
        return min(1.0, self.avg_goodput / self.avg_throughput)

    @property
    def avg_capacity_ratio(self) -> float:
        """Mean effective capacity relative to nominal hardware rate."""
        if not self.capacity_ratios_per_step:
            return 1.0
        return float(np.mean(self.capacity_ratios_per_step))

    @property
    def avg_active_link_events(self) -> float:
        """Mean number of active runtime impairments per step."""
        if not self.active_link_events_per_step:
            return 0.0
        return float(np.mean(self.active_link_events_per_step))

    def summary(self) -> Dict[str, float]:
        """Full summary dict with all required metrics."""
        return {
            # 1. Tail FCT
            "tail_fct_p99": self.tail_fct_p99,
            "tail_fct_p95": self.tail_fct_p95,
            "mean_fct": self.mean_fct,
            "mean_slowdown": self.mean_slowdown,
            # 2. Link Utilisation Balance
            "jains_fairness": self.jains_fairness,
            "avg_utilization_std": self.avg_utilization_std,
            "link_balance_ratio": self.link_balance_ratio,
            # 3. Throughput
            "avg_throughput_ratio": self.avg_throughput_ratio,
            "avg_throughput_mbps": self.avg_throughput,
            "throughput_improvement_pct": self.throughput_improvement_over_ecmp,
            "total_dropped_mb": self.total_dropped,
            # 4. Convergence
            "convergence_step": float(self.convergence_step),
            "adaptation_speed": self.adaptation_speed,
            # 5. TCP Health
            "total_retransmissions": float(self.final_retransmissions),
            "total_drops_mb": float(self.final_drops_mb),
            "total_ecn_marks": float(self.final_ecn_marks),
            "avg_cwnd_fraction": self.avg_cwnd_fraction,
            "goodput_ratio": self.goodput_ratio,
            "avg_capacity_ratio": self.avg_capacity_ratio,
            "avg_active_link_events": self.avg_active_link_events,
            # Aggregate
            "avg_reward": self.avg_reward,
            "total_reward": self.total_reward,
            "stability_index": self.stability_index,
            "weight_changes": float(self.weight_changes),
        }


def aggregate_episodes(
    episodes: List[EpisodeMetrics],
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple episodes.

    Returns:
        Dict mapping metric_name → {mean, std, min, max}.
    """
    if not episodes:
        return {}

    summaries = [ep.summary() for ep in episodes]
    keys = summaries[0].keys()

    result: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [s[key] for s in summaries]
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return result
