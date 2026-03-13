"""
Reward function for adaptive routing — V5 The Adaptive Conductor.

Single focused reward: dense goodput + dense drop penalty + sparse FCT
pressure + anti-lazy starvation.  Proven by A–G ablation to outperform
single-objective variants because it gives the critic multiple
correlated gradient signals on every step.
"""

from dataclasses import dataclass
import numpy as np

from simulator.network import NetworkState


@dataclass
class RewardConfig:
    """V5 reward configuration.

    All fields can be overridden from YAML under ``environment.reward:``.
    """

    # Reference capacity for goodput normalisation (Mbps).
    # Should approximate achievable goodput at typical load, NOT total fabric BW.
    # Micro-lab (4L×2S): ~800_000, Production (16L×4S): ~6_000_000.
    reference_capacity_mbps: float = 800_000.0
    max_expected_flows: float = 20.0

    # Drop normalisation: Mbps of drops that corresponds to a serious
    # congestion event.  Keeps the penalty visible even when drops are
    # small relative to total capacity.
    drop_denominator_mbps: float = 5_000.0

    # Core multipliers (goodput + drops = actionable signal)
    goodput_weight: float = 3.0
    drop_weight: float = -5.0

    # Auxiliary terms (small, optional — avoid dominating gradient)
    flow_penalty_weight: float = 0.0     # disabled: flow arrivals are uncontrollable
    starvation_penalty: float = 0.0      # disabled: goodput reward already penalises idling


class RewardCalculator:
    """Computes a scalar reward from the current network state."""

    def __init__(self, config: RewardConfig):
        self.config = config

    def compute(self, state: NetworkState, weight_changed: bool = False) -> float:
        cfg = self.config

        # 1. POSITIVE: goodput (data successfully moved)
        cap = max(cfg.reference_capacity_mbps, 1.0)
        goodput_norm = state.total_goodput / cap

        # 2. DENSE PENALTY: drops (fabric + edge buffer overflows)
        # Use log-scale so the gradient is visible across the full range
        # (685K vs 100K vs 5K drops all produce different penalties).
        dropped = state.dropped_traffic + getattr(state, "edge_dropped_mb", 0.0)
        drop_denom = max(cfg.drop_denominator_mbps, 1.0)
        drop_norm = np.log1p(dropped / drop_denom)

        # 3. SPARSE PENALTY: FCT pressure (finish the flows)
        edge_drops = getattr(state, "edge_dropped_flows", 0)
        raw_flows = state.num_active_flows + edge_drops
        flow_norm = min(raw_flows / max(cfg.max_expected_flows, 1.0), 2.0)

        # 4. ANTI-LAZY: starvation (idle fabric with waiting backlog)
        starvation = 0.0
        avg_util = (
            float(state.link_utilizations.mean())
            if len(state.link_utilizations) > 0
            else 0.0
        )
        has_backlog = (
            hasattr(state, "leaf_backlog")
            and len(state.leaf_backlog) > 0
            and state.leaf_backlog.sum() > 0
        )
        if avg_util < 0.15 and state.num_active_flows > 0 and has_backlog:
            starvation = 1.0

        reward = (
            cfg.goodput_weight * goodput_norm
            + cfg.drop_weight * drop_norm
            + cfg.flow_penalty_weight * flow_norm
            + cfg.starvation_penalty * starvation
        )

        return reward

    def reset(self):
        """Reset any internal state between episodes."""
        pass

