"""
Reward function for adaptive routing — V4 Adaptive Conductor.

V4 reward: normalised Flow Completion Time pressure.

    Reward = -(num_active_flows + edge_drops) / max_expected_flows

Scaled to roughly [-1, 0] so the Critic network sees stable gradients.
``max_expected_flows`` is a soft normalisation constant (not a hard cap).
"""

from dataclasses import dataclass

import numpy as np

from simulator.network import NetworkState


@dataclass
class RewardConfig:
    """Reward configuration.

    V4 default: scaled -num_active_flows / max_expected_flows.
    Set ``use_legacy_reward=True`` to fall back to the V3 weighted sum.
    """

    use_legacy_reward: bool = False

    # V4: scaling factor applied after normalisation (default -1.0)
    flow_penalty_weight: float = -1.0

    # V4: soft normaliser — expected peak active flows during an episode.
    # Reward ≈ flow_penalty_weight × (active / max_expected).  Keeps
    # reward in roughly [-1, 0] for stable critic training.
    max_expected_flows: float = 100.0

    # Legacy V3 weights (only used when use_legacy_reward=True)
    throughput_weight: float = 3.0
    drop_weight: float = -5.0
    hotspot_weight: float = -1.0
    hotspot_threshold: float = 0.90
    fairness_weight: float = 0.0


class RewardCalculator:
    """Computes a scalar reward from the current network state.

    V4 default: reward = flow_penalty_weight × num_active_flows
    Legacy:     weighted sum of throughput, drops, hotspot.
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def compute(self, state: NetworkState, weight_changed: bool = False) -> float:
        """Return per-step reward.

        Args:
            state:          Current network snapshot.
            weight_changed: Ignored (no flapping penalty).
        """
        cfg = self.config

        if not cfg.use_legacy_reward:
            # ── V4: Normalised FCT pressure ─────────────
            # Reward = weight × (active + edge_drops) / max_expected
            # Keeps reward in roughly [-1, 0] for stable critic.
            edge_drops = getattr(state, 'edge_dropped_flows', 0)
            raw = state.num_active_flows + edge_drops
            normalised = raw / max(cfg.max_expected_flows, 1.0)
            return cfg.flow_penalty_weight * normalised

        # ── Legacy V3 reward ────────────────────────────
        if state.total_demand > 0:
            throughput = state.total_throughput / state.total_demand
            offered_volume_mb = state.total_demand * state.step_duration_s
            drop_normalizer_mb = max(2.0 * offered_volume_mb, 1e-8)
            drop_ratio = min(
                max(state.dropped_traffic, 0.0) / drop_normalizer_mb,
                1.0,
            )
        else:
            throughput = 1.0
            drop_ratio = 0.0

        hotspot = max(0.0, state.max_utilization - cfg.hotspot_threshold)

        reward = (
            cfg.throughput_weight * throughput
            + cfg.drop_weight * drop_ratio
            + cfg.hotspot_weight * hotspot
        )

        if abs(cfg.fairness_weight) > 1e-12:
            utils = state.link_utilizations
            if len(utils) > 0 and utils.sum() > 1e-8:
                n = len(utils)
                fairness = float(
                    utils.sum() ** 2 / (n * (utils ** 2).sum() + 1e-8)
                )
            else:
                fairness = 1.0
            reward += cfg.fairness_weight * fairness

        return reward

    def reset(self):
        """Reset any internal state between episodes."""
        pass

