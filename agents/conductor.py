"""
Conductor heuristic agent — V4 Stage B.

A simple threshold-based admission controller + least-loaded router
that proves the edge-buffer physics work before training the GNN.

Rule:
    - If ANY spine serving this leaf exceeds 90% utilisation:
        admission_rate = 0.5 (throttle injection by 50%)
    - Else:
        admission_rate = 1.0 (full injection)
    - Always route to the least-loaded spine (inverse-utilisation).

Expected validation results (Stage B vs ECMP/Stage A):
    - Zero / near-zero drops
    - Zero / near-zero retransmissions
    - Lower tail FCT than ECMP
    - Proves admission control is effective
"""

import numpy as np

from .base import BaseAgent


class ConductorAgent(BaseAgent):
    """Heuristic conductor: admission gating + least-loaded routing.

    Args:
        throttle_threshold: Spine utilisation above which to throttle.
        throttled_admission: Admission rate when throttled (0 to 1).
        sensitivity: How aggressively to prefer lighter spines.
    """

    def __init__(
        self,
        num_leaves: int,
        num_spines: int,
        throttle_threshold: float = 0.90,
        throttled_admission: float = 0.5,
        sensitivity: float = 2.0,
    ):
        super().__init__(num_leaves, num_spines)
        self.throttle_threshold = throttle_threshold
        self.throttled_admission = throttled_admission
        self.sensitivity = sensitivity

        self._num_nodes = num_leaves + num_spines
        self._num_edges = 2 * num_leaves * num_spines
        self._num_links = 2 * num_leaves * num_spines
        self._node_feat_dim = 9
        self._edge_feat_dim = 7
        self._intent_size = num_leaves * num_leaves

    def _link_feature_slices(self, observation: np.ndarray):
        """Parse the flat observation into link-level slices."""
        offset = (
            self._num_nodes * self._node_feat_dim
            + self._num_edges * self._edge_feat_dim
            + self._intent_size
        )
        utils = observation[offset:offset + self._num_links]
        offset += self._num_links
        _queues = observation[offset:offset + self._num_links]
        offset += self._num_links
        _ecn = observation[offset:offset + self._num_links]
        offset += self._num_links
        caps = observation[offset:offset + self._num_links]
        offset += self._num_links
        ups = observation[offset:offset + self._num_links]
        return utils, caps, ups

    def act(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        utilizations, capacity_ratios, up_flags = self._link_feature_slices(observation)

        # Uplink utilizations: first L×S values
        uplink_utils = utilizations[: self.num_leaves * self.num_spines]
        uplink_matrix = uplink_utils.reshape(self.num_leaves, self.num_spines)
        uplink_caps = capacity_ratios[: self.num_leaves * self.num_spines].reshape(
            self.num_leaves, self.num_spines
        )
        uplink_up = up_flags[: self.num_leaves * self.num_spines].reshape(
            self.num_leaves, self.num_spines
        )

        S = self.num_spines
        action = np.zeros((self.num_leaves, 1 + S), dtype=np.float32)

        for leaf in range(self.num_leaves):
            spine_utils = uplink_matrix[leaf]
            spine_caps = uplink_caps[leaf]
            spine_up = uplink_up[leaf]
            effective_health = np.maximum(spine_caps * spine_up, 1e-3)

            # ── Admission control ───────────────────────
            if spine_utils.max() > self.throttle_threshold:
                # Map throttled_admission to raw action space:
                # admission_rate = (raw + 1) / 2
                # raw = 2 * admission_rate - 1
                action[leaf, 0] = 2.0 * self.throttled_admission - 1.0
            else:
                action[leaf, 0] = 1.0  # → admission_rate = 1.0

            # ── Least-loaded spine routing ──────────────
            inverse = (1.0 - spine_utils) * effective_health
            spine_action = (inverse * 2.0 - 1.0) * self.sensitivity
            action[leaf, 1:] = np.clip(spine_action, -1.0, 1.0)

        return action.flatten().astype(np.float32)

    @property
    def name(self) -> str:
        return "Conductor"
