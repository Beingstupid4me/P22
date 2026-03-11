"""
Threshold-based routing agent (heuristic baseline) — V4.

Mirrors the logic in ``tester_2.py``: if a spine link exceeds
a utilisation threshold, shift traffic to the least-loaded
alternative spine.  Includes a cooldown to prevent flapping.

V4: Also outputs admission_rate = 1.0 by default (full injection),
    set to 0.5 when any spine is hotspotted.
"""

import numpy as np

from .base import BaseAgent


class ThresholdAgent(BaseAgent):
    """Industry-standard '75 % trigger' heuristic.

    Args:
        threshold: Utilisation fraction that triggers re-routing.
        cooldown: Minimum steps between consecutive switches.
    """

    def __init__(
        self,
        num_leaves: int,
        num_spines: int,
        threshold: float = 0.75,
        cooldown: int = 5,
    ):
        super().__init__(num_leaves, num_spines)
        self.threshold = threshold
        self.cooldown = cooldown
        self._steps_since_change = cooldown  # allow immediate first action
        self._current_action = np.zeros(self.action_dim, dtype=np.float32)
        # Set default admission to +1.0 (full rate) for all leaves
        S = num_spines
        for leaf in range(num_leaves):
            self._current_action[leaf * (1 + S)] = 1.0

        self._num_nodes = num_leaves + num_spines
        self._num_edges = 2 * num_leaves * num_spines
        self._num_links = 2 * num_leaves * num_spines
        self._node_feat_dim = 9   # V4: was 8, now 9 (backlog added)
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
        self._steps_since_change += 1

        if self._steps_since_change < self.cooldown:
            return self._current_action

        utilizations, capacity_ratios, up_flags = self._link_feature_slices(observation)

        # Uplink utilizations: interleaved in edge order (up, down, up, down…)
        # First L×S entries when edges are (leaf0→spine0, spine0→leaf0, leaf0→spine1, …)
        # Actually in our topology, edges are ordered as L*S uplinks then L*S downlinks
        # but link features use a different ordering. Let's use the first L*S*2 entries
        # and take every other one for uplinks.
        # Actually the link arrays are from get_all_links() which returns uplinks then downlinks.
        uplink_utils = utilizations[: self.num_leaves * self.num_spines]
        uplink_matrix = uplink_utils.reshape(self.num_leaves, self.num_spines)
        uplink_caps = capacity_ratios[: self.num_leaves * self.num_spines].reshape(self.num_leaves, self.num_spines)
        uplink_up = up_flags[: self.num_leaves * self.num_spines].reshape(self.num_leaves, self.num_spines)

        S = self.num_spines
        action = np.zeros((self.num_leaves, 1 + S), dtype=np.float32)
        changed = False

        for leaf in range(self.num_leaves):
            spine_utils = uplink_matrix[leaf]
            spine_caps = uplink_caps[leaf]
            spine_up = uplink_up[leaf]
            effective_health = spine_caps * spine_up

            # V4: Default admission = full rate (+1 → mapped to 1.0)
            action[leaf, 0] = 1.0

            if spine_utils.max() > self.threshold or effective_health.min() < 0.99:
                # Boost the least-loaded spine, suppress the rest
                score = (1.0 - spine_utils) * np.maximum(effective_health, 1e-3)
                least_loaded = int(score.argmax())
                action[leaf, 1:] = -0.5
                action[leaf, 1 + least_loaded] = 1.0
                # V4: Throttle admission when hotspotted
                if spine_utils.max() > 0.90:
                    action[leaf, 0] = 0.0  # → admission_rate = 0.5
                changed = True
            # else: spine weights stay 0 → ECMP

        if changed:
            self._steps_since_change = 0

        self._current_action = action.flatten()
        return self._current_action

    def reset(self):
        self._steps_since_change = self.cooldown
        self._current_action = np.zeros(self.action_dim, dtype=np.float32)
        S = self.num_spines
        for leaf in range(self.num_leaves):
            self._current_action[leaf * (1 + S)] = 1.0


    @property
    def name(self) -> str:
        return f"Threshold({self.threshold})"
