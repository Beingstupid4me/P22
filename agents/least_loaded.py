"""
Least-loaded path routing agent — V4.

Continuously adjusts weights inversely proportional to current
link utilisation — always preferring the emptiest spine.

V4: Outputs admission_rate = 1.0 (full injection) + inverse-util
    spine weights.
"""

import numpy as np

from .base import BaseAgent


class LeastLoadedAgent(BaseAgent):
    """Inverse-utilisation weighted routing.

    Args:
        sensitivity: How aggressively to prefer lighter paths.
    """

    def __init__(
        self,
        num_leaves: int,
        num_spines: int,
        sensitivity: float = 2.0,
    ):
        super().__init__(num_leaves, num_spines)
        self.sensitivity = sensitivity

        self._num_nodes = num_leaves + num_spines
        self._num_edges = 2 * num_leaves * num_spines
        self._num_links = 2 * num_leaves * num_spines
        self._node_feat_dim = 9   # V4: was 8
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

        # Uplink utilizations only
        uplink_utils = utilizations[: self.num_leaves * self.num_spines]
        uplink_matrix = uplink_utils.reshape(self.num_leaves, self.num_spines)
        uplink_caps = capacity_ratios[: self.num_leaves * self.num_spines].reshape(self.num_leaves, self.num_spines)
        uplink_up = up_flags[: self.num_leaves * self.num_spines].reshape(self.num_leaves, self.num_spines)

        # Prefer healthy, high-capacity, low-utilisation spines
        effective_health = np.maximum(uplink_caps * uplink_up, 1e-3)
        inverse = (1.0 - uplink_matrix) * effective_health
        # Map to action range [-1, 1]
        spine_action = (inverse * 2.0 - 1.0) * self.sensitivity
        spine_action = np.clip(spine_action, -1.0, 1.0)

        # V4: Build full action: admission_rate (+1 = full) + spine weights
        S = self.num_spines
        action = np.zeros((self.num_leaves, 1 + S), dtype=np.float32)
        action[:, 0] = 1.0   # Full admission rate
        action[:, 1:] = spine_action

        return action.flatten().astype(np.float32)

    @property
    def name(self) -> str:
        return "LeastLoaded"
