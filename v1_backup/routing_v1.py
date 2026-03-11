"""
Routing engine with weighted path selection.

Manages a weight matrix that controls how traffic from each
source leaf is distributed across the available spine paths.
This directly mirrors BGP weight manipulation in a real fabric.
"""

import numpy as np
from typing import Dict

from .topology import ClosTopology


class RoutingEngine:
    """Weighted routing logic for a Clos topology.

    Each source leaf has a weight per spine.  The weights determine
    the traffic split ratio — exactly analogous to setting BGP
    neighbor weights on a real router (as in the GNS3 prototype).

    Example:
        weights[leaf_0][spine_0] = 500  →  higher share
        weights[leaf_0][spine_1] = 100  →  lower  share
    """

    def __init__(self, topology: ClosTopology):
        self.topology = topology
        self.num_leaves = topology.num_leaves
        self.num_spines = topology.num_spines

        # Weight matrix: shape (num_leaves, num_spines), initialised to ECMP
        self._weights = np.ones((self.num_leaves, self.num_spines))

    # ── Properties ──────────────────────────────────────

    @property
    def weights(self) -> np.ndarray:
        """Current weight matrix (copy)."""
        return self._weights.copy()

    @property
    def split_ratios(self) -> np.ndarray:
        """Normalised split ratios per source leaf (rows sum to 1)."""
        row_sums = self._weights.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-8)
        return self._weights / row_sums

    # ── Weight manipulation ─────────────────────────────

    def set_weights(self, weights: np.ndarray):
        """Set routing weights from a (num_leaves, num_spines) array."""
        assert weights.shape == (self.num_leaves, self.num_spines), (
            f"Expected shape ({self.num_leaves}, {self.num_spines}), "
            f"got {weights.shape}"
        )
        self._weights = np.maximum(weights, 1e-8)

    def set_weights_flat(self, weights_flat: np.ndarray):
        """Set weights from a flat 1-D array."""
        self.set_weights(weights_flat.reshape(self.num_leaves, self.num_spines))

    # ── Path queries ────────────────────────────────────

    def get_flow_distribution(self, src: int, dst: int) -> Dict[int, float]:
        """Traffic split ratios for a (src, dst) leaf pair.

        Returns:
            Dict mapping spine_index → fraction of traffic.
        """
        if src == dst:
            return {}
        ratios = self.split_ratios[src]
        return {k: float(ratios[k]) for k in range(self.num_spines)}

    def assign_flow_to_spine(self, flow) -> int:
        """Assign a flow to exactly ONE spine using 5-tuple hash.

        Uses the flow's network identity (src_ip, dst_ip, src_port,
        dst_port, protocol) to build a deterministic hash value,
        then maps it to a spine through the source leaf's weight
        CDF (cumulative distribution function).

        Under ECMP (equal weights):
            Pure hash → uniform distribution → natural collisions.
            With N flows through S spines, some spines randomly get
            more flows than others — exactly like real hardware.

        Under weighted routing (RL agent):
            Weights skew the CDF, allowing the agent to steer
            specific hash-ranges to less-loaded spines.

        The same flow always maps to the same spine as long as
        weights don't change (consistent hashing).
        """
        if flow.src == flow.dst:
            return 0

        # ── 5-tuple hash (matches real switch ECMP behaviour) ──
        h = hash((flow.src_ip, flow.dst_ip, flow.src_port,
                  flow.dst_port, flow.protocol))

        # Map hash to [0, 1)
        hash_frac = (abs(h) % 65536) / 65536.0

        # Weighted selection via CDF of source leaf's split ratios
        ratios = self.split_ratios[flow.src]
        cumulative = 0.0
        for spine_idx in range(self.num_spines):
            cumulative += ratios[spine_idx]
            if hash_frac < cumulative:
                return spine_idx
        return self.num_spines - 1

    # ── Reset ───────────────────────────────────────────

    def reset(self):
        """Reset to equal weights (ECMP)."""
        self._weights = np.ones((self.num_leaves, self.num_spines))
