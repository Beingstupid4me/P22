"""
Routing engine with WCMP flowlet switching — V2.

V1 used sticky spine assignment: a flow was hashed once and stayed
on that spine forever.  V2 implements WCMP (Weighted Cost Multi-Path)
with flowlet-granularity re-evaluation:

        - Each flow's spine is re-evaluated EVERY step (flowlet model).
    - The spine is chosen by mapping the flow's 5-tuple hash into
      WCMP hash buckets whose widths are proportional to the weights.
    - When weights change, flows naturally migrate to less-loaded
      spines within one step (simulating a real flowlet gap).
        - Path changes themselves do NOT inject synthetic TCP penalties;
            TCP reacts only to link-generated ECN/drop signals.

This gives the RL agent fine-grained, per-step traffic steering
instead of the "set once and pray" model of V1.
"""

import numpy as np
from typing import Dict

from .topology import ClosTopology


class RoutingEngine:
    """WCMP flowlet routing engine for a Clos topology.

    Each source leaf has a weight per spine.  Every step, each flow
    re-hashes through the current WCMP bucket table, allowing weight
    changes to take effect immediately.

    WCMP Hash Buckets:
        Given weights [0.5, 0.3, 0.2] for 3 spines, the CDF is
        [0.5, 0.8, 1.0].  A flow whose hash maps to 0.6 lands in
        spine-1.  If weights change to [0.2, 0.5, 0.3], the same
        hash now maps to spine-1 (CDF [0.2, 0.7, 1.0], 0.6 still
        in bucket 1) — or it might migrate, which is the point.
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
        """Traffic split ratios for a (src, dst) leaf pair."""
        if src == dst:
            return {}
        ratios = self.split_ratios[src]
        return {k: float(ratios[k]) for k in range(self.num_spines)}

    def assign_flowlet_to_spine(self, flow) -> int:
        """V2: Assign a flow to a spine using WCMP hash buckets.

        Called EVERY step (not just once).  The flow's 5-tuple hash
        is deterministic, but the spine it maps to depends on the
        current weight distribution.  When the RL agent changes
        weights, flows can migrate to different spines — exactly
        like real flowlet switching where inter-packet gaps allow
        the switch to re-hash to a new ECMP bucket.

        WCMP algorithm:
            1. Hash the 5-tuple → [0, 1) fraction.
            2. Walk the source-leaf's weight CDF.
            3. First bucket whose cumulative weight exceeds the
               hash fraction gets the flow.

        This is equivalent to having ``num_spines`` hash buckets
        whose widths are proportional to the weights.
        """
        if flow.src == flow.dst:
            return 0

        # ── 5-tuple hash (matches real switch ECMP behaviour) ──
        h = hash((flow.src_ip, flow.dst_ip, flow.src_port,
                  flow.dst_port, flow.protocol))

        # Map hash to [0, 1)
        hash_frac = (abs(h) % 65536) / 65536.0

        # WCMP CDF selection using current split ratios
        ratios = self.split_ratios[flow.src]
        cumulative = 0.0
        for spine_idx in range(self.num_spines):
            cumulative += ratios[spine_idx]
            if hash_frac < cumulative:
                return spine_idx
        return self.num_spines - 1

    # Legacy alias for backward compatibility
    assign_flow_to_spine = assign_flowlet_to_spine

    # ── Reset ───────────────────────────────────────────

    def reset(self):
        """Reset to equal weights (ECMP)."""
        self._weights = np.ones((self.num_leaves, self.num_spines))
