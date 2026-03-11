"""
ECMP (Equal-Cost Multi-Path) baseline agent — V4 Stage A.

Always outputs maximum admission (rate = 1.0) and equal spine
weights — the standard static hashing baseline.  In V4 this
serves as the "dumb agent" sanity check: it should reproduce
legacy TCP collapse behaviour, proving the edge buffer physics
are transparent when admission is fully open.
"""

import numpy as np

from .base import BaseAgent


class ECMPAgent(BaseAgent):
    """Standard ECMP — full admission + equal split across all spines.

    V4 action layout per leaf: [admission_raw, spine0, …, spineS-1]
    admission_raw = +1.0 → mapped to admission_rate = 1.0
    spine values  =  0.0 → softmax → uniform distribution
    """

    def act(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)
        S = self.num_spines
        for leaf in range(self.num_leaves):
            base = leaf * (1 + S)
            action[base] = 1.0   # admission_raw = +1 → rate = 1.0
            # spine slots stay 0 → equal weights via softmax
        return action

    @property
    def name(self) -> str:
        return "ECMP"
