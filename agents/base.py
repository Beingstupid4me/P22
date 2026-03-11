"""
Abstract base class for all routing agents.

Every agent — whether a simple heuristic or a trained RL model —
must implement this interface so that the benchmarking runner can
evaluate them interchangeably.

V4: Action dim is L × (1 + S) — admission rate + spine weights per leaf.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    """Minimal interface for a routing decision-maker.

    Attributes:
        num_leaves: Number of leaf switches in the topology.
        num_spines: Number of spine switches in the topology.
        action_dim: Flat size of the action vector: L × (1 + S).
    """

    def __init__(self, num_leaves: int, num_spines: int):
        self.num_leaves = num_leaves
        self.num_spines = num_spines
        self.action_dim = num_leaves * (1 + num_spines)

    @abstractmethod
    def act(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """Choose an action given the current observation.

        Args:
            observation: Environment observation vector.

        Returns:
            Action array of shape ``(action_dim,)``.
        """
        ...

    def reset(self):
        """Reset any internal state between episodes."""
        pass

    @property
    def name(self) -> str:
        """Human-readable agent identifier."""
        return self.__class__.__name__
