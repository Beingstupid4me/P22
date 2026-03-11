"""
Gymnasium-compatible RL environment for adaptive network routing.

Wraps the NetworkSimulator (with TCP congestion control) with a
standard gym.Env interface so that any Gymnasium-compatible agent
(PPO, SAC, …) can train on it.

Interface mapping to the real GNS3 fabric:
    Observation  →  equivalent to SSH telemetry (link utilisation, queues, ECN)
    Action       →  equivalent to BGP weight commands via vtysh
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

from simulator.network import NetworkSimulator, NetworkState
from simulator.topology import TopologyConfig
from simulator.traffic import TrafficConfig
from simulator.link import TCPConfig
from environment.rewards import RewardCalculator, RewardConfig


class NetworkRoutingEnv(gym.Env):
    """RL environment for adaptive routing in Clos topologies.

    Observation Space (Box):
        Per-link utilisation   : (num_links,)    ∈ [0, 1]
        Per-link queue depth   : (num_links,)    ∈ [0, 1]
        Per-link ECN fraction  : (num_links,)    ∈ [0, 1]
        Current routing weights: (L × S,)        ∈ [0, 1]
        × observation_history (stacked)

    Action Space (Box):
        Raw weight scores      : (L × S,)        ∈ [-1, 1]
        Transformed to per-leaf softmax split ratios internally.

    Reward:
        Multi-objective — see ``environment.rewards``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        topology_config: Optional[TopologyConfig] = None,
        traffic_config: Optional[TrafficConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        tcp_config: Optional[TCPConfig] = None,
        max_steps: int = 500,
        observation_history: int = 1,
        step_duration_seconds: float = 1.0,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.topo_config = topology_config or TopologyConfig()
        self.traffic_config = traffic_config or TrafficConfig()
        self.reward_config = reward_config or RewardConfig()
        self.tcp_config = tcp_config or TCPConfig()
        self.max_steps = max_steps
        self.obs_history = observation_history
        self.step_duration = step_duration_seconds
        self._seed = seed
        self.render_mode = render_mode

        # Build simulator (determines observation / action dimensions)
        self.simulator = NetworkSimulator(
            self.topo_config, self.traffic_config, self.tcp_config, seed,
            step_duration_seconds=step_duration_seconds,
        )
        self.reward_calc = RewardCalculator(self.reward_config)

        # ── Dimensions ──────────────────────────────────
        self._num_links = self.simulator.num_links
        self._num_leaves = self.simulator.num_leaves
        self._num_spines = self.simulator.num_spines
        self._weight_dim = self._num_leaves * self._num_spines

        # Single-step observation: utilisation + queue + ecn + weights
        self._single_obs_dim = self._num_links * 3 + self._weight_dim
        # Full observation (with history stacking)
        self._obs_dim = self._single_obs_dim * self.obs_history

        # ── Spaces ──────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._weight_dim,),
            dtype=np.float32,
        )

        # ── Episode state ──────────────────────────────
        self._current_step: int = 0
        self._obs_buffer: list = []
        self._prev_action: Optional[np.ndarray] = None
        self._episode_metrics: list = []

    # ── Gym API ─────────────────────────────────────────

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        effective_seed = seed if seed is not None else self._seed
        state = self.simulator.reset(effective_seed)
        self.reward_calc.reset()

        self._current_step = 0
        self._prev_action = None
        self._episode_metrics = []

        # Fill observation buffer with zeros for history
        obs = self._build_observation(state)
        self._obs_buffer = [
            np.zeros(self._single_obs_dim, dtype=np.float32)
        ] * (self.obs_history - 1)
        self._obs_buffer.append(obs)

        return self._get_full_observation(), self._build_info(state)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Maps the raw action to routing weights, steps the simulator,
        computes reward, and returns the standard Gymnasium tuple.
        """
        # Transform action → routing weights
        weights = self._action_to_weights(action)

        # Detect meaningful weight change
        weight_changed = (
            self._prev_action is not None
            and not np.allclose(action, self._prev_action, atol=0.05)
        )

        # Advance simulator
        state = self.simulator.step(weights)
        self._current_step += 1

        # Reward
        reward = self.reward_calc.compute(state, weight_changed)

        # Observation
        obs = self._build_observation(state)
        self._obs_buffer.append(obs)
        if len(self._obs_buffer) > self.obs_history:
            self._obs_buffer.pop(0)

        full_obs = self._get_full_observation()

        # Per-step metrics
        self._episode_metrics.append({
            "throughput_ratio": state.total_throughput / max(state.total_demand, 1e-8),
            "max_utilization": state.max_utilization,
            "utilization_std": state.utilization_std,
            "queue_depth_avg": float(state.link_queue_depths.mean()),
            "num_flows": state.num_active_flows,
            "num_elephant_flows": state.num_elephant_flows,
            "elephant_fcts": list(getattr(state, "elephant_fcts_this_step", [])),
            "dropped_traffic": getattr(state, "dropped_traffic", 0.0),
            "total_retransmissions": getattr(state, "total_retransmissions", 0),
            "total_drops_mb": getattr(state, "total_drops_mb", 0.0),
            "total_ecn_marks": getattr(state, "total_ecn_marks", 0),
            "avg_cwnd_fraction": getattr(state, "avg_cwnd_fraction", 1.0),
            "goodput": getattr(state, "total_goodput", 0.0),
            "reward": reward,
        })

        # Termination
        terminated = False
        truncated = self._current_step >= self.max_steps

        self._prev_action = action.copy()

        info = self._build_info(state)
        if truncated or terminated:
            info["episode_metrics"] = self._episode_metrics

        return full_obs, reward, terminated, truncated, info

    # ── Action transform ────────────────────────────────

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """Convert raw action ∈ [-1, 1] to valid routing weights.

        Uses per-leaf softmax to ensure weights sum to a valid
        distribution, then scales to [0.1, 1.0].
        """
        raw = action.reshape(self._num_leaves, self._num_spines)
        scaled = raw * 3.0  # temperature
        # Numerically stable softmax per leaf
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_w = np.exp(shifted)
        weights = exp_w / exp_w.sum(axis=1, keepdims=True)
        # Scale to reasonable routing weight range
        weights = weights * 0.9 + 0.1
        return weights

    # ── Observation helpers ─────────────────────────────

    def _build_observation(self, state: NetworkState) -> np.ndarray:
        """Build a single-step observation vector.

        Includes link utilisations, queue depths, ECN fractions,
        and normalised routing weights.
        """
        weight_max = max(state.routing_weights.max(), 1e-8)
        ecn_fracs = getattr(state, "link_ecn_fractions", np.zeros_like(state.link_utilizations))
        obs = np.concatenate([
            state.link_utilizations.astype(np.float32),
            state.link_queue_depths.astype(np.float32),
            ecn_fracs.astype(np.float32),
            (state.routing_weights.flatten() / weight_max).astype(np.float32),
        ])
        return np.clip(obs, 0.0, 1.0)

    def _get_full_observation(self) -> np.ndarray:
        """Stack history into the full observation vector."""
        return np.concatenate(self._obs_buffer).astype(np.float32)

    def _build_info(self, state: NetworkState) -> Dict[str, Any]:
        """Build the info dictionary returned by step/reset."""
        info: Dict[str, Any] = {
            "step": state.step,
            "total_throughput": state.total_throughput,
            "total_demand": state.total_demand,
            "total_goodput": getattr(state, "total_goodput", 0.0),
            "max_utilization": state.max_utilization,
            "utilization_std": state.utilization_std,
            "num_active_flows": state.num_active_flows,
            "num_elephant_flows": state.num_elephant_flows,
            "weight_changes": self.simulator.weight_changes,
            "link_utilizations": state.link_utilizations.tolist(),
            "dropped_traffic": getattr(state, "dropped_traffic", 0.0),
            # TCP health
            "total_retransmissions": getattr(state, "total_retransmissions", 0),
            "total_drops_mb": getattr(state, "total_drops_mb", 0.0),
            "total_ecn_marks": getattr(state, "total_ecn_marks", 0),
            "avg_cwnd_fraction": getattr(state, "avg_cwnd_fraction", 1.0),
            "flows_in_slow_start": getattr(state, "flows_in_slow_start", 0),
            "flows_in_congestion_avoidance": getattr(state, "flows_in_congestion_avoidance", 0),
            "flows_in_fast_recovery": getattr(state, "flows_in_fast_recovery", 0),
        }

        # FCT data for metrics collection
        fcts = getattr(state, "elephant_fcts_this_step", [])
        info["elephant_fcts_this_step"] = list(fcts)

        # Compute slowdowns from completed elephants
        completed = getattr(self.simulator, "completed_elephant_flows", [])
        slowdowns = [f.slowdown for f in completed if f.slowdown is not None]
        info["elephant_slowdowns"] = slowdowns[-10:]

        return info

    # ── Episode summary ─────────────────────────────────

    def get_episode_summary(self) -> Dict[str, float]:
        """Summary statistics for the completed episode."""
        if not self._episode_metrics:
            return {}

        m = self._episode_metrics

        # Collect all elephant FCTs across the episode
        all_fcts = []
        for step_data in m:
            all_fcts.extend(step_data.get("elephant_fcts", []))

        summary = {
            "avg_throughput_ratio": float(np.mean([x["throughput_ratio"] for x in m])),
            "avg_max_utilization": float(np.mean([x["max_utilization"] for x in m])),
            "avg_utilization_std": float(np.mean([x["utilization_std"] for x in m])),
            "avg_queue_depth": float(np.mean([x["queue_depth_avg"] for x in m])),
            "total_reward": float(sum(x["reward"] for x in m)),
            "avg_reward": float(np.mean([x["reward"] for x in m])),
            "total_weight_changes": self.simulator.weight_changes,
            "total_dropped_traffic": float(sum(x.get("dropped_traffic", 0.0) for x in m)),
            "elephant_flows_completed": len(all_fcts),
            # TCP health summary
            "total_retransmissions": max(x.get("total_retransmissions", 0) for x in m),
            "total_drops_mb": max(x.get("total_drops_mb", 0.0) for x in m),
            "total_ecn_marks": max(x.get("total_ecn_marks", 0) for x in m),
            "avg_cwnd_fraction": float(np.mean([x.get("avg_cwnd_fraction", 1.0) for x in m])),
            "avg_goodput": float(np.mean([x.get("goodput", 0.0) for x in m])),
        }

        if all_fcts:
            summary["mean_fct"] = float(np.mean(all_fcts))
            summary["p99_fct"] = float(np.percentile(all_fcts, 99))
            summary["p95_fct"] = float(np.percentile(all_fcts, 95))

        return summary
