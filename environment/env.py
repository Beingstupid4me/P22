"""
Gymnasium-compatible RL environment — V4 Adaptive Conductor.

V4 key changes:
    1. Observation includes graph-structured features with leaf backlog:
       [node_features(9), edge_features, intent_matrix,
        link_utils, link_queues, link_ecn, routing_weights]
    2. Action space outputs BOTH admission rate AND split ratios
       per leaf: [admission_rate, spine1, spine2, spine3, spine4].
    3. Intent matrix gives the agent advance burst notice.
    4. Link health / capacity asymmetry are exposed explicitly.
    5. Reward is purely -(num_active_flows) to incentivise fast FCT.
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
    """V4 RL environment for adaptive routing + admission control.

    Observation Space (Box) — packed flat for GNN unpacking:
        Node features     : (num_nodes × node_feat_dim)  ∈ [0, 1]
        Edge features     : (num_edges × edge_feat_dim)  ∈ [0, 1]
        Intent matrix     : (num_leaves × num_leaves)    ∈ [0, 1]
        Link utilisations : (num_links,)                 ∈ [0, 1]
        Link queue depths : (num_links,)                 ∈ [0, 1]
        Link ECN fractions: (num_links,)                 ∈ [0, 1]
        Link capacities   : (num_links,)                 ∈ [0, 1]
        Link up-flags     : (num_links,)                 ∈ [0, 1]
        Routing weights   : (L × S,)                     ∈ [0, 1]
        × observation_history (stacked)

    Action Space (Box):
        Per leaf: [admission_rate, spine1, spine2, …, spineS]
        Shape: (L × (1 + S),)                            ∈ [-1, 1]
        admission_rate mapped via (x+1)/2 → [0,1]
        spine weights via per-leaf softmax.
    """

    metadata = {"render_modes": ["human"]}

    # V4: Node feature dim bumped to 9 (added normalized_backlog)
    NODE_FEAT_DIM = 9
    EDGE_FEAT_DIM = 7

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

        # Build simulator
        self.simulator = NetworkSimulator(
            self.topo_config, self.traffic_config, self.tcp_config, seed,
            step_duration_seconds=step_duration_seconds,
        )
        self.reward_calc = RewardCalculator(self.reward_config)

        # ── Dimensions ──────────────────────────────────
        self._num_links = self.simulator.num_links
        self._num_leaves = self.simulator.num_leaves
        self._num_spines = self.simulator.num_spines
        self._num_nodes = self._num_leaves + self._num_spines
        self._num_edges = 2 * self._num_leaves * self._num_spines
        self._weight_dim = self._num_leaves * self._num_spines

        # V4: Action dim = L × (1 + S) = admission_rate per leaf + spine weights
        self._action_dim = self._num_leaves * (1 + self._num_spines)

        # V4: Single-step observation layout:
        #   node_features + edge_features + intent_matrix +
        #   link_utils + link_queues + link_ecn +
        #   link_capacity_ratios + link_up_flags + routing_weights
        self._node_feat_size = self._num_nodes * self.NODE_FEAT_DIM
        self._edge_feat_size = self._num_edges * self.EDGE_FEAT_DIM
        self._intent_size = self._num_leaves * self._num_leaves

        self._single_obs_dim = (
            self._node_feat_size +
            self._edge_feat_size +
            self._intent_size +
            self._num_links * 5 +   # utils, queues, ecn, capacity, up
            self._weight_dim
        )
        self._obs_dim = self._single_obs_dim * self.obs_history

        # ── Spaces ──────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._action_dim,),
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

        V4: Action is parsed into admission_rates + spine weights.
        """
        # V4: Parse action → admission rates + routing weights
        admission_rates, weights = self._action_to_admission_and_weights(action)

        # Set admission rates on the simulator (edge buffer gate)
        self.simulator.set_admission_rates(admission_rates)

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
            "avg_capacity_ratio": float(state.link_capacity_ratios.mean()),
            "active_link_events": int(getattr(state, "active_link_events", 0)),
            "num_flows": state.num_active_flows,
            "num_elephant_flows": state.num_elephant_flows,
            "elephant_fcts": list(getattr(state, "elephant_fcts_this_step", [])),
            "dropped_traffic": getattr(state, "dropped_traffic", 0.0),
            "total_retransmissions": getattr(state, "total_retransmissions", 0),
            "total_drops_mb": getattr(state, "total_drops_mb", 0.0),
            "total_ecn_marks": getattr(state, "total_ecn_marks", 0),
            "avg_cwnd_fraction": getattr(state, "avg_cwnd_fraction", 1.0),
            "goodput": getattr(state, "total_goodput", 0.0),
            "has_intent": bool(state.intent_matrix.any()),
            "leaf_backlog_total": float(state.leaf_backlog.sum()) if hasattr(state, 'leaf_backlog') and state.leaf_backlog.size > 0 else 0.0,
            "edge_dropped_mb": getattr(state, 'edge_dropped_mb', 0.0),
            "edge_dropped_flows": getattr(state, 'edge_dropped_flows', 0),
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

    # ── V4: Action transform ───────────────────────────

    def _action_to_admission_and_weights(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse V4 action into admission rates + routing weights.

        Action layout per leaf (1 + S values):
            [admission_raw, spine0_raw, spine1_raw, …, spineS-1_raw]

        Returns:
            admission_rates: (num_leaves,)    ∈ [0, 1]
            weights:         (num_leaves, num_spines) softmax-normed
        """
        S = self._num_spines
        raw = action.reshape(self._num_leaves, 1 + S)

        # Admission rate: linear map [-1,1] → [0,1].
        # Heuristic agents already use this convention (Conductor: 2*rate-1).
        admission_rates = np.clip((raw[:, 0] + 1.0) * 0.5, 0.01, 1.0)

        # Spine weights: per-leaf softmax with temperature
        # Spine weights: per-leaf softmax with higher temperature
        # so small action differences create meaningful weight shifts
        spine_raw = raw[:, 1:]  # (L, S)
        scaled = spine_raw * 5.0  # temperature: action=±0.4 → 90/10 split
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_w = np.exp(shifted)
        weights = exp_w / exp_w.sum(axis=1, keepdims=True)
        weights = weights * 0.99 + 0.01   # min 1% per spine (numerical safety only)

        return admission_rates.astype(np.float32), weights

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """Legacy V2/V3 action transform (spine weights only).

        Used internally when loading old models. For V4, use
        ``_action_to_admission_and_weights`` instead.
        """
        raw = action.reshape(self._num_leaves, self._num_spines)
        scaled = raw * 3.0
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_w = np.exp(shifted)
        weights = exp_w / exp_w.sum(axis=1, keepdims=True)
        weights = weights * 0.9 + 0.1
        return weights

    # ── V2: Observation builder ─────────────────────────

    def _build_observation(self, state: NetworkState) -> np.ndarray:
        """Build a single-step observation with graph features + intent.

        Layout: [node_features | edge_features | intent_matrix |
                 link_utils | link_queues | link_ecn |
                 link_capacity | link_up | routing_weights]
        """
        # Node features (clipped to [0, 1])
        node_feats = np.clip(state.node_features.flatten(), 0.0, 1.0).astype(np.float32)
        # Pad if needed
        if node_feats.shape[0] < self._node_feat_size:
            node_feats = np.pad(node_feats, (0, self._node_feat_size - node_feats.shape[0]))

        # Edge features (clipped)
        edge_feats = np.clip(state.edge_features.flatten(), 0.0, 1.0).astype(np.float32)
        if edge_feats.shape[0] < self._edge_feat_size:
            edge_feats = np.pad(edge_feats, (0, self._edge_feat_size - edge_feats.shape[0]))

        # Intent matrix (already normalised [0, 1])
        intent = state.intent_matrix.flatten().astype(np.float32)
        if intent.shape[0] < self._intent_size:
            intent = np.pad(intent, (0, self._intent_size - intent.shape[0]))

        # Link-level features
        ecn_fracs = getattr(state, "link_ecn_fractions", np.zeros_like(state.link_utilizations))

        # Routing weights: compute actual split ratios (rows sum to 1)
        rw = state.routing_weights  # (L, S) raw weights
        row_sums = np.maximum(rw.sum(axis=1, keepdims=True), 1e-8)
        split_ratios = (rw / row_sums).flatten().astype(np.float32)

        obs = np.concatenate([
            node_feats[:self._node_feat_size],
            edge_feats[:self._edge_feat_size],
            intent[:self._intent_size],
            state.link_utilizations.astype(np.float32),
            state.link_queue_depths.astype(np.float32),
            ecn_fracs.astype(np.float32),
            state.link_capacity_ratios.astype(np.float32),
            state.link_up_flags.astype(np.float32),
            split_ratios,
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
            "link_capacity_ratios": state.link_capacity_ratios.tolist(),
            "link_up_flags": state.link_up_flags.tolist(),
            "avg_capacity_ratio": float(state.link_capacity_ratios.mean()),
            "active_link_events": int(getattr(state, "active_link_events", 0)),
            "dropped_traffic": getattr(state, "dropped_traffic", 0.0),
            # TCP health
            "total_retransmissions": getattr(state, "total_retransmissions", 0),
            "total_drops_mb": getattr(state, "total_drops_mb", 0.0),
            "total_ecn_marks": getattr(state, "total_ecn_marks", 0),
            "avg_cwnd_fraction": getattr(state, "avg_cwnd_fraction", 1.0),
            "flows_in_slow_start": getattr(state, "flows_in_slow_start", 0),
            "flows_in_congestion_avoidance": getattr(state, "flows_in_congestion_avoidance", 0),
            "flows_in_fast_recovery": getattr(state, "flows_in_fast_recovery", 0),
            # V2
            "has_intent": bool(state.intent_matrix.any()),
            # V4: Admission control
            "leaf_backlog": state.leaf_backlog.tolist() if hasattr(state, 'leaf_backlog') and state.leaf_backlog.size > 0 else [],
            "leaf_admission_rates": state.leaf_admission_rates.tolist() if hasattr(state, 'leaf_admission_rates') and state.leaf_admission_rates.size > 0 else [],
            # V4-fix: Edge buffer overflow
            "edge_dropped_mb": getattr(state, 'edge_dropped_mb', 0.0),
            "edge_dropped_flows": getattr(state, 'edge_dropped_flows', 0),
        }

        # FCT data
        fcts = getattr(state, "elephant_fcts_this_step", [])
        info["elephant_fcts_this_step"] = list(fcts)

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

        all_fcts = []
        for step_data in m:
            all_fcts.extend(step_data.get("elephant_fcts", []))

        intent_steps = sum(1 for x in m if x.get("has_intent", False))

        summary = {
            "avg_throughput_ratio": float(np.mean([x["throughput_ratio"] for x in m])),
            "avg_max_utilization": float(np.mean([x["max_utilization"] for x in m])),
            "avg_utilization_std": float(np.mean([x["utilization_std"] for x in m])),
            "avg_queue_depth": float(np.mean([x["queue_depth_avg"] for x in m])),
            "avg_capacity_ratio": float(np.mean([x.get("avg_capacity_ratio", 1.0) for x in m])),
            "total_reward": float(sum(x["reward"] for x in m)),
            "avg_reward": float(np.mean([x["reward"] for x in m])),
            "total_weight_changes": self.simulator.weight_changes,
            "avg_active_link_events": float(np.mean([x.get("active_link_events", 0) for x in m])),
            "total_dropped_traffic": float(sum(x.get("dropped_traffic", 0.0) for x in m)),
            "elephant_flows_completed": len(all_fcts),
            "intent_active_steps": intent_steps,
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
