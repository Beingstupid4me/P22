"""
High-level network simulator — the Digital Twin (V2/V4).

V2/V4 key changes from V1:
     1. FLOWLET re-evaluation: every step, each flow re-hashes
         through the current WCMP buckets (no more sticky spine).
     2. INTENT MATRIX: the traffic generator's intent (future burst
         demand forecast) is included in the network state snapshot.
     3. Graph-ready state: node/edge features for GNN consumption.
     4. Spatial intent projection onto edges: predicted future demand
         is exposed per edge, not only as a flat leaf-leaf matrix.
     5. Realism engine: static link asymmetry + dynamic brownouts /
         hard port failures sampled per episode.

Each step runs the full feedback loop:
    1. Apply routing weights (WCMP bucket update).
    2. Generate / expire flows (unit-correct data transfer).
    3. Re-assign EVERY flow to a spine via WCMP flowlet hash.
    4. Compute per-link loads from TCP sending rates.
    5. Links compute ECN marks and drops (buffer-based).
    6. Feed congestion signals back to flows (worst link wins).
    7. Flows adjust cwnd via AIMD.
    8. Compute per-flow throughput (fair-share on bottleneck link).
    9. Update queues, build state snapshot with intent.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .topology import ClosTopology, TopologyConfig
from .link import Link, Flow, TCPConfig
from .traffic import TrafficGenerator, TrafficConfig
from .routing import RoutingEngine


# ── State snapshot ──────────────────────────────────────


@dataclass
class NetworkState:
    """Immutable snapshot of the network at a point in time.

    V2 additions:
        - intent_matrix: (num_leaves × num_leaves) demand forecast
        - node_features / edge_features: for GNN observation
    """

    step: int

    # Per-link arrays (ordered by ClosTopology.get_all_links)
    link_utilizations: np.ndarray       # [0, 1] clipped utilisation
    link_queue_depths: np.ndarray       # [0, 1] normalised queue depth
    link_loads: np.ndarray              # raw load in Mbps
    link_ecn_fractions: np.ndarray      # ECN marking fraction per link
    link_capacity_ratios: np.ndarray    # effective / nominal capacity per link
    link_up_flags: np.ndarray           # 1 if link operational else 0

    # Routing
    routing_weights: np.ndarray         # current weight matrix

    # Traffic summary
    num_active_flows: int
    num_elephant_flows: int
    num_mice_flows: int

    # Aggregate metrics (computed from FLOWS, not links)
    total_throughput: float
    total_demand: float
    total_goodput: float
    max_utilization: float
    utilization_std: float

    # FCT metrics (this step)
    flows_completed_this_step: int = 0
    elephant_fcts_this_step: List[int] = field(default_factory=list)
    dropped_traffic: float = 0.0      # Mb dropped this step across all links
    step_duration_s: float = 1.0

    # TCP health
    total_retransmissions: int = 0
    total_drops_mb: float = 0.0
    total_ecn_marks: int = 0
    avg_cwnd_fraction: float = 1.0
    flows_in_slow_start: int = 0
    flows_in_congestion_avoidance: int = 0
    flows_in_fast_recovery: int = 0

    # V2: Intent matrix (num_leaves × num_leaves)
    intent_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

    # V2: Graph features for GNN
    node_features: np.ndarray = field(default_factory=lambda: np.array([]))
    edge_index: np.ndarray = field(default_factory=lambda: np.array([]))
    edge_features: np.ndarray = field(default_factory=lambda: np.array([]))

    # V4: impairment summary
    active_link_events: int = 0

    # V4: Leaf-level admission control
    leaf_backlog: np.ndarray = field(default_factory=lambda: np.array([]))
    leaf_admission_rates: np.ndarray = field(default_factory=lambda: np.array([]))

    # V4-fix: Edge buffer overflow tracking
    edge_dropped_mb: float = 0.0        # Mb dropped at edge this step
    edge_dropped_flows: int = 0          # Flows killed by edge overflow this step


@dataclass
class LinkEvent:
    """Episode-scoped link impairment event."""

    src: str
    dst: str
    start_step: int
    end_step: int
    capacity_scale: float
    is_up: bool
    kind: str

    def is_active(self, step: int) -> bool:
        return self.start_step <= step < self.end_step


# ── Simulator ───────────────────────────────────────────


class NetworkSimulator:
    """Complete network simulation with WCMP flowlet routing + intent.

    V2/V4 key realism features:
        1. Flowlet WCMP: flows re-hash every step through weighted
           buckets.  Weight changes take effect immediately.
        2. Intent matrix: upcoming burst demand is pre-announced.
        3. Graph-structured state for GNN observation.
        4. No synthetic TCP penalty is injected for path changes.
           TCP only reacts to ECN/drop signals produced by links.
        5. Static asymmetry + dynamic link events mimic real fabrics.
        6. TCP AIMD reacts to per-link congestion signals.
    """

    def __init__(
        self,
        topology_config: Optional[TopologyConfig] = None,
        traffic_config: Optional[TrafficConfig] = None,
        tcp_config: Optional[TCPConfig] = None,
        seed: Optional[int] = None,
        step_duration_seconds: float = 1.0,
    ):
        self.topo_config = topology_config or TopologyConfig()
        self.traffic_config = traffic_config or TrafficConfig()
        self.tcp_config = tcp_config or TCPConfig()
        self.step_duration = step_duration_seconds
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self.topology = ClosTopology(self.topo_config)
        self.traffic_gen = TrafficGenerator(
            self.traffic_config, self.topology.num_leaves, seed
        )
        self.routing = RoutingEngine(self.topology)

        self._step: int = 0
        self._prev_weights: Optional[np.ndarray] = None
        self._weight_changes: int = 0
        self._link_events: List[LinkEvent] = []
        self._active_link_events: int = 0

        # V4: Leaf admission rates (per-leaf injection control)
        self._leaf_admission_rates = np.ones(self.topology.num_leaves, dtype=np.float32)

        # V2: Pre-compute graph structure (edge_index for GNN)
        self._edge_index = self._build_edge_index()

    # ── Public API ──────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> NetworkState:
        """Reset simulator to initial (idle) state."""
        effective_seed = seed if seed is not None else self._seed
        self._rng = np.random.default_rng(effective_seed)

        self.topology.reset()
        self._sample_static_link_profiles()
        self._link_events = self._sample_link_events()
        self._apply_link_profiles_for_step(0)
        self.traffic_gen.reset(effective_seed)
        self.routing.reset()
        self._step = 0
        self._prev_weights = None
        self._weight_changes = 0
        self._leaf_admission_rates = np.ones(self.topology.num_leaves, dtype=np.float32)
        return self._get_state([], [], 0.0)

    def set_admission_rates(self, rates: np.ndarray):
        """V4: Set per-leaf admission rates ∈ [0, 1].

        Each leaf's flows will have their sending rate capped at
        ``admission_rate * flow.bandwidth``.  This simulates the
        edge buffer / priority flow control gate at each ToR switch.
        """
        assert rates.shape == (self.topology.num_leaves,), (
            f"Expected shape ({self.topology.num_leaves},), got {rates.shape}"
        )
        self._leaf_admission_rates = np.clip(rates, 0.0, 1.0).astype(np.float32)

    def step(self, weights: Optional[np.ndarray] = None) -> NetworkState:
        """Advance simulation by one discrete time step.

        V2 changes:
            - Step 4 now uses flowlet re-evaluation (no sticky spine).
            - State includes intent matrix and graph features.
        """
        # 1. Apply new routing weights
        if weights is not None:
            self._prev_weights = self.routing.weights
            self.routing.set_weights(weights)
            if self._prev_weights is not None:
                if not np.allclose(weights, self._prev_weights, atol=0.01):
                    self._weight_changes += 1

        # 1b. Apply static/dynamic link conditions for this step
        self._apply_link_profiles_for_step(self._step + 1)

        # Track completed flows
        completed_before = self.traffic_gen.num_completed

        # 2. Generate / expire traffic
        active_flows = self.traffic_gen.step(self.step_duration)

        # 3. Initialise TCP for new flows
        for flow in active_flows:
            flow.init_tcp(self.tcp_config)
            flow.step_duration_s = self.step_duration

        # ── V4-fix: Edge buffer overflow (Fix 1) ────────
        # Per-leaf backlog must not exceed edge_buffer_capacity_mb.
        # Excess flows are terminated and their undelivered data is
        # registered as edge drops (like a server NIC ring overflow).
        edge_buf_cap = self.topo_config.edge_buffer_capacity_mb
        edge_dropped_mb_step = 0.0
        edge_dropped_flows_step = 0

        leaf_remaining: Dict[int, List[Tuple[float, Flow]]] = {}
        for flow in active_flows:
            if flow.total_data_mb > 0 and flow.src < self.topology.num_leaves:
                remaining = max(0.0, flow.total_data_mb - flow.transferred_mb)
                leaf_remaining.setdefault(flow.src, []).append((remaining, flow))

        flows_to_drop: Set[int] = set()
        for leaf_idx, entries in leaf_remaining.items():
            total_backlog = sum(r for r, _ in entries)
            if total_backlog <= edge_buf_cap:
                continue
            # Sort by remaining data ascending — drop smallest first
            # (they contribute least to completion, clearing buffer fast)
            entries.sort(key=lambda x: x[0])
            excess = total_backlog - edge_buf_cap
            for remaining, flow in entries:
                if excess <= 0:
                    break
                # Edge-drop this flow
                edge_dropped_mb_step += remaining
                edge_dropped_flows_step += 1
                flows_to_drop.add(flow.id)
                flow.completed_step = self._step + 1  # mark done
                excess -= remaining

        if flows_to_drop:
            # Move dropped flows to completed list and remove from active
            surviving = []
            for flow in active_flows:
                if flow.id in flows_to_drop:
                    self.traffic_gen.completed_flows.append(flow)
                else:
                    surviving.append(flow)
            active_flows = surviving
            self.traffic_gen.active_flows = active_flows

        self._edge_dropped_mb = edge_dropped_mb_step
        self._edge_dropped_flows = edge_dropped_flows_step

        # 4. Clear link loads, then assign EVERY flow via WCMP flowlet
        for link in self.topology.links.values():
            link.current_load = 0.0

        # Build per-link flow mapping for congestion feedback
        link_flows: Dict[Tuple[str, str], List[Flow]] = {}

        # ── Pass 1: compute per-flow desired send rate + per-leaf total ──
        flow_send_rates: Dict[int, float] = {}
        leaf_total_desired = np.zeros(self.topology.num_leaves, dtype=np.float64)

        for flow in active_flows:
            if flow.src == flow.dst:
                continue
            admission = self._leaf_admission_rates[flow.src]
            desired = min(flow.sending_rate, admission * flow.bandwidth)
            flow_send_rates[flow.id] = desired
            leaf_total_desired[flow.src] += desired

        # ── V4-fix: Per-leaf uplink clamp (Fix 3) ───────
        # A leaf physically cannot inject more than the sum of its
        # uplink capacities (num_spines × link_capacity).  Enforce
        # credit-based / PFC-style line-rate clamping.
        leaf_uplink_cap = np.zeros(self.topology.num_leaves, dtype=np.float64)
        for i in range(self.topology.num_leaves):
            leaf_node = self.topology.leaf_nodes[i]
            for spine_node in self.topology.spine_nodes:
                leaf_uplink_cap[i] += self.topology.links[(leaf_node, spine_node)].capacity

        leaf_scale = np.ones(self.topology.num_leaves, dtype=np.float64)
        for i in range(self.topology.num_leaves):
            if leaf_total_desired[i] > leaf_uplink_cap[i] and leaf_total_desired[i] > 1e-8:
                leaf_scale[i] = leaf_uplink_cap[i] / leaf_total_desired[i]

        # ── Pass 2: assign flows, apply scaled send rates to links ──
        for flow in active_flows:
            if flow.src == flow.dst:
                continue

            # V2: Flowlet re-evaluation every step
            spine_idx = self.routing.assign_flowlet_to_spine(flow)
            flow.assigned_spine = spine_idx

            # Apply per-leaf uplink scale (Fix 3)
            send_rate = flow_send_rates.get(flow.id, 0.0) * leaf_scale[flow.src]

            # Store effective rate for throughput computation (Fix 2+3)
            flow.effective_send_rate = send_rate

            src_node = self.topology.leaf_nodes[flow.src]
            spine_node = self.topology.spine_nodes[spine_idx]
            dst_node = self.topology.leaf_nodes[flow.dst]

            # Uplink: leaf → spine
            up_key = (src_node, spine_node)
            self.topology.links[up_key].current_load += send_rate
            link_flows.setdefault(up_key, []).append(flow)

            # Downlink: spine → leaf
            dn_key = (spine_node, dst_node)
            self.topology.links[dn_key].current_load += send_rate
            link_flows.setdefault(dn_key, []).append(flow)

        # 5. Compute ECN and drop signals on each link
        for link in self.topology.links.values():
            link.compute_congestion_signals(self.tcp_config, dt=self.step_duration)

        # 6. Feed congestion signals back to flows
        self._apply_congestion_feedback(link_flows)

        # 7. Flows run TCP AIMD
        for flow in active_flows:
            flow.tcp_step(self.tcp_config, self.step_duration)

        # 8. Compute actual flow throughputs (fair-share bottleneck)
        self._compute_flow_throughputs(active_flows)

        # 9. Update queue depths
        for link in self.topology.links.values():
            link.update_queue(self.step_duration)

        # 10. Compute dropped traffic
        dropped = sum(l.drop_volume for l in self.topology.links.values())

        # Gather FCT data
        completed_after = self.traffic_gen.num_completed
        newly_completed = self.traffic_gen.completed_flows[completed_before:completed_after]
        elephant_fcts = [
            f.fct for f in newly_completed if f.flow_type == "elephant" and f.fct > 0
        ]

        self._step += 1
        return self._get_state(active_flows, elephant_fcts, dropped,
                               edge_dropped_mb_step, edge_dropped_flows_step)

    # ── Realism engine: asymmetry + events ─────────────

    def _sample_static_link_profiles(self):
        """Sample per-episode static asymmetry for every link."""
        cfg = self.topo_config

        spine_scales: Dict[str, float] = {}
        if cfg.spine_capacity_variance > 0:
            for spine in self.topology.spine_nodes:
                spine_scales[spine] = float(
                    np.clip(
                        1.0 + self._rng.uniform(-cfg.spine_capacity_variance, cfg.spine_capacity_variance),
                        0.05,
                        2.0,
                    )
                )
        else:
            for spine in self.topology.spine_nodes:
                spine_scales[spine] = 1.0

        for (src, dst), link in self.topology.links.items():
            cap_scale = 1.0
            if cfg.static_capacity_variance > 0:
                cap_scale *= float(
                    np.clip(
                        1.0 + self._rng.uniform(-cfg.static_capacity_variance, cfg.static_capacity_variance),
                        0.05,
                        2.0,
                    )
                )

            if src in spine_scales:
                cap_scale *= spine_scales[src]
            elif dst in spine_scales:
                cap_scale *= spine_scales[dst]

            delay_scale = 1.0
            if cfg.static_delay_variance > 0:
                delay_scale = float(
                    np.clip(
                        1.0 + self._rng.uniform(-cfg.static_delay_variance, cfg.static_delay_variance),
                        0.25,
                        4.0,
                    )
                )

            buffer_scale = 1.0
            if cfg.static_buffer_variance > 0:
                buffer_scale = float(
                    np.clip(
                        1.0 + self._rng.uniform(-cfg.static_buffer_variance, cfg.static_buffer_variance),
                        0.10,
                        4.0,
                    )
                )

            link.apply_static_profile(
                capacity_scale=cap_scale,
                delay_scale=delay_scale,
                buffer_scale=buffer_scale,
            )

    def _sample_link_events(self) -> List[LinkEvent]:
        """Sample runtime port brownouts / failures for this episode."""
        cfg = self.topo_config
        if not cfg.enable_link_events or cfg.max_dynamic_events <= 0:
            return []

        candidates = [
            (leaf, spine)
            for leaf in self.topology.leaf_nodes
            for spine in self.topology.spine_nodes
        ]
        self._rng.shuffle(candidates)
        events: List[LinkEvent] = []

        hard_prob = max(cfg.hard_failure_probability, 0.0)
        brown_prob = max(cfg.brownout_probability, 0.0)
        total_prob = hard_prob + brown_prob
        hard_threshold = hard_prob / total_prob if total_prob > 0 else 0.0

        for pair_idx in range(min(cfg.max_dynamic_events, len(candidates))):
            if self._rng.random() > cfg.link_event_probability:
                continue

            leaf_node, spine_node = candidates[pair_idx]
            start_low = min(cfg.event_start_min, cfg.event_start_max)
            start_high = max(cfg.event_start_min, cfg.event_start_max)
            duration_low = max(1, min(cfg.event_duration_min, cfg.event_duration_max))
            duration_high = max(duration_low, max(cfg.event_duration_min, cfg.event_duration_max))

            start_step = int(self._rng.integers(start_low, start_high + 1))
            duration = int(self._rng.integers(duration_low, duration_high + 1))
            end_step = start_step + duration

            roll = self._rng.random()
            if total_prob > 0 and roll < hard_threshold:
                is_up = False
                cap_scale = 0.0
                kind = "hard_failure"
            else:
                is_up = True
                cap_scale = float(
                    self._rng.uniform(cfg.brownout_capacity_min, cfg.brownout_capacity_max)
                )
                cap_scale = float(np.clip(cap_scale, 0.01, 1.0))
                kind = "brownout"

            directions = [(leaf_node, spine_node)]
            if cfg.bidirectional_link_events:
                directions.append((spine_node, leaf_node))

            for src, dst in directions:
                events.append(
                    LinkEvent(
                        src=src,
                        dst=dst,
                        start_step=start_step,
                        end_step=end_step,
                        capacity_scale=cap_scale,
                        is_up=is_up,
                        kind=kind,
                    )
                )

        return events

    def _apply_link_profiles_for_step(self, step: int):
        """Apply all active runtime impairment events for ``step``."""
        runtime_profile: Dict[Tuple[str, str], Tuple[bool, float, str]] = {
            key: (True, 1.0, "healthy") for key in self.topology.links.keys()
        }

        active_count = 0
        for event in self._link_events:
            if not event.is_active(step):
                continue
            active_count += 1
            key = (event.src, event.dst)
            if key not in runtime_profile:
                continue

            cur_up, cur_scale, cur_tag = runtime_profile[key]
            if not event.is_up:
                runtime_profile[key] = (False, 0.0, event.kind)
            else:
                runtime_profile[key] = (
                    cur_up,
                    min(cur_scale, event.capacity_scale),
                    event.kind if cur_tag == "healthy" else cur_tag,
                )

        for key, (is_up, scale, tag) in runtime_profile.items():
            self.topology.links[key].apply_dynamic_profile(
                is_up=is_up,
                capacity_scale=scale,
                tag=tag,
            )

        self._active_link_events = active_count

    # ── TCP feedback ────────────────────────────────────

    def _apply_congestion_feedback(
        self,
        link_flows: Dict[Tuple[str, str], List[Flow]],
    ):
        """Propagate ECN marks and drops from links back to flows."""
        flow_drop: Set[int] = set()
        flow_ecn: Set[int] = set()

        for (src, dst), flows in link_flows.items():
            link = self.topology.links[(src, dst)]

            for flow in flows:
                fid = flow.id
                if link.packets_dropped:
                    flow_drop.add(fid)
                elif link.ecn_marked:
                    flow_ecn.add(fid)

        for flow in self.traffic_gen.active_flows:
            if flow.id in flow_drop:
                flow.dropped = True
                flow.ecn_marked = False
            elif flow.id in flow_ecn:
                flow.ecn_marked = True

    # ── Throughput computation ──────────────────────────

    def _compute_flow_throughputs(self, flows: List[Flow]):
        """Compute per-flow throughput with fair-share congestion.

        V4-fix: Uses effective_send_rate (admission-capped + uplink-clamped)
        instead of raw flow.sending_rate. This ensures flows with low
        admission rates actually transfer less data.
        """
        for flow in flows:
            if flow.assigned_spine < 0 or flow.src == flow.dst:
                flow.actual_throughput = 0.0
                continue

            # Use the admission-capped, uplink-clamped rate
            send_rate = flow.effective_send_rate
            spine_idx = flow.assigned_spine

            src_node = self.topology.leaf_nodes[flow.src]
            spine_node = self.topology.spine_nodes[spine_idx]
            dst_node = self.topology.leaf_nodes[flow.dst]

            up = self.topology.links[(src_node, spine_node)]
            down = self.topology.links[(spine_node, dst_node)]

            up_factor = min(1.0, up.capacity / max(up.current_load, 1e-8))
            down_factor = min(1.0, down.capacity / max(down.current_load, 1e-8))

            flow.actual_throughput = send_rate * min(up_factor, down_factor)

            up_lat = up.latency
            dn_lat = down.latency
            flow.rtt_estimate = max(0.1, (up_lat + dn_lat) / 1000.0)

    # ── V2: Graph feature builders ──────────────────────

    def _build_edge_index(self) -> np.ndarray:
        """Build a (2 × num_edges) edge index for GNN.

        Node ordering: leaves first, then spines.
            leaf_i  → index i
            spine_j → index num_leaves + j

        Edges include both uplinks (leaf→spine) and downlinks (spine→leaf).
        """
        edges = []
        for i in range(self.topology.num_leaves):
            for j in range(self.topology.num_spines):
                leaf_idx = i
                spine_idx = self.topology.num_leaves + j
                # Uplink: leaf → spine
                edges.append([leaf_idx, spine_idx])
                # Downlink: spine → leaf
                edges.append([spine_idx, leaf_idx])

        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        return np.array(edges, dtype=np.int64).T  # shape (2, num_edges)

    def _build_node_features(self, active_flows: List[Flow]) -> np.ndarray:
        """Build per-node feature matrix for GNN.

        Node features (per node):
            [is_leaf, is_spine, num_flows_at_node, avg_queue_depth,
             max_link_util, predicted_future_load,
             avg_capacity_ratio, up_fraction, normalized_backlog]

        Shape: (num_leaves + num_spines, 9)
        """
        n_leaves = self.topology.num_leaves
        n_spines = self.topology.num_spines
        n_nodes = n_leaves + n_spines

        features = np.zeros((n_nodes, 9), dtype=np.float32)

        # is_leaf, is_spine
        features[:n_leaves, 0] = 1.0   # is_leaf
        features[n_leaves:, 1] = 1.0   # is_spine

        # Count flows per leaf (source)
        flow_counts = np.zeros(n_leaves, dtype=np.float32)
        for f in active_flows:
            if f.src < n_leaves:
                flow_counts[f.src] += 1
        max_flows = max(flow_counts.max(), 1.0)
        features[:n_leaves, 2] = flow_counts / max_flows

        # Per-node queue and utilization aggregates
        for i in range(n_leaves):
            leaf_node = self.topology.leaf_nodes[i]
            utils = []
            queues = []
            caps = []
            ups = []
            for j in range(n_spines):
                spine_node = self.topology.spine_nodes[j]
                up_key = (leaf_node, spine_node)
                if up_key in self.topology.links:
                    link = self.topology.links[up_key]
                    utils.append(link.utilization_clipped)
                    queues.append(link.buffer_fill_fraction)
                    caps.append(link.capacity_ratio)
                    ups.append(link.health_ratio)
            features[i, 3] = np.mean(queues) if queues else 0.0
            features[i, 4] = max(utils) if utils else 0.0
            features[i, 6] = np.mean(caps) if caps else 1.0
            features[i, 7] = np.mean(ups) if ups else 1.0

        for j in range(n_spines):
            spine_node = self.topology.spine_nodes[j]
            utils = []
            queues = []
            caps = []
            ups = []
            for i in range(n_leaves):
                leaf_node = self.topology.leaf_nodes[i]
                dn_key = (spine_node, leaf_node)
                if dn_key in self.topology.links:
                    link = self.topology.links[dn_key]
                    utils.append(link.utilization_clipped)
                    queues.append(link.buffer_fill_fraction)
                    caps.append(link.capacity_ratio)
                    ups.append(link.health_ratio)
            features[n_leaves + j, 3] = np.mean(queues) if queues else 0.0
            features[n_leaves + j, 4] = max(utils) if utils else 0.0
            features[n_leaves + j, 6] = np.mean(caps) if caps else 1.0
            features[n_leaves + j, 7] = np.mean(ups) if ups else 1.0

        # Intent load per leaf / spine
        intent = self.traffic_gen.intent_matrix
        if intent.size > 0 and intent.shape[0] == n_leaves:
            row_sums = intent.sum(axis=1)
            max_row = max(row_sums.max(), 1e-8)
            features[:n_leaves, 5] = row_sums / max_row

            projected_spine_load = (row_sums[:, None] * self.routing.split_ratios).sum(axis=0)
            max_spine = max(projected_spine_load.max(), 1e-8)
            features[n_leaves:, 5] = projected_spine_load / max_spine

        # V4: Normalized leaf backlog (remaining data in Mb at each leaf)
        #   backlog[i] = sum of (total_data_mb - transferred_mb) for flows at leaf i
        #   Normalized by a reference capacity (spine_capacity × step_duration × num_spines)
        leaf_backlog_mb = np.zeros(n_leaves, dtype=np.float32)
        for f in active_flows:
            if f.src < n_leaves and f.total_data_mb > 0:
                remaining = max(0.0, f.total_data_mb - f.transferred_mb)
                leaf_backlog_mb[f.src] += remaining
        max_backlog = max(leaf_backlog_mb.max(), 1e-8)
        features[:n_leaves, 8] = leaf_backlog_mb / max_backlog
        # Store raw backlog for state snapshot
        self._leaf_backlog_mb = leaf_backlog_mb

        return features

    def _build_edge_features(self) -> np.ndarray:
        """Build per-edge feature matrix for GNN.

        Edge features (per directed edge):
            [utilization, queue_fill, ecn_fraction, weight,
             predicted_intent_demand, capacity_ratio, link_up]

        Shape: (num_edges, 7)
        Edges are in the same order as _edge_index.
        """
        n_leaves = self.topology.num_leaves
        n_spines = self.topology.num_spines
        n_edges = 2 * n_leaves * n_spines  # up + down

        features = np.zeros((n_edges, 7), dtype=np.float32)

        # Spatial intent projection:
        #   uplink(i, j)   = outgoing_intent(i) routed via split(i, j)
        #   downlink(j, d) = Σ_src intent(src, d) * split(src, j)
        intent = self.traffic_gen.intent_matrix
        split = self.routing.split_ratios
        predicted_up = np.zeros((n_leaves, n_spines), dtype=np.float32)
        predicted_down = np.zeros((n_spines, n_leaves), dtype=np.float32)

        if intent.size > 0 and intent.shape == (n_leaves, n_leaves):
            outgoing = intent.sum(axis=1)
            for src in range(n_leaves):
                predicted_up[src, :] = outgoing[src] * split[src, :]
                for dst in range(n_leaves):
                    demand = intent[src, dst]
                    if demand <= 0.0:
                        continue
                    predicted_down[:, dst] += demand * split[src, :]

        pred_scale = max(
            float(predicted_up.max()) if predicted_up.size > 0 else 0.0,
            float(predicted_down.max()) if predicted_down.size > 0 else 0.0,
            1e-8,
        )
        predicted_up /= pred_scale
        predicted_down /= pred_scale

        idx = 0
        for i in range(n_leaves):
            for j in range(n_spines):
                leaf_node = self.topology.leaf_nodes[i]
                spine_node = self.topology.spine_nodes[j]

                # Uplink: leaf → spine
                up_key = (leaf_node, spine_node)
                if up_key in self.topology.links:
                    link = self.topology.links[up_key]
                    features[idx, 0] = link.utilization_clipped
                    features[idx, 1] = link.buffer_fill_fraction
                    features[idx, 2] = link.ecn_fraction
                    # Weight for this leaf→spine pair
                    features[idx, 3] = self.routing.split_ratios[i, j]
                    features[idx, 4] = predicted_up[i, j]
                    features[idx, 5] = link.capacity_ratio
                    features[idx, 6] = link.health_ratio
                idx += 1

                # Downlink: spine → leaf
                dn_key = (spine_node, leaf_node)
                if dn_key in self.topology.links:
                    link = self.topology.links[dn_key]
                    features[idx, 0] = link.utilization_clipped
                    features[idx, 1] = link.buffer_fill_fraction
                    features[idx, 2] = link.ecn_fraction
                    features[idx, 3] = self.routing.split_ratios[i, j]
                    features[idx, 4] = predicted_down[j, i]
                    features[idx, 5] = link.capacity_ratio
                    features[idx, 6] = link.health_ratio
                idx += 1

        return features

    # ── FCT accessors ──────────────────────────────────

    @property
    def all_completed_flows(self) -> List[Flow]:
        return self.traffic_gen.completed_flows

    @property
    def completed_elephant_flows(self) -> List[Flow]:
        return [f for f in self.traffic_gen.completed_flows if f.flow_type == "elephant"]

    @property
    def elephant_fcts(self) -> List[int]:
        return [f.fct for f in self.completed_elephant_flows if f.fct > 0]

    @property
    def tail_fct(self) -> float:
        fcts = self.elephant_fcts
        return float(np.percentile(fcts, 99)) if fcts else 0.0

    @property
    def mean_fct(self) -> float:
        fcts = self.elephant_fcts
        return float(np.mean(fcts)) if fcts else 0.0

    @property
    def elephant_slowdowns(self) -> List[float]:
        return [f.slowdown for f in self.completed_elephant_flows]

    # ── State builder ──────────────────────────────────

    def _get_state(
        self,
        active_flows: List[Flow],
        elephant_fcts: List[int],
        dropped: float,
        edge_dropped_mb: float = 0.0,
        edge_dropped_flows: int = 0,
    ) -> NetworkState:
        """Build state snapshot with V2 intent + graph features."""
        all_links = self.topology.get_all_links()

        utilizations = np.array([l.utilization_clipped for l in all_links])
        queue_depths = np.array([
            l.queue_depth / max(l.buffer_size, 1e-8) for l in all_links
        ])
        loads = np.array([l.current_load for l in all_links])
        ecn_fracs = np.array([l.ecn_fraction for l in all_links])
        capacity_ratios = np.array([l.capacity_ratio for l in all_links])
        up_flags = np.array([l.health_ratio for l in all_links])

        # Flow-level aggregates (no double counting)
        routed = [f for f in active_flows if f.src != f.dst]
        total_demand = sum(f.bandwidth for f in routed)
        total_throughput = sum(f.actual_throughput for f in routed)
        total_goodput = sum(f.goodput for f in routed)

        # TCP health metrics
        elephants = [f for f in active_flows if f.flow_type == "elephant"]
        total_retrans = sum(f.retransmissions for f in active_flows)
        total_drops_cumulative = sum(l.total_drops_mb for l in all_links)
        total_ecn_cumulative = sum(l.total_ecn_marks for l in all_links)

        avg_cwnd = 1.0
        if elephants:
            cwnd_fracs = [f.cwnd / max(f.bandwidth, 1e-8) for f in elephants]
            avg_cwnd = float(np.mean(cwnd_fracs))

        ss_count = sum(1 for f in active_flows if f.tcp_phase == "slow_start")
        ca_count = sum(1 for f in active_flows if f.tcp_phase == "congestion_avoidance")
        fr_count = sum(1 for f in active_flows if f.tcp_phase == "fast_recovery")

        # V2: Graph features
        node_feats = self._build_node_features(active_flows)
        edge_feats = self._build_edge_features()

        return NetworkState(
            step=self._step,
            link_utilizations=utilizations,
            link_queue_depths=queue_depths,
            link_loads=loads,
            link_ecn_fractions=ecn_fracs,
            link_capacity_ratios=capacity_ratios,
            link_up_flags=up_flags,
            routing_weights=self.routing.weights,
            num_active_flows=len(active_flows),
            num_elephant_flows=self.traffic_gen.num_elephant_flows,
            num_mice_flows=self.traffic_gen.num_mice_flows,
            total_throughput=total_throughput,
            total_demand=total_demand,
            total_goodput=total_goodput,
            max_utilization=float(utilizations.max()) if len(utilizations) > 0 else 0.0,
            utilization_std=float(utilizations.std()) if len(utilizations) > 0 else 0.0,
            flows_completed_this_step=len(elephant_fcts),
            elephant_fcts_this_step=elephant_fcts,
            dropped_traffic=dropped,
            step_duration_s=self.step_duration,
            total_retransmissions=total_retrans,
            total_drops_mb=total_drops_cumulative,
            total_ecn_marks=total_ecn_cumulative,
            avg_cwnd_fraction=avg_cwnd,
            flows_in_slow_start=ss_count,
            flows_in_congestion_avoidance=ca_count,
            flows_in_fast_recovery=fr_count,
            # V2
            intent_matrix=self.traffic_gen.intent_matrix,
            node_features=node_feats,
            edge_index=self._edge_index,
            edge_features=edge_feats,
            active_link_events=self._active_link_events,
            # V4: Leaf admission control
            leaf_backlog=getattr(self, '_leaf_backlog_mb', np.zeros(self.topology.num_leaves)),
            leaf_admission_rates=self._leaf_admission_rates.copy(),
            # V4-fix: Edge buffer overflow
            edge_dropped_mb=edge_dropped_mb,
            edge_dropped_flows=edge_dropped_flows,
        )

    # ── Accessors ───────────────────────────────────────

    @property
    def weight_changes(self) -> int:
        return self._weight_changes

    @property
    def num_links(self) -> int:
        return self.topology.num_links

    @property
    def num_leaves(self) -> int:
        return self.topology.num_leaves

    @property
    def num_spines(self) -> int:
        return self.topology.num_spines

    @property
    def num_nodes(self) -> int:
        """Total graph nodes: leaves + spines."""
        return self.topology.num_leaves + self.topology.num_spines

    @property
    def num_edges(self) -> int:
        """Total directed edges in the graph."""
        return self._edge_index.shape[1] if self._edge_index.size > 0 else 0
