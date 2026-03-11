"""
High-level network simulator — the Digital Twin.

Combines topology, traffic generation, routing, and **TCP congestion
control** into a single simulation engine.

Each step runs the full feedback loop:
    1. Apply routing weights.
    2. Generate / expire flows (unit-correct data transfer).
    3. Assign each flow to exactly ONE spine via 5-tuple ECMP hash.
    4. Compute per-link loads from TCP sending rates.
    5. Links compute ECN marks and drops (buffer-based).
    6. Feed congestion signals back to flows (worst link wins).
    7. Flows adjust cwnd via AIMD.
    8. Compute per-flow throughput (fair-share on bottleneck link).
    9. Update queues, build state snapshot.

The single-spine assignment (Fix A) ensures that flows are NOT
fractionally split — every flow hashes to one path, creating the
real ECMP hash-collision congestion that destroys goodput under
elephant-heavy AI workloads.
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
    """Immutable snapshot of the network at a point in time."""

    step: int

    # Per-link arrays (ordered by ClosTopology.get_all_links)
    link_utilizations: np.ndarray       # [0, 1] clipped utilisation
    link_queue_depths: np.ndarray       # [0, 1] normalised queue depth
    link_loads: np.ndarray              # raw load in Mbps
    link_ecn_fractions: np.ndarray      # ECN marking fraction per link

    # Routing
    routing_weights: np.ndarray         # current weight matrix

    # Traffic summary
    num_active_flows: int
    num_elephant_flows: int
    num_mice_flows: int

    # Aggregate metrics  (computed from FLOWS, not links — no double-count)
    total_throughput: float             # sum of per-flow actual throughput
    total_demand: float                 # sum of per-flow bandwidth demand
    total_goodput: float                # sum of per-flow goodput
    max_utilization: float              # hottest single link
    utilization_std: float              # load-balance indicator

    # FCT metrics (this step)
    flows_completed_this_step: int = 0
    elephant_fcts_this_step: List[int] = field(default_factory=list)
    dropped_traffic: float = 0.0

    # TCP health
    total_retransmissions: int = 0      # Cumulative retransmit events
    total_drops_mb: float = 0.0         # Cumulative dropped volume
    total_ecn_marks: int = 0            # Cumulative ECN marks
    avg_cwnd_fraction: float = 1.0      # Avg cwnd/bandwidth across elephants
    flows_in_slow_start: int = 0
    flows_in_congestion_avoidance: int = 0
    flows_in_fast_recovery: int = 0


# ── Simulator ───────────────────────────────────────────


class NetworkSimulator:
    """Complete network simulation engine with TCP + ECMP flow hashing.

    Key realism features:
        1. Each flow hashes to exactly ONE spine via 5-tuple ECMP.
           No fractional splitting — collisions happen naturally.
        2. Throughput is computed per-FLOW (not per-link) to avoid
           double-counting uplink + downlink.
        3. Unit conversion is correct: Mbps × seconds / 8 = MB.
        4. TCP AIMD reacts to per-link congestion signals.

    Example collision scenario (why ECMP fails):
        - Leaf-0 sends 3 elephants at 320 Gbps each
        - Hash puts 2 on spine-0: leaf-0→spine-0 = 640 Gbps
        - Link capacity = 400 Gbps → buffer fills → drops
        - Both flows cut cwnd by 50% → throughput collapses
        - Meanwhile spine-1 has 1 flow at 320 Gbps (80% util)
        - RL agent learns to redistribute → no collision
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

        self.topology = ClosTopology(self.topo_config)
        self.traffic_gen = TrafficGenerator(
            self.traffic_config, self.topology.num_leaves, seed
        )
        self.routing = RoutingEngine(self.topology)

        self._step: int = 0
        self._prev_weights: Optional[np.ndarray] = None
        self._weight_changes: int = 0

    # ── Public API ──────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> NetworkState:
        """Reset simulator to initial (idle) state."""
        self.topology.reset()
        self.traffic_gen.reset(seed)
        self.routing.reset()
        self._step = 0
        self._prev_weights = None
        self._weight_changes = 0
        return self._get_state([], [], 0.0)

    def step(self, weights: Optional[np.ndarray] = None) -> NetworkState:
        """Advance simulation by one discrete time step.

        Full TCP feedback loop with single-spine ECMP hashing:
            1. Apply routing weights
            2. Generate / expire flows (with correct MB transfer)
            3. Init TCP for new flows
            4. Assign each flow to ONE spine (5-tuple hash)
            5. Compute per-link loads
            6. Links compute ECN marks and drops
            7. Feed congestion signals back to flows
            8. Flows run AIMD (adjust cwnd for next step)
            9. Compute per-flow throughput (fair-share bottleneck)
            10. Update queue depths
        """
        # 1. Apply new routing weights
        if weights is not None:
            self._prev_weights = self.routing.weights
            self.routing.set_weights(weights)
            if self._prev_weights is not None:
                if not np.allclose(weights, self._prev_weights, atol=0.01):
                    self._weight_changes += 1

        # Track completed flows
        completed_before = self.traffic_gen.num_completed

        # 2. Generate / expire traffic (correct Mbps→MB conversion)
        active_flows = self.traffic_gen.step(self.step_duration)

        # 3. Initialise TCP for new flows
        for flow in active_flows:
            flow.init_tcp(self.tcp_config)
            flow.step_duration_s = self.step_duration  # for slowdown calc

        # 4. Clear link loads, then assign each flow to ONE spine
        for link in self.topology.links.values():
            link.current_load = 0.0

        # Build per-link flow mapping for congestion feedback
        link_flows: Dict[Tuple[str, str], List[Flow]] = {}

        for flow in active_flows:
            if flow.src == flow.dst:
                continue

            # ── Sticky spine assignment ──────────────────
            # New flows (assigned_spine == -1) get hashed to a spine
            # using the current weight distribution.  Existing flows
            # keep their original spine for the rest of their life.
            # This prevents reactive weight changes from rerouting
            # mid-stream, which is how real ECMP works: the nexthop
            # only changes when the ECMP set membership changes, not
            # on every weight update.
            if flow.assigned_spine < 0:
                spine_idx = self.routing.assign_flow_to_spine(flow)
                flow.assigned_spine = spine_idx
            else:
                spine_idx = flow.assigned_spine

            send_rate = flow.sending_rate  # TCP-controlled rate

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

        # 7. Flows run TCP AIMD (adjust cwnd for next step)
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
        return self._get_state(active_flows, elephant_fcts, dropped)

    # ── TCP feedback ────────────────────────────────────

    def _apply_congestion_feedback(
        self,
        link_flows: Dict[Tuple[str, str], List[Flow]],
    ):
        """Propagate ECN marks and drops from links back to flows.

        A flow gets the WORST congestion signal from any link on
        its path (uplink or downlink).  Drop supersedes ECN.

        Since each flow is on exactly ONE spine (not split), it
        traverses exactly 2 links: one uplink + one downlink.
        The worst signal across both determines the feedback.
        """
        flow_drop: Set[int] = set()
        flow_ecn: Set[int] = set()

        for (src, dst), flows in link_flows.items():
            link = self.topology.links[(src, dst)]

            for flow in flows:
                fid = flow.id

                # Drop signal: link is tail-dropping packets
                if link.packets_dropped:
                    flow_drop.add(fid)

                # ECN signal: buffer above marking threshold
                elif link.ecn_marked:
                    flow_ecn.add(fid)

        # Apply worst signal to each flow
        for flow in self.traffic_gen.active_flows:
            if flow.id in flow_drop:
                flow.dropped = True
                flow.ecn_marked = False  # Drop supersedes ECN
            elif flow.id in flow_ecn:
                flow.ecn_marked = True

    # ── Throughput computation ──────────────────────────

    def _compute_flow_throughputs(self, flows: List[Flow]):
        """Compute per-flow throughput with fair-share congestion.

        Each flow is on exactly one spine.  Its actual throughput
        is the TCP sending rate scaled by the bottleneck link's
        fair-share factor:

            factor = min(1.0, capacity / load)

        If the uplink (leaf→spine) or downlink (spine→leaf) is
        overloaded, all flows on that link receive proportionally
        less throughput.  This is the hard capacity cap.
        """
        for flow in flows:
            if flow.assigned_spine < 0 or flow.src == flow.dst:
                flow.actual_throughput = 0.0
                continue

            send_rate = flow.sending_rate
            spine_idx = flow.assigned_spine

            src_node = self.topology.leaf_nodes[flow.src]
            spine_node = self.topology.spine_nodes[spine_idx]
            dst_node = self.topology.leaf_nodes[flow.dst]

            up = self.topology.links[(src_node, spine_node)]
            down = self.topology.links[(spine_node, dst_node)]

            # Fair share: capacity / total_load gives the fraction
            # each Mbps of demand actually receives
            up_factor = min(1.0, up.capacity / max(up.current_load, 1e-8))
            down_factor = min(1.0, down.capacity / max(down.current_load, 1e-8))

            # Bottleneck determines throughput (hard capacity cap)
            flow.actual_throughput = send_rate * min(up_factor, down_factor)

            # Update flow RTT estimate based on path latency
            up_lat = up.latency
            dn_lat = down.latency
            flow.rtt_estimate = max(0.1, (up_lat + dn_lat) / 1000.0)

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
    ) -> NetworkState:
        """Build a snapshot of the current network state.

        IMPORTANT: total_throughput and total_demand are computed
        from FLOWS (not links) to avoid the double-counting that
        made the throughput ratio artificially close to 1.0.
        """
        all_links = self.topology.get_all_links()

        utilizations = np.array([l.utilization_clipped for l in all_links])
        queue_depths = np.array([
            l.queue_depth / max(l.buffer_size, 1e-8) for l in all_links
        ])
        loads = np.array([l.current_load for l in all_links])
        ecn_fracs = np.array([l.ecn_fraction for l in all_links])

        # ── Flow-level aggregates (no double counting) ──
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

        return NetworkState(
            step=self._step,
            link_utilizations=utilizations,
            link_queue_depths=queue_depths,
            link_loads=loads,
            link_ecn_fractions=ecn_fracs,
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
            total_retransmissions=total_retrans,
            total_drops_mb=total_drops_cumulative,
            total_ecn_marks=total_ecn_cumulative,
            avg_cwnd_fraction=avg_cwnd,
            flows_in_slow_start=ss_count,
            flows_in_congestion_avoidance=ca_count,
            flows_in_fast_recovery=fr_count,
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
