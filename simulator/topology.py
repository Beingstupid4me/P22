"""
Topology engine for generating Clos (Leaf-Spine) fabrics.

Supports variable switch radix, oversubscription ratios, and
multi-tier topologies. The graph is stored as a NetworkX DiGraph
where each edge carries a Link object.

V4 realism extensions:
    - static per-link asymmetry (capacity / delay / buffer skew)
    - per-spine structural bias
    - scheduled brownouts and hard port failures
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx

from .link import Link


@dataclass
class TopologyConfig:
    """Configuration for Clos topology generation.

    Attributes:
        num_leaves: Number of Top-of-Rack (leaf) switches.
        num_spines: Number of spine switches.
        link_capacity_mbps: Capacity per link in Mbps.
        propagation_delay_ms: Base propagation delay per link in ms.
        buffer_size_mb: Buffer capacity per link in Mb-equivalent.
        static_capacity_variance: Per-link capacity skew sampled each episode.
        static_delay_variance: Per-link propagation skew sampled each episode.
        static_buffer_variance: Per-link buffer skew sampled each episode.
        spine_capacity_variance: Per-spine capacity bias sampled each episode.
        enable_link_events: Whether to inject dynamic port events.
        link_event_probability: Probability of sampling each event slot.
        max_dynamic_events: Max number of events per episode.
        event_start_min / event_start_max: Start-step range for an event.
        event_duration_min / event_duration_max: Duration range in steps.
        brownout_probability: Probability an event is a brownout.
        hard_failure_probability: Probability an event is a full outage.
        brownout_capacity_min / brownout_capacity_max: Remaining capacity
            fraction during a brownout.
        bidirectional_link_events: Apply event to both directions of a port.
    """

    num_leaves: int = 4
    num_spines: int = 2
    link_capacity_mbps: float = 1000.0
    propagation_delay_ms: float = 0.1
    buffer_size_mb: float = 10.0

    # Static asymmetry (sampled once per episode)
    static_capacity_variance: float = 0.0
    static_delay_variance: float = 0.0
    static_buffer_variance: float = 0.0
    spine_capacity_variance: float = 0.0

    # V4: Edge buffer capacity per leaf (Mb). Backlog exceeding this is dropped.
    edge_buffer_capacity_mb: float = 50000.0   # ~6.25 GB per leaf (server RAM / NIC ring)

    # Dynamic port / link events
    enable_link_events: bool = False
    link_event_probability: float = 0.0
    max_dynamic_events: int = 0
    event_start_min: int = 50
    event_start_max: int = 250
    event_duration_min: int = 25
    event_duration_max: int = 100
    brownout_probability: float = 0.7
    hard_failure_probability: float = 0.3
    brownout_capacity_min: float = 0.25
    brownout_capacity_max: float = 0.60
    bidirectional_link_events: bool = True


class ClosTopology:
    """Generates and manages a multi-tier Clos (Leaf-Spine) fabric.

    A Clos topology has full-mesh connectivity between the leaf and
    spine tiers.  Every leaf connects to every spine with a
    bidirectional link.

        Leaf-0 ──── Spine-0 ──── Leaf-2
               ╲  ╱         ╲  ╱
                ╲╱           ╲╱
               ╱╲           ╱╲
              ╱  ╲         ╱  ╲
        Leaf-1 ──── Spine-1 ──── Leaf-3
    """

    def __init__(self, config: TopologyConfig):
        self.config = config
        self.graph: nx.DiGraph = nx.DiGraph()
        self.links: Dict[Tuple[str, str], Link] = {}
        self.leaf_nodes: List[str] = []
        self.spine_nodes: List[str] = []
        self._build()

    # ── Construction ────────────────────────────────────

    def _build(self):
        """Build the full-mesh leaf-spine topology."""
        cfg = self.config

        # Create leaf nodes
        for i in range(cfg.num_leaves):
            nid = f"leaf_{i}"
            self.graph.add_node(nid, type="leaf", index=i)
            self.leaf_nodes.append(nid)

        # Create spine nodes
        for j in range(cfg.num_spines):
            nid = f"spine_{j}"
            self.graph.add_node(nid, type="spine", index=j)
            self.spine_nodes.append(nid)

        # Full mesh between leaves and spines (bidirectional)
        for leaf in self.leaf_nodes:
            for spine in self.spine_nodes:
                # Uplink: leaf → spine
                up = Link(
                    src=leaf,
                    dst=spine,
                    capacity=cfg.link_capacity_mbps,
                    propagation_delay=cfg.propagation_delay_ms,
                    buffer_size=cfg.buffer_size_mb,
                )
                self.graph.add_edge(leaf, spine)
                self.links[(leaf, spine)] = up

                # Downlink: spine → leaf
                down = Link(
                    src=spine,
                    dst=leaf,
                    capacity=cfg.link_capacity_mbps,
                    propagation_delay=cfg.propagation_delay_ms,
                    buffer_size=cfg.buffer_size_mb,
                )
                self.graph.add_edge(spine, leaf)
                self.links[(spine, leaf)] = down

    # ── Queries ─────────────────────────────────────────

    @property
    def num_links(self) -> int:
        return len(self.links)

    @property
    def num_leaves(self) -> int:
        return len(self.leaf_nodes)

    @property
    def num_spines(self) -> int:
        return len(self.spine_nodes)

    def get_link(self, src: str, dst: str) -> Link:
        """Return the Link object for a given (src, dst) pair."""
        return self.links[(src, dst)]

    def get_paths(self, src_leaf: str, dst_leaf: str) -> List[List[str]]:
        """Get all 2-hop paths between two leaves through spines."""
        if src_leaf == dst_leaf:
            return []
        return [[src_leaf, spine, dst_leaf] for spine in self.spine_nodes]

    def get_all_links(self) -> List[Link]:
        """Return all links in a deterministic order.

        Order: leaf-to-spine (uplinks) then spine-to-leaf (downlinks).
        Within each group, ordered by (leaf_index, spine_index).
        """
        ordered: List[Link] = []
        # Uplinks
        for leaf in self.leaf_nodes:
            for spine in self.spine_nodes:
                ordered.append(self.links[(leaf, spine)])
        # Downlinks
        for spine in self.spine_nodes:
            for leaf in self.leaf_nodes:
                ordered.append(self.links[(spine, leaf)])
        return ordered

    # ── State management ────────────────────────────────

    def reset(self):
        """Reset all link states to idle."""
        for link in self.links.values():
            link.reset()
