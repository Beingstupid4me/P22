"""
AI Training Workload Traffic Generator.

Simulates the dominant communication patterns of distributed LLM
training at scale:

    1. All-Reduce (gradient sync)  — every GPU sends to every other
    2. All-to-All  (expert routing) — MoE / pipeline-parallel shuffles
    3. Ring-AllReduce               — sequential ring-based gradient passing
    4. Background heartbeat         — sparse, low-bandwidth control plane

The model is ELEPHANT-HEAVY by design.  AI training fabrics are
defined by massive, synchronized bursts (gradients), NOT by many
small flows.  Mice flows are minimal background noise.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .link import Flow


@dataclass
class TrafficConfig:
    """Configuration for AI training traffic generation.

    Attributes:
        elephant_flow_bandwidth_mbps: Bandwidth per gradient flow (high).
        mice_flow_bandwidth_mbps: Bandwidth per heartbeat flow (low).
        mice_flow_rate: P(new mice flow) per leaf per step (keep LOW).
        elephant_burst_interval: Steps between synchronized training bursts.
        elephant_burst_size: Flows per burst (> num_spines to force congestion).
        elephant_flow_duration: Max lifetime of elephant flows (hard cap).
        mice_flow_duration: Lifetime of mice flows in steps.
        elephant_data_mb: Total data each elephant flow must push (Mb).
        bandwidth_variance: Relative std-dev on flow bandwidth.
        allreduce_probability: P(All-Reduce pattern) vs other patterns.
        alltoall_probability: P(All-to-All pattern).
        ring_probability: P(Ring-AllReduce pattern).
        min_burst_participants: Minimum leaves in a burst.
    """

    # 300B model scale: 600 GB gradients → 1.2 TB per All-Reduce
    # Scaled to 400 Gbps links: 320 Gbps = 80% of link rate
    elephant_flow_bandwidth_mbps: float = 320000.0
    mice_flow_bandwidth_mbps: float = 1000.0
    mice_flow_rate: float = 0.02
    elephant_burst_interval: int = 5
    elephant_burst_size: int = 16
    elephant_flow_duration: int = 20
    mice_flow_duration: int = 80
    elephant_data_mb: float = 75000.0     # 75 GB shard of the 1.2 TB sync
    bandwidth_variance: float = 0.05

    # Communication pattern mix (must sum to ~1.0)
    allreduce_probability: float = 0.50
    alltoall_probability: float = 0.25
    ring_probability: float = 0.25

    # Burst parameters
    min_burst_participants: int = 8

    # Host / IP generation
    hosts_per_rack: int = 8           # GPUs per leaf (for IP assignment)


class TrafficGenerator:
    """Generates elephant-heavy traffic mimicking LLM training.

    The traffic is dominated by massive, synchronized gradient
    exchange bursts that stress ECMP into hash collisions.
    Background mice flows are minimal (heartbeats, control).

    Pattern descriptions:
        All-Reduce:  Every participant sends to every other participant.
                     With N participants, creates N*(N-1) flows.
                     This is the worst case for ECMP.
        All-to-All:  Every participant sends to one random other.
                     Creates N flows. Models MoE expert routing.
        Ring:        Each participant sends to the next in a ring.
                     Creates N flows. Models ring-allreduce.
    """

    def __init__(
        self,
        config: TrafficConfig,
        num_leaves: int,
        seed: Optional[int] = None,
    ):
        self.config = config
        self.num_leaves = num_leaves
        self.rng = np.random.default_rng(seed)
        self.active_flows: List[Flow] = []
        self.completed_flows: List[Flow] = []
        self._flow_counter: int = 0
        self._step: int = 0

    # ── Public API ──────────────────────────────────────

    def step(self, step_duration_s: float = 1.0) -> List[Flow]:
        """Advance one time step: expire old flows, create new ones.

        Args:
            step_duration_s: Duration of one simulation step in seconds.
                Used to convert throughput (Mbps) to transferred data (MB)
                inside each flow's ``tick()`` method.

        Returns:
            Currently active flows after this step.
        """
        self._step += 1

        # Expire / complete flows
        still_active = []
        for f in self.active_flows:
            if f.tick(self._step, step_duration_s):
                still_active.append(f)
            else:
                self.completed_flows.append(f)
        self.active_flows = still_active

        # Sparse background heartbeat (minimal mice)
        self._generate_heartbeat()

        # Periodic synchronized training bursts (dominant traffic)
        if self._step % self.config.elephant_burst_interval == 0:
            self._generate_training_burst()

        return self.active_flows

    def reset(self, seed: Optional[int] = None):
        """Reset generator to initial state."""
        self.active_flows.clear()
        self.completed_flows.clear()
        self._flow_counter = 0
        self._step = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    # ── Properties ──────────────────────────────────────

    @property
    def num_active_flows(self) -> int:
        return len(self.active_flows)

    @property
    def num_elephant_flows(self) -> int:
        return sum(1 for f in self.active_flows if f.flow_type == "elephant")

    @property
    def num_mice_flows(self) -> int:
        return sum(1 for f in self.active_flows if f.flow_type == "mice")

    @property
    def num_completed(self) -> int:
        return len(self.completed_flows)

    # ── Training burst generator ────────────────────────

    def _generate_training_burst(self):
        """Spawn a synchronized training communication burst.

        Selects a pattern (All-Reduce, All-to-All, Ring) and creates
        the corresponding set of elephant flows across a subset of
        leaves simulating GPU nodes in a training job.
        """
        cfg = self.config

        # Determine number of participants (at least min_burst_participants)
        n_min = min(cfg.min_burst_participants, self.num_leaves)
        n_max = self.num_leaves
        n_participants = int(self.rng.integers(n_min, n_max + 1))
        participants = list(
            self.rng.choice(self.num_leaves, size=n_participants, replace=False)
        )

        # Pick communication pattern
        roll = self.rng.random()
        if roll < cfg.allreduce_probability:
            self._pattern_allreduce(participants)
        elif roll < cfg.allreduce_probability + cfg.alltoall_probability:
            self._pattern_alltoall(participants)
        else:
            self._pattern_ring(participants)

    def _pattern_allreduce(self, participants: List[int]):
        """All-Reduce: every participant sends to every other.

        Creates N × (N−1) flows — one per (src, dst) pair.  Each flow
        carries ``elephant_data_mb / N`` megabytes (the source node's
        gradient shard).  All N−1 outgoing flows from a single source
        compete for that leaf's uplinks via ECMP hashing, creating
        the realistic uplink oversubscription that causes TCP collapse.

        Example with 12 participants, 4 spines:
            - Each source sends 11 flows → hash to 4 uplinks → ~2.75 per uplink
            - After slow-start ramp: 2.75 × 256 Gbps = 704 Gbps per uplink
            - Uplink capacity = 400 Gbps → 176% overload → ECN → drops → AIMD
        """
        cfg = self.config
        n = len(participants)
        data_per_flow = cfg.elephant_data_mb / max(n, 1)  # shard / N

        for src in participants:
            for dst in participants:
                if src == dst:
                    continue
                bw = cfg.elephant_flow_bandwidth_mbps * (
                    1.0 + self.rng.normal(0, cfg.bandwidth_variance)
                )
                bw = max(100.0, bw)
                self._add_flow(
                    int(src), int(dst), bw, "elephant",
                    cfg.elephant_flow_duration,
                    data_mb=data_per_flow, pattern="allreduce",
                )

    def _pattern_alltoall(self, participants: List[int]):
        """All-to-All: each participant sends to one random other.

        Models MoE token routing or pipeline-parallel data shuffles.
        """
        cfg = self.config
        shuffled = list(participants)
        self.rng.shuffle(shuffled)
        created = 0
        for i, src in enumerate(participants):
            if created >= cfg.elephant_burst_size:
                break
            dst = shuffled[(i + 1) % len(shuffled)]
            if int(src) == int(dst):
                continue
            bw = cfg.elephant_flow_bandwidth_mbps * (
                1.0 + self.rng.normal(0, cfg.bandwidth_variance)
            )
            bw = max(100.0, bw)
            self._add_flow(
                int(src), int(dst), bw, "elephant", cfg.elephant_flow_duration,
                data_mb=cfg.elephant_data_mb * 0.7, pattern="alltoall",
            )
            created += 1

    def _pattern_ring(self, participants: List[int]):
        """Ring-AllReduce: each node sends to the next in a ring.

        Sequential pipeline — creates exactly N flows forming a ring.
        """
        cfg = self.config
        ring = list(participants)
        self.rng.shuffle(ring)
        created = 0
        for i in range(len(ring)):
            if created >= cfg.elephant_burst_size:
                break
            src = int(ring[i])
            dst = int(ring[(i + 1) % len(ring)])
            if src == dst:
                continue
            bw = cfg.elephant_flow_bandwidth_mbps * 0.8 * (
                1.0 + self.rng.normal(0, cfg.bandwidth_variance)
            )
            bw = max(100.0, bw)
            self._add_flow(
                src, dst, bw, "elephant", cfg.elephant_flow_duration,
                data_mb=cfg.elephant_data_mb * 0.5, pattern="ring",
            )
            created += 1

    # ── Background heartbeat ────────────────────────────

    def _generate_heartbeat(self):
        """Sparse background mice flows — minimal control plane traffic."""
        cfg = self.config
        for src in range(self.num_leaves):
            if self.rng.random() < cfg.mice_flow_rate:
                dst = self._random_dst(src)
                if dst is None:
                    continue
                bw = cfg.mice_flow_bandwidth_mbps * (
                    1.0 + self.rng.normal(0, cfg.bandwidth_variance)
                )
                self._add_flow(
                    src, dst, max(0.5, bw), "mice", cfg.mice_flow_duration,
                    data_mb=0, pattern="heartbeat",
                )

    # ── Helpers ─────────────────────────────────────────

    @staticmethod
    def _make_ip(leaf_idx: int, host_id: int) -> int:
        """Build a 32-bit IPv4 address: ``10.<leaf>.<0>.<host>``."""
        return (10 << 24) | ((leaf_idx & 0xFF) << 16) | (host_id & 0xFF)

    def _add_flow(
        self,
        src: int,
        dst: int,
        bandwidth: float,
        flow_type: str,
        duration: int,
        data_mb: float = 0.0,
        pattern: str = "",
    ):
        """Create and register a new flow with network identity."""
        cfg = self.config
        # ── Generate 5-tuple network identity ─────────
        src_host = int(self.rng.integers(0, cfg.hosts_per_rack))
        dst_host = int(self.rng.integers(0, cfg.hosts_per_rack))
        src_ip = self._make_ip(src, src_host)
        dst_ip = self._make_ip(dst, dst_host)
        src_port = int(self.rng.integers(49152, 65536))    # ephemeral
        # elephants use fixed NCCL port; mice get random service port
        dst_port = 5001 if flow_type == "elephant" else int(self.rng.integers(1024, 49152))

        flow = Flow(
            id=self._flow_counter,
            src=src,
            dst=dst,
            bandwidth=bandwidth,
            flow_type=flow_type,
            remaining_steps=duration,
            start_step=self._step,
            total_data_mb=data_mb,
            pattern=pattern,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=6,
        )
        self.active_flows.append(flow)
        self._flow_counter += 1

    def _random_dst(self, src: int) -> Optional[int]:
        """Pick a random destination ≠ src."""
        options = [i for i in range(self.num_leaves) if i != src]
        if not options:
            return None
        return int(self.rng.choice(options))

