"""
AI Training Workload Traffic Generator — V2/V3 with Spatial Intent.

Simulates the dominant communication patterns of distributed LLM
training at scale:

    1. All-Reduce (gradient sync)  — every GPU sends to every other
    2. All-to-All  (expert routing) — MoE / pipeline-parallel shuffles
    3. Ring-AllReduce               — sequential ring-based gradient passing
    4. Background heartbeat         — sparse, low-bandwidth control plane

V2 Enhancement — Intent Matrix (Pre-Intention / NCCL Hook):
    Real NCCL collectives are scheduled ahead of time.  The intent
    matrix announces upcoming bursts ``intent_lookahead`` steps in
    advance, giving the GNN brain time to pre-position weights
    BEFORE the flood arrives.  This is the "NCCL hook" concept:
    the fabric controller knows what's coming.

    Intent is a (num_leaves × num_leaves) matrix where entry (i, j)
    is the predicted demand that leaf-i intends to send to leaf-j
    in a future burst.
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
        elephant_burst_size: Legacy cap used by one-to-one burst patterns.
        elephant_flow_duration: Max lifetime of elephant flows (hard cap).
        mice_flow_duration: Lifetime of mice flows in steps.
        elephant_data_mb: Total data each elephant flow must push (Mb).
        bandwidth_variance: Relative std-dev on flow bandwidth.
        allreduce_probability: P(All-Reduce pattern) vs other patterns.
        alltoall_probability: P(All-to-All pattern).
        ring_probability: P(Ring-AllReduce pattern).
        min_burst_participants: Minimum leaves in a burst.
        max_burst_participants: Maximum leaves in a burst.
        subgroup_locality_probability: Probability burst participants come
            from a local sub-group rather than the full fabric.
        subgroup_window: Candidate local window size in leaves.
        intent_lookahead: Steps before burst to publish intent matrix.
    """

    # 300B model scale: 600 GB gradients → 1.2 TB per All-Reduce
    elephant_flow_bandwidth_mbps: float = 320000.0
    mice_flow_bandwidth_mbps: float = 1000.0
    mice_flow_rate: float = 0.02
    elephant_burst_interval: int = 5
    elephant_burst_size: int = 7
    elephant_flow_duration: int = 100
    mice_flow_duration: int = 80
    elephant_data_mb: float = 75000.0     # 75 GB shard of the 1.2 TB sync
    bandwidth_variance: float = 0.05

    # Communication pattern mix (must sum to ~1.0)
    allreduce_probability: float = 0.30
    alltoall_probability: float = 0.35
    ring_probability: float = 0.35

    # Burst parameters
    min_burst_participants: int = 4
    max_burst_participants: int = 7
    subgroup_locality_probability: float = 0.75
    subgroup_window: int = 8

    # Host / IP generation
    hosts_per_rack: int = 8

    # V2/V3: Intent / NCCL hook
    intent_lookahead: int = 2         # Announce burst 2 steps early


@dataclass
class PlannedFlow:
    """Blueprint entry for a future elephant flow."""

    src: int
    dst: int
    bandwidth: float
    duration: int
    data_mb: float
    pattern: str


@dataclass
class BurstPlan:
    """Deterministic description of one upcoming burst."""

    participants: List[int]
    pattern: str
    flows: List[PlannedFlow]


class TrafficGenerator:
    """Generates elephant-heavy traffic mimicking LLM training.

    V2 adds an Intent Matrix — a (num_leaves × num_leaves) demand
    forecast published ``intent_lookahead`` steps before each burst.
    This gives the RL agent advance notice to pre-position weights.

    Pattern descriptions:
        All-Reduce:  N*(N-1) flows — dense collective.
        All-to-All:  N flows — sparse expert-routing style.
        Ring:        N flows — sparse pipeline/ring exchange.

    V3 curriculum change:
        Bursts are no longer full-fabric 16-way walls by default.
        Instead, each burst samples a small sub-group (typically 4–7
        leaves, usually locality-biased).  This creates idle "holes"
        in the fabric — exactly the setting where an orchestrator can
        beat blind ECMP hashing.
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

        # V2: Intent matrix state
        self._intent_matrix = np.zeros(
            (num_leaves, num_leaves), dtype=np.float32
        )
        self._pending_intent: Optional[np.ndarray] = None
        self._intent_publish_step: int = -1  # step when intent becomes visible
        self._intent_clear_step: int = -1    # step when burst fires (clear intent)

    # ── Public API ──────────────────────────────────────

    def step(self, step_duration_s: float = 1.0) -> List[Flow]:
        """Advance one time step: expire old flows, create new ones.

        V2 intent lifecycle:
            1. ``intent_lookahead`` steps before a burst, the intent
               matrix is pre-computed and published.
            2. The agent sees the intent for ``intent_lookahead`` steps.
            3. When the burst actually fires, the intent is cleared.
        """
        self._step += 1

        # ── V2: Intent pre-computation ──────────────────
        # Check if the NEXT burst step minus lookahead equals this step
        cfg = self.config
        next_burst_step = self._next_burst_step()
        publish_step = next_burst_step - cfg.intent_lookahead

        if self._step == publish_step and publish_step > 0:
            # Pre-compute what the burst will look like
            self._pending_intent = self._precompute_burst_intent()
            self._intent_matrix = self._pending_intent.copy()
            self._intent_publish_step = self._step
            self._intent_clear_step = next_burst_step

        # Clear intent when burst fires
        if self._step == self._intent_clear_step:
            self._intent_matrix = np.zeros(
                (self.num_leaves, self.num_leaves), dtype=np.float32
            )
            self._pending_intent = None

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
        if self._step % cfg.elephant_burst_interval == 0:
            self._generate_training_burst()

        return self.active_flows

    def reset(self, seed: Optional[int] = None):
        """Reset generator to initial state."""
        self.active_flows.clear()
        self.completed_flows.clear()
        self._flow_counter = 0
        self._step = 0
        # V2: Reset intent state
        self._intent_matrix = np.zeros(
            (self.num_leaves, self.num_leaves), dtype=np.float32
        )
        self._pending_intent = None
        self._intent_publish_step = -1
        self._intent_clear_step = -1
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    # ── Properties ──────────────────────────────────────

    @property
    def intent_matrix(self) -> np.ndarray:
        """V2: Current intent matrix (num_leaves × num_leaves).

        Entry (i, j) is the predicted bandwidth-normalised demand leaf-i
        intends to send to leaf-j in the upcoming burst.  The matrix is
        pairwise / spatial: source and destination are preserved, not
        collapsed into a single scalar volume.
        """
        return self._intent_matrix.copy()

    @property
    def has_intent(self) -> bool:
        """True if a burst intent is currently published."""
        return self._pending_intent is not None

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

    # ── V2: Intent pre-computation ──────────────────────

    def _next_burst_step(self) -> int:
        """Return the next step at which a burst will fire."""
        interval = self.config.elephant_burst_interval
        if interval <= 0:
            return -1
        # Next multiple of interval after current step
        if self._step % interval == 0:
            return self._step + interval
        return ((self._step // interval) + 1) * interval

    def _precompute_burst_intent(self) -> np.ndarray:
        """Preview the next burst plan without perturbing the RNG.

        This ensures the published intent matrix exactly matches the
        future burst that will fire, including participant selection,
        pattern choice, and sampled per-flow bandwidths.

        Returns:
            (num_leaves × num_leaves) normalised demand matrix.
        """
        rng_state = self.rng.bit_generator.state
        plan = self._build_burst_plan()
        self.rng.bit_generator.state = rng_state
        return self._intent_from_plan(plan)

    # ── Training burst generator ────────────────────────

    def _generate_training_burst(self):
        """Spawn a synchronized training communication burst."""
        plan = self._build_burst_plan()
        for planned in plan.flows:
            self._add_flow(
                planned.src,
                planned.dst,
                planned.bandwidth,
                "elephant",
                planned.duration,
                data_mb=planned.data_mb,
                pattern=planned.pattern,
            )

    def _build_burst_plan(self) -> BurstPlan:
        """Sample a burst once and reuse it for both intent + execution."""
        cfg = self.config
        participants = self._sample_participants()

        roll = self.rng.random()
        if roll < cfg.allreduce_probability:
            pattern = "allreduce"
            flows = self._plan_allreduce(participants)
        elif roll < cfg.allreduce_probability + cfg.alltoall_probability:
            pattern = "alltoall"
            flows = self._plan_alltoall(participants)
        else:
            pattern = "ring"
            flows = self._plan_ring(participants)

        return BurstPlan(participants=participants, pattern=pattern, flows=flows)

    def _sample_participants(self) -> List[int]:
        """Sample a sparse participant set, usually from a local sub-group."""
        cfg = self.config
        n_min = max(2, min(cfg.min_burst_participants, self.num_leaves))
        n_max = max(n_min, min(cfg.max_burst_participants, self.num_leaves))
        n_participants = int(self.rng.integers(n_min, n_max + 1))

        use_local_group = (
            self.num_leaves > n_participants
            and self.rng.random() < cfg.subgroup_locality_probability
        )

        if use_local_group:
            window = max(n_participants, min(cfg.subgroup_window, self.num_leaves))
            anchor = int(self.rng.integers(0, self.num_leaves))
            candidates = [
                (anchor + offset) % self.num_leaves for offset in range(window)
            ]
            sampled = self.rng.choice(candidates, size=n_participants, replace=False)
        else:
            sampled = self.rng.choice(self.num_leaves, size=n_participants, replace=False)

        participants = [int(x) for x in sampled]
        participants.sort()
        return participants

    def _sample_elephant_bandwidth(self, multiplier: float = 1.0) -> float:
        """Sample one elephant bandwidth draw."""
        cfg = self.config
        bw = cfg.elephant_flow_bandwidth_mbps * multiplier * (
            1.0 + self.rng.normal(0, cfg.bandwidth_variance)
        )
        return max(100.0, bw)

    def _intent_from_plan(self, plan: BurstPlan) -> np.ndarray:
        """Convert a burst plan into a spatial intent matrix."""
        cfg = self.config
        demand = np.zeros((self.num_leaves, self.num_leaves), dtype=np.float32)
        normalizer = max(cfg.elephant_flow_bandwidth_mbps, 1e-8)

        for flow in plan.flows:
            demand[flow.src, flow.dst] += flow.bandwidth / normalizer

        return np.clip(demand, 0.0, 1.0)

    def _plan_allreduce(self, participants: List[int]) -> List[PlannedFlow]:
        """All-Reduce: every participant sends to every other."""
        cfg = self.config
        n = len(participants)
        data_per_flow = cfg.elephant_data_mb / max(n, 1)
        flows: List[PlannedFlow] = []

        for src in participants:
            for dst in participants:
                if src == dst:
                    continue
                flows.append(
                    PlannedFlow(
                        src=int(src),
                        dst=int(dst),
                        bandwidth=self._sample_elephant_bandwidth(),
                        duration=cfg.elephant_flow_duration,
                        data_mb=data_per_flow,
                        pattern="allreduce",
                    )
                )
        return flows

    def _plan_alltoall(self, participants: List[int]) -> List[PlannedFlow]:
        """All-to-All: one sparse flow per participant."""
        cfg = self.config
        shuffled = list(participants)
        self.rng.shuffle(shuffled)
        flows: List[PlannedFlow] = []

        for i, src in enumerate(participants):
            if len(flows) >= cfg.elephant_burst_size:
                break
            dst = int(shuffled[(i + 1) % len(shuffled)])
            if int(src) == dst:
                continue
            flows.append(
                PlannedFlow(
                    src=int(src),
                    dst=dst,
                    bandwidth=self._sample_elephant_bandwidth(),
                    duration=cfg.elephant_flow_duration,
                    data_mb=cfg.elephant_data_mb * 0.7,
                    pattern="alltoall",
                )
            )
        return flows

    def _plan_ring(self, participants: List[int]) -> List[PlannedFlow]:
        """Ring-AllReduce: one flow per participant, ordered ring."""
        cfg = self.config
        ring = list(participants)
        self.rng.shuffle(ring)
        flows: List[PlannedFlow] = []

        for i in range(len(ring)):
            if len(flows) >= cfg.elephant_burst_size:
                break
            src = int(ring[i])
            dst = int(ring[(i + 1) % len(ring)])
            if src == dst:
                continue
            flows.append(
                PlannedFlow(
                    src=src,
                    dst=dst,
                    bandwidth=self._sample_elephant_bandwidth(multiplier=0.8),
                    duration=cfg.elephant_flow_duration,
                    data_mb=cfg.elephant_data_mb * 0.5,
                    pattern="ring",
                )
            )
        return flows

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
        src_host = int(self.rng.integers(0, cfg.hosts_per_rack))
        dst_host = int(self.rng.integers(0, cfg.hosts_per_rack))
        src_ip = self._make_ip(src, src_host)
        dst_ip = self._make_ip(dst, dst_host)
        src_port = int(self.rng.integers(49152, 65536))
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
        """Pick a random destination != src."""
        options = [i for i in range(self.num_leaves) if i != src]
        if not options:
            return None
        return int(self.rng.choice(options))

