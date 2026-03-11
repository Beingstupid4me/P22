"""
Link and Flow models with realistic TCP congestion control.

Implements:
    - TCP AIMD (Additive Increase / Multiplicative Decrease)
    - Slow Start + Congestion Avoidance phases
    - ECN (Explicit Congestion Notification) marking
    - Packet drops and retransmission tracking
    - Buffer overflow with tail-drop and RED-like early drops
    - Flow Completion Time (FCT) tracking with data-limited completion

The congestion model is fluid-flow (bandwidth, not packets) but
applies the same dynamics as real TCP at a macro level.  This
makes the simulator produce the "TCP collapse" behaviour that
destroys ECMP performance under elephant-heavy AI workloads.
"""

from dataclasses import dataclass, field
from typing import List
import math


# ── TCP Congestion Control Parameters ──────────────────


@dataclass
class TCPConfig:
    """Tunable parameters for the TCP congestion model.

    Defaults are calibrated for 400 Gbps data-centre networks
    (similar to DCQCN / Swift / DCTCP dynamics).
    """

    # Slow Start
    initial_cwnd_fraction: float = 0.1      # Fraction of link capacity
    ss_growth_factor: float = 2.0           # Doubles each RTT (exponential)

    # Congestion Avoidance (Additive Increase)
    ai_increment: float = 0.05             # Fraction of capacity per step

    # Multiplicative Decrease
    md_factor: float = 0.5                 # Cut window in half on loss

    # ECN / marking thresholds (fraction of buffer)
    ecn_marking_threshold: float = 0.3     # Start marking at 30% buffer
    drop_threshold: float = 0.95           # Start tail-dropping at 95%

    # Retransmission
    retransmit_penalty: float = 1.0        # Steps wasted per retransmit event

    # Recovery
    fast_recovery_exit: float = 0.7        # Exit fast-recovery at 70% cwnd

    # Minimum sending rate (fraction of capacity)
    min_rate_fraction: float = 0.01        # 1% floor to avoid total stall


# ── Flow (with TCP state) ──────────────────────────────


@dataclass
class Flow:
    """A network flow with built-in TCP congestion control.

    Each flow independently runs AIMD congestion control:
        - Starts in SLOW_START, exponentially growing its window.
        - On ECN mark → transitions to CONGESTION_AVOIDANCE (AI phase).
        - On packet drop → multiplicative decrease + retransmit.
        - Tracks total data, retransmissions, and FCT.

    Attributes:
        id: Unique flow identifier.
        src: Source leaf index.
        dst: Destination leaf index.
        bandwidth: MAXIMUM demanded bandwidth in Mbps (line rate).
        flow_type: 'elephant' or 'mice'.
        remaining_steps: Hard time-limit cap.
        start_step: Step when flow was created.
        total_data_mb: Total data to transfer (Mb). 0 = unlimited.
        transferred_mb: Data transferred so far (Mb).
        actual_throughput: Achieved throughput this step (Mbps).
        pattern: Communication pattern that spawned this flow.
        completed_step: Step when flow completed (-1 if active).
    """

    id: int
    src: int
    dst: int
    bandwidth: float                    # Max sending rate (line rate)
    flow_type: str
    remaining_steps: int
    start_step: int
    total_data_mb: float = 0.0
    transferred_mb: float = 0.0
    actual_throughput: float = 0.0
    effective_send_rate: float = 0.0      # V4-fix: after admission + uplink clamp
    pattern: str = ""
    completed_step: int = -1

    # Network identity (5-tuple for ECMP hashing)
    src_ip: int = 0                     # Source IPv4 as 32-bit integer
    dst_ip: int = 0                     # Destination IPv4 as 32-bit integer
    src_port: int = 0                   # Source port (ephemeral)
    dst_port: int = 0                   # Destination port (service)
    protocol: int = 6                   # IP protocol (6 = TCP)

    # Spine assignment (set by routing engine per step)
    assigned_spine: int = -1            # Which spine this flow routes through

    # ── TCP Congestion Control State ────────────────────

    cwnd: float = -1.0                  # Congestion window (Mbps). -1 = uninitialised
    ssthresh: float = -1.0             # Slow-start threshold (Mbps)
    tcp_phase: str = "slow_start"       # slow_start | congestion_avoidance | fast_recovery
    ecn_marked: bool = False            # ECN flag from last step
    dropped: bool = False               # Drop flag from last step
    retransmissions: int = 0            # Total retransmit events
    retransmit_data_mb: float = 0.0     # Data that had to be retransmitted
    rtt_estimate: float = 1.0           # Estimated RTT in steps (updated by link latency)

    def init_tcp(self, tcp_config: TCPConfig):
        """Initialise TCP state relative to flow's line rate."""
        if self.cwnd < 0:
            self.cwnd = self.bandwidth * tcp_config.initial_cwnd_fraction
            self.ssthresh = self.bandwidth * 0.7   # 70% of line rate
            self.tcp_phase = "slow_start"

    @property
    def sending_rate(self) -> float:
        """Current sending rate (capped by cwnd and line rate)."""
        if self.cwnd < 0:
            return self.bandwidth   # If TCP not initialised, send at max
        return min(self.cwnd, self.bandwidth)

    def tcp_step(self, tcp_config: TCPConfig, step_duration_s: float = 1.0):
        """Run one step of TCP congestion control based on feedback.

        Called AFTER link feedback (ecn_marked / dropped) is set.
        ``step_duration_s`` is needed to convert throughput (Mbps) to
        data volume (MB) for retransmission bookkeeping.
        """
        min_rate = self.bandwidth * tcp_config.min_rate_fraction

        if self.dropped:
            # ── Multiplicative Decrease ──────────────────
            self.ssthresh = max(self.cwnd * tcp_config.md_factor, min_rate)
            self.cwnd = self.ssthresh
            self.tcp_phase = "fast_recovery"
            self.retransmissions += 1
            # Retransmit penalty: fraction of this step's data is lost (MB)
            step_data_mb = self.actual_throughput * step_duration_s / 8.0
            lost_data = step_data_mb * tcp_config.retransmit_penalty * 0.1
            self.retransmit_data_mb += lost_data
            self.transferred_mb = max(0.0, self.transferred_mb - lost_data)
            self.dropped = False

        elif self.ecn_marked:
            # ── ECN reaction (gentler than drop) ─────────
            if self.tcp_phase != "fast_recovery":
                self.ssthresh = max(self.cwnd * tcp_config.md_factor, min_rate)
                self.cwnd = self.ssthresh
                self.tcp_phase = "congestion_avoidance"
            self.ecn_marked = False

        else:
            # ── No congestion signal → grow window ───────
            if self.tcp_phase == "slow_start":
                # Exponential growth
                self.cwnd = min(
                    self.cwnd * tcp_config.ss_growth_factor,
                    self.bandwidth,
                )
                if self.cwnd >= self.ssthresh:
                    self.tcp_phase = "congestion_avoidance"

            elif self.tcp_phase == "congestion_avoidance":
                # Additive Increase
                self.cwnd = min(
                    self.cwnd + self.bandwidth * tcp_config.ai_increment,
                    self.bandwidth,
                )

            elif self.tcp_phase == "fast_recovery":
                # Gradually recover
                target = self.ssthresh / tcp_config.fast_recovery_exit
                if self.cwnd < target:
                    self.cwnd = min(
                        self.cwnd + self.bandwidth * tcp_config.ai_increment * 0.5,
                        self.bandwidth,
                    )
                    self.tcp_phase = "congestion_avoidance"

        self.cwnd = max(self.cwnd, min_rate)

    def tick(self, current_step: int, step_duration_s: float = 1.0) -> bool:
        """Advance one time step. Returns True if flow is still active.

        Converts throughput (Mbps) to data transferred (MB):
            MB_transferred = throughput_Mbps × duration_s / 8

        Completion rules (V4-fix: Ghost-Flow exploit prevention):
          - Data-limited flows (total_data_mb > 0) ONLY finish when all
            data has been transferred. They cannot silently time-out.
          - Time-limited flows (total_data_mb == 0, e.g. mice) expire
            when remaining_steps reaches zero.
        """
        throughput_mbytes = self.actual_throughput * step_duration_s / 8.0
        self.transferred_mb += throughput_mbytes

        if self.total_data_mb > 0 and self.transferred_mb >= self.total_data_mb:
            self.completed_step = current_step
            return False

        # Data-limited flows: do NOT expire by time — they must finish
        # their transfer or be edge-dropped.  This prevents the agent
        # from setting admission=0 and waiting for free timeouts.
        if self.total_data_mb > 0:
            self.remaining_steps = max(self.remaining_steps - 1, 1)
            return True

        # Time-limited flows (mice / unlimited): expire normally
        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            self.completed_step = current_step
            return False
        return True

    @property
    def is_data_complete(self) -> bool:
        return self.total_data_mb > 0 and self.transferred_mb >= self.total_data_mb

    @property
    def fct(self) -> int:
        """Flow Completion Time: steps from creation to completion."""
        if self.completed_step > 0:
            return self.completed_step - self.start_step
        return 0

    # Step duration used for slowdown computation (set per-episode)
    step_duration_s: float = 1.0

    @property
    def slowdown(self) -> float:
        """FCT slowdown: actual / ideal (1.0 = no congestion).

        Ideal steps = total_data_mb / (bandwidth_Mbps × step_duration_s / 8).
        At full line rate, one step transfers
        ``bandwidth × step_duration / 8`` megabytes.
        """
        if self.completed_step <= 0 or self.bandwidth <= 0:
            return 1.0
        mb_per_step = self.bandwidth * self.step_duration_s / 8.0
        ideal_steps = max(1.0, self.total_data_mb / mb_per_step)
        actual_steps = self.completed_step - self.start_step
        return max(1.0, actual_steps / ideal_steps)

    @property
    def goodput(self) -> float:
        """Effective goodput excluding retransmitted data."""
        if self.transferred_mb <= 0:
            return 0.0
        useful = self.transferred_mb
        total = useful + self.retransmit_data_mb
        if total <= 0:
            return 0.0
        return self.actual_throughput * (useful / total)


# ── Link (with ECN, drops, and buffer dynamics) ────────


@dataclass
class Link:
    """Directed network link with realistic congestion behaviour.

    Models:
        - Fluid-flow congestion (load vs capacity)
        - Buffer fill / drain dynamics
        - ECN marking when buffer exceeds threshold
        - Tail-drop when buffer is near-full
        - Per-step drop/retransmission accounting
        - Latency = propagation + queueing

    Attributes:
        src: Source node ID (e.g. 'leaf_0').
        dst: Destination node ID (e.g. 'spine_1').
        capacity: Maximum throughput in Mbps.
        propagation_delay: Base propagation delay in ms.
        buffer_size: Buffer capacity in Mb-equivalent.

    Realism extensions:
        - static per-episode capacity / delay / buffer skew
        - dynamic brownouts (capacity reduction while link stays up)
        - dynamic hard failures (port down, traffic blackholed)
    """

    src: str
    dst: str
    capacity: float
    propagation_delay: float
    buffer_size: float

    # Physical baselines (captured at construction)
    base_capacity: float = field(init=False)
    base_propagation_delay: float = field(init=False)
    base_buffer_size: float = field(init=False)

    # Static / dynamic health profile
    static_capacity_scale: float = 1.0
    static_delay_scale: float = 1.0
    static_buffer_scale: float = 1.0
    dynamic_capacity_scale: float = 1.0
    is_up: bool = True
    impairment_tag: str = "healthy"

    # ── Dynamic state ───────────────────────────────────
    current_load: float = 0.0
    queue_depth: float = 0.0
    _prev_utilization: float = 0.0

    # Per-step congestion signals
    ecn_marked: bool = False
    packets_dropped: bool = False
    drop_volume: float = 0.0            # Mbps of traffic dropped this step
    ecn_fraction: float = 0.0           # Fraction of traffic ECN-marked

    # Cumulative stats
    total_drops_mb: float = 0.0
    total_ecn_marks: int = 0
    total_retransmit_events: int = 0

    def __post_init__(self):
        self.base_capacity = self.capacity
        self.base_propagation_delay = self.propagation_delay
        self.base_buffer_size = self.buffer_size

    # ── Physical profile helpers ───────────────────────

    def apply_static_profile(
        self,
        capacity_scale: float = 1.0,
        delay_scale: float = 1.0,
        buffer_scale: float = 1.0,
    ):
        """Apply persistent per-episode asymmetry."""
        self.static_capacity_scale = max(capacity_scale, 0.01)
        self.static_delay_scale = max(delay_scale, 0.01)
        self.static_buffer_scale = max(buffer_scale, 0.01)
        self._refresh_physical_state()

    def apply_dynamic_profile(
        self,
        *,
        is_up: bool = True,
        capacity_scale: float = 1.0,
        tag: str = "healthy",
    ):
        """Apply runtime impairment state for the current step."""
        self.is_up = bool(is_up)
        self.dynamic_capacity_scale = max(capacity_scale, 0.0)
        self.impairment_tag = tag
        self._refresh_physical_state()

    def _refresh_physical_state(self):
        """Recompute effective physical properties."""
        if not self.is_up:
            self.capacity = 0.0
            self.propagation_delay = self.base_propagation_delay * self.static_delay_scale
            self.buffer_size = 0.0
            return

        self.capacity = (
            self.base_capacity
            * self.static_capacity_scale
            * self.dynamic_capacity_scale
        )
        self.propagation_delay = self.base_propagation_delay * self.static_delay_scale
        self.buffer_size = self.base_buffer_size * self.static_buffer_scale

    # ── Derived properties ──────────────────────────────

    @property
    def utilization(self) -> float:
        """Raw utilization ratio (can exceed 1.0 when overloaded)."""
        if self.capacity <= 0:
            return 0.0
        return self.current_load / self.capacity

    @property
    def capacity_ratio(self) -> float:
        """Effective capacity relative to the original hardware rate."""
        if self.base_capacity <= 0:
            return 0.0
        return min(1.0, self.capacity / self.base_capacity)

    @property
    def health_ratio(self) -> float:
        """Binary operational health indicator."""
        return 1.0 if self.is_up else 0.0

    @property
    def utilization_clipped(self) -> float:
        """Utilization clamped to [0, 1]."""
        return min(self.utilization, 1.0)

    @property
    def effective_throughput(self) -> float:
        """Actual delivered throughput (capped at capacity)."""
        return min(self.current_load, self.capacity)

    @property
    def dropped_traffic(self) -> float:
        """Traffic dropped this step."""
        return self.drop_volume

    @property
    def latency(self) -> float:
        """Total latency: propagation + queueing delay (ms)."""
        if self.capacity <= 0:
            return self.propagation_delay
        queue_delay = (self.queue_depth / self.capacity) * 1000.0
        return self.propagation_delay + queue_delay

    @property
    def congested(self) -> bool:
        """True when load exceeds capacity."""
        return self.current_load > self.capacity

    @property
    def buffer_fill_fraction(self) -> float:
        """How full the buffer is [0, 1]."""
        if self.buffer_size <= 0:
            return 1.0 if self.queue_depth > 0 else 0.0
        return min(self.queue_depth / self.buffer_size, 1.0)

    # ── Congestion feedback ─────────────────────────────

    def compute_congestion_signals(self, tcp_config: TCPConfig, dt: float = 1.0):
        """Determine ECN marks and drops based on buffer state.

        Called AFTER loads are applied, BEFORE queue update.
        Sets ecn_marked, packets_dropped, drop_volume, ecn_fraction.

        Args:
            tcp_config: TCP tuning parameters.
            dt: Step duration in seconds (needed to convert overflow
                rate to data volume for correct drop accounting).
        """
        fill = self.buffer_fill_fraction

        # Reset per-step flags
        self.ecn_marked = False
        self.packets_dropped = False
        self.drop_volume = 0.0
        self.ecn_fraction = 0.0

        # Hard-down port: all arriving traffic is lost immediately.
        if not self.is_up or self.capacity <= 0:
            lost_volume = self.current_load * dt + self.queue_depth
            if lost_volume > 0:
                self.packets_dropped = True
                self.drop_volume = lost_volume
                self.total_drops_mb += lost_volume
            self.queue_depth = 0.0
            return

        # ECN marking: proportional to buffer fill above threshold
        if fill >= tcp_config.ecn_marking_threshold:
            self.ecn_marked = True
            # Fraction of traffic that gets ECN-marked (probabilistic)
            ecn_range = tcp_config.drop_threshold - tcp_config.ecn_marking_threshold
            if ecn_range > 0:
                self.ecn_fraction = min(
                    1.0,
                    (fill - tcp_config.ecn_marking_threshold) / ecn_range,
                )
            else:
                self.ecn_fraction = 1.0
            self.total_ecn_marks += 1

        # Tail-drop: when buffer is nearly full AND load exceeds capacity
        overflow = self.current_load - self.capacity   # Mbps (excess rate)
        if overflow > 0 and fill >= tcp_config.drop_threshold:
            overflow_volume = overflow * dt                # Mb (data arriving this step)
            bufferable = max(0.0, self.buffer_size - self.queue_depth)  # Mb (space left)
            self.drop_volume = max(0.0, overflow_volume - bufferable)   # Mb dropped
            if self.drop_volume > 0:
                self.packets_dropped = True
                self.total_drops_mb += self.drop_volume

    def update_queue(self, dt: float = 1.0):
        """Update queue depth for one time step.

        Queue builds up when load > capacity and drains otherwise.
        Queue is capped at buffer_size (excess was already dropped).
        """
        if not self.is_up or self.capacity <= 0:
            self.queue_depth = 0.0
            self._prev_utilization = 0.0
            return

        excess_rate = self.current_load - self.capacity
        if excess_rate > 0:
            # Fill buffer (but don't exceed capacity — drops handled separately)
            self.queue_depth = min(
                self.queue_depth + excess_rate * dt,
                self.buffer_size,
            )
        else:
            # Drain buffer
            self.queue_depth = max(
                self.queue_depth + excess_rate * dt,
                0.0,
            )
        self._prev_utilization = self.utilization_clipped

    def reset(self):
        """Reset link to idle state."""
        self.current_load = 0.0
        self.queue_depth = 0.0
        self._prev_utilization = 0.0
        self.ecn_marked = False
        self.packets_dropped = False
        self.drop_volume = 0.0
        self.ecn_fraction = 0.0
        self.total_drops_mb = 0.0
        self.total_ecn_marks = 0
        self.total_retransmit_events = 0
        self.dynamic_capacity_scale = 1.0
        self.is_up = True
        self.impairment_tag = "healthy"
        self._refresh_physical_state()
