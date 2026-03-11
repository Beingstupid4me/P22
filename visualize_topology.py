#!/usr/bin/env python3
"""
Generate network.png — a publication-quality visualization of the
simulated Clos (Leaf-Spine) topology with link capacities and node info.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml
import os

# ── Load config ─────────────────────────────────────────

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "default.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

topo = cfg["topology"]
traffic = cfg["traffic"]
tcp = cfg["tcp"]
env = cfg["environment"]

NUM_LEAVES = topo["num_leaves"]
NUM_SPINES = topo["num_spines"]
LINK_CAP = topo["link_capacity_mbps"]
BUFFER = topo["buffer_size_mb"]
PROP_DELAY = topo["propagation_delay_ms"]

# ── Layout ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(-1, NUM_LEAVES + 1)
ax.set_ylim(-1.5, 5.5)
ax.set_aspect("equal")
ax.axis("off")

# Title
fig.suptitle(
    "300B-Parameter Training Pod — Clos Topology (16L × 4S)",
    fontsize=18, fontweight="bold", y=0.97,
)

# ── Node positions ──────────────────────────────────────

leaf_y = 0.0
spine_y = 3.5

# Center leaves across width  (spacing to fit 16 nodes)
leaf_spacing = (NUM_LEAVES) / (NUM_LEAVES)
leaf_xs = [0.5 + i * leaf_spacing for i in range(NUM_LEAVES)]

# Center spines above leaves
spine_total_width = (NUM_SPINES - 1) * 3.0
spine_start = (NUM_LEAVES / 2.0) - spine_total_width / 2.0 + 0.5
spine_xs = [spine_start + j * 3.0 for j in range(NUM_SPINES)]

# ── Draw links (before nodes so nodes overlay the lines) ─

link_count = NUM_LEAVES * NUM_SPINES * 2  # bidirectional
for i, lx in enumerate(leaf_xs):
    for j, sx in enumerate(spine_xs):
        ax.plot(
            [lx, sx], [leaf_y + 0.3, spine_y - 0.3],
            color="#B0B8C4", linewidth=0.7, alpha=0.5, zorder=1,
        )

# ── Draw spine nodes ───────────────────────────────────

spine_color = "#2E86C1"
for j, sx in enumerate(spine_xs):
    circle = plt.Circle(
        (sx, spine_y), 0.35, color=spine_color, ec="white",
        linewidth=2, zorder=3,
    )
    ax.add_patch(circle)
    ax.text(
        sx, spine_y, f"S{j}", ha="center", va="center",
        fontsize=11, fontweight="bold", color="white", zorder=4,
    )

# ── Draw leaf nodes ────────────────────────────────────

leaf_color = "#27AE60"
for i, lx in enumerate(leaf_xs):
    rect = mpatches.FancyBboxPatch(
        (lx - 0.35, leaf_y - 0.25), 0.7, 0.5,
        boxstyle="round,pad=0.08",
        facecolor=leaf_color, edgecolor="white", linewidth=2, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        lx, leaf_y, f"L{i}", ha="center", va="center",
        fontsize=9, fontweight="bold", color="white", zorder=4,
    )

# ── Host / GPU annotations below leaves ────────────────

hosts = traffic["hosts_per_rack"]
for i, lx in enumerate(leaf_xs):
    ax.text(
        lx, leaf_y - 0.55, f"{hosts} GPUs",
        ha="center", va="top", fontsize=6.5, color="#555", zorder=4,
    )

# ── Capacity label on one highlighted link ─────────────

mid_leaf = NUM_LEAVES // 2
mid_spine = NUM_SPINES // 2
mlx = leaf_xs[mid_leaf]
msy = spine_xs[mid_spine]
ax.annotate(
    f"{LINK_CAP/1000:.0f} Gbps\n(per link)",
    xy=((mlx + msy) / 2, (leaf_y + spine_y) / 2),
    fontsize=9, ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9C4", ec="#F9A825", lw=1),
    zorder=5,
)

# ── Info table ──────────────────────────────────────────

info_lines = [
    ("Topology", f"{NUM_LEAVES} Leaves × {NUM_SPINES} Spines"),
    ("Directed Links", f"{link_count}"),
    ("Link Capacity", f"{LINK_CAP/1000:.0f} Gbps"),
    ("Buffer / Link", f"{BUFFER:.0f} Mb ({BUFFER/8:.0f} MB)"),
    ("Propagation Delay", f"{PROP_DELAY*1000:.0f} µs"),
    ("Elephant BW", f"{traffic['elephant_flow_bandwidth_mbps']/1000:.0f} Gbps (80% line rate)"),
    ("Elephant Data", f"{traffic['elephant_data_mb']/1000:.0f} GB (per flow shard)"),
    ("Burst Interval", f"Every {traffic['elephant_burst_interval']} steps ({traffic['elephant_burst_interval']*env['step_duration_seconds']*1000:.0f} ms)"),
    ("Step Duration", f"{env['step_duration_seconds']*1000:.0f} ms"),
    ("TCP MD Factor", f"{tcp['md_factor']}"),
    ("ECN Threshold", f"{tcp['ecn_marking_threshold']*100:.0f}% buffer fill"),
    ("Drop Threshold", f"{tcp['drop_threshold']*100:.0f}% buffer fill"),
    ("Total GPUs", f"{NUM_LEAVES * hosts}"),
]

table_x = NUM_LEAVES - 2.5
table_y = 5.2
ax.text(
    table_x, table_y, "Simulation Parameters",
    fontsize=11, fontweight="bold", ha="left", va="top",
    transform=ax.transData,
)
for idx, (key, val) in enumerate(info_lines):
    y = table_y - 0.33 * (idx + 1)
    ax.text(table_x, y, f"{key}:", fontsize=7.5, ha="left", va="top",
            fontweight="bold", color="#333")
    ax.text(table_x + 2.3, y, val, fontsize=7.5, ha="left", va="top",
            color="#555")

# ── Tier labels ─────────────────────────────────────────

ax.text(
    -0.5, spine_y, "Spine\nTier", fontsize=12, ha="center", va="center",
    fontweight="bold", color=spine_color,
)
ax.text(
    -0.5, leaf_y, "Leaf\n(ToR)\nTier", fontsize=12, ha="center", va="center",
    fontweight="bold", color=leaf_color,
)

# ── Traffic pattern annotation ──────────────────────────

patterns_text = (
    "Communication Patterns:\n"
    f"  • All-Reduce: {traffic['allreduce_probability']*100:.0f}%  (N×(N-1) flows — worst case)\n"
    f"  • All-to-All: {traffic['alltoall_probability']*100:.0f}%  (MoE routing)\n"
    f"  • Ring:       {traffic['ring_probability']*100:.0f}%  (pipeline parallel)"
)
ax.text(
    0.5, 5.2, patterns_text,
    fontsize=8, ha="left", va="top", family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", fc="#EBF5FB", ec="#2E86C1", lw=1),
    zorder=5,
)

# ── Collision callout ───────────────────────────────────

callout_x = spine_xs[0]
ax.annotate(
    "ECMP collision:\n2× 320G → 640G\non 400G link\n→ TCP collapse!",
    xy=(callout_x, spine_y + 0.4),
    xytext=(callout_x - 1.5, spine_y + 1.3),
    fontsize=7.5, ha="center", va="bottom",
    color="#C0392B", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="#FDEDEC", ec="#C0392B", lw=1),
    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5),
    zorder=5,
)

# ── Legend ───────────────────────────────────────────────

legend_items = [
    mpatches.Patch(color=leaf_color, label=f"Leaf Switch (ToR) — {NUM_LEAVES} total"),
    mpatches.Patch(color=spine_color, label=f"Spine Switch — {NUM_SPINES} total"),
    mpatches.Patch(color="#B0B8C4", label=f"Links — {LINK_CAP/1000:.0f} Gbps each"),
]
ax.legend(
    handles=legend_items, loc="lower center",
    ncol=3, fontsize=9, framealpha=0.9,
    bbox_to_anchor=(0.45, -0.08),
)

# ── Save ───────────────────────────────────────────────

out_path = os.path.join(os.path.dirname(__file__), "network.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved topology image to {out_path}")
