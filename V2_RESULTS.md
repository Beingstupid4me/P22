# V2 Benchmark Archive — Collective-Aware Graph Flowlet Orchestrator

**Date:** 2026-03-06  
**Topology:** 16 leaves × 4 spines, 400 Gbps links  
**Traffic regime:** 320 Gbps elephants, intent lookahead = 2, dense 8–16 participant bursts  
**Policy:** batched GCN + intent encoder + WCMP flowlet routing  
**Reward:** throughput + Jain's fairness + hotspot penalty

**Archived config:** [config/v2_benchmark.yaml](config/v2_benchmark.yaml)

---

## What V2 Was Testing

V2 fixed the original PPO failure and introduced the correct orchestrator ingredients:

1. **Pairwise intent matrix** published before the burst
2. **Graph policy** (`GCNConv`) instead of a flat MLP
3. **WCMP flowlet routing** instead of sticky per-flow ECMP
4. **Low-entropy PPO** with state-dependent `log_std`

The question for V2 was simple:

> If the control plane is now structurally correct, can it beat ECMP on the default 16L×4S AI-training workload?

---

## Benchmark Results

### Tail FCT

| Metric | ECMP | Threshold (0.75) | LeastLoaded | PPO-GNN |
|---|---:|---:|---:|---:|
| Tail FCT (p99) | 49.950 ± 3.023 | 100.000 ± 0.000 | 100.000 ± 0.000 | 50.533 ± 3.335 |
| Tail FCT (p95) | 26.533 ± 2.187 | 100.000 ± 0.000 | 62.032 ± 5.493 | 26.972 ± 2.314 |
| Mean FCT | 10.688 ± 0.426 | 39.385 ± 5.080 | 23.124 ± 1.377 | 10.779 ± 0.431 |
| Mean Slowdown | 5.128 ± 0.159 | 18.035 ± 2.325 | 9.790 ± 0.624 | 5.184 ± 0.169 |

### Utilisation Balance

| Metric | ECMP | Threshold (0.75) | LeastLoaded | PPO-GNN |
|---|---:|---:|---:|---:|
| Jain's Fairness | 0.593 ± 0.016 | 0.462 ± 0.010 | 0.543 ± 0.012 | 0.593 ± 0.016 |
| Util Std Dev | 0.307 ± 0.003 | 0.363 ± 0.004 | 0.353 ± 0.003 | 0.307 ± 0.003 |
| Link Balance | 0.005 ± 0.003 | 0.003 ± 0.003 | 0.003 ± 0.003 | 0.005 ± 0.003 |

### Throughput

| Metric | ECMP | Threshold (0.75) | LeastLoaded | PPO-GNN |
|---|---:|---:|---:|---:|
| Throughput Ratio | 0.297 ± 0.018 | 0.076 ± 0.014 | 0.137 ± 0.017 | 0.294 ± 0.018 |
| Throughput (Mbps) | 11,240,266.114 ± 393,505.162 | 9,125,042.345 ± 255,660.799 | 10,428,205.794 ± 295,318.174 | 11,235,938.273 ± 394,097.718 |
| vs ECMP (%) | 0.000 ± 0.000 | -18.818 ± 2.275 | -7.225 ± 2.627 | -0.039 ± 3.506 |
| Dropped (MB) | 77,302,614.119 ± 9,925,097.599 | 208,730,185.054 ± 39,338,156.886 | 98,126,498.748 ± 16,739,359.287 | 77,060,892.902 ± 9,459,942.113 |

### TCP Health

| Metric | ECMP | Threshold (0.75) | LeastLoaded | PPO-GNN |
|---|---:|---:|---:|---:|
| Retransmissions | 365.433 ± 220.978 | 9,297.300 ± 5,622.433 | 2,945.633 ± 1,848.179 | 375.367 ± 224.572 |
| Total Drops (MB) | 77,302,614.119 ± 9,925,097.599 | 208,730,185.054 ± 39,338,156.886 | 98,126,498.748 ± 16,739,359.287 | 77,060,892.902 ± 9,459,942.113 |
| ECN Marks | 6,770.033 ± 541.191 | 8,867.033 ± 638.656 | 8,222.433 ± 692.792 | 6,773.667 ± 550.622 |
| Avg CWND Frac | 0.329 ± 0.016 | 0.100 ± 0.015 | 0.161 ± 0.017 | 0.326 ± 0.017 |
| Goodput Ratio | 0.909 ± 0.006 | 0.902 ± 0.004 | 0.920 ± 0.005 | 0.909 ± 0.006 |

### Convergence + Aggregate

| Metric | ECMP | Threshold (0.75) | LeastLoaded | PPO-GNN |
|---|---:|---:|---:|---:|
| Conv. Step | 7.133 ± 10.689 | 24.533 ± 40.232 | 4.000 ± 6.608 | 7.467 ± 10.632 |
| Adapt. Speed | 0.986 ± 0.021 | 0.951 ± 0.080 | 0.992 ± 0.013 | 0.985 ± 0.021 |
| Avg Reward | 2.188 ± 0.032 | 1.349 ± 0.026 | 1.713 ± 0.018 | 2.182 ± 0.032 |
| Total Reward | 1,094.208 ± 16.154 | 674.638 ± 13.152 | 856.482 ± 8.848 | 1,091.235 ± 15.800 |
| Stability | 0.002 ± 0.000 | 0.166 ± 0.008 | 0.988 ± 0.005 | 0.079 ± 0.014 |

---

## Executive Readout

### What worked

V2 **fixed the broken training dynamics** from Experiment 1:

- `std` was no longer frozen
- PPO no longer collapsed into random noise
- the GNN learned a sane policy
- PPO-GNN caught up to ECMP on throughput, fairness, FCT, and reward

That is a real improvement. V2 turned a broken controller into a viable one.

### What did not work

V2 still **did not beat ECMP**.

The gap was tiny enough to call the result a statistical tie:

- throughput: 0.297 vs 0.294
- fairness: 0.593 vs 0.593
- p99 FCT: 49.95 vs 50.53
- avg reward: 2.188 vs 2.182

So the orchestrator was no longer failing because of PPO. It was failing because of the **workload geometry**.

---

## Root Cause — We Were Fighting ECMP on Its Home Turf

The default V2 traffic was still too symmetric:

- many bursts activated a large fraction of the 16 leaves
- all-reduce remained dominant
- almost every spine became busy anyway
- there were very few true “holes” for the controller to exploit

In that regime, ECMP is already hard to beat.

If the fabric is uniformly overfull, the GNN sees the same thing ECMP sees:

> every path is bad, so moving traffic changes very little.

That is why PPO-GNN converged to **“roughly ECMP”** rather than clearly outperforming it.

---

## V2 Drawbacks Recorded for Reproducibility

### 1. Dense, symmetric bursts masked ECMP's weakness

ECMP is worst when a *small* number of large flows collide on the same spine while other spines sit idle.

V2 often created the opposite:

- many participants
- many simultaneous elephant flows
- all spines busy
- little exploitable asymmetry

### 2. Pairwise intent existed, but edge-level forecasting was missing

The GNN already saw the leaf→leaf intent matrix, but the simulator did not project that intent onto the actual leaf→spine and spine→leaf edges.

That made the “future congestion picture” harder to decode than it needed to be.

### 3. Stability is not the right criticism for a flowlet controller

The benchmark's `Stability` metric measured **weight-change frequency**, not harmful control-plane churn.

For flowlet routing, fast split-ratio updates are often *the point*, provided they do not create artificial TCP penalties.

### 4. Safe flowlet switching needed to be explicit in the docs

The simulator did **not** inject a synthetic TCP multiplicative-decrease penalty just because a path changed. TCP only reacted to actual ECN/drop signals from links.

That behavior was correct, but it needed to be stated clearly because the benchmark tables made the controller look “unstable” even when the routing changes were semantically safe.

---

## What V3 Changes

V3 keeps the V2 orchestrator and changes the training/evaluation curriculum:

1. **Sparse subgroup bursts (4–7 leaves)** instead of near full-fabric floods
2. **Locality-biased participant selection** so unused parts of the fabric stay empty
3. **More all-to-all and ring traffic** where hash collisions are visible
4. **Projected edge intent demand** added to edge features so the GNN sees future pressure where it matters
5. **Safe flowlet semantics made explicit**: path changes do not create synthetic TCP penalties

The goal of V3 is not to make ECMP look bad artificially.

The goal is to test the controller on the class of workloads where a centralized orchestrator should actually have an advantage:

> sparse, asymmetric, collision-prone bursts with real empty capacity elsewhere in the fabric.

---

## Practical Consequence

Any GNN model trained before V3 should be considered **architecture-incompatible** with the new default environment because the edge feature dimension changed.

Retraining is required.
