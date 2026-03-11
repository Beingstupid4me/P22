# V3 Benchmark Archive — Sparse-Collision Curriculum

**Date:** 2026-03-06  
**Topology:** 16 leaves × 4 spines, 400 Gbps links  
**Traffic regime:** sparse 4–7 participant bursts, locality-biased, projected intent edges  
**Policy:** PPO-GNN with batched GCN, state-dependent `log_std`, WCMP flowlets  
**Config lineage:** pre-realism default before dynamic asymmetry / fault injection

---

## Why V3 Existed

V3 changed the training curriculum from dense full-fabric bursts to sparse subgroup collectives.

Goal:

> expose ECMP's collision problem by creating holes in the fabric.

This was directionally correct, but still not enough to produce a decisive PPO-GNN win.

---

## Benchmark Results

### Tail FCT

| Metric | ECMP | Threshold (0.75) | LeastLoaded | PPO-GNN |
|---|---:|---:|---:|---:|
| Tail FCT (p99) | 32.654 ± 2.347 | 63.286 ± 8.906 | 39.378 ± 2.801 | 33.184 ± 1.902 |
| Tail FCT (p95) | 24.342 ± 1.008 | 41.448 ± 2.982 | 28.402 ± 1.417 | 24.602 ± 1.003 |
| Mean FCT | 12.408 ± 0.482 | 18.592 ± 0.889 | 14.165 ± 0.563 | 12.526 ± 0.473 |
| Mean Slowdown | 2.473 ± 0.124 | 3.721 ± 0.273 | 2.816 ± 0.160 | 2.502 ± 0.138 |

### Utilisation Balance

| Metric | ECMP | Threshold (0.75) | LeastLoaded | PPO-GNN |
|---|---:|---:|---:|---:|
| Jain's Fairness | 0.246 ± 0.010 | 0.238 ± 0.009 | 0.248 ± 0.010 | 0.246 ± 0.010 |
| Util Std Dev | 0.263 ± 0.004 | 0.275 ± 0.005 | 0.265 ± 0.004 | 0.264 ± 0.004 |
| Link Balance | 0.003 ± 0.003 | 0.003 ± 0.003 | 0.003 ± 0.003 | 0.003 ± 0.003 |

### Throughput

| Metric | ECMP | Threshold (0.75) | LeastLoaded | PPO-GNN |
|---|---:|---:|---:|---:|
| Throughput Ratio | 0.567 ± 0.018 | 0.383 ± 0.027 | 0.501 ± 0.021 | 0.562 ± 0.019 |
| Throughput (Mbps) | 4,258,934.108 ± 162,666.494 | 4,205,021.179 ± 156,817.488 | 4,239,988.008 ± 161,185.148 | 4,257,873.550 ± 162,741.693 |
| vs ECMP (%) | 0.000 ± 0.000 | -1.266 ± 3.682 | -0.445 ± 3.785 | -0.025 ± 3.821 |
| Dropped (MB) | 9,693,341.121 ± 1,351,407.187 | 11,762,587.026 ± 1,531,995.319 | 11,448,044.636 ± 1,510,838.717 | 9,781,633.664 ± 1,390,141.625 |

### TCP Health

| Metric | ECMP | Threshold (0.75) | LeastLoaded | PPO-GNN |
|---|---:|---:|---:|---:|
| Retransmissions | 43.300 ± 24.554 | 164.967 ± 84.230 | 73.233 ± 33.978 | 44.033 ± 24.275 |
| Total Drops (MB) | 9,693,341.121 ± 1,351,407.187 | 11,762,587.026 ± 1,531,995.319 | 11,448,044.636 ± 1,510,838.717 | 9,781,633.664 ± 1,390,141.625 |
| ECN Marks | 1,212.233 ± 132.073 | 1,538.933 ± 149.083 | 1,372.067 ± 123.460 | 1,229.267 ± 131.593 |
| Avg CWND Frac | 0.593 ± 0.016 | 0.408 ± 0.026 | 0.529 ± 0.020 | 0.588 ± 0.017 |
| Goodput Ratio | 0.961 ± 0.002 | 0.957 ± 0.003 | 0.960 ± 0.002 | 0.961 ± 0.002 |

### Convergence + Aggregate

| Metric | ECMP | Threshold (0.75) | LeastLoaded | PPO-GNN |
|---|---:|---:|---:|---:|
| Conv. Step | 10.133 ± 12.505 | 37.433 ± 60.247 | 8.933 ± 16.260 | 7.700 ± 13.617 |
| Adapt. Speed | 0.980 ± 0.025 | 0.925 ± 0.120 | 0.982 ± 0.033 | 0.985 ± 0.027 |
| Avg Reward | 1.746 ± 0.022 | 1.312 ± 0.045 | 1.597 ± 0.031 | 1.734 ± 0.028 |
| Total Reward | 873.199 ± 10.960 | 656.068 ± 22.724 | 798.675 ± 15.381 | 866.760 ± 13.849 |
| Action Churn | 0.002 ± 0.000 | 0.172 ± 0.008 | 0.965 ± 0.010 | 0.002 ± 0.000 |

---

## Readout

V3 succeeded in lowering overall congestion and drops compared with V2, but PPO-GNN still ended up in a statistical tie with ECMP.

### What improved

- lower drop volume than V2
- lower retransmissions than V2
- lower tail FCT than V2
- PPO-GNN remained stable and no longer behaved randomly

### What still failed to separate PPO-GNN from ECMP

Even under sparse subgroup traffic, the topology itself was still too idealized:

- all links started with the same hardware capacity
- there were no weak spines or weak ports
- there were no brownouts or sudden failures
- all available paths were equally good unless traffic alone made them bad

That meant ECMP's equal split remained a very strong default.

### Conclusion

The next step was not a new PPO trick.

The next step was to make the fabric itself more realistic:

1. static link asymmetry
2. per-spine strength differences
3. runtime brownouts
4. sudden hard port failures

That is the purpose of the new [config/realistic_asymmetric.yaml](config/realistic_asymmetric.yaml) scenario.
