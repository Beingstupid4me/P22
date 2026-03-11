"""
Benchmark runner for comparative evaluation of routing agents.

Runs the same traffic patterns through each agent and collects
standardised metrics — including ECMP baseline comparison —
for side-by-side evaluation.
"""

from typing import Dict, List, Optional

from environment.env import NetworkRoutingEnv
from agents.base import BaseAgent
from benchmarking.metrics import EpisodeMetrics, aggregate_episodes


class BenchmarkRunner:
    """Runs multi-episode evaluations and prints comparison tables.

    The ECMP agent is automatically detected (by name) and used
    as the throughput baseline for computing relative improvement.

    Args:
        env: The shared evaluation environment.
        num_episodes: Episodes per agent.
        seed: Base random seed (incremented per episode for reproducibility).
    """

    def __init__(
        self,
        env: NetworkRoutingEnv,
        num_episodes: int = 50,
        seed: int = 42,
    ):
        self.env = env
        self.num_episodes = num_episodes
        self.base_seed = seed
        self._ecmp_avg_throughput: Optional[float] = None

    # ── Public API ──────────────────────────────────────

    def evaluate_agent(
        self,
        agent: BaseAgent,
        verbose: bool = False,
        store_ecmp_baseline: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate a single agent over ``num_episodes`` episodes."""
        episodes: List[EpisodeMetrics] = []

        for ep in range(self.num_episodes):
            seed = self.base_seed + ep
            metrics = self._run_episode(agent, seed)

            # Inject ECMP baseline if available
            if self._ecmp_avg_throughput is not None:
                metrics.ecmp_avg_throughput = self._ecmp_avg_throughput

            episodes.append(metrics)

            if verbose:
                s = metrics.summary()
                print(
                    f"  Episode {ep + 1:>3}/{self.num_episodes}: "
                    f"reward={s['avg_reward']:>7.3f}  "
                    f"throughput={s['avg_throughput_ratio']:.3f}  "
                    f"tail_fct_p99={s['tail_fct_p99']:.0f}  "
                    f"jains={s['jains_fairness']:.4f}"
                )

        # If this is the ECMP agent, store baseline
        if store_ecmp_baseline and episodes:
            avg_tp = float(
                sum(ep.avg_throughput for ep in episodes) / len(episodes)
            )
            self._ecmp_avg_throughput = avg_tp

        return aggregate_episodes(episodes)

    def compare_agents(
        self, agents: List[BaseAgent], verbose: bool = True
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Evaluate all agents and return comparative results.

        ECMP is evaluated first to establish the throughput baseline
        used for computing relative improvement for all other agents.

        Returns:
            ``{agent_name: {metric_name: {mean, std, min, max}}}``
        """
        results: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Sort: ECMP first to establish baseline
        ecmp_agents = [a for a in agents if "ecmp" in a.name.lower()]
        other_agents = [a for a in agents if "ecmp" not in a.name.lower()]
        ordered_agents = ecmp_agents + other_agents

        for agent in ordered_agents:
            if verbose:
                print(f"\nEvaluating: {agent.name}")
            agent.reset()
            is_ecmp = "ecmp" in agent.name.lower()
            results[agent.name] = self.evaluate_agent(
                agent, verbose, store_ecmp_baseline=is_ecmp
            )

        if verbose:
            self._print_comparison(results)

        return results

    # ── Internal ────────────────────────────────────────

    def _run_episode(self, agent: BaseAgent, seed: int) -> EpisodeMetrics:
        """Run one episode and collect metrics."""
        obs, info = self.env.reset(seed=seed)
        agent.reset()

        metrics = EpisodeMetrics()
        done = False

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            metrics.record(info, reward)
            done = terminated or truncated

        metrics.weight_changes = info.get("weight_changes", 0)
        return metrics

    # ── Display ─────────────────────────────────────────

    @staticmethod
    def _print_comparison(
        results: Dict[str, Dict[str, Dict[str, float]]]
    ):
        """Print a formatted comparison table to stdout."""
        print("\n" + "=" * 90)
        print("BENCHMARK RESULTS - LLM Training Centre Simulation")
        print("=" * 90)

        agents = list(results.keys())

        # Group metrics logically
        metric_groups = {
            "--- Tail FCT ---": [
                ("tail_fct_p99", "Tail FCT (p99)"),
                ("tail_fct_p95", "Tail FCT (p95)"),
                ("mean_fct", "Mean FCT"),
                ("mean_slowdown", "Mean Slowdown"),
            ],
            "--- Utilisation Balance ---": [
                ("jains_fairness", "Jain's Fairness"),
                ("avg_utilization_std", "Util Std Dev"),
                ("link_balance_ratio", "Link Balance"),
            ],
            "--- Throughput ---": [
                ("avg_throughput_ratio", "Throughput Ratio"),
                ("avg_throughput_mbps", "Throughput (Mbps)"),
                ("throughput_improvement_pct", "vs ECMP (%)"),
                ("total_dropped_mb", "Dropped (MB)"),
            ],
            "--- TCP Health ---": [
                ("total_retransmissions", "Retransmissions"),
                ("total_drops_mb", "Total Drops (MB)"),
                ("total_ecn_marks", "ECN Marks"),
                ("avg_cwnd_fraction", "Avg CWND Frac"),
                ("goodput_ratio", "Goodput Ratio"),
            ],
            "--- Fabric Realism ---": [
                ("avg_capacity_ratio", "Avg Cap Ratio"),
                ("avg_active_link_events", "Active Events"),
            ],
            "--- Convergence ---": [
                ("convergence_step", "Conv. Step"),
                ("adaptation_speed", "Adapt. Speed"),
            ],
            "--- Aggregate ---": [
                ("avg_reward", "Avg Reward"),
                ("total_reward", "Total Reward"),
                ("stability_index", "Action Churn"),
            ],
        }

        # Header
        header = f"{'Metric':<28}" + "".join(f"{a:>20}" for a in agents)
        print(header)
        print("-" * len(header))

        # Rows
        for group_name, metric_list in metric_groups.items():
            print(f"\n{group_name}")
            for key, label in metric_list:
                row = f"  {label:<26}"
                for agent in agents:
                    if key in results[agent]:
                        mean = results[agent][key]["mean"]
                        std = results[agent][key]["std"]
                        row += f"{mean:>11.3f} ±{std:<7.3f}"
                    else:
                        row += f"{'N/A':>20}"
                print(row)

        print("\n" + "=" * 90)
