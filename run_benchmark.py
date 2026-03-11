"""
Benchmarking entry point — V2 with GNN agent support.

Usage:
    python run_benchmark.py
    python run_benchmark.py --ppo-model trained_models/ppo_gnn_routing_final
    python run_benchmark.py --policy gnn --episodes 20
"""

import argparse

import yaml

from simulator.topology import TopologyConfig
from simulator.traffic import TrafficConfig
from simulator.link import TCPConfig
from environment.rewards import RewardConfig
from environment.env import NetworkRoutingEnv
from agents.ecmp import ECMPAgent
from agents.threshold import ThresholdAgent
from agents.least_loaded import LeastLoadedAgent
from agents.conductor import ConductorAgent
from agents.ppo import PPOAgent
from benchmarking.runner import BenchmarkRunner
from utils.visualization import plot_benchmark_comparison
from utils.logger import setup_logger


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Benchmark routing agents (V2)")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--ppo-model", type=str, default=None,
                        help="Path to a trained PPO model")
    parser.add_argument("--policy", type=str, default="gnn",
                        choices=["gnn", "mlp"],
                        help="Policy type for loaded PPO model")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-plot", type=str, default="results/benchmark.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    logger = setup_logger("benchmark")

    # ── Config ──────────────────────────────────────────
    cfg = load_config(args.config)

    topo_cfg = TopologyConfig(**cfg.get("topology", {}))
    traffic_cfg = TrafficConfig(**cfg.get("traffic", {}))
    tcp_cfg = TCPConfig(**cfg.get("tcp", {}))

    reward_dict = cfg.get("environment", {}).get("reward", {})
    reward_cfg = RewardConfig(**reward_dict)

    env_params = cfg.get("environment", {})
    bench_params = cfg.get("benchmarking", {})

    # ── Environment ─────────────────────────────────
    env = NetworkRoutingEnv(
        topology_config=topo_cfg,
        traffic_config=traffic_cfg,
        reward_config=reward_cfg,
        tcp_config=tcp_cfg,
        max_steps=env_params.get("max_steps", 500),
        observation_history=env_params.get("observation_history", 1),
        step_duration_seconds=env_params.get("step_duration_seconds", 1.0),
    )

    num_leaves = env._num_leaves
    num_spines = env._num_spines

    # ── Agents ──────────────────────────────────────────
    agents = [
        ECMPAgent(num_leaves, num_spines),
        ThresholdAgent(num_leaves, num_spines, threshold=0.75),
        LeastLoadedAgent(num_leaves, num_spines),
        ConductorAgent(num_leaves, num_spines),
    ]

    if args.ppo_model:
        try:
            ppo = PPOAgent.load(args.ppo_model, env=env, policy_type=args.policy)
            ppo.num_leaves = num_leaves
            ppo.num_spines = num_spines
            agents.append(ppo)
            logger.info(f"Loaded PPO-{args.policy.upper()} model from {args.ppo_model}")
        except Exception as e:
            logger.warning(f"Could not load PPO model: {e}")

    # ── Run ─────────────────────────────────────────────
    num_episodes = args.episodes or bench_params.get("num_episodes", 50)

    runner = BenchmarkRunner(env, num_episodes=num_episodes, seed=args.seed)
    results = runner.compare_agents(agents, verbose=True)

    # ── Plot ────────────────────────────────────────────
    if not args.no_plot:
        try:
            plot_benchmark_comparison(
                results, save_path=args.save_plot, show=True,
            )
            logger.info(f"Plot saved to {args.save_plot}")
        except Exception as e:
            logger.warning(f"Could not create plot: {e}")


if __name__ == "__main__":
    main()
