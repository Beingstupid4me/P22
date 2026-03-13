"""
Reward Profile Search — Train A–G and benchmark side-by-side.

Usage:
    python reward_search.py                          # Full run (50k each)
    python reward_search.py --timesteps 10000        # Quick test
    python reward_search.py --skip-train             # Benchmark only (models exist)
    python reward_search.py --profiles A D G         # Subset only
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml
import numpy as np

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
from utils.logger import setup_logger

PROFILE_NAMES = {
    "A": "Speed Demon",
    "B": "Cautious Guard",
    "C": "Pure Scheduler",
    "D": "Shaped Architect",
    "E": "Util Balancer",
    "F": "Completion Sprinter",
    "G": "Adaptive Pacer",
}


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_profile(profile: str, config_path: str, timesteps: int,
                   save_dir: str, seed: int, policy: str) -> str:
    """Train one model with the given reward profile. Returns save path."""
    save_path = str(Path(save_dir) / f"ppo_{policy}_routing_{profile}_final")

    cmd = [
        sys.executable, "train.py",
        "--config", config_path,
        "--timesteps", str(timesteps),
        "--save-dir", save_dir,
        "--seed", str(seed),
        "--policy", policy,
        "--reward-profile", profile,
    ]
    print(f"\n{'='*60}")
    print(f"  Training Profile {profile}: {PROFILE_NAMES[profile]}")
    print(f"  Steps: {timesteps:,}  |  Saving to: {save_path}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        print(f"  [FAILED] Profile {profile} training failed (exit {result.returncode})")
        return ""
    print(f"  [OK] Profile {profile} trained.")
    return save_path


def benchmark_all(config_path: str, model_paths: dict, episodes: int,
                   seed: int, policy: str):
    """Benchmark all heuristics + trained PPO models side-by-side."""
    cfg = load_config(config_path)

    topo_cfg = TopologyConfig(**cfg.get("topology", {}))
    traffic_cfg = TrafficConfig(**cfg.get("traffic", {}))
    tcp_cfg = TCPConfig(**cfg.get("tcp", {}))

    # Use profile C as the "default" reward for benchmarking env
    reward_cfg = RewardConfig()

    env_params = cfg.get("environment", {})
    env = NetworkRoutingEnv(
        topology_config=topo_cfg,
        traffic_config=traffic_cfg,
        reward_config=reward_cfg,
        tcp_config=tcp_cfg,
        max_steps=env_params.get("max_steps", 500),
        observation_history=env_params.get("observation_history", 1),
        step_duration_seconds=env_params.get("step_duration_seconds", 1.0),
        seed=seed,
    )

    num_leaves = env._num_leaves
    num_spines = env._num_spines

    # Heuristic baselines
    agents = [
        ECMPAgent(num_leaves, num_spines),
        ThresholdAgent(num_leaves, num_spines, threshold=0.75),
        LeastLoadedAgent(num_leaves, num_spines),
        ConductorAgent(num_leaves, num_spines),
    ]

    # Load trained models
    for profile, path in sorted(model_paths.items()):
        if not path or not Path(path + ".pt").exists():
            print(f"  [SKIP] Profile {profile}: model not found at {path}.pt")
            continue
        try:
            ppo = PPOAgent.load(path, env=env, policy_type=policy)
            ppo.num_leaves = num_leaves
            ppo.num_spines = num_spines
            ppo.name = f"PPO-{profile} ({PROFILE_NAMES[profile]})"
            agents.append(ppo)
            print(f"  [OK] Loaded Profile {profile}: {ppo.name}")
        except Exception as e:
            print(f"  [SKIP] Profile {profile}: {e}")

    print(f"\n{'='*60}")
    print(f"  Benchmarking {len(agents)} agents × {episodes} episodes")
    print(f"{'='*60}\n")

    runner = BenchmarkRunner(env, num_episodes=episodes, seed=seed)
    results = runner.compare_agents(agents, verbose=True)

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    try:
        from utils.visualization import plot_benchmark_comparison
        plot_benchmark_comparison(
            results,
            save_path=str(results_dir / "reward_search.png"),
            show=False,
        )
        print(f"\n  Plot saved to results/reward_search.png")
    except Exception as e:
        print(f"  Could not create plot: {e}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  REWARD PROFILE SEARCH RESULTS")
    print(f"{'='*70}")
    print(f"  {'Agent':<35} {'Reward':>10} {'Tail FCT':>10} {'Goodput':>10} {'Drops':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name, metrics in results.items():
        avg_reward = metrics.get("avg_reward", {}).get("mean", 0.0)
        avg_tail = metrics.get("tail_fct_p99", {}).get("mean", 0.0)
        avg_goodput = metrics.get("goodput_ratio", {}).get("mean", 0.0)
        avg_drops = metrics.get("total_drops_mb", {}).get("mean", 0.0)
        print(f"  {name:<35} {avg_reward:>10.3f} {avg_tail:>10.1f} {avg_goodput:>10.3f} {avg_drops:>10.1f}")

    print(f"{'='*70}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Reward Profile Search")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", type=str, default="gnn")
    parser.add_argument("--save-dir", type=str, default="trained_models")
    parser.add_argument("--profiles", nargs="+", default=list(PROFILE_NAMES.keys()),
                        help="Profiles to train (default: all A-G)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, only run benchmark")
    args = parser.parse_args()

    logger = setup_logger("reward_search")

    profiles = [p.upper() for p in args.profiles]
    model_paths = {}

    if not args.skip_train:
        print(f"\n{'#'*60}")
        print(f"  PHASE 1: Training {len(profiles)} reward profiles")
        print(f"  Profiles: {', '.join(f'{p} ({PROFILE_NAMES[p]})' for p in profiles)}")
        print(f"  Timesteps per model: {args.timesteps:,}")
        print(f"{'#'*60}")

        for profile in profiles:
            path = train_profile(
                profile, args.config, args.timesteps,
                args.save_dir, args.seed, args.policy,
            )
            model_paths[profile] = path
    else:
        # Discover existing models
        for profile in profiles:
            path = str(Path(args.save_dir) / f"ppo_{args.policy}_routing_{profile}_final")
            model_paths[profile] = path

    print(f"\n{'#'*60}")
    print(f"  PHASE 2: Benchmarking all agents")
    print(f"{'#'*60}")

    benchmark_all(
        args.config, model_paths, args.episodes,
        args.seed, args.policy,
    )


if __name__ == "__main__":
    main()
