"""
Training entry point for the PPO routing agent.

Usage:
    python train.py
    python train.py --config config/default.yaml --timesteps 500000
    python train.py --save-dir trained_models --log-dir logs
"""

import argparse
from pathlib import Path

import yaml

from simulator.topology import TopologyConfig
from simulator.traffic import TrafficConfig
from simulator.link import TCPConfig
from environment.rewards import RewardConfig
from environment.env import NetworkRoutingEnv
from agents.ppo import PPOAgent
from utils.logger import setup_logger


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train PPO routing agent")
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total training timesteps",
    )
    parser.add_argument(
        "--save-dir", type=str, default="trained_models",
        help="Directory for saved models",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="TensorBoard log directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logger("train", log_file=f"{args.log_dir}/train.log")

    # ── Load config ─────────────────────────────────────
    cfg = load_config(args.config)

    topo_cfg = TopologyConfig(**cfg.get("topology", {}))
    traffic_cfg = TrafficConfig(**cfg.get("traffic", {}))
    tcp_cfg = TCPConfig(**cfg.get("tcp", {}))

    reward_dict = cfg.get("environment", {}).get("reward", {})
    reward_cfg = RewardConfig(**reward_dict)

    env_params = cfg.get("environment", {})
    train_params = cfg.get("training", {})

    # ── Create environment ──────────────────────────────
    env = NetworkRoutingEnv(
        topology_config=topo_cfg,
        traffic_config=traffic_cfg,
        reward_config=reward_cfg,
        tcp_config=tcp_cfg,
        max_steps=env_params.get("max_steps", 500),
        observation_history=env_params.get("observation_history", 1),
        step_duration_seconds=env_params.get("step_duration_seconds", 1.0),
        seed=args.seed,
    )

    logger.info(
        f"Environment: {env._num_links} links, "
        f"{env._num_leaves} leaves, {env._num_spines} spines"
    )
    logger.info(f"Observation dim: {env._obs_dim}, Action dim: {env._weight_dim}")

    # ── Train ───────────────────────────────────────────
    total_timesteps = args.timesteps or train_params.get("total_timesteps", 1_000_000)
    save_path = str(Path(args.save_dir) / "ppo_routing_final")

    logger.info(f"Starting PPO training for {total_timesteps:,} timesteps…")

    agent = PPOAgent.train(
        env=env,
        total_timesteps=total_timesteps,
        learning_rate=train_params.get("learning_rate", 3e-4),
        n_steps=train_params.get("n_steps", 2048),
        batch_size=train_params.get("batch_size", 64),
        gamma=train_params.get("gamma", 0.99),
        gae_lambda=train_params.get("gae_lambda", 0.95),
        clip_range=train_params.get("clip_range", 0.2),
        ent_coef=train_params.get("ent_coef", 0.01),
        vf_coef=train_params.get("vf_coef", 0.5),
        save_path=save_path,
        log_dir=args.log_dir,
        verbose=1,
    )

    logger.info(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    main()
