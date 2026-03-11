"""
Training entry point — V2 with GNN policy support.

Usage:
    python train.py                              # GNN (default)
    python train.py --policy mlp                 # V1-style MLP
    python train.py --config config/default.yaml --timesteps 500000
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
    parser = argparse.ArgumentParser(description="Train PPO routing agent (V2)")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="trained_models")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy", type=str, default=None,
        choices=["gnn", "mlp"],
        help="Policy type: gnn (V2) or mlp (V1)",
    )
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
    logger.info(f"Observation dim: {env._obs_dim}, Action dim: {env._action_dim}")

    # ── Determine policy type ───────────────────────────
    policy_type = args.policy or train_params.get("policy_type", "gnn")
    logger.info(f"Policy type: {policy_type}")

    # ── Train ───────────────────────────────────────────
    total_timesteps = args.timesteps or train_params.get("total_timesteps", 1_000_000)
    save_path = str(Path(args.save_dir) / f"ppo_{policy_type}_routing_final")

    logger.info(f"Starting PPO-{policy_type.upper()} training for {total_timesteps:,} timesteps…")

    agent = PPOAgent.train(
        env=env,
        total_timesteps=total_timesteps,
        learning_rate=train_params.get("learning_rate", 1e-4),
        n_steps=train_params.get("n_steps", 2048),
        batch_size=train_params.get("batch_size", 128),
        gamma=train_params.get("gamma", 0.99),
        gae_lambda=train_params.get("gae_lambda", 0.95),
        clip_range=train_params.get("clip_range", 0.2),
        ent_coef=train_params.get("ent_coef", 0.001),
        vf_coef=train_params.get("vf_coef", 0.5),
        max_grad_norm=train_params.get("max_grad_norm", 0.5),
        n_epochs=train_params.get("n_epochs", 4),
        target_kl=train_params.get("target_kl", 0.02),
        save_path=save_path,
        log_dir=args.log_dir,
        verbose=1,
        policy_type=policy_type,
        gcn_hidden=train_params.get("gcn_hidden", 64),
        gcn_layers=train_params.get("gcn_layers", 2),
        features_dim=train_params.get("features_dim", 128),
        log_std_init=train_params.get("log_std_init", -1.0),
    )

    logger.info(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    main()
