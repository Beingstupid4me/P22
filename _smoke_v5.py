"""Quick V5 micro-lab smoke test."""
import yaml
from simulator.topology import TopologyConfig
from simulator.traffic import TrafficConfig
from simulator.link import TCPConfig
from environment.rewards import RewardConfig
from environment.env import NetworkRoutingEnv
import numpy as np

with open("config/micro_lab.yaml") as f:
    cfg = yaml.safe_load(f)

topo_cfg = TopologyConfig(**cfg.get("topology", {}))
traffic_cfg = TrafficConfig(**cfg.get("traffic", {}))
tcp_cfg = TCPConfig(**cfg.get("tcp", {}))
reward_cfg = RewardConfig(**cfg.get("environment", {}).get("reward", {}))
env_params = cfg.get("environment", {})

env = NetworkRoutingEnv(
    topology_config=topo_cfg,
    traffic_config=traffic_cfg,
    reward_config=reward_cfg,
    tcp_config=tcp_cfg,
    max_steps=env_params.get("max_steps", 500),
    observation_history=env_params.get("observation_history", 1),
    step_duration_seconds=env_params.get("step_duration_seconds", 1.0),
    seed=42,
)

print(f"Topology: {env._num_leaves}L x {env._num_spines}S")
print(f"Links: {env._num_links}")
print(f"Obs dim: {env._obs_dim}, Action dim: {env._action_dim}")
print(f"Reward cfg: cap={reward_cfg.reference_capacity_mbps/1e6:.1f}Tbps, max_flows={reward_cfg.max_expected_flows}")

obs, info = env.reset(seed=42)
print(f"Obs shape: {obs.shape}")

# Run 20 steps with random actions
rewards = []
for i in range(20):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    rewards.append(reward)

print(f"20-step rewards: min={min(rewards):.4f} max={max(rewards):.4f} mean={np.mean(rewards):.4f}")
print("Smoke test PASSED")
