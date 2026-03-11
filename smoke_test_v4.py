"""Quick V4 smoke test — validates all changes work end-to-end."""
import sys
sys.path.insert(0, ".")

import yaml
from simulator.topology import TopologyConfig
from simulator.traffic import TrafficConfig
from simulator.link import TCPConfig
from environment.env import NetworkRoutingEnv
from environment.rewards import RewardConfig
from agents.ecmp import ECMPAgent
from agents.conductor import ConductorAgent
from agents.threshold import ThresholdAgent
from agents.least_loaded import LeastLoadedAgent
import numpy as np

print("=" * 60)
print("V4 SMOKE TEST")
print("=" * 60)

# Load full config (16L × 4S)
with open("config/default.yaml") as f:
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
)

L = env._num_leaves
S = env._num_spines
expected_action = L * (1 + S)

print(f"[OK] Env created")
print(f"     Obs dim: {env._obs_dim}")
print(f"     Action dim (expected {expected_action}): {env._action_dim}")
print(f"     Action shape: {env.action_space.shape}")
print(f"     NODE_FEAT_DIM: {env.NODE_FEAT_DIM}")
print(f"     Leaves={L}, Spines={S}")

obs, info = env.reset()
print(f"[OK] Reset: obs shape = {obs.shape}")

# 2. ECMP (Stage A)
ecmp = ECMPAgent(env._num_leaves, env._num_spines)
print(f"\n--- Stage A: ECMP (Dumb Agent) ---")
print(f"     Action dim: {ecmp.action_dim}")
action = ecmp.act(obs)
print(f"     Action shape: {action.shape}")
assert action.shape[0] == expected_action, f"Expected {expected_action}, got {action.shape[0]}"

obs, r, t, tr, info = env.step(action)
print(f"     Step 1: reward={r:.2f}, flows={info['num_active_flows']}")

for i in range(20):
    obs, r, t, tr, info = env.step(ecmp.act(obs))
print(f"     Step 21: reward={r:.2f}, flows={info['num_active_flows']}")
print(f"[OK] ECMP passed")

# 3. Conductor (Stage B)
obs, _ = env.reset()
cond = ConductorAgent(env._num_leaves, env._num_spines)
print(f"\n--- Stage B: Conductor ---")
action = cond.act(obs)
assert action.shape[0] == expected_action, f"Expected {expected_action}, got {action.shape[0]}"

for i in range(20):
    obs, r, t, tr, info = env.step(cond.act(obs))
print(f"     Step 20: reward={r:.2f}, flows={info['num_active_flows']}")
print(f"[OK] Conductor passed")

# 4. Threshold
obs, _ = env.reset()
thresh = ThresholdAgent(env._num_leaves, env._num_spines)
action = thresh.act(obs)
assert action.shape[0] == expected_action
for i in range(5):
    obs, r, t, tr, info = env.step(thresh.act(obs))
print(f"\n[OK] Threshold agent passed (action dim = {action.shape[0]})")

# 5. LeastLoaded
obs, _ = env.reset()
ll = LeastLoadedAgent(env._num_leaves, env._num_spines)
action = ll.act(obs)
assert action.shape[0] == expected_action
for i in range(5):
    obs, r, t, tr, info = env.step(ll.act(obs))
print(f"[OK] LeastLoaded agent passed (action dim = {action.shape[0]})")

# 6. Check reward is V4 style (negative integer-ish)
print(f"\n--- Reward check ---")
print(f"     Last reward: {r:.2f} (should be negative or zero = -num_flows)")

# 7. Check backlog in info
if 'leaf_backlog' in info:
    print(f"     Leaf backlog present: {len(info['leaf_backlog'])} leaves")
if 'leaf_admission_rates' in info:
    print(f"     Admission rates present: {len(info['leaf_admission_rates'])} leaves")

print(f"\n{'=' * 60}")
print(f"ALL V4 SMOKE TESTS PASSED!")
print(f"{'=' * 60}")
