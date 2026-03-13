"""
PPO agent with V2 GNN policy support.

Supports two policy modes:
    1. "mlp"  — Classic SB3 MlpPolicy (V1 compatible)
    2. "gnn"  — Custom GNN actor-critic with GCN feature extraction

The GNN mode uses our custom training loop (not SB3's built-in PPO)
because SB3 doesn't natively support PyG graph convolutions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import Optional, Dict, List
from pathlib import Path
from collections import deque

from .base import BaseAgent


class PPOAgent(BaseAgent):
    """PPO-based adaptive routing agent.

    V2 supports both MLP (via SB3) and GNN (custom loop) policies.
    """

    def __init__(self, num_leaves: int, num_spines: int, model=None,
                 policy_type: str = "gnn"):
        super().__init__(num_leaves, num_spines)
        self.model = model
        self.policy_type = policy_type
        self._policy = None  # GNN policy network
        self._label = None   # Optional display name override
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Inference ───────────────────────────────────────

    def act(self, observation: np.ndarray, deterministic: bool = True, **kwargs) -> np.ndarray:
        """Select action using the trained policy."""
        if self.policy_type == "mlp":
            if self.model is None:
                raise RuntimeError("PPO agent has no trained model.")
            action, _ = self.model.predict(observation, deterministic=deterministic)
            return action
        else:
            # GNN policy
            if self._policy is None:
                raise RuntimeError("GNN policy not initialised. Call train() or load().")
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self._device)
            with torch.no_grad():
                action_mean, log_std, _ = self._policy(obs_tensor)
                if deterministic:
                    action = action_mean
                else:
                    std = log_std.exp()
                    action = torch.distributions.Normal(action_mean, std).sample()
                    action = torch.clamp(action, -1.0, 1.0)
            return action.cpu().numpy().squeeze(0)

    # ── GNN Training ────────────────────────────────────

    @classmethod
    def train(
        cls,
        env,
        total_timesteps: int = 1_000_000,
        learning_rate: float = 1e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.001,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        target_kl: Optional[float] = 0.02,
        save_path: Optional[str] = None,
        log_dir: Optional[str] = None,
        verbose: int = 1,
        policy_type: str = "gnn",
        gcn_hidden: int = 64,
        gcn_layers: int = 2,
        features_dim: int = 128,
        log_std_init: float = -1.0,
        **kwargs,
    ) -> "PPOAgent":
        """Train a PPO agent with either MLP or GNN policy.

        For policy_type="mlp", delegates to SB3.
        For policy_type="gnn", uses custom PPO training loop.
        """
        if policy_type == "mlp":
            return cls._train_mlp(
                env, total_timesteps, learning_rate, n_steps,
                batch_size, gamma, gae_lambda, clip_range,
                ent_coef, vf_coef, save_path, log_dir, verbose, **kwargs
            )

        # ── GNN policy training ─────────────────────────
        from .gnn_policy import GNNActorCriticPolicy

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_leaves = env.unwrapped._num_leaves
        num_spines = env.unwrapped._num_spines
        action_dim = env.action_space.shape[0]

        policy = GNNActorCriticPolicy(
            observation_space=env.observation_space,
            action_dim=action_dim,
            num_leaves=num_leaves,
            num_spines=num_spines,
            gcn_hidden=gcn_hidden,
            gcn_layers=gcn_layers,
            features_dim=features_dim,
            log_std_init=log_std_init,
        ).to(device)

        optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)

        # Allow quick smoke runs with total_timesteps < rollout size.
        n_steps = max(1, min(n_steps, total_timesteps))

        # Rollout storage
        obs_buf = np.zeros((n_steps, *env.observation_space.shape), dtype=np.float32)
        act_buf = np.zeros((n_steps, action_dim), dtype=np.float32)
        rew_buf = np.zeros(n_steps, dtype=np.float32)
        done_buf = np.zeros(n_steps, dtype=np.float32)
        val_buf = np.zeros(n_steps, dtype=np.float32)
        logp_buf = np.zeros(n_steps, dtype=np.float32)

        obs, _ = env.reset()
        total_updates = total_timesteps // n_steps
        global_step = 0
        reward_history = deque(maxlen=100)
        episode_reward = 0.0

        for update in range(total_updates):
            # ── Collect rollout ─────────────────────────
            policy.eval()

            for step in range(n_steps):
                obs_buf[step] = obs
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

                with torch.no_grad():
                    action, log_prob, _, value = policy.get_action_and_value(obs_tensor)

                act = action.cpu().numpy().squeeze(0)
                act_buf[step] = act
                val_buf[step] = value.cpu().item()
                logp_buf[step] = log_prob.cpu().item()

                obs, reward, terminated, truncated, info = env.step(act)
                rew_buf[step] = reward
                done_buf[step] = float(terminated or truncated)
                episode_reward += reward
                global_step += 1

                if terminated or truncated:
                    reward_history.append(episode_reward)
                    episode_reward = 0.0
                    obs, _ = env.reset()

            # ── GAE advantage computation ───────────────
            with torch.no_grad():
                next_value = policy.get_value(
                    torch.FloatTensor(obs).unsqueeze(0).to(device)
                ).cpu().item()

            advantages = np.zeros(n_steps, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    next_val = next_value
                    next_nonterminal = 1.0 - done_buf[t]
                else:
                    next_val = val_buf[t + 1]
                    next_nonterminal = 1.0 - done_buf[t]

                delta = rew_buf[t] + gamma * next_val * next_nonterminal - val_buf[t]
                last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
                advantages[t] = last_gae

            returns = advantages + val_buf

            # ── PPO update ──────────────────────────────
            policy.train()

            # Flatten and create tensors
            b_obs = torch.FloatTensor(obs_buf).to(device)
            b_act = torch.FloatTensor(act_buf).to(device)
            b_logp = torch.FloatTensor(logp_buf).to(device)
            b_adv = torch.FloatTensor(advantages).to(device)
            b_ret = torch.FloatTensor(returns).to(device)

            # Normalise advantages
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            indices = np.arange(n_steps)
            clip_fracs = []
            final_epoch = 0

            for epoch in range(n_epochs):
                final_epoch = epoch + 1
                np.random.shuffle(indices)
                epoch_kl = []
                for start in range(0, n_steps, batch_size):
                    end = min(start + batch_size, n_steps)
                    mb_idx = indices[start:end]

                    _, new_logp, entropy, new_value = policy.get_action_and_value(
                        b_obs[mb_idx], b_act[mb_idx]
                    )

                    # Policy loss (clipped)
                    log_ratio = new_logp - b_logp[mb_idx]
                    ratio = log_ratio.exp()
                    clip_frac = ((ratio - 1.0).abs() > clip_range).float().mean().item()
                    clip_fracs.append(clip_frac)

                    # Approximate KL for early stopping
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        epoch_kl.append(approx_kl)

                    mb_adv = b_adv[mb_idx]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    v_loss = 0.5 * ((new_value - b_ret[mb_idx]) ** 2).mean()

                    # Entropy bonus
                    entropy_loss = entropy.mean()

                    # Total loss
                    loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                    optimizer.step()

                # KL early stopping (end of each epoch)
                avg_epoch_kl = np.mean(epoch_kl) if epoch_kl else 0.0
                if target_kl is not None and avg_epoch_kl > target_kl:
                    break

            # ── Logging ─────────────────────────────────
            if verbose and (update + 1) % max(1, total_updates // 20) == 0:
                avg_reward = np.mean(reward_history) if reward_history else 0.0
                avg_clip = np.mean(clip_fracs) if clip_fracs else 0.0
                with torch.no_grad():
                    _sample = b_obs[:min(64, len(b_obs))]
                    _, _log_std, _ = policy(_sample)
                    _std = _log_std.exp()
                    std_mean = _std.mean().item()
                    std_min  = _std.min().item()
                    std_max  = _std.max().item()
                print(
                    f"[Update {update + 1}/{total_updates}] "
                    f"steps={global_step:,}  "
                    f"avg_reward={avg_reward:.2f}  "
                    f"clip_frac={avg_clip:.3f}  "
                    f"std={std_mean:.3f} [{std_min:.3f},{std_max:.3f}]  "
                    f"epochs={final_epoch}/{n_epochs}  "
                    f"pg_loss={pg_loss.item():.4f}  "
                    f"v_loss={v_loss.item():.4f}"
                )

        # ── Save ────────────────────────────────────────
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "num_leaves": num_leaves,
                "num_spines": num_spines,
                "action_dim": action_dim,
                "gcn_hidden": gcn_hidden,
                "gcn_layers": gcn_layers,
                "features_dim": features_dim,
                "log_std_init": log_std_init,
            }, save_path + ".pt")
            print(f"Model saved to {save_path}.pt")

        agent = cls(num_leaves, num_spines, policy_type="gnn")
        agent._policy = policy
        agent._device = device
        return agent

    @classmethod
    def _train_mlp(cls, env, total_timesteps, learning_rate, n_steps,
                   batch_size, gamma, gae_lambda, clip_range,
                   ent_coef, vf_coef, save_path, log_dir, verbose, **kwargs):
        """V1-style MLP training via SB3."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback

        callbacks = []
        if save_path:
            ckpt_dir = str(Path(save_path).parent / "checkpoints")
            callbacks.append(
                CheckpointCallback(
                    save_freq=max(total_timesteps // 20, 10_000),
                    save_path=ckpt_dir,
                    name_prefix="ppo_routing",
                )
            )

        model = PPO(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            verbose=verbose,
            tensorboard_log=log_dir,
            **kwargs,
        )
        model.learn(total_timesteps=total_timesteps, callback=callbacks if callbacks else None)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(save_path)

        agent = cls(
            num_leaves=env.unwrapped._num_leaves,
            num_spines=env.unwrapped._num_spines,
            model=model,
            policy_type="mlp",
        )
        return agent

    # ── Persistence ─────────────────────────────────────

    @classmethod
    def load(cls, path: str, env=None, policy_type: str = "gnn") -> "PPOAgent":
        """Load a trained model from disk."""
        if policy_type == "mlp":
            from stable_baselines3 import PPO
            model = PPO.load(path, env=env)
            act_dim = model.action_space.shape[0]
            agent = cls.__new__(cls)
            agent.action_dim = act_dim
            agent.model = model
            agent.policy_type = "mlp"
            agent.num_leaves = 0
            agent.num_spines = 0
            agent._policy = None
            agent._device = torch.device("cpu")
            return agent

        # GNN load
        from .gnn_policy import GNNActorCriticPolicy

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_path = path if path.endswith(".pt") else path + ".pt"
        checkpoint = torch.load(load_path, map_location=device, weights_only=True)

        num_leaves = checkpoint["num_leaves"]
        num_spines = checkpoint["num_spines"]
        action_dim = checkpoint["action_dim"]

        obs_space = env.observation_space if env else gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        policy = GNNActorCriticPolicy(
            observation_space=obs_space,
            action_dim=action_dim,
            num_leaves=num_leaves,
            num_spines=num_spines,
            gcn_hidden=checkpoint.get("gcn_hidden", 64),
            gcn_layers=checkpoint.get("gcn_layers", 2),
            features_dim=checkpoint.get("features_dim", 128),
            log_std_init=checkpoint.get("log_std_init", -1.0),
        ).to(device)

        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()

        agent = cls(num_leaves, num_spines, policy_type="gnn")
        agent._policy = policy
        agent._device = device
        return agent

    def save(self, path: str):
        """Save the trained model to disk."""
        if self.policy_type == "mlp":
            if self.model is None:
                raise RuntimeError("No model to save.")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
        else:
            if self._policy is None:
                raise RuntimeError("No GNN policy to save.")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            save_path = path if path.endswith(".pt") else path + ".pt"
            torch.save({
                "policy_state_dict": self._policy.state_dict(),
                "num_leaves": self.num_leaves,
                "num_spines": self.num_spines,
                "action_dim": self.action_dim,
            }, save_path)

    @property
    def name(self) -> str:
        return self._label or f"PPO-{self.policy_type.upper()}"

    @name.setter
    def name(self, value: str):
        self._label = value
