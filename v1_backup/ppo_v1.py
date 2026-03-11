"""
PPO (Proximal Policy Optimisation) agent using Stable-Baselines3.

This is the primary RL agent — trained in the Digital Twin and
designed for zero-shot transfer to the GNS3 fabric.
"""

import numpy as np
from typing import Optional
from pathlib import Path

from .base import BaseAgent


class PPOAgent(BaseAgent):
    """PPO-based adaptive routing agent.

    Wraps Stable-Baselines3 ``PPO`` for training, inference,
    saving, and loading.

    Usage:
        # Train a new agent
        agent = PPOAgent.train(env, total_timesteps=1_000_000)

        # Use in evaluation
        action = agent.act(observation)

        # Save / load
        agent.save("models/ppo_routing")
        agent = PPOAgent.load("models/ppo_routing", env=env)
    """

    def __init__(self, num_leaves: int, num_spines: int, model=None):
        super().__init__(num_leaves, num_spines)
        self.model = model

    # ── Inference ───────────────────────────────────────

    def act(self, observation: np.ndarray, deterministic: bool = True, **kwargs) -> np.ndarray:
        """Select action using the trained policy."""
        if self.model is None:
            raise RuntimeError(
                "PPO agent has no trained model. "
                "Call PPOAgent.train() or PPOAgent.load() first."
            )
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    # ── Training ────────────────────────────────────────

    @classmethod
    def train(
        cls,
        env,
        total_timesteps: int = 1_000_000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        save_path: Optional[str] = None,
        log_dir: Optional[str] = None,
        verbose: int = 1,
        **ppo_kwargs,
    ) -> "PPOAgent":
        """Train a new PPO agent from scratch.

        Args:
            env: A Gymnasium environment (``NetworkRoutingEnv``).
            total_timesteps: Total training interactions.
            save_path: Where to save the final model.
            log_dir: TensorBoard log directory.
            verbose: SB3 verbosity level.
            **ppo_kwargs: Additional SB3 PPO kwargs.

        Returns:
            Trained ``PPOAgent`` instance.
        """
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
            "MlpPolicy",
            env,
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
            **ppo_kwargs,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(save_path)

        agent = cls(
            num_leaves=env.unwrapped._num_leaves,
            num_spines=env.unwrapped._num_spines,
            model=model,
        )
        return agent

    # ── Persistence ─────────────────────────────────────

    @classmethod
    def load(cls, path: str, env=None) -> "PPOAgent":
        """Load a trained PPO model from disk.

        Args:
            path: File path (without .zip extension).
            env: Optional environment for continued training.

        Returns:
            ``PPOAgent`` with the loaded model.
        """
        from stable_baselines3 import PPO

        model = PPO.load(path, env=env)
        act_dim = model.action_space.shape[0]

        agent = cls.__new__(cls)
        agent.action_dim = act_dim
        agent.model = model
        agent.num_leaves = 0  # set externally if needed
        agent.num_spines = 0
        return agent

    def save(self, path: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    @property
    def name(self) -> str:
        return "PPO"
