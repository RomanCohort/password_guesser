"""
RL Training Loop

Trains the reflective RL agent on simulated penetration testing environments.
"""

from typing import Dict, Optional, List
import time
import logging
import json
import os

import numpy as np

from .environment import PenTestEnvironment
from .reflective_agent import ReflectiveRLAgent, ReplayBuffer
from .action import ActionSpace

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Trains the reflective RL agent using PPO with self-reflection.

    Features:
    - Episode-based training with environment reset
    - Periodic evaluation
    - Checkpoint saving
    - Training metrics logging
    """

    def __init__(
        self,
        agent: ReflectiveRLAgent,
        env: Optional[PenTestEnvironment] = None,
        max_steps_per_episode: int = 100,
        train_batch_size: int = 32,
        train_epochs: int = 4,
        eval_frequency: int = 10,
        checkpoint_frequency: int = 50,
        checkpoint_dir: str = "checkpoints",
    ):
        self.agent = agent
        self.env = env or PenTestEnvironment()
        self.max_steps = max_steps_per_episode
        self.train_batch_size = train_batch_size
        self.train_epochs = train_epochs
        self.eval_frequency = eval_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_dir = checkpoint_dir

        self.metrics: List[Dict] = []
        self.best_reward = float('-inf')

    def train_episode(self) -> Dict:
        """
        Run one training episode.

        Returns:
            Episode metrics
        """
        state = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        successes = 0
        failures = 0

        for step in range(self.max_steps):
            # Select action
            action, action_idx = self.agent.select_action(
                state, env=self.env, deterministic=False
            )

            # Execute action
            next_state, reward, done, info = self.env.step(action)

            # Record experience
            self.agent.record_step(
                state, action, action_idx, reward, next_state, done
            )

            episode_reward += reward
            episode_steps += 1

            if reward > 0:
                successes += 1
            else:
                failures += 1

            state = next_state

            if done:
                break

        # Train on collected experience
        train_loss = 0.0
        if len(self.agent.replay_buffer) >= self.train_batch_size:
            for _ in range(self.train_epochs):
                loss = self.agent.train_step(self.train_batch_size)
                train_loss += loss
            train_loss /= self.train_epochs

        metrics = {
            "episode_reward": episode_reward,
            "episode_steps": episode_steps,
            "successes": successes,
            "failures": failures,
            "train_loss": train_loss,
            "compromised_hosts": len(state.compromised_hosts),
            "total_vulnerabilities": state.total_vulnerabilities,
            "credentials_found": len(state.credentials),
            "reflection_count": len(self.agent.reflections),
        }

        self.metrics.append(metrics)
        return metrics

    def train(self, n_episodes: int, verbose: bool = True) -> Dict:
        """
        Train for multiple episodes.

        Args:
            n_episodes: Number of training episodes
            verbose: Print progress

        Returns:
            Training summary
        """
        logger.info(f"Starting training for {n_episodes} episodes...")
        start_time = time.time()

        for ep in range(n_episodes):
            metrics = self.train_episode()

            if verbose and (ep + 1) % 10 == 0:
                avg_reward = np.mean([m["episode_reward"] for m in self.metrics[-10:]])
                logger.info(
                    f"Episode {ep + 1}/{n_episodes} | "
                    f"Reward: {metrics['episode_reward']:.2f} | "
                    f"Avg(10): {avg_reward:.2f} | "
                    f"Compromised: {metrics['compromised_hosts']} | "
                    f"Loss: {metrics['train_loss']:.4f}"
                )

            # Periodic evaluation
            if (ep + 1) % self.eval_frequency == 0:
                eval_metrics = self.evaluate(n_episodes=3)
                if eval_metrics["avg_reward"] > self.best_reward:
                    self.best_reward = eval_metrics["avg_reward"]
                    self._save_checkpoint("best.pt")

            # Periodic checkpoint
            if (ep + 1) % self.checkpoint_frequency == 0:
                self._save_checkpoint(f"episode_{ep + 1}.pt")

        elapsed = time.time() - start_time

        summary = {
            "total_episodes": n_episodes,
            "total_time": elapsed,
            "best_reward": self.best_reward,
            "final_avg_reward": np.mean([m["episode_reward"] for m in self.metrics[-10:]]),
            "total_reflections": len(self.agent.reflections),
        }

        logger.info(f"Training complete. {summary}")
        return summary

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict:
        """
        Evaluate agent without training.

        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic action selection

        Returns:
            Evaluation metrics
        """
        rewards = []
        steps_list = []
        compromised_counts = []

        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            steps = 0

            for step in range(self.max_steps):
                action, action_idx = self.agent.select_action(
                    state, deterministic=deterministic
                )
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                steps += 1
                state = next_state
                if done:
                    break

            rewards.append(episode_reward)
            steps_list.append(steps)
            compromised_counts.append(len(state.compromised_hosts))

        eval_metrics = {
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
            "avg_steps": np.mean(steps_list),
            "avg_compromised": np.mean(compromised_counts),
            "compromise_rate": np.mean([c > 0 for c in compromised_counts]),
        }

        return eval_metrics

    def _save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, filename)
        self.agent.save(path)
        logger.info(f"Checkpoint saved: {path}")

    def save_metrics(self, path: str) -> None:
        """Save training metrics to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
