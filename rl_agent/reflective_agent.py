"""
Reflective RL Agent

RL agent with self-reflective reasoning capabilities:
- Combines neural policy with LLM-based strategic planning
- Reflects on failed actions to improve strategy
- Maintains an experience replay buffer with reflection annotations
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import json
import time
import logging
import random

import numpy as np

from .state import PenTestState
from .action import PenTestAction, ActionType, ActionSpace
from .environment import PenTestEnvironment
from .policy import PenTestPolicyNetwork

logger = logging.getLogger(__name__)


@dataclass
class Reflection:
    """Record of agent self-reflection after an action sequence."""
    timestamp: float
    actions_taken: List[Dict]
    results: List[str]  # 'success', 'failure', 'partial'
    total_reward: float
    observation: str
    lessons_learned: List[str] = field(default_factory=list)
    suggested_modifications: List[str] = field(default_factory=list)
    alternative_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "actions_taken": self.actions_taken,
            "results": self.results,
            "total_reward": self.total_reward,
            "observation": self.observation,
            "lessons_learned": self.lessons_learned,
            "suggested_modifications": self.suggested_modifications,
            "alternative_actions": self.alternative_actions,
        }


@dataclass
class Experience:
    """A single experience tuple for RL training."""
    state_vector: np.ndarray
    action_index: int
    reward: float
    next_state_vector: np.ndarray
    done: bool
    action_mask: np.ndarray
    reflection_weight: float = 1.0  # Weight boosted by reflection


class ReplayBuffer:
    """Experience replay buffer with reflection-based prioritization."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Experience] = []

    def add(self, experience: Experience) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        if len(self.buffer) < batch_size:
            return self.buffer

        # Prioritized sampling based on reflection weight
        weights = np.array([e.reflection_weight for e in self.buffer])
        weights = weights / weights.sum()

        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
            p=weights,
        )
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class ReflectiveRLAgent:
    """
    RL Agent with reflective reasoning for penetration testing.

    Combines:
    1. Neural policy network for action selection
    2. LLM-based strategic planning (optional)
    3. Self-reflection after action sequences
    4. Experience replay with reflection-weighted prioritization
    """

    def __init__(
        self,
        policy: Optional[PenTestPolicyNetwork] = None,
        action_space: Optional[ActionSpace] = None,
        llm_planner=None,
        state_dim: int = 256,
        action_dim: int = 900,
        reflection_frequency: int = 5,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        device: str = "auto",
        experience_store=None,  # Optional persistent experience store
    ):
        if device == "auto":
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            import torch
            self.device = torch.device(device)

        self.action_space = action_space or ActionSpace()
        self.action_dim = action_dim

        # Policy network
        if policy is not None:
            self.policy = policy.to(self.device)
        else:
            self.policy = PenTestPolicyNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
            ).to(self.device)

        # LLM planner for strategic decisions
        self.llm_planner = llm_planner
        self.reflection_frequency = reflection_frequency

        # RL hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon

        # Optimizer
        import torch.optim as optim
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Experience tracking
        self.replay_buffer = ReplayBuffer()
        self.episode_history: List[Dict] = []
        self.reflections: List[Reflection] = []

        # Persistent experience store for cross-session learning
        self.experience_store = experience_store
        self.step_count = 0

    def select_action(
        self,
        state: PenTestState,
        env: Optional[PenTestEnvironment] = None,
        deterministic: bool = False,
    ) -> Tuple[PenTestAction, int]:
        """
        Select next action using policy + optional LLM guidance.

        Returns:
            (selected_action, action_index)
        """
        valid_actions = self.action_space.get_valid_actions(state)

        if not valid_actions:
            raise ValueError("No valid actions available")

        state_vector = state.to_vector()
        action_mask = self.action_space.get_action_mask(state)

        # Get action from policy network
        action_idx = self.policy.select_action(
            state_vector,
            action_mask,
            deterministic=deterministic,
        )

        # Try to map index to valid action
        import torch
        selected_action = None
        for action in valid_actions:
            if self.action_space.action_to_index(action) == action_idx:
                selected_action = action
                break

        # Fallback: pick random valid action
        if selected_action is None:
            selected_action = random.choice(valid_actions)
            action_idx = self.action_space.action_to_index(selected_action)

        # Optional: use LLM to validate/enhance the choice
        if self.llm_planner and env is not None and not deterministic:
            try:
                llm_action = self.llm_planner.suggest_next_action(state, valid_actions)
                if llm_action is not None:
                    # Blend: 70% policy, 30% LLM suggestion
                    if random.random() < 0.3:
                        selected_action = llm_action
                        action_idx = self.action_space.action_to_index(llm_action)
            except Exception as e:
                logger.debug(f"LLM planning failed: {e}")

        return selected_action, action_idx

    def record_step(
        self,
        state: PenTestState,
        action: PenTestAction,
        action_idx: int,
        reward: float,
        next_state: PenTestState,
        done: bool,
    ) -> None:
        """Record a step for later training."""
        self.step_count += 1

        self.episode_history.append({
            "state": state.to_dict(),
            "action": action.to_dict(),
            "action_idx": action_idx,
            "reward": reward,
            "next_state": next_state.to_dict(),
            "done": done,
        })

        # Add to replay buffer
        self.replay_buffer.add(Experience(
            state_vector=state.to_vector(),
            action_index=action_idx,
            reward=reward,
            next_state_vector=next_state.to_vector(),
            done=done,
            action_mask=self.action_space.get_action_mask(state),
        ))

        # Trigger reflection periodically
        if self.step_count % self.reflection_frequency == 0:
            self._trigger_reflection()

    def _trigger_reflection(self) -> Optional[Reflection]:
        """Trigger self-reflection based on recent episode history."""
        if len(self.episode_history) < 2:
            return None

        recent = self.episode_history[-self.reflection_frequency:]
        actions_taken = [step["action"] for step in recent]
        rewards = [step["reward"] for step in recent]
        total_reward = sum(rewards)

        results = []
        for reward in rewards:
            if reward > 1.0:
                results.append("success")
            elif reward > 0:
                results.append("partial")
            else:
                results.append("failure")

        # Generate observation
        success_count = sum(1 for r in results if r == "success")
        failure_count = sum(1 for r in results if r == "failure")

        observation = (
            f"Last {len(recent)} actions: {success_count} successes, "
            f"{failure_count} failures. Total reward: {total_reward:.2f}"
        )

        lessons = []
        suggestions = []

        # Simple rule-based reflection
        if failure_count > success_count:
            lessons.append("Too many failures - consider changing strategy")
            suggestions.append("Try different scan types or target different hosts")

        if total_reward < 0:
            lessons.append("Negative cumulative reward - actions are counterproductive")
            suggestions.append("Focus on reconnaissance before exploitation")

        # Check if stuck in a loop
        action_types = [a["type"] for a in actions_taken]
        if len(set(action_types)) == 1:
            lessons.append(f"Repeating same action type: {action_types[0]}")
            suggestions.append("Diversify action selection")

        # LLM-enhanced reflection (if available)
        if self.llm_planner:
            try:
                llm_reflection = self._llm_reflect(actions_taken, results, total_reward)
                if llm_reflection:
                    lessons.extend(llm_reflection.get("lessons", []))
                    suggestions.extend(llm_reflection.get("suggestions", []))
            except Exception as e:
                logger.debug(f"LLM reflection failed: {e}")

        reflection = Reflection(
            timestamp=time.time(),
            actions_taken=actions_taken,
            results=results,
            total_reward=total_reward,
            observation=observation,
            lessons_learned=lessons,
            suggested_modifications=suggestions,
        )

        self.reflections.append(reflection)

        # Update replay buffer priorities based on reflection
        self._apply_reflection_weights(reflection)

        return reflection

    def _llm_reflect(
        self,
        actions: List[Dict],
        results: List[str],
        total_reward: float,
    ) -> Optional[Dict]:
        """Use LLM for deeper reflection analysis."""
        if self.llm_planner is None:
            return None

        try:
            # Format actions for LLM
            action_summary = "\n".join(
                f"- {a['type']} -> {a['target']} ({r})"
                for a, r in zip(actions, results)
            )

            prompt = f"""Analyze the following penetration testing actions and results:

{action_summary}

Total reward: {total_reward:.2f}

Provide:
1. Lessons learned from failures
2. Suggested strategy modifications
3. Alternative action recommendations

Respond in JSON format:
{{
    "lessons": ["lesson1", "lesson2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "alternatives": ["alternative1"]
}}"""

            response = self.llm_planner.extractor._call_api(
                messages=[
                    {"role": "system", "content": "You are a penetration testing strategy advisor."},
                    {"role": "user", "content": prompt},
                ],
                use_json_mode=True,
                temperature=0.3,
            )

            if response is None:
                logger.debug("LLM reflection: no response from API")
                return None

            return self.llm_planner.extractor._parse_json_response(response)
        except Exception as e:
            logger.debug(f"LLM reflection error: {e}")
            return None

    def _apply_reflection_weights(self, reflection: Reflection) -> None:
        """Boost replay buffer weights for experiences related to reflection."""
        boost_factor = 2.0 if reflection.total_reward < 0 else 1.5

        for exp in self.replay_buffer.buffer[-len(reflection.actions_taken):]:
            exp.reflection_weight *= boost_factor

    def train_step(self, batch_size: int = 32) -> float:
        """
        Perform one PPO training step.

        Returns:
            Training loss
        """
        import torch
        import torch.nn.functional as F

        if len(self.replay_buffer) < batch_size:
            return 0.0

        batch = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(
            np.stack([e.state_vector for e in batch])
        ).to(self.device)
        actions = torch.LongTensor(
            [e.action_index for e in batch]
        ).to(self.device)
        rewards = torch.FloatTensor(
            [e.reward for e in batch]
        ).to(self.device)
        next_states = torch.FloatTensor(
            np.stack([e.next_state_vector for e in batch])
        ).to(self.device)
        dones = torch.FloatTensor(
            [float(e.done) for e in batch]
        ).to(self.device)
        masks = torch.FloatTensor(
            np.stack([e.action_mask for e in batch])
        ).to(self.device)

        # Compute returns (simple MC)
        returns = rewards + self.gamma * (
            self.policy(next_states)[1].detach() * (1 - dones)
        )
        advantages = returns - self.policy(states)[1].detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        new_log_probs, values, entropy = self.policy.evaluate_actions(
            states, actions, masks
        )

        # Get old log probs (from current policy, detached)
        with torch.no_grad():
            old_logits, _ = self.policy(states)
            from torch.distributions import Categorical
            old_logits_masked = old_logits.masked_fill(masks == 0, float('-inf'))
            old_dist = Categorical(logits=old_logits_masked)
            old_log_probs = old_dist.log_prob(actions)

        # Clipped surrogate loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy_loss = -entropy.mean() * 0.01

        # Total loss
        loss = actor_loss + 0.5 * value_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()

    def save_reflection_log(self, path: str) -> None:
        """Save reflection log to JSON file."""
        data = [r.to_dict() for r in self.reflections]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save(self, path: str) -> None:
        """Save agent state."""
        import torch
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "reflections_count": len(self.reflections),
        }, path)

    def load(self, path: str) -> None:
        """Load agent state."""
        import torch
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint.get("step_count", 0)
