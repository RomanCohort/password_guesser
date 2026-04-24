"""
RL Policy Networks

Neural network architectures for the penetration testing RL agent:
- Actor-Critic policy network (PPO-style)
- Q-Network (DQN-style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, List, Optional
import numpy as np


class PenTestPolicyNetwork(nn.Module):
    """
    Actor-Critic policy network for penetration testing.

    Actor: selects actions based on state
    Critic: estimates state value function
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 900,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            state_dim: Dimension of state vector
            action_dim: Total number of possible actions
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared encoder
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*layers)

        # Actor head (action selection)
        self.action_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.GELU(),
            nn.Linear(prev_dim // 2, action_dim),
        )

        # Critic head (state value)
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.GELU(),
            nn.Linear(prev_dim // 2, 1),
        )

        # Auxiliary heads
        self.technique_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.GELU(),
            nn.Linear(64, 14),  # 14 ATT&CK tactics
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Smaller gain for output layers
        for module in self.action_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)

    def forward(self, state_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state_vector: State tensor of shape (batch, state_dim)

        Returns:
            (action_logits, state_value)
        """
        features = self.encoder(state_vector)
        action_logits = self.action_head(features)
        state_value = self.value_head(features)
        return action_logits, state_value

    def get_action(
        self,
        state_vector: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action from policy.

        Args:
            state_vector: State tensor
            action_mask: Binary mask for valid actions
            deterministic: If True, return argmax instead of sampling

        Returns:
            (action, log_prob, state_value)
        """
        action_logits, state_value = self.forward(state_vector)

        # Apply action mask (set invalid actions to -inf)
        if action_mask is not None:
            action_logits = action_logits.masked_fill(
                action_mask == 0, float('-inf')
            )

        # Sample or take argmax
        if deterministic:
            action = action_logits.argmax(dim=-1)
            log_prob = torch.zeros_like(action, dtype=torch.float32)
        else:
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, state_value.squeeze(-1)

    def evaluate_actions(
        self,
        state_vector: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.

        Used for PPO update step.
        """
        action_logits, state_value = self.forward(state_vector)

        if action_mask is not None:
            action_logits = action_logits.masked_fill(
                action_mask == 0, float('-inf')
            )

        dist = Categorical(logits=action_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, state_value.squeeze(-1), entropy

    def select_action(
        self,
        state_vector: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """
        Convenience method for single-action selection.

        Args:
            state_vector: Numpy state vector
            action_mask: Numpy binary mask
            deterministic: Whether to use greedy selection

        Returns:
            Selected action index
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state_vector).unsqueeze(0)
            mask_t = torch.FloatTensor(action_mask).unsqueeze(0) if action_mask is not None else None

            action, _, _ = self.get_action(state_t, mask_t, deterministic)
            return action.item()


class PenTestQNetwork(nn.Module):
    """
    Q-Network for DQN-style learning.

    Estimates Q-values for each action given a state.
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 900,
        hidden_dims: Optional[List[int]] = None,
        dueling: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.dueling = dueling

        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim

        self.features = nn.Sequential(*layers)

        if dueling:
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(prev_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(prev_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
            )
        else:
            self.q_head = nn.Linear(prev_dim, action_dim)

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions."""
        features = self.features(state_vector)

        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            return value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            return self.q_head(features)

    def get_q_values(
        self,
        state_vector: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get Q-values as numpy array."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state_vector).unsqueeze(0)
            q_values = self.forward(state_t).squeeze(0).numpy()

            if action_mask is not None:
                q_values[action_mask == 0] = float('-inf')

            return q_values
