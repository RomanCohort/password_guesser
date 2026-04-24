"""
MLP Encoder for feature to latent space mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


class MLPEncoder(nn.Module):
    """
    Multi-layer perceptron for encoding target features into latent vectors.

    Maps heterogeneous feature vectors to a continuous latent space
    that conditions the password generation model.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = None,
        output_dim: int = 64,
        dropout: float = 0.2,
        activation: str = "relu"
    ):
        """
        Initialize MLP encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output latent dimension
            dropout: Dropout rate
            activation: Activation function ('relu', 'leaky_relu', 'gelu', 'silu')
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer with normalization for bounded latent space
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # Bound output to [-1, 1] for stable training

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
        }
        return activations.get(name, nn.ReLU())

    def _init_weights(self):
        """Initialize network weights with Kaiming/He initialization (optimal for ReLU/GELU)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming initialization for ReLU-like activations
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim] or [input_dim]

        Returns:
            Latent vector [batch_size, output_dim]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        return self.network(x)


class ConditionalMLPEncoder(MLPEncoder):
    """
    MLP Encoder with conditional outputs for password generation.

    Outputs both latent vector and additional conditioning parameters:
    - latent: Main latent vector for MAMBA conditioning
    - length_logits: Predicted password length distribution
    - pattern_logits: Predicted pattern type distribution
    """

    PATTERNS = [
        'lower', 'upper', 'digit', 'alpha', 'alphanumeric',
        'name_digit', 'name_digit_special', 'leet'
    ]

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = None,
        output_dim: int = 64,
        max_length: int = 32,
        dropout: float = 0.2
    ):
        super().__init__(input_dim, hidden_dims, output_dim, dropout)

        self.max_length = max_length
        self.num_patterns = len(self.PATTERNS)

        # Additional output heads
        self.length_head = nn.Linear(output_dim, max_length)
        self.pattern_head = nn.Linear(output_dim, self.num_patterns)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = True
    ) -> dict:
        """
        Forward pass with auxiliary outputs.

        Args:
            x: Input features [batch_size, input_dim]
            return_aux: Whether to return auxiliary predictions

        Returns:
            Dictionary with latent and optionally auxiliary outputs
        """
        latent = super().forward(x)

        result = {'latent': latent}

        if return_aux:
            result['length_logits'] = self.length_head(latent)
            result['pattern_logits'] = self.pattern_head(latent)

        return result

    def predict_length(self, x: torch.Tensor) -> torch.Tensor:
        """Predict most likely password length"""
        latent = super().forward(x)
        length_logits = self.length_head(latent)
        return torch.argmax(length_logits, dim=-1) + 1  # 1-indexed

    def predict_pattern(self, x: torch.Tensor) -> str:
        """Predict most likely pattern type"""
        latent = super().forward(x)
        pattern_logits = self.pattern_head(latent)
        pattern_idx = torch.argmax(pattern_logits, dim=-1)
        return self.PATTERNS[pattern_idx.item()]


class FeatureMLP(nn.Module):
    """
    Specialized MLP for processing different feature types separately
    before fusion.

    Has separate sub-networks for:
    - Name features
    - Date features
    - Number features
    - Keyword features

    Then fuses them through a final MLP.
    """

    def __init__(
        self,
        name_dim: int = 64,
        date_dim: int = 32,
        number_dim: int = 16,
        keyword_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()

        # Feature-specific encoders
        self.name_encoder = nn.Sequential(
            nn.Linear(name_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.date_encoder = nn.Sequential(
            nn.Linear(date_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.number_encoder = nn.Sequential(
            nn.Linear(number_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.keyword_encoder = nn.Sequential(
            nn.Linear(keyword_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion network
        fusion_input_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 4 + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        name_features: torch.Tensor,
        date_features: torch.Tensor,
        number_features: torch.Tensor,
        keyword_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Process separate feature types and fuse.

        Args:
            name_features: Name-related features [batch, name_dim]
            date_features: Date-related features [batch, date_dim]
            number_features: Number-related features [batch, number_dim]
            keyword_features: Keyword features [batch, keyword_dim]

        Returns:
            Fused latent representation [batch, output_dim]
        """
        # Encode each feature type
        name_encoded = self.name_encoder(name_features)
        date_encoded = self.date_encoder(date_features)
        number_encoded = self.number_encoder(number_features)
        keyword_encoded = self.keyword_encoder(keyword_features)

        # Concatenate and fuse
        fused = torch.cat([
            name_encoded,
            date_encoded,
            number_encoded,
            keyword_encoded
        ], dim=-1)

        return self.fusion(fused)


def create_mlp_encoder(config: dict) -> MLPEncoder:
    """Factory function to create MLP encoder from config"""
    mlp_config = config.get('model', {}).get('mlp', {})

    return MLPEncoder(
        input_dim=mlp_config.get('input_dim', 256),
        hidden_dims=mlp_config.get('hidden_dims', [512, 256, 128]),
        output_dim=mlp_config.get('output_dim', 64),
        dropout=mlp_config.get('dropout', 0.2)
    )
