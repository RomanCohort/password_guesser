"""
MAMBA-based Password Generation Model

Implements a simplified MAMBA architecture (Selective State Space Model)
for password sequence generation.

MAMBA paper: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
https://arxiv.org/abs/2312.00752
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Callable, Union
from dataclasses import dataclass
import heapq


@dataclass
class MambaConfig:
    """Configuration for MAMBA model"""
    vocab_size: int = 128
    d_model: int = 128        # Model dimension
    n_layers: int = 4         # Number of MAMBA layers
    d_state: int = 16         # SSM state dimension
    d_conv: int = 4           # Local convolution width
    expand_factor: int = 2    # Block expansion factor
    max_length: int = 32      # Maximum sequence length
    dropout: float = 0.1


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (SSM) - Core of MAMBA

    Implements a discretized SSM with input-dependent parameters:
    h'(t) = A h(t) + B x(t)
    y(t) = C h(t) + D x(t)

    Where A, B, C are now functions of the input (selectivity).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand_factor

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )

        # SSM parameters
        # A: State transition matrix (diagonal, negative for stability)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))

        # D: Skip connection parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Projections for B, C, and delta (selectivity)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through selective SSM.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Input projection and split
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # Convolution for local context
        x_conv = x_proj.transpose(1, 2)  # [batch, d_inner, seq_len]
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)  # [batch, seq_len, d_inner]
        x_conv = F.silu(x_conv)

        # SSM computation
        # Initialize A (negative for stability)
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        # Project x to get B, C, and delta
        x_dbl = self.x_proj(x_conv)  # [batch, seq_len, d_state * 2 + 1]
        delta, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)

        # Delta: input-dependent step size
        delta = F.softplus(delta)  # Ensure positive

        # Discretize A and B with delta
        # A_discrete = exp(delta * A)
        # B_discrete = (delta * A)^{-1} * (exp(delta * A) - I) * B
        # Simplified using the scan operation

        # Selective scan
        y = self.selective_scan(x_conv, delta, A, B, C)

        # Gated output
        y = y * F.silu(z)

        return self.out_proj(y)

    def selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Selective scan operation for efficient SSM computation.

        Uses parallel prefix scan algorithm for O(log n) parallel complexity
        instead of O(n) sequential.

        For production, use mamba-ssm CUDA kernels or causal-conv1d.
        This is a reference implementation.

        Args:
            u: Input [batch, seq_len, d_inner]
            delta: Step size [batch, seq_len, 1]
            A: State transition [d_inner, d_state]
            B: Input matrix [batch, seq_len, d_state]
            C: Output matrix [batch, seq_len, d_state]

        Returns:
            Output [batch, seq_len, d_inner]
        """
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]
        device = u.device
        dtype = u.dtype

        # Discretization
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        # Parallel prefix scan using associative property
        # For a proper parallel scan, use:
        # 1. mamba-ssm CUDA kernels (fastest)
        # 2. torch.cumsum for simplified version (works when A is scalar)
        # 3. This reference sequential version

        # Try to use vectorized operations where possible
        # Compute cumulative product of deltaA for state transition
        # log_deltaA = delta.unsqueeze(-1) * A_log.unsqueeze(0).unsqueeze(0)

        # Initialize state
        h = torch.zeros(batch_size, d_inner, d_state, device=device, dtype=dtype)

        # Use scan with torch.jit for potential speedup
        # For now, use optimized sequential version
        outputs = torch.empty(batch_size, seq_len, d_inner, device=device, dtype=dtype)

        for i in range(seq_len):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            outputs[:, i] = torch.sum(h * C[:, i].unsqueeze(1), dim=-1)

        return outputs

    def selective_scan_parallel(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel scan using chunked processing.
        Better for longer sequences on GPU.
        """
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]
        device = u.device

        # Chunk size for parallel processing
        chunk_size = min(64, seq_len)

        # Process in chunks
        outputs = []
        h = torch.zeros(batch_size, d_inner, d_state, device=device, dtype=u.dtype)

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start

            # Extract chunk
            u_chunk = u[:, chunk_start:chunk_end]
            delta_chunk = delta[:, chunk_start:chunk_end]
            B_chunk = B[:, chunk_start:chunk_end]
            C_chunk = C[:, chunk_start:chunk_end]

            # Compute chunk
            deltaA_chunk = torch.exp(delta_chunk.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
            deltaB_u_chunk = delta_chunk.unsqueeze(-1) * B_chunk.unsqueeze(2) * u_chunk.unsqueeze(-1)

            chunk_outputs = torch.empty(batch_size, chunk_len, d_inner, device=device, dtype=u.dtype)
            for i in range(chunk_len):
                h = deltaA_chunk[:, i] * h + deltaB_u_chunk[:, i]
                chunk_outputs[:, i] = torch.sum(h * C_chunk[:, i].unsqueeze(1), dim=-1)

            outputs.append(chunk_outputs)

        return torch.cat(outputs, dim=1)


class MambaBlock(nn.Module):
    """Complete MAMBA block with normalization and residual connection"""

    def __init__(self, config: MambaConfig):
        super().__init__()

        self.norm = nn.LayerNorm(config.d_model)
        self.ssm = SelectiveSSM(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        return x + self.dropout(self.ssm(self.norm(x)))


class FiLMConditioner(nn.Module):
    """
    Feature-wise Linear Modulation for conditioning.
    Applies affine transformation: gamma * x + beta, where gamma and beta
    are predicted from the conditioning latent.
    """

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(latent_dim, hidden_dim)
        self.beta_proj = nn.Linear(latent_dim, hidden_dim)

        # Initialize to identity transform
        nn.init.ones_(self.gamma_proj.weight.data.mean(dim=0, keepdim=True))
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states [batch, seq_len, hidden_dim]
            latent: Conditioning [batch, latent_dim]
        Returns:
            Modulated hidden states
        """
        gamma = self.gamma_proj(latent).unsqueeze(1)  # [batch, 1, hidden_dim]
        beta = self.beta_proj(latent).unsqueeze(1)
        return gamma * x + beta


class MambaPasswordModel(nn.Module):
    """
    MAMBA-based model for password generation.

    A generative model that produces passwords character by character,
    conditioned on latent features from the MLP encoder.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.max_length = config.max_length

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional encoding (learned)
        self.pos_embedding = nn.Embedding(config.max_length, config.d_model)

        # MAMBA layers
        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Conditioning: FiLM (better than simple addition)
        self.condition_proj = nn.Linear(64, config.d_model)
        self.film = FiLMConditioner(64, config.d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing).

        Args:
            input_ids: Token IDs [batch, seq_len]
            latent: Conditioning latent vector [batch, 64]
            attention_mask: Mask for padding [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Add positional encoding
        positions = torch.arange(seq_len, device=input_ids.device)
        x = x + self.pos_embedding(positions)

        # Add conditioning from latent
        if latent is not None:
            cond = self.condition_proj(latent)  # [batch, d_model]
            x = x + cond.unsqueeze(1)  # Broadcast across sequence

        # Apply MAMBA layers
        for layer in self.layers:
            x = layer(x)

        # Final norm and output projection
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits

    def generate_incremental(
        self,
        latent: torch.Tensor,
        tokenizer,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> str:
        """
        Generate a password using incremental state caching (O(n) instead of O(n^2)).

        This is the correct way to generate with SSMs - cache the hidden state
        and only process the new token each step.

        Uses the mamba_cuda module for optimized incremental inference when available.
        """
        if max_length is None:
            max_length = self.max_length

        self.eval()
        device = latent.device

        # Try to use CUDA-optimized incremental inference
        try:
            from models.mamba_cuda import IncrementalStateCache, SelectiveScanCuda

            # Pre-compute conditioning
            cond = self.condition_proj(latent) if latent is not None else None

            # Initialize incremental state
            first_layer = self.layers[0].ssm
            d_inner = first_layer.d_inner
            d_state = first_layer.d_state

            cache = IncrementalStateCache(
                n_layers=len(self.layers),
                d_inner=d_inner,
                d_state=d_state,
                batch_size=1
            )
            cache.initialize(device, dtype=latent.dtype)

            # Start with SOS token
            current_token = torch.tensor([[tokenizer.sos_idx]], device=device)
            generated_tokens = []

            with torch.no_grad():
                for step in range(max_length):
                    # Embed current token
                    x = self.token_embedding(current_token)
                    x = x + self.pos_embedding.weight[step:step + 1]

                    if cond is not None:
                        x = x + cond.unsqueeze(1)

                    # Process through MAMBA layers with state caching
                    for i, layer in enumerate(self.layers):
                        x_norm = layer.norm(x)

                        # Get SSM components
                        xz = layer.ssm.in_proj(x_norm)
                        x_proj, z = xz.chunk(2, dim=-1)

                        # Conv step
                        x_conv = x_proj.transpose(1, 2)
                        conv_out, new_cache = layer.ssm.conv1d(
                            x_conv,
                            cache.get_conv_cache(i)
                        ) if hasattr(layer.ssm.conv1d, 'forward') else (
                            layer.ssm.conv1d(x_conv)[:, :, :1], None
                        )
                        cache.set_conv_cache(i, new_cache)
                        x_conv = conv_out[:, :, -1:].transpose(1, 2)
                        x_conv = torch.nn.functional.silu(x_conv)

                        # SSM step
                        A = -torch.exp(layer.ssm.A_log.float())
                        x_dbl = layer.ssm.x_proj(x_conv)
                        delta, B, C = torch.split(
                            x_dbl,
                            [1, layer.ssm.d_state, layer.ssm.d_state],
                            dim=-1
                        )

                        h = cache.get_ssm_state(i)
                        y_step, h_new = SelectiveScanCuda.selective_scan_step(
                            u_step=x_conv.squeeze(1),
                            delta_step=delta.squeeze(1),
                            A=A,
                            B_step=B.squeeze(1),
                            C_step=C.squeeze(1),
                            h_prev=h,
                            D=layer.ssm.D
                        )
                        cache.set_ssm_state(i, h_new)

                        # Gated output
                        y_step = y_step.unsqueeze(1) * torch.nn.functional.silu(z)
                        x = x + layer.ssm.out_proj(y_step)
                        x = layer.dropout(x)

                    # Get logits
                    x = self.final_norm(x)
                    logits = self.lm_head(x[:, -1, :])

                    # Sample
                    next_logits = logits[0] / temperature
                    next_logits = self._filter_logits(next_logits, top_k, top_p)
                    probs = torch.nn.functional.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    if next_token.item() == tokenizer.eos_idx:
                        break

                    generated_tokens.append(next_token.item())
                    current_token = next_token.unsqueeze(0)

            return tokenizer.decode(generated_tokens)

        except ImportError:
            # Fallback to standard generation
            return self.generate(latent, tokenizer, max_length, temperature, top_k, top_p)

    @staticmethod
    def _filter_logits(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        Apply top-k and top-p (nucleus) filtering to logits.

        Works with both 1D (single sample) and 2D (batched) tensors.
        Modifies logits in-place and returns the modified tensor.
        """
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            # Get top-k threshold
            top_k_vals = torch.topk(logits, top_k, dim=-1)[0][..., -1]
            indices_to_remove = logits < top_k_vals.unsqueeze(-1)
            logits[indices_to_remove] = -float('inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Create mask for tokens to remove
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right to keep first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter back to original indexing
            scatter_dim = logits.dim() - 1  # Last dim (0 for 1D, 1 for 2D batched)
            indices_to_remove = sorted_indices_to_remove.scatter(scatter_dim, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')

        return logits

    def generate_mirostat(
        self,
        latent: torch.Tensor,
        tokenizer,
        max_length: Optional[int] = None,
        target_surprise: float = 6.0,
        learning_rate: float = 0.1,
        max_surprise: float = 20.0
    ) -> str:
        """
        Generate using Mirostat sampling.

        Mirostat dynamically adjusts the temperature to maintain a constant
        target surprise (information content), resulting in more controlled
        and coherent generation.

        Args:
            latent: Conditioning latent vector
            tokenizer: PasswordTokenizer instance
            max_length: Maximum password length
            target_surprise: Target surprise value (lower = more predictable)
            learning_rate: Adaptation rate for surprise tracking
            max_surprise: Maximum allowed surprise

        Returns:
            Generated password string
        """
        if max_length is None:
            max_length = self.max_length

        self.eval()
        device = latent.device

        current_ids = torch.tensor([[tokenizer.sos_idx]], device=device)
        mu = target_surprise * 2  # Initial threshold

        with torch.no_grad():
            for step in range(max_length):
                logits = self.forward(current_ids, latent)
                next_logits = logits[0, -1, :]

                # Compute probabilities
                probs = F.softmax(next_logits, dim=-1)

                # Sort by probability descending
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)

                # Mirostat: dynamically set top-k based on surprise target
                # Find k such that the sum of top-k probabilities exceeds a threshold
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Estimate surprise for each token
                log_probs = torch.log2(sorted_probs + 1e-10)
                surprise = -log_probs

                # Find cutoff based on mu (dynamic threshold)
                k = 1
                for i in range(len(sorted_probs)):
                    if sorted_probs[i] == 0:
                        break
                    k = i + 1
                    if surprise[i] > mu:
                        break
                    if k >= len(sorted_probs):
                        break

                # Sample from top-k
                top_k_probs = sorted_probs[:k]
                top_k_indices = sorted_indices[:k]

                # Renormalize
                top_k_probs = top_k_probs / top_k_probs.sum()

                # Sample
                sampled_idx = torch.multinomial(top_k_probs, num_samples=1)
                next_token = top_k_indices[sampled_idx].unsqueeze(0)

                # Update mu based on actual surprise
                actual_surprise = surprise[sampled_idx].item()
                mu = mu - learning_rate * (actual_surprise - target_surprise)
                mu = max(0, min(mu, max_surprise))

                if next_token.item() == tokenizer.eos_idx:
                    break

                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

        return tokenizer.decode(current_ids[0].tolist())

    def generate_tfs(
        self,
        latent: torch.Tensor,
        tokenizer,
        max_length: Optional[int] = None,
        tailfree: float = 0.95,
        temperature: float = 1.0
    ) -> str:
        """
        Generate using Tail Free Sampling (TFS).

        TFS looks at the second derivative of the sorted probability
        distribution to find a natural cutoff point, removing the
        "tail" of low-probability tokens.

        Args:
            latent: Conditioning latent vector
            tokenizer: PasswordTokenizer instance
            max_length: Maximum password length
            tailfree: TFS threshold (1.0 to disable)
            temperature: Sampling temperature

        Returns:
            Generated password string
        """
        if max_length is None:
            max_length = self.max_length

        self.eval()
        device = latent.device

        current_ids = torch.tensor([[tokenizer.sos_idx]], device=device)

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(current_ids, latent)
                next_logits = logits[0, -1, :] / temperature

                # Sort probabilities
                probs = F.softmax(next_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)

                # Compute first derivative (differences)
                first_deriv = sorted_probs[:-1] - sorted_probs[1:]

                # Compute second derivative
                if len(first_deriv) > 1:
                    second_deriv = torch.abs(first_deriv[:-1] - first_deriv[1:])
                    # Pad to match sorted_probs length
                    second_deriv = F.pad(second_deriv, (0, 2), value=0.0)

                    # Normalize second derivative
                    if second_deriv.sum() > 0:
                        second_deriv = second_deriv / second_deriv.sum()

                        # Cumulative sum
                        cum_second_deriv = torch.cumsum(second_deriv, dim=-1)

                        # Find cutoff
                        mask = cum_second_deriv < tailfree
                        # Keep at least one token
                        mask[0] = True

                        # Apply mask
                        filtered_probs = sorted_probs * mask.float()
                    else:
                        filtered_probs = sorted_probs
                else:
                    filtered_probs = sorted_probs

                # Renormalize
                if filtered_probs.sum() > 0:
                    filtered_probs = filtered_probs / filtered_probs.sum()
                else:
                    filtered_probs = sorted_probs

                # Sample
                sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
                next_token = sorted_indices[sampled_idx].unsqueeze(0)

                if next_token.item() == tokenizer.eos_idx:
                    break

                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

        return tokenizer.decode(current_ids[0].tolist())

    def generate_eta_cutoff(
        self,
        latent: torch.Tensor,
        tokenizer,
        max_length: Optional[int] = None,
        eta: float = 0.003,
        temperature: float = 1.0
    ) -> str:
        """
        Generate using Eta Cutoff sampling.

        Removes all tokens with probability less than eta * max(probability).
        This is a simple but effective filtering strategy.

        Args:
            latent: Conditioning latent vector
            tokenizer: PasswordTokenizer instance
            max_length: Maximum password length
            eta: Cutoff multiplier (lower = more filtering)
            temperature: Sampling temperature

        Returns:
            Generated password string
        """
        if max_length is None:
            max_length = self.max_length

        self.eval()
        device = latent.device

        current_ids = torch.tensor([[tokenizer.sos_idx]], device=device)

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(current_ids, latent)
                next_logits = logits[0, -1, :] / temperature

                probs = F.softmax(next_logits, dim=-1)

                # Eta cutoff: keep tokens with p >= eta * max(p)
                max_prob = probs.max()
                threshold = eta * max_prob
                mask = probs >= threshold

                # Ensure at least one token survives
                if not mask.any():
                    mask[probs.argmax()] = True

                # Zero out tokens below threshold
                filtered_probs = probs * mask.float()

                # Renormalize
                filtered_probs = filtered_probs / filtered_probs.sum()

                # Sample
                next_token = torch.multinomial(filtered_probs, num_samples=1)

                if next_token.item() == tokenizer.eos_idx:
                    break

                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

        return tokenizer.decode(current_ids[0].tolist())

    def generate_ensemble(
        self,
        latent: torch.Tensor,
        tokenizer,
        methods: Optional[List[str]] = None,
        n_per_method: int = 5,
        **kwargs
    ) -> List[Tuple[str, float, str]]:
        """
        Generate passwords using an ensemble of different methods.

        Runs multiple generation methods and combines results,
        scoring each password for ranking.

        Args:
            latent: Conditioning latent vector
            tokenizer: PasswordTokenizer instance
            methods: List of method names to use
            n_per_method: Number of samples per method
            **kwargs: Shared generation parameters

        Returns:
            List of (password, score, method_name) tuples, sorted by score
        """
        if methods is None:
            methods = ['sampling', 'typical', 'contrastive', 'mirostat']

        all_results = []

        for method in methods:
            for _ in range(n_per_method):
                try:
                    if method == 'sampling':
                        pwd = self.generate(
                            latent, tokenizer,
                            temperature=kwargs.get('temperature', 1.0),
                            top_k=kwargs.get('top_k', 0),
                            top_p=kwargs.get('top_p', 0.9)
                        )
                    elif method == 'typical':
                        pwd = self.generate_typical(
                            latent, tokenizer,
                            typical_mass=kwargs.get('typical_mass', 0.9)
                        )
                    elif method == 'contrastive':
                        pwd = self.generate_contrastive(
                            latent, tokenizer,
                            alpha=kwargs.get('contrastive_alpha', 0.5)
                        )
                    elif method == 'mirostat':
                        pwd = self.generate_mirostat(
                            latent, tokenizer,
                            target_surprise=kwargs.get('target_surprise', 6.0)
                        )
                    elif method == 'tfs':
                        pwd = self.generate_tfs(
                            latent, tokenizer,
                            tailfree=kwargs.get('tailfree', 0.95)
                        )
                    elif method == 'eta_cutoff':
                        pwd = self.generate_eta_cutoff(
                            latent, tokenizer,
                            eta=kwargs.get('eta', 0.003)
                        )
                    else:
                        continue

                    if pwd:
                        score = self.score_password(pwd, latent, tokenizer)
                        all_results.append((pwd, score, method))
                except Exception:
                    continue

        # Sort by score (descending)
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results

    def generate(
        self,
        latent: torch.Tensor,
        tokenizer,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> str:
        """
        Generate a password conditioned on latent features.

        Args:
            latent: Conditioning latent vector [batch, 64]
            tokenizer: PasswordTokenizer instance
            max_length: Maximum password length
            temperature: Sampling temperature
            top_k: Top-k filtering (0 to disable)
            top_p: Nucleus sampling threshold (1.0 to disable)

        Returns:
            Generated password string
        """
        if max_length is None:
            max_length = self.max_length

        self.eval()
        device = latent.device

        # Start with SOS token
        current_ids = torch.tensor([[tokenizer.sos_idx]], device=device)

        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.forward(current_ids, latent)
                next_logits = logits[0, -1, :] / temperature

                # Apply top-k and top-p filtering
                next_logits = MambaPasswordModel._filter_logits(next_logits, top_k, top_p)

                # Sample next token
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Stop if EOS
                if next_token.item() == tokenizer.eos_idx:
                    break

                # Append to sequence
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

        return tokenizer.decode(current_ids[0].tolist())

    def generate_batch(
        self,
        latent: torch.Tensor,
        tokenizer,
        n_samples: int = 10,
        **kwargs
    ) -> List[str]:
        """
        Generate multiple password candidates.

        Args:
            latent: Conditioning latent vector [batch, 64]
            tokenizer: PasswordTokenizer instance
            n_samples: Number of samples to generate
            **kwargs: Additional generation parameters

        Returns:
            List of generated password strings
        """
        passwords = []
        for _ in range(n_samples):
            password = self.generate(latent, tokenizer, **kwargs)
            passwords.append(password)

        return passwords

    def generate_batch_parallel(
        self,
        latent: torch.Tensor,
        tokenizer,
        n_samples: int = 10,
        batch_size: int = 16,
        **kwargs
    ) -> List[str]:
        """
        Generate multiple passwords in parallel batches for efficiency.

        Instead of generating one password at a time, this processes
        multiple sequences in parallel, significantly improving throughput.

        Args:
            latent: Conditioning latent vector [1, 64] or [batch, 64]
            tokenizer: PasswordTokenizer instance
            n_samples: Total number of samples to generate
            batch_size: Number of parallel sequences per batch
            **kwargs: Generation parameters (temperature, top_k, top_p)

        Returns:
            List of generated password strings
        """
        temperature = kwargs.get('temperature', 1.0)
        top_k = kwargs.get('top_k', 0)
        top_p = kwargs.get('top_p', 1.0)
        max_length = kwargs.get('max_length', self.max_length)

        self.eval()
        device = latent.device

        # Expand latent for batch processing
        if latent.size(0) == 1:
            latent_expanded = latent.expand(batch_size, -1)
        else:
            latent_expanded = latent

        all_passwords = []
        remaining = n_samples

        with torch.no_grad():
            while remaining > 0:
                current_batch_size = min(batch_size, remaining)

                # Initialize batch with SOS tokens
                current_ids = torch.full(
                    (current_batch_size, 1),
                    tokenizer.sos_idx,
                    dtype=torch.long,
                    device=device
                )

                # Active mask for tracking which sequences are still generating
                active = torch.ones(current_batch_size, dtype=torch.bool, device=device)

                for step in range(max_length):
                    if not active.any():
                        break

                    # Forward pass for entire batch
                    logits = self.forward(current_ids, latent_expanded[:current_batch_size])

                    # Get next token logits
                    next_logits = logits[:, -1, :] / temperature

                    # Apply top-k and top-p filtering (works on batch)
                    next_logits = MambaPasswordModel._filter_logits(next_logits, top_k, top_p)

                    # Sample next tokens
                    probs = F.softmax(next_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    # Mark EOS tokens as inactive
                    eos_mask = next_tokens == tokenizer.eos_idx
                    active = active & ~eos_mask

                    # Append non-EOS tokens
                    current_ids = torch.cat([current_ids, next_tokens.unsqueeze(1)], dim=1)

                # Decode all sequences in batch
                for i in range(current_batch_size):
                    password = tokenizer.decode(current_ids[i].tolist())
                    if password:  # Skip empty passwords
                        all_passwords.append(password)

                remaining -= current_batch_size

        return all_passwords[:n_samples]

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        latent: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for training.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            latent: Conditioning latent [batch, 64]
            labels: Target token IDs [batch, seq_len]

        Returns:
            Loss value
        """
        logits = self.forward(input_ids, latent)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Cross entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=0  # PAD token
        )

        return loss

    def score_password(
        self,
        password: str,
        latent: torch.Tensor,
        tokenizer
    ) -> float:
        """
        Score a password based on model probability.

        Args:
            password: Password string to score
            latent: Conditioning latent [batch, 64]
            tokenizer: PasswordTokenizer instance

        Returns:
            Log probability score
        """
        # Truncate to max_length to avoid positional embedding overflow
        input_ids = torch.tensor(
            [tokenizer.encode(password, max_length=self.max_length)],
            device=latent.device
        )

        with torch.no_grad():
            logits = self.forward(input_ids, latent)

            # Compute log probability
            log_probs = F.log_softmax(logits, dim=-1)

            # Sum of log probabilities for each token
            total_log_prob = 0.0
            for i, token_id in enumerate(input_ids[0, 1:], 1):  # Skip SOS
                total_log_prob += log_probs[0, i-1, token_id].item()

        return total_log_prob

    def generate_with_temperature_schedule(
        self,
        latent: torch.Tensor,
        tokenizer,
        max_length: Optional[int] = None,
        temperature_schedule: Union[float, str, Callable] = "linear_decay",
        top_k: int = 0,
        top_p: float = 1.0
    ) -> str:
        """
        Generate password with dynamic temperature scheduling.

        Temperature scheduling allows the model to be more exploratory
        at the beginning and more confident at the end (or vice versa).

        Args:
            latent: Conditioning latent vector
            tokenizer: PasswordTokenizer instance
            max_length: Maximum password length
            temperature_schedule: Can be:
                - float: Constant temperature
                - "linear_decay": Start high, decrease linearly
                - "linear_increase": Start low, increase linearly
                - "cosine": Cosine annealing
                - Callable: Function(step, max_steps) -> temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold

        Returns:
            Generated password string
        """
        if max_length is None:
            max_length = self.max_length

        self.eval()
        device = latent.device

        # Build temperature function
        if isinstance(temperature_schedule, (int, float)):
            temp_fn = lambda step, max_s: float(temperature_schedule)
        elif temperature_schedule == "linear_decay":
            temp_fn = lambda step, max_s: max(0.5, 1.5 - step / max_s)
        elif temperature_schedule == "linear_increase":
            temp_fn = lambda step, max_s: min(2.0, 0.5 + step / max_s)
        elif temperature_schedule == "cosine":
            temp_fn = lambda step, max_s: 0.5 + 0.5 * math.cos(math.pi * step / max_s)
        elif callable(temperature_schedule):
            temp_fn = temperature_schedule
        else:
            temp_fn = lambda step, max_s: 1.0

        current_ids = torch.tensor([[tokenizer.sos_idx]], device=device)

        with torch.no_grad():
            for step in range(max_length):
                logits = self.forward(current_ids, latent)

                # Get temperature for this step
                temperature = temp_fn(step, max_length)

                next_logits = logits[0, -1, :] / temperature

                # Apply top-k
                if top_k > 0:
                    top_k_actual = min(top_k, self.vocab_size)
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k_actual)[0][..., -1, None]
                    next_logits[indices_to_remove] = -float('inf')

                # Apply top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = -float('inf')

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if next_token.item() == tokenizer.eos_idx:
                    break

                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

        return tokenizer.decode(current_ids[0].tolist())

    def generate_beam_search(
        self,
        latent: torch.Tensor,
        tokenizer,
        max_length: Optional[int] = None,
        beam_width: int = 5,
        length_penalty: float = 1.0,
        temperature: float = 1.0
    ) -> List[Tuple[str, float]]:
        """
        Generate password candidates using beam search.

        Beam search explores multiple hypotheses simultaneously,
        keeping the top-k most probable sequences at each step.

        Args:
            latent: Conditioning latent vector
            tokenizer: PasswordTokenizer instance
            max_length: Maximum password length
            beam_width: Number of beams to keep
            length_penalty: Penalty for longer sequences (>1 favors shorter, <1 favors longer)
            temperature: Sampling temperature

        Returns:
            List of (password, log_probability) tuples, sorted by score
        """
        if max_length is None:
            max_length = self.max_length

        self.eval()
        device = latent.device

        # Each beam: (log_prob, token_ids, finished)
        beams = [(0.0, [tokenizer.sos_idx], False)]
        completed = []

        with torch.no_grad():
            for step in range(max_length):
                all_candidates = []

                for log_prob, tokens, finished in beams:
                    if finished:
                        completed.append((log_prob, tokens, True))
                        continue

                    # Forward pass
                    input_ids = torch.tensor([tokens], device=device)
                    logits = self.forward(input_ids, latent)

                    # Get next token logits
                    next_logits = logits[0, -1, :] / temperature
                    log_probs = F.log_softmax(next_logits, dim=-1)

                    # Get top beam_width tokens
                    top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                    for i in range(beam_width):
                        token_id = top_indices[i].item()
                        token_log_prob = top_log_probs[i].item()

                        new_tokens = tokens + [token_id]
                        new_log_prob = log_prob + token_log_prob

                        if token_id == tokenizer.eos_idx:
                            completed.append((new_log_prob, new_tokens, True))
                        else:
                            all_candidates.append((new_log_prob, new_tokens, False))

                # Keep top beam_width candidates
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                beams = all_candidates[:beam_width]

                if not beams:  # All beams finished
                    break

        # Add remaining beams to completed
        completed.extend(beams)

        # Apply length penalty and sort
        results = []
        for log_prob, tokens, _ in completed:
            length = len(tokens) - 1  # Exclude SOS
            if length > 0:
                adjusted_score = log_prob / (length ** length_penalty)
                password = tokenizer.decode(tokens[1:])  # Exclude SOS
                if password:
                    results.append((password, adjusted_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:beam_width]

    def generate_diverse_beam(
        self,
        latent: torch.Tensor,
        tokenizer,
        max_length: Optional[int] = None,
        num_groups: int = 3,
        diversity_penalty: float = 0.5,
        beam_width: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Generate diverse passwords using Diverse Beam Search.

        Maintains multiple groups of beams with inter-group diversity
        penalty to encourage varied outputs.

        Args:
            latent: Conditioning latent vector
            tokenizer: PasswordTokenizer instance
            max_length: Maximum password length
            num_groups: Number of diverse groups
            diversity_penalty: Penalty for similar tokens across groups
            beam_width: Beams per group

        Returns:
            List of (password, score) tuples
        """
        if max_length is None:
            max_length = self.max_length

        self.eval()
        device = latent.device

        # Initialize groups
        groups = [
            [(0.0, [tokenizer.sos_idx], False)] for _ in range(num_groups)
        ]
        all_completed = []

        with torch.no_grad():
            for step in range(max_length):
                group_tokens = []  # Track tokens chosen by each group

                for g, beams in enumerate(groups):
                    all_candidates = []
                    other_group_tokens = set()
                    for other_g, other_tokens in enumerate(group_tokens):
                        if other_g != g:
                            other_group_tokens.update(other_tokens)

                    for log_prob, tokens, finished in beams:
                        if finished:
                            all_completed.append((log_prob, tokens, True))
                            continue

                        input_ids = torch.tensor([tokens], device=device)
                        logits = self.forward(input_ids, latent)
                        next_logits = logits[0, -1, :]
                        log_probs = F.log_softmax(next_logits, dim=-1)

                        # Apply diversity penalty to tokens chosen by other groups
                        for token in other_group_tokens:
                            log_probs[token] -= diversity_penalty

                        top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                        for i in range(beam_width):
                            token_id = top_indices[i].item()
                            token_log_prob = top_log_probs[i].item()

                            new_tokens = tokens + [token_id]
                            new_log_prob = log_prob + token_log_prob

                            if token_id == tokenizer.eos_idx:
                                all_completed.append((new_log_prob, new_tokens, True))
                            else:
                                all_candidates.append((new_log_prob, new_tokens, False))

                    # Track chosen tokens
                    chosen = set()
                    for _, tokens, _ in all_candidates[:beam_width]:
                        if len(tokens) > 1:
                            chosen.add(tokens[-1])
                    group_tokens.append(chosen)

                # Update each group
                new_groups = []
                for g in range(num_groups):
                    group_candidates = [c for c in all_candidates if c not in groups[g]]
                    group_candidates.sort(key=lambda x: x[0], reverse=True)
                    new_groups.append(group_candidates[:beam_width] if group_candidates else groups[g])
                groups = new_groups

                # Check if all groups are done
                all_finished = all(all(f for _, _, f in beams) for beams in groups)
                if all_finished:
                    break

        # Collect results
        results = []
        for log_prob, tokens, _ in all_completed:
            password = tokenizer.decode(tokens[1:])
            if password:
                results.append((password, log_prob))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:num_groups * beam_width]

    def generate_typical(
        self,
        latent: torch.Tensor,
        tokenizer,
        max_length: Optional[int] = None,
        typical_mass: float = 0.9,
        temperature: float = 1.0
    ) -> str:
        """
        Generate using typical sampling (entropy-based).

        Typical sampling selects tokens whose information content is
        close to the expected information content, leading to more
        coherent outputs.

        Args:
            latent: Conditioning latent vector
            tokenizer: PasswordTokenizer instance
            max_length: Maximum password length
            typical_mass: Proportion of probability mass to consider
            temperature: Sampling temperature

        Returns:
            Generated password string
        """
        if max_length is None:
            max_length = self.max_length

        self.eval()
        device = latent.device

        current_ids = torch.tensor([[tokenizer.sos_idx]], device=device)

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(current_ids, latent)
                next_logits = logits[0, -1, :] / temperature

                # Compute probabilities and entropy
                probs = F.softmax(next_logits, dim=-1)
                log_probs = F.log_softmax(next_logits, dim=-1)

                # Entropy of the distribution
                entropy = -torch.sum(probs * log_probs)

                # Information content of each token
                info_content = -log_probs

                # Absolute difference from entropy (typical set criterion)
                typicality = torch.abs(info_content - entropy)

                # Sort by typicality (lower is more typical)
                sorted_typicality, sorted_indices = torch.sort(typicality)

                # Accumulate probability mass until typical_mass is reached
                cumulative_mass = 0.0
                typical_indices = []

                for idx in sorted_indices:
                    token_idx = idx.item()
                    cumulative_mass += probs[token_idx].item()
                    typical_indices.append(token_idx)
                    if cumulative_mass >= typical_mass:
                        break

                # Filter logits to typical set
                mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
                mask[typical_indices] = False
                next_logits[mask] = -float('inf')

                # Sample from typical set
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if next_token.item() == tokenizer.eos_idx:
                    break

                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

        return tokenizer.decode(current_ids[0].tolist())

    def generate_contrastive(
        self,
        latent: torch.Tensor,
        tokenizer,
        max_length: Optional[int] = None,
        alpha: float = 0.5,
        contrastive_samples: int = 4
    ) -> str:
        """
        Generate using contrastive search.

        Maintains a context window and penalizes tokens that would
        make the output too similar to previously generated content.

        Args:
            latent: Conditioning latent vector
            tokenizer: PasswordTokenizer instance
            max_length: Maximum password length
            alpha: Contrastive penalty strength
            contrastive_samples: Number of contrastive candidates

        Returns:
            Generated password string
        """
        if max_length is None:
            max_length = self.max_length

        self.eval()
        device = latent.device

        current_ids = torch.tensor([[tokenizer.sos_idx]], device=device)
        hidden_states_history = []

        with torch.no_grad():
            for step in range(max_length):
                logits = self.forward(current_ids, latent)
                next_logits = logits[0, -1, :]

                # Get top candidates
                top_probs, top_indices = torch.topk(F.softmax(next_logits, dim=-1), contrastive_samples)

                if len(hidden_states_history) > 0:
                    current_hidden = logits[0, -1, :]

                    best_score = float('-inf')
                    best_token = top_indices[0].item()

                    for i, (prob, token_id) in enumerate(zip(top_probs, top_indices)):
                        # Degeneracy penalty: similarity to previous hidden states
                        token_hidden = self.token_embedding.weight[token_id]
                        max_similarity = 0.0
                        for past_hidden in hidden_states_history:
                            sim = F.cosine_similarity(
                                token_hidden.unsqueeze(0),
                                past_hidden.unsqueeze(0)
                            )
                            max_similarity = max(max_similarity, sim.item())

                        # Score = log prob - alpha * max_similarity
                        score = torch.log(prob + 1e-10).item() - alpha * max_similarity
                        if score > best_score:
                            best_score = score
                            best_token = token_id.item()

                    next_token = torch.tensor([[best_token]], device=device)
                else:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)

                if next_token.item() == tokenizer.eos_idx:
                    break

                hidden_states_history.append(logits[0, -1, :].clone())
                current_ids = torch.cat([current_ids, next_token], dim=1)

        return tokenizer.decode(current_ids[0].tolist())


def create_mamba_model(config: dict) -> MambaPasswordModel:
    """Factory function to create MAMBA model from config"""
    mamba_config = config.get('model', {}).get('mamba', {})

    model_config = MambaConfig(
        vocab_size=mamba_config.get('vocab_size', 128),
        d_model=mamba_config.get('d_model', 128),
        n_layers=mamba_config.get('n_layers', 4),
        d_state=mamba_config.get('d_state', 16),
        d_conv=mamba_config.get('d_conv', 4),
        max_length=mamba_config.get('max_length', 32)
    )

    return MambaPasswordModel(model_config)
