"""
Model Quantization for Password Guessing Models

Features:
- INT8 dynamic quantization
- INT8 static quantization with calibration
- Quantization-aware training (QAT) support
- 2-4x inference speedup

Usage:
    from optimization.quantization import QuantizedModel, quantize_model

    # Dynamic quantization
    quantized = quantize_model(model, dtype='qint8')

    # Static quantization (requires calibration)
    quantized = quantize_model_static(model, calibration_loader)
"""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, quantize_qat
from torch.quantization import MinMaxObserver, QConfig
from typing import Optional, Dict, Any, List, Callable
import copy


class QuantizedMambaModel(nn.Module):
    """
    Wrapper for quantized MAMBA model.

    Handles the quantization/dequantization process and
    provides a unified interface for inference.
    """

    def __init__(
        self,
        model: nn.Module,
        quantized_model: nn.Module,
        quantization_type: str = 'dynamic'
    ):
        super().__init__()
        self.original_model = model
        self.quantized_model = quantized_model
        self.quantization_type = quantization_type

    def forward(self, *args, **kwargs):
        """Forward through quantized model"""
        return self.quantized_model(*args, **kwargs)

    def get_model_size(self) -> Dict[str, float]:
        """Get model size in MB"""
        def get_size(model):
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return (param_size + buffer_size) / 1024 / 1024

        return {
            'original_size_mb': get_size(self.original_model),
            'quantized_size_mb': get_size(self.quantized_model),
            'compression_ratio': get_size(self.original_model) / max(get_size(self.quantized_model), 0.001)
        }


def quantize_model_dynamic(
    model: nn.Module,
    dtype: str = 'qint8',
    qconfig_spec: Optional[Dict] = None
) -> QuantizedMambaModel:
    """
    Apply dynamic quantization to the model.

    Dynamic quantization converts weights to lower precision at runtime,
    while activations remain in float. Best for LSTM, Transformer, and
    other models with dynamic behavior.

    Args:
        model: The model to quantize
        dtype: Quantization dtype ('qint8', 'float16')
        qconfig_spec: Custom quantization config specification

    Returns:
        QuantizedMambaModel wrapper
    """
    model.eval()
    original_model = copy.deepcopy(model)

    # Map dtype string to torch dtype
    dtype_map = {
        'qint8': torch.qint8,
        'float16': torch.float16,
    }
    torch_dtype = dtype_map.get(dtype, torch.qint8)

    # Default: quantize all Linear layers
    if qconfig_spec is None:
        qconfig_spec = {nn.Linear}

    # Apply dynamic quantization
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec=qconfig_spec,
        dtype=torch_dtype
    )

    return QuantizedMambaModel(
        model=original_model,
        quantized_model=quantized,
        quantization_type='dynamic'
    )


def quantize_model_static(
    model: nn.Module,
    calibration_loader,
    num_calibration_batches: int = 10,
    qconfig: Optional[QConfig] = None
) -> QuantizedMambaModel:
    """
    Apply static quantization with calibration.

    Static quantization requires a calibration dataset to determine
    optimal quantization parameters. Provides best performance but
    requires representative data.

    Args:
        model: The model to quantize
        calibration_loader: DataLoader for calibration
        num_calibration_batches: Number of batches for calibration
        qconfig: Custom quantization config

    Returns:
        QuantizedMambaModel wrapper
    """
    model.eval()
    original_model = copy.deepcopy(model)

    # Set qconfig
    if qconfig is None:
        qconfig = torch.quantization.get_default_qconfig('fbgemm')

    model.qconfig = qconfig

    # Prepare for quantization
    torch.quantization.prepare(model, inplace=True)

    # Calibration
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= num_calibration_batches:
                break

            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    input_ids, labels, features = batch
                else:
                    input_ids, labels = batch
                    features = torch.zeros(input_ids.size(0), 64)
                model(input_ids, features)
            else:
                model(batch)

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)

    return QuantizedMambaModel(
        model=original_model,
        quantized_model=model,
        quantization_type='static'
    )


class QuantizationAwareTraining(nn.Module):
    """
    Quantization-Aware Training (QAT) wrapper.

    Simulates quantization during training to allow the model
    to adapt to quantization effects.
    """

    def __init__(
        self,
        model: nn.Module,
        qconfig: Optional[QConfig] = None
    ):
        super().__init__()
        self.model = model

        # Set qconfig
        if qconfig is None:
            qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

        self.model.qconfig = qconfig

        # Prepare for QAT
        torch.quantization.prepare_qat(self.model, inplace=True)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def freeze(self) -> nn.Module:
        """
        Freeze quantization parameters and convert to quantized model.

        Call this after training is complete.
        """
        self.model.eval()
        torch.quantization.convert(self.model, inplace=True)
        return self.model

    def train(self, mode: bool = True):
        """Override train to handle QAT mode"""
        self.training = mode
        self.model.train(mode)
        return self


class QuantizedPasswordGenerator:
    """
    Password generation with quantized model.

    Handles the inference pipeline for quantized models,
    including any necessary dequantization for outputs.
    """

    def __init__(
        self,
        quantized_model: QuantizedMambaModel,
        tokenizer,
        device: str = 'cpu'
    ):
        self.model = quantized_model
        self.tokenizer = tokenizer
        self.device = device
        # Quantized models typically run on CPU
        self.model.quantized_model.to(device)

    @torch.no_grad()
    def generate(
        self,
        latent: torch.Tensor,
        max_length: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> str:
        """Generate password with quantized model"""
        self.model.quantized_model.eval()

        latent = latent.to(self.device)
        current_ids = torch.tensor([[self.tokenizer.sos_idx]], device=self.device)

        for _ in range(max_length):
            # Forward through quantized model
            logits = self.model.quantized_model(current_ids, latent)

            # Handle quantized output
            if hasattr(logits, 'dequantize'):
                logits = logits.dequantize()

            next_logits = logits[0, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, min(top_k, next_logits.size(-1)))[0][..., -1, None]
                next_logits[indices_to_remove] = -float('inf')

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = -float('inf')

            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == self.tokenizer.eos_idx:
                break

            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

        return self.tokenizer.decode(current_ids[0].tolist())

    def generate_batch(
        self,
        latent: torch.Tensor,
        n_samples: int = 10,
        **kwargs
    ) -> List[str]:
        """Generate multiple passwords"""
        passwords = []
        for _ in range(n_samples):
            password = self.generate(latent, **kwargs)
            passwords.append(password)
        return passwords


def benchmark_quantization(
    model: nn.Module,
    sample_input: torch.Tensor,
    sample_latent: torch.Tensor,
    num_runs: int = 100
) -> Dict[str, Any]:
    """
    Benchmark quantized vs original model performance.

    Args:
        model: Original model
        sample_input: Sample input tensor
        sample_latent: Sample latent tensor
        num_runs: Number of inference runs

    Returns:
        Dictionary with benchmark results
    """
    import time

    model.eval()
    device = next(model.parameters()).device

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input.to(device), sample_latent.to(device))

    # Benchmark original
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(sample_input.to(device), sample_latent.to(device))
    original_time = time.perf_counter() - start_time

    # Quantize model
    quantized_wrapper = quantize_model_dynamic(model)

    # Benchmark quantized
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = quantized_wrapper(sample_input, sample_latent)
    quantized_time = time.perf_counter() - start_time

    # Get sizes
    sizes = quantized_wrapper.get_model_size()

    return {
        'original_time_ms': original_time / num_runs * 1000,
        'quantized_time_ms': quantized_time / num_runs * 1000,
        'speedup': original_time / quantized_time,
        'original_size_mb': sizes['original_size_mb'],
        'quantized_size_mb': sizes['quantized_size_mb'],
        'compression_ratio': sizes['compression_ratio'],
    }


def apply_quantization_aware_training(
    model: nn.Module,
    train_loader,
    optimizer,
    num_epochs: int = 1,
    device: str = 'cuda'
) -> nn.Module:
    """
    Apply quantization-aware training.

    Fine-tunes the model with simulated quantization to adapt
    to quantization effects.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer instance
        num_epochs: Number of QAT epochs
        device: Training device

    Returns:
        Quantized model ready for deployment
    """
    model = QuantizationAwareTraining(model)
    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    input_ids, labels, features = batch
                else:
                    input_ids, labels = batch
                    features = torch.zeros(input_ids.size(0), 64)

            input_ids = input_ids.to(device)
            labels = labels.to(device)
            features = features.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, features)

            # Compute loss
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=0
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"QAT Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # Freeze and convert
    model.eval()
    quantized_model = model.freeze()

    return quantized_model


# Convenience function
def quantize_model(
    model: nn.Module,
    mode: str = 'dynamic',
    calibration_loader=None,
    **kwargs
) -> QuantizedMambaModel:
    """
    Quantize model using specified mode.

    Args:
        model: Model to quantize
        mode: 'dynamic' or 'static'
        calibration_loader: Required for static mode
        **kwargs: Additional arguments

    Returns:
        QuantizedMambaModel wrapper
    """
    if mode == 'dynamic':
        return quantize_model_dynamic(model, **kwargs)
    elif mode == 'static':
        if calibration_loader is None:
            raise ValueError("calibration_loader required for static quantization")
        return quantize_model_static(model, calibration_loader, **kwargs)
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")
