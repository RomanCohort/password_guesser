"""
Password Guesser - Package Init
"""

from models import (
    MambaPasswordModel,
    MambaConfig,
    MLPEncoder,
    ConditionalMLPEncoder,
    LLMInfoExtractor,
    LLMConfig,
    PasswordDataset,
    FrequencyWeightedPasswordDataset,
    PasswordAugmentor,
    create_mamba_model,
    create_mlp_encoder,
)

from optimization import (
    PasswordDEOptimizer,
    MultiStrategyDEOptimizer,
    SHADE,
    ParallelDEOptimizer,
    QuantizedMambaModel,
    quantize_model,
    DistillationTrainer,
    prune_model,
    get_model_sparsity,
    EvaluationCache,
    CheckpointManager,
)

from utils import PasswordTokenizer
from utils.feature_utils import TargetFeatures

__version__ = "1.0.0"
__all__ = [
    # Models
    "MambaPasswordModel",
    "MambaConfig",
    "MLPEncoder",
    "ConditionalMLPEncoder",
    "LLMInfoExtractor",
    "LLMConfig",
    "PasswordDataset",
    "FrequencyWeightedPasswordDataset",
    "PasswordAugmentor",
    "create_mamba_model",
    "create_mlp_encoder",
    # Optimization
    "PasswordDEOptimizer",
    "MultiStrategyDEOptimizer",
    "SHADE",
    "ParallelDEOptimizer",
    "QuantizedMambaModel",
    "quantize_model",
    "DistillationTrainer",
    "prune_model",
    "get_model_sparsity",
    "EvaluationCache",
    "CheckpointManager",
    # Utils
    "PasswordTokenizer",
    "TargetFeatures",
]
