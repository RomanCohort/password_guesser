"""
Password Guessing System
Based on MLP + MAMBA + Differential Evolution
With LLM-based information extraction

Modules:
- models: Neural network models (MAMBA, MLP, LLM extractor)
- optimization: Differential evolution, quantization, distillation, pruning
- utils: Tokenizers, feature extraction, logging, profiling
- rules: Hashcat-style password rules engine
- pcfg: Probabilistic Context-Free Grammar generator
- evaluation: Password strength assessment and metrics
- data: Data loading, augmentation, and preprocessing
- training: Distributed training and monitoring
- config: Configuration validation and management
- web: FastAPI web interface
"""

__version__ = "1.1.0"

# Lazy imports to avoid circular dependencies
__all__ = [
    # Models
    "MambaPasswordModel",
    "MambaConfig",
    "create_mamba_model",
    "MLPEncoder",
    "ConditionalMLPEncoder",
    "create_mlp_encoder",
    "LLMInfoExtractor",
    "LLMConfig",
    "PasswordDataset",
    "PasswordAugmentor",
    # Optimization
    "PasswordDEOptimizer",
    "MultiStrategyDEOptimizer",
    "SHADE",
    "ParallelDEOptimizer",
    "QuantizedMambaModel",
    "quantize_model",
    "DistillationTrainer",
    "prune_model",
    "EvaluationCache",
    "CheckpointManager",
    # Utils
    "PasswordTokenizer",
    "TargetFeatures",
    "FeatureVectorizer",
    # Rules
    "PasswordRuleEngine",
    "HashcatRuleParser",
    # PCFG
    "PCFGGenerator",
    "PCFGTrainer",
    # Evaluation
    "PasswordStrengthEvaluator",
    "EvaluationMetrics",
    # Data
    "DataPipeline",
    "PasswordAugmentor",
    # Training
    "DistributedTrainer",
    "TrainingMonitor",
    # Config
    "ConfigModel",
    "validate_config",
]


def __getattr__(name):
    """Lazy import for modules."""
    if name in __all__:
        # Models
        if name in ["MambaPasswordModel", "MambaConfig", "create_mamba_model",
                    "MLPEncoder", "ConditionalMLPEncoder", "create_mlp_encoder",
                    "LLMInfoExtractor", "LLMConfig", "PasswordDataset", "PasswordAugmentor"]:
            from models import (
                MambaPasswordModel, MambaConfig, create_mamba_model,
                MLPEncoder, ConditionalMLPEncoder, create_mlp_encoder,
                LLMInfoExtractor, LLMConfig, PasswordDataset, PasswordAugmentor
            )
            return locals()[name]
        # Optimization
        elif name in ["PasswordDEOptimizer", "MultiStrategyDEOptimizer", "SHADE",
                      "ParallelDEOptimizer", "QuantizedMambaModel", "quantize_model",
                      "DistillationTrainer", "prune_model", "EvaluationCache", "CheckpointManager"]:
            from optimization import (
                PasswordDEOptimizer, MultiStrategyDEOptimizer, SHADE,
                ParallelDEOptimizer, QuantizedMambaModel, quantize_model,
                DistillationTrainer, prune_model, EvaluationCache, CheckpointManager
            )
            return locals()[name]
        # Utils
        elif name in ["PasswordTokenizer", "TargetFeatures", "FeatureVectorizer"]:
            from utils import PasswordTokenizer, TargetFeatures, FeatureVectorizer
            return locals()[name]
        # Rules
        elif name in ["PasswordRuleEngine", "HashcatRuleParser"]:
            from rules import PasswordRuleEngine, HashcatRuleParser
            return locals()[name]
        # PCFG
        elif name in ["PCFGGenerator", "PCFGTrainer"]:
            from pcfg import PCFGGenerator, PCFGTrainer
            return locals()[name]
        # Evaluation
        elif name in ["PasswordStrengthEvaluator", "EvaluationMetrics"]:
            from evaluation import PasswordStrengthEvaluator, EvaluationMetrics
            return locals()[name]
        # Data
        elif name == "DataPipeline":
            from data import DataPipeline
            return DataPipeline
        # Training
        elif name in ["DistributedTrainer", "TrainingMonitor"]:
            from training import DistributedTrainer, TrainingMonitor
            return locals()[name]
        # Config
        elif name in ["ConfigModel", "validate_config"]:
            from config import ConfigModel, validate_config
            return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
