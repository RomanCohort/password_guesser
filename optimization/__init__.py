"""Optimization modules for password guessing"""

from .differential_evolution import (
    PasswordDEOptimizer,
    PasswordCandidate,
    MutationStrategy,
    HybridDEOptimizer,
    AdaptiveDEOptimizer,
    StructuredPasswordDEOptimizer,
    MultiStrategyDEOptimizer,
    SHADE,
    ParallelDEOptimizer,
)

from .quantization import (
    QuantizedMambaModel,
    QuantizedPasswordGenerator,
    QuantizationAwareTraining,
    quantize_model,
    quantize_model_dynamic,
    quantize_model_static,
    benchmark_quantization,
)

from .distillation import (
    DistillationConfig,
    DistillationLoss,
    KnowledgeDistillation,
    DistillationTrainer,
    ProgressiveDistillation,
)

from .pruning import (
    PruningConfig,
    MagnitudePruning,
    StructuredPruning,
    GradientPruning,
    IterativePruning,
    PruningScheduler,
    prune_model,
    get_model_sparsity,
    count_parameters,
    print_model_summary,
)

from .system import (
    EvaluationCache,
    CheckpointManager,
    CheckpointInfo,
    IncrementalTrainer,
    PerformanceMonitor,
)

__all__ = [
    # Differential Evolution
    "PasswordDEOptimizer",
    "PasswordCandidate",
    "MutationStrategy",
    "HybridDEOptimizer",
    "AdaptiveDEOptimizer",
    "StructuredPasswordDEOptimizer",
    "MultiStrategyDEOptimizer",
    "SHADE",
    "ParallelDEOptimizer",
    # Quantization
    "QuantizedMambaModel",
    "QuantizedPasswordGenerator",
    "QuantizationAwareTraining",
    "quantize_model",
    "quantize_model_dynamic",
    "quantize_model_static",
    "benchmark_quantization",
    # Distillation
    "DistillationConfig",
    "DistillationLoss",
    "KnowledgeDistillation",
    "DistillationTrainer",
    "ProgressiveDistillation",
    # Pruning
    "PruningConfig",
    "MagnitudePruning",
    "StructuredPruning",
    "GradientPruning",
    "IterativePruning",
    "PruningScheduler",
    "prune_model",
    "get_model_sparsity",
    "count_parameters",
    "print_model_summary",
    # System
    "EvaluationCache",
    "CheckpointManager",
    "CheckpointInfo",
    "IncrementalTrainer",
    "PerformanceMonitor",
]
