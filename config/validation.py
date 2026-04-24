"""
Configuration validation using Pydantic models.

Defines strongly-typed configuration schemas for every subsystem
(LLM, MLP model, Mamba model, optimisation, training, and web server)
and provides helpers to load and validate YAML/JSON config files.
"""

import os
import json
from typing import List, Optional, Dict, Any

import yaml
from pydantic import BaseModel, Field, validator


class LLMConfig(BaseModel):
    """Configuration for the LLM information-extraction backend."""

    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: str = ""
    api_base: str = "https://api.deepseek.com/v1"
    openai_format: bool = True
    json_mode_supported: bool = True  # Set False for APIs without response_format support
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=1, le=8000)
    max_retries: int = Field(3, ge=1, le=10)
    retry_delay: float = Field(1.0, ge=0.1, le=60.0)

    # Additional provider-specific settings
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class MLPModelConfig(BaseModel):
    """Configuration for the MLP sub-model."""

    input_dim: int = Field(256, ge=1)
    hidden_dims: List[int] = [512, 256, 128]
    output_dim: int = Field(64, ge=1)
    dropout: float = Field(0.2, ge=0.0, le=0.5)

    @validator('hidden_dims')
    def hidden_dims_not_empty(cls, v):
        if not v:
            raise ValueError("hidden_dims must contain at least one integer")
        for dim in v:
            if dim < 1:
                raise ValueError(f"Each hidden dim must be >= 1, got {dim}")
        return v


class MambaModelConfig(BaseModel):
    """Configuration for the Mamba sub-model."""

    vocab_size: int = Field(128, ge=1)
    d_model: int = Field(128, ge=1)
    n_layers: int = Field(4, ge=1, le=32)
    d_state: int = Field(16, ge=1)
    d_conv: int = Field(4, ge=1)
    max_length: int = Field(32, ge=1, le=256)


class ModelConfig(BaseModel):
    """Aggregate model configuration."""

    mlp: MLPModelConfig = MLPModelConfig()
    mamba: MambaModelConfig = MambaModelConfig()


class OptimizationConfig(BaseModel):
    """Configuration for differential-evolution optimisation."""

    population_size: int = Field(100, ge=10)
    max_generations: int = Field(50, ge=1)
    mutation_rate: float = Field(0.8, ge=0.0, le=1.0)
    crossover_rate: float = Field(0.7, ge=0.0, le=1.0)
    elite_ratio: float = Field(0.1, ge=0.0, le=0.5)


class TrainingConfig(BaseModel):
    """Configuration for the training loop."""

    batch_size: int = Field(64, ge=1)
    learning_rate: float = Field(0.001, gt=0.0)
    epochs: int = Field(100, ge=1)
    device: str = "cuda"
    amp: bool = False
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 10
    scheduler: str = "cosine"

    @validator('device')
    def device_valid(cls, v):
        allowed = {'cuda', 'cpu', 'auto'}
        if v not in allowed and not v.startswith('cuda:'):
            raise ValueError(f"device must be one of {allowed} or 'cuda:N', got '{v}'")
        return v

    @validator('scheduler')
    def scheduler_valid(cls, v):
        allowed = {'cosine', 'linear', 'constant', 'step', 'plateau'}
        if v not in allowed:
            raise ValueError(f"scheduler must be one of {allowed}, got '{v}'")
        return v


class WebConfig(BaseModel):
    """Configuration for the web API server."""

    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)
    workers: int = 1
    auth_enabled: bool = False
    rate_limit: int = 100  # requests per minute
    api_keys: List[str] = []

    @validator('workers')
    def workers_positive(cls, v):
        if v < 1:
            raise ValueError("workers must be >= 1")
        return v


class PenTestConfigModel(BaseModel):
    """Configuration for the penetration testing module."""

    enabled: bool = False
    max_concurrent_attacks: int = Field(5, ge=1, le=20)
    timeout_per_action: int = Field(300, ge=30, le=3600)
    max_steps: int = Field(100, ge=10, le=1000)
    auto_mode: bool = True
    reflection_frequency: int = Field(5, ge=1, le=50)
    allowed_targets: List[str] = []
    tools_path: Dict[str, str] = {}


class RLConfigModel(BaseModel):
    """Configuration for the reinforcement learning agent."""

    algorithm: str = "ppo"
    learning_rate: float = Field(0.0003, gt=0.0, le=0.1)
    gamma: float = Field(0.99, ge=0.9, le=1.0)
    gae_lambda: float = Field(0.95, ge=0.8, le=1.0)
    batch_size: int = Field(32, ge=8, le=256)
    train_epochs: int = Field(4, ge=1, le=20)
    state_dim: int = Field(256, ge=64, le=1024)
    action_dim: int = Field(900, ge=100, le=5000)
    reflection_enabled: bool = True
    device: str = "auto"

    @validator('algorithm')
    def algorithm_valid(cls, v):
        allowed = {'ppo', 'dqn', 'a2c', 'reinforce'}
        if v not in allowed:
            raise ValueError(f"algorithm must be one of {allowed}, got '{v}'")
        return v


class KnowledgeConfigModel(BaseModel):
    """Configuration for the knowledge graph module."""

    cve_api_base: str = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    attack_data_path: str = "data/enterprise-attack.json"
    exploit_db_path: str = ""
    cache_enabled: bool = True
    preload_cves: int = Field(100, ge=0, le=5000)


class ConfigModel(BaseModel):
    """Complete configuration model combining all subsystem configs."""

    llm: LLMConfig = LLMConfig()
    model: ModelConfig = ModelConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    training: TrainingConfig = TrainingConfig()
    web: WebConfig = WebConfig()

    # Penetration testing configurations
    pentest: PenTestConfigModel = PenTestConfigModel()
    rl: RLConfigModel = RLConfigModel()
    knowledge: KnowledgeConfigModel = KnowledgeConfigModel()


def validate_config(config: dict) -> ConfigModel:
    """Validate a raw configuration dictionary.

    Args:
        config: Dictionary (typically loaded from YAML/JSON).

    Returns:
        A validated :class:`ConfigModel` instance.

    Raises:
        pydantic.ValidationError: If validation fails.
    """
    return ConfigModel(**config)


def load_and_validate_config(path: str) -> ConfigModel:
    """Load a YAML or JSON config file and validate it.

    The file format is determined by its extension (``.yaml``/``.yml``
    vs ``.json``).

    Args:
        path: Path to the configuration file.

    Returns:
        A validated :class:`ConfigModel` instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If validation fails.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()

    ext = os.path.splitext(path)[1].lower()
    if ext in ('.yaml', '.yml'):
        data = yaml.safe_load(raw)
    elif ext == '.json':
        data = json.loads(raw)
    else:
        # Default to YAML
        data = yaml.safe_load(raw)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping, got {type(data).__name__}")

    return validate_config(data)
