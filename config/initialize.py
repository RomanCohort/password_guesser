"""
Centralized Application Initialization

Single entry point for initializing configuration, logging, and all subsystems.
Use this instead of loading config directly with yaml.safe_load().
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def initialize(
    config_path: str = "config.yaml",
    log_level: str = None,
    log_file: str = None,
    json_logs: bool = False,
) -> dict:
    """
    Initialize the application: load config, setup logging, apply env overrides.

    This is the single entry point for all applications (CLI, web, training).

    Args:
        config_path: Path to YAML/JSON config file
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file name
        json_logs: Use JSON structured logging

    Returns:
        Validated configuration dictionary
    """
    # 1. Load raw config
    config = _load_config(config_path)

    # 2. Apply environment variable overrides
    config = _apply_env_overrides(config)

    # 3. Setup logging
    _setup_logging(
        level=log_level or config.get("logging", {}).get("level", "INFO"),
        log_file=log_file,
        json_format=json_logs,
    )

    logger.info(f"Application initialized with config from {config_path}")
    return config


def _load_config(config_path: str) -> dict:
    """Load config from YAML/JSON file with fallback to defaults."""
    if not os.path.exists(config_path):
        logger.debug(f"Config file {config_path} not found, using defaults")
        return {}

    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}


def _apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides."""
    try:
        from config.env import EnvConfig
        return EnvConfig.apply_env_overrides(config)
    except ImportError:
        return config


def _setup_logging(level: str, log_file: str = None, json_format: bool = False) -> None:
    """Setup structured logging."""
    try:
        from utils.logging import setup_logging
        setup_logging(level=level, log_file=log_file, json_format=json_format)
    except ImportError:
        # Fallback to basic logging
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        )
