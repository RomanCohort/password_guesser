"""
Environment variable configuration.

Maps well-known environment variables (prefixed with ``PG_``) to nested
configuration keys so that secrets and deployment-specific overrides can be
provided without modifying config files.
"""

import os
from typing import Dict, Any, Optional, Callable


class EnvConfig:
    """Environment variable configuration loader.

    Each entry in :attr:`ENV_MAP` maps an environment variable name to a
    tuple of ``(section, key[, type_converter])``.  When loaded, the values
    are parsed and merged into a nested configuration dictionary.
    """

    ENV_MAP: Dict[str, tuple] = {
        'PG_LLM_API_KEY': ('llm', 'api_key'),
        'PG_LLM_API_BASE': ('llm', 'api_base'),
        'PG_LLM_MODEL': ('llm', 'model'),
        'PG_TRAINING_DEVICE': ('training', 'device'),
        'PG_TRAINING_EPOCHS': ('training', 'epochs', int),
        'PG_TRAINING_BATCH_SIZE': ('training', 'batch_size', int),
        'PG_TRAINING_LR': ('training', 'learning_rate', float),
        'PG_WEB_HOST': ('web', 'host'),
        'PG_WEB_PORT': ('web', 'port', int),
        'PG_WEB_AUTH_ENABLED': (
            'web', 'auth_enabled', lambda x: x.lower() == 'true'
        ),
        'PG_WEB_API_KEYS': ('web', 'api_keys', lambda x: x.split(',')),
    }

    @staticmethod
    def load_env_config() -> Dict[str, Any]:
        """Build a nested configuration dictionary from environment variables.

        Only variables that are actually set in the current process
        environment are included.

        Returns:
            A nested dict such as ``{'llm': {'api_key': '...'}, ...}``.
        """
        config: Dict[str, Any] = {}

        for env_var, mapping in EnvConfig.ENV_MAP.items():
            value = os.environ.get(env_var)
            if value is None:
                continue

            # Unpack mapping
            section = mapping[0]
            key = mapping[1]
            converter = mapping[2] if len(mapping) > 2 else None

            if converter is not None:
                try:
                    value = converter(value)
                except (ValueError, TypeError):
                    continue

            if section not in config:
                config[section] = {}
            config[section][key] = value

        return config

    @staticmethod
    def apply_env_overrides(config: dict) -> dict:
        """Merge environment variable overrides into an existing config dict.

        Values from environment variables take precedence over those in the
        supplied *config* dictionary.  The original dict is **not** mutated.

        Args:
            config: Base configuration dictionary.

        Returns:
            A new dictionary with env overrides applied.
        """
        import copy
        result = copy.deepcopy(config)

        env_config = EnvConfig.load_env_config()
        for section, values in env_config.items():
            if section not in result:
                result[section] = {}
            result[section].update(values)

        return result


def load_env_config() -> dict:
    """Module-level convenience wrapper for :meth:`EnvConfig.load_env_config`."""
    return EnvConfig.load_env_config()
