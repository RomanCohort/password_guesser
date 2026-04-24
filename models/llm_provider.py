"""
LLM Provider Abstraction Layer

Provides a flexible interface for integrating different LLM backends:
- DeepSeek (default)
- OpenAI
- vLLM / Local models with OpenAI-compatible API
- Custom LLM APIs

Usage:
    # Default (DeepSeek)
    provider = get_provider(LLMConfig())

    # Custom provider
    config = LLMConfig(
        provider="custom",
        api_base="http://your-llm-server:8000/v1",
        openai_format=True,  # if API follows OpenAI format
    )
    provider = get_provider(config)

    # Use the provider
    response = provider.call(
        messages=[{"role": "user", "content": "Hello"}],
        use_json_mode=True,
    )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
import json
import time
import requests
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)

    def parse_json(self) -> dict:
        """Parse JSON from response content, handling markdown wrappers."""
        content = self.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())


@dataclass
class LLMConfig:
    """LLM configuration - supports multiple providers."""

    # Provider settings
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: str = ""
    api_base: str = "https://api.deepseek.com/v1"

    # API format settings
    openai_format: bool = True  # Set True for OpenAI-compatible APIs
    json_mode_supported: bool = True  # Set False if API doesn't support response_format

    # Generation settings
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    stop_sequences: List[str] = field(default_factory=list)

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_on_status: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])

    # Additional provider-specific settings (passed as-is)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set provider-specific defaults."""
        # DeepSeek defaults
        if self.provider == "deepseek":
            if not self.api_base:
                self.api_base = "https://api.deepseek.com/v1"
            if not self.model:
                self.model = "deepseek-chat"
            self.json_mode_supported = True

        # OpenAI defaults
        elif self.provider == "openai":
            if not self.api_base:
                self.api_base = "https://api.openai.com/v1"
            if not self.model:
                self.model = "gpt-4"
            self.json_mode_supported = True

        # vLLM / Local models
        elif self.provider in ("vllm", "local", "custom"):
            self.json_mode_supported = False  # Usually not supported
            if not self.api_base:
                self.api_base = "http://localhost:8000/v1"


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implement this class to support custom LLM backends.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def call(
        self,
        messages: List[Dict[str, str]],
        use_json_mode: bool = False,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Call the LLM with a list of messages.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            use_json_mode: Request JSON-formatted output
            temperature: Override temperature for this call
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with content and metadata
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM endpoint is available."""
        pass

    def supports_json_mode(self) -> bool:
        """Check if this provider supports JSON mode."""
        return self.config.json_mode_supported


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Provider for OpenAI-compatible APIs.

    Supports:
    - OpenAI
    - DeepSeek
    - vLLM
    - Any API following the OpenAI chat completions format
    """

    def call(
        self,
        messages: List[Dict[str, str]],
        use_json_mode: bool = False,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Call OpenAI-compatible chat completions API.

        Args:
            messages: Chat messages
            use_json_mode: Request JSON output (requires json_mode_supported)
            temperature: Override temperature
            **kwargs: Additional parameters (e.g., top_p, stop)

        Returns:
            LLMResponse
        """
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Add optional parameters
        if self.config.top_p != 1.0:
            payload["top_p"] = self.config.top_p
        if self.config.stop_sequences:
            payload["stop"] = self.config.stop_sequences

        # Add extra parameters
        payload.update(self.config.extra_params)
        payload.update(kwargs)

        # JSON mode (if supported)
        if use_json_mode and self.config.json_mode_supported:
            payload["response_format"] = {"type": "json_object"}

        # Retry loop
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    f"{self.config.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                response.raise_for_status()
                result = response.json()

                choice = result['choices'][0]
                return LLMResponse(
                    content=choice['message']['content'],
                    model=result.get('model', self.config.model),
                    usage=result.get('usage', {}),
                    raw_response=result,
                )

            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"API timeout, attempt {attempt + 1}/{self.config.max_retries}")

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    last_error = e
                    retry_after = float(response.headers.get('Retry-After', self.config.retry_delay * (2 ** attempt)))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue

                # JSON mode not supported - fallback
                if response.status_code == 400 and use_json_mode:
                    logger.info("JSON mode not supported, falling back to standard mode")
                    return self.call(messages, use_json_mode=False, temperature=temperature, **kwargs)

                # Check if we should retry
                if response.status_code in self.config.retry_on_status:
                    last_error = e
                    logger.warning(f"HTTP {response.status_code}, attempt {attempt + 1}/{self.config.max_retries}")
                else:
                    raise

            except json.JSONDecodeError as e:
                last_error = e
                logger.error(f"Failed to parse API response: {e}")

            except Exception as e:
                last_error = e
                logger.error(f"API call error: {e}")

            # Exponential backoff
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2 ** attempt)
                time.sleep(delay)

        raise RuntimeError(f"API call failed after {self.config.max_retries} retries: {last_error}")

    async def async_call(
        self,
        messages: List[Dict[str, str]],
        use_json_mode: bool = False,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Async version of call() for non-blocking LLM API requests.

        Uses httpx for async HTTP requests.
        """
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed, falling back to sync call")
            return self.call(messages, use_json_mode, temperature, **kwargs)

        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if self.config.top_p != 1.0:
            payload["top_p"] = self.config.top_p
        if self.config.stop_sequences:
            payload["stop"] = self.config.stop_sequences

        payload.update(self.config.extra_params)
        payload.update(kwargs)

        if use_json_mode and self.config.json_mode_supported:
            payload["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.post(
                        f"{self.config.api_base}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    result = response.json()

                    choice = result['choices'][0]
                    return LLMResponse(
                        content=choice['message']['content'],
                        model=result.get('model', self.config.model),
                        usage=result.get('usage', {}),
                        raw_response=result,
                    )

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"API timeout, attempt {attempt + 1}/{self.config.max_retries}")

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    last_error = e
                    retry_after = float(e.response.headers.get('Retry-After', self.config.retry_delay * (2 ** attempt)))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                if e.response.status_code == 400 and use_json_mode:
                    logger.info("JSON mode not supported, falling back to standard mode")
                    return await self.async_call(messages, use_json_mode=False, temperature=temperature, **kwargs)

                if e.response.status_code in self.config.retry_on_status:
                    last_error = e
                    logger.warning(f"HTTP {e.response.status_code}, attempt {attempt + 1}/{self.config.max_retries}")
                else:
                    raise

            except Exception as e:
                last_error = e
                logger.error(f"Async API call error: {e}")

            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)

        raise RuntimeError(f"Async API call failed after {self.config.max_retries} retries: {last_error}")

    def is_available(self) -> bool:
        """Check if the API endpoint is reachable."""
        try:
            response = requests.get(
                f"{self.config.api_base.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {},
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"LLM availability check failed: {e}")
            return False


class CustomProvider(BaseLLMProvider):
    """
    Provider for custom LLM APIs.

    This is a template for implementing custom providers.
    Override the `call` method to match your API format.

    Example usage:
        class MyCustomProvider(CustomProvider):
            def call(self, messages, use_json_mode=False, temperature=None, **kwargs):
                # Implement your custom API call
                response = requests.post(
                    self.config.api_base + "/generate",
                    json={"prompt": self._format_messages(messages), ...}
                )
                return LLMResponse(content=response.json()["text"], ...)
    """

    def call(
        self,
        messages: List[Dict[str, str]],
        use_json_mode: bool = False,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Override this method for your custom API.

        The default implementation attempts OpenAI-compatible format.
        """
        # Default: try OpenAI-compatible format
        provider = OpenAICompatibleProvider(self.config)
        return provider.call(messages, use_json_mode, temperature, **kwargs)

    async def async_call(
        self,
        messages: List[Dict[str, str]],
        use_json_mode: bool = False,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Async version for custom providers."""
        provider = OpenAICompatibleProvider(self.config)
        return await provider.async_call(messages, use_json_mode, temperature, **kwargs)

    def is_available(self) -> bool:
        """Check availability."""
        try:
            response = requests.get(self.config.api_base, timeout=5)
            return response.status_code < 500
        except Exception as e:
            logger.debug(f"Custom provider availability check failed: {e}")
            return False


# Provider registry
_PROVIDERS: Dict[str, type] = {
    "deepseek": OpenAICompatibleProvider,
    "openai": OpenAICompatibleProvider,
    "vllm": OpenAICompatibleProvider,
    "local": OpenAICompatibleProvider,
    "custom": CustomProvider,
}


def register_provider(name: str, provider_class: type):
    """Register a custom provider class."""
    _PROVIDERS[name.lower()] = provider_class


def get_provider(config: LLMConfig) -> BaseLLMProvider:
    """
    Get an LLM provider instance based on configuration.

    Args:
        config: LLM configuration

    Returns:
        Provider instance
    """
    provider_name = config.provider.lower()

    if provider_name in _PROVIDERS:
        return _PROVIDERS[provider_name](config)

    # Unknown provider - try OpenAI-compatible format
    logger.warning(f"Unknown provider '{config.provider}', attempting OpenAI-compatible format")
    return OpenAICompatibleProvider(config)


def create_provider(
    provider: str = "deepseek",
    model: str = "",
    api_key: str = "",
    api_base: str = "",
    **kwargs
) -> BaseLLMProvider:
    """
    Factory function to create a provider with simple parameters.

    Args:
        provider: Provider name (deepseek, openai, vllm, custom)
        model: Model name
        api_key: API key
        api_base: API base URL
        **kwargs: Additional configuration

    Returns:
        Provider instance
    """
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
        **kwargs
    )
    return get_provider(config)
