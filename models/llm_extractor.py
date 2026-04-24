"""
LLM-based information extraction using DeepSeek
with structured JSON output and multi-stage extraction.

Features:
- DeepSeek response_format=json for reliable structured output
- Multi-stage extraction: Extract -> Validate -> Refine
- Retry with exponential backoff
- Flexible LLM provider support (DeepSeek, OpenAI, vLLM, custom)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import logging

from utils.feature_utils import TargetFeatures

# Import provider abstraction
from models.llm_provider import (
    LLMConfig as ProviderLLMConfig,
    LLMResponse,
    BaseLLMProvider,
    OpenAICompatibleProvider,
    get_provider,
    register_provider,
)

logger = logging.getLogger(__name__)


# Re-export LLMConfig for backward compatibility
@dataclass
class LLMConfig:
    """LLM configuration - backward compatible wrapper."""
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: str = ""
    api_base: str = "https://api.deepseek.com/v1"
    openai_format: bool = True
    temperature: float = 0.7
    max_tokens: int = 2000
    max_retries: int = 3
    retry_delay: float = 1.0

    def to_provider_config(self) -> ProviderLLMConfig:
        """Convert to provider config."""
        return ProviderLLMConfig(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            openai_format=self.openai_format,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )


def _normalize_config(config) -> LLMConfig:
    """
    Normalize config from either dataclass or Pydantic model.

    Handles both:
    - LLMConfig dataclass from models/llm_extractor.py
    - LLMConfig Pydantic model from config/validation.py
    """
    # Fields accepted by our dataclass LLMConfig
    accepted_fields = {
        'provider', 'model', 'api_key', 'api_base', 'openai_format',
        'temperature', 'max_tokens', 'max_retries', 'retry_delay'
    }

    if hasattr(config, 'model_dump'):
        # Pydantic v2 model
        data = config.model_dump()
    elif hasattr(config, 'dict'):
        # Pydantic v1 model
        data = config.dict()
    elif isinstance(config, dict):
        data = config
    elif isinstance(config, LLMConfig):
        return config
    else:
        # Try to extract common attributes
        return LLMConfig(
            provider=getattr(config, 'provider', 'deepseek'),
            model=getattr(config, 'model', 'deepseek-chat'),
            api_key=getattr(config, 'api_key', ''),
            api_base=getattr(config, 'api_base', 'https://api.deepseek.com/v1'),
            openai_format=getattr(config, 'openai_format', True),
            temperature=getattr(config, 'temperature', 0.7),
            max_tokens=getattr(config, 'max_tokens', 2000),
            max_retries=getattr(config, 'max_retries', 3),
            retry_delay=getattr(config, 'retry_delay', 1.0),
        )

    # Filter to only accepted fields
    filtered = {k: v for k, v in data.items() if k in accepted_fields}
    return LLMConfig(**filtered)


def _to_provider_config(config) -> ProviderLLMConfig:
    """Convert any config type to provider config."""
    normalized = _normalize_config(config)
    return normalized.to_provider_config()


# JSON Schema for structured extraction
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "full_name": {"type": "string"},
        "first_name": {"type": "string"},
        "last_name": {"type": "string"},
        "nickname": {"type": "string"},
        "birthday": {"type": "string", "description": "YYYYMMDD format"},
        "anniversary": {"type": "string", "description": "YYYYMMDD format"},
        "phone": {"type": "string"},
        "email_prefix": {"type": "string"},
        "hobbies": {"type": "array", "items": {"type": "string"}},
        "favorite_words": {"type": "array", "items": {"type": "string"}},
        "favorite_numbers": {"type": "array", "items": {}},
        "sports_teams": {"type": "array", "items": {"type": "string"}},
        "pet_names": {"type": "array", "items": {"type": "string"}},
        "city": {"type": "string"},
        "country": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["full_name", "first_name", "last_name", "birthday", "phone"]
}


class LLMInfoExtractor:
    """Extract structured information from raw text using LLM"""

    SYSTEM_PROMPT = """你是一个信息安全研究助手。你的任务是从给定的目标个人信息中提取可能与密码相关的结构化信息。

请仔细分析文本，提取以下类别的信息：
1. 姓名信息：全名、名、姓、昵称
2. 日期信息：生日、纪念日等（格式化为YYYYMMDD）
3. 联系方式：电话号码、邮箱前缀
4. 兴趣爱好：运动、游戏、音乐等
5. 喜爱的词汇/数字：特殊意义的词或数字
6. 宠物名：宠物或亲近的人的名字
7. 地点：城市、国家
8. 其他关键词：可能用于密码的其他词汇

请以JSON格式输出，格式如下：
{
    "full_name": "",
    "first_name": "",
    "last_name": "",
    "nickname": "",
    "birthday": "",
    "anniversary": "",
    "phone": "",
    "email_prefix": "",
    "hobbies": [],
    "favorite_words": [],
    "favorite_numbers": [],
    "sports_teams": [],
    "pet_names": [],
    "city": "",
    "country": "",
    "keywords": []
}

注意：
- 日期统一转换为YYYYMMDD格式（如：19900315）
- 如果信息不存在，保持空字符串或空数组
- 只输出JSON，不要有其他文字"""

    EXTRACTION_PROMPT = """请从以下个人信息中提取可能与密码相关的结构化信息：

---
{target_info}
---

请以JSON格式输出提取的信息。"""

    VALIDATION_PROMPT = """请检查以下提取结果是否完整和准确。

原始信息：
---
{target_info}
---

提取结果：
```json
{extracted_json}
```

请检查：
1. 是否有遗漏的重要信息（姓名、日期、联系方式等）？
2. 日期格式是否正确（应为YYYYMMDD）？
3. 电话号码是否完整？
4. 是否有隐藏的关联信息被忽略？

如果有问题，请输出修正后的完整JSON（格式相同）。
如果提取结果已经完整准确，请输出原始JSON即可。
只输出JSON，不要有其他文字。"""

    REFINE_PROMPT = """请基于已提取的信息，进一步挖掘可能被忽略的密码相关线索。

原始信息：
---
{target_info}
---

已提取信息：
{extracted_json}

请关注以下容易被忽略的信息：
1. 用户ID、网名、游戏ID等在线身份
2. 账号名习惯（如邮箱前缀、社交账号名）
3. 家庭成员信息（配偶、父母、子女的名字或生日）
4. 学校/公司名称
5. 常用数字组合（门牌号、车牌号等）
6. 文化/宗教相关的特殊词汇

请将新发现的字段合并到原JSON中输出。如果字段在原有结构中不存在，放入keywords数组。
只输出合并后的完整JSON。"""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        config_path: Optional[str] = None,
        provider: Optional[BaseLLMProvider] = None,
    ):
        """
        Initialize LLM extractor.

        Args:
            config: LLM configuration (dataclass or Pydantic model)
            config_path: Path to YAML config file
            provider: Pre-configured provider instance (for custom LLMs)
        """
        if provider is not None:
            # Use provided provider directly
            self.provider = provider
            self.config = LLMConfig(
                provider=provider.config.provider,
                model=provider.config.model,
                api_key=provider.config.api_key,
                api_base=provider.config.api_base,
            )
        elif config_path:
            self._load_config(config_path)
        elif config:
            self.config = _normalize_config(config)
        else:
            self.config = LLMConfig()

        # Initialize provider if not provided
        if not hasattr(self, 'provider'):
            provider_config = _to_provider_config(self.config)
            self.provider = get_provider(provider_config)

    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        llm_cfg = cfg.get('llm', {})
        self.config = LLMConfig(
            provider=llm_cfg.get('provider', 'deepseek'),
            model=llm_cfg.get('model', 'deepseek-chat'),
            api_key=llm_cfg.get('api_key', ''),
            api_base=llm_cfg.get('api_base', 'https://api.deepseek.com/v1'),
            openai_format=llm_cfg.get('openai_format', True),
            temperature=llm_cfg.get('temperature', 0.7),
            max_tokens=llm_cfg.get('max_tokens', 2000),
        )
        self.provider = get_provider(_to_provider_config(self.config))

    def _call_api(
        self,
        messages: list,
        use_json_mode: bool = True,
        temperature: Optional[float] = None
    ) -> str:
        """
        Call the LLM API with retry, exponential backoff, and optional JSON mode.

        Uses the provider abstraction layer for flexible LLM backend support.

        Args:
            messages: Chat messages
            use_json_mode: Use JSON response format
            temperature: Override temperature for this call
        """
        try:
            response = self.provider.call(
                messages=messages,
                use_json_mode=use_json_mode,
                temperature=temperature,
            )
            return response.content
        except Exception as e:
            logger.error(f"Provider call failed: {e}")
            raise

    @staticmethod
    def _parse_json_response(response: str) -> dict:
        """Parse JSON from LLM response, handling markdown wrappers."""
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        return json.loads(response.strip())

    @staticmethod
    def _build_features(data: dict) -> TargetFeatures:
        """Build TargetFeatures from parsed JSON data."""
        birthday = data.get('birthday', '')
        return TargetFeatures(
            full_name=data.get('full_name', ''),
            first_name=data.get('first_name', ''),
            last_name=data.get('last_name', ''),
            nickname=data.get('nickname', ''),
            birthday=birthday,
            birth_year=birthday[:4] if len(birthday) >= 4 else '',
            birth_month=birthday[4:6] if len(birthday) >= 6 else '',
            birth_day=birthday[6:8] if len(birthday) >= 8 else '',
            anniversary=data.get('anniversary', ''),
            phone=data.get('phone', ''),
            email_prefix=data.get('email_prefix', ''),
            hobbies=data.get('hobbies', []),
            favorite_words=data.get('favorite_words', []),
            favorite_numbers=[str(n) for n in data.get('favorite_numbers', [])],
            sports_teams=data.get('sports_teams', []),
            pet_names=data.get('pet_names', []),
            city=data.get('city', ''),
            country=data.get('country', ''),
            keywords=data.get('keywords', []),
        )

    def extract(self, target_info: str) -> TargetFeatures:
        """
        Extract structured features from target information.

        Single-shot extraction for basic use.
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.EXTRACTION_PROMPT.format(target_info=target_info)}
        ]

        try:
            response = self._call_api(messages, use_json_mode=True)
            data = self._parse_json_response(response)
            return self._build_features(data)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return TargetFeatures()
        except Exception as e:
            print(f"Extraction error: {e}")
            return TargetFeatures()

    def extract_multistage(
        self,
        target_info: str,
        stages: int = 3,
        verbose: bool = True
    ) -> TargetFeatures:
        """
        Multi-stage extraction: Extract -> Validate -> Refine.

        Stage 1 (Extract): Initial structured extraction
        Stage 2 (Validate): Check for missing/incorrect fields
        Stage 3 (Refine): Deep-dive for overlooked information

        Args:
            target_info: Raw text with target's personal information
            stages: Number of stages to run (1-3)
            verbose: Print progress

        Returns:
            TargetFeatures with progressively refined information
        """
        # Stage 1: Initial Extraction
        if verbose:
            print("[Stage 1/3] Initial extraction...")
        features = self._stage_extract(target_info)

        if stages < 2:
            return features

        # Stage 2: Validation
        if verbose:
            print("[Stage 2/3] Validating extraction completeness...")
        features = self._stage_validate(target_info, features)

        if stages < 3:
            return features

        # Stage 3: Refinement
        if verbose:
            print("[Stage 3/3] Deep refinement for hidden clues...")
        features = self._stage_refine(target_info, features)

        return features

    def _stage_extract(self, target_info: str) -> TargetFeatures:
        """Stage 1: Initial extraction"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.EXTRACTION_PROMPT.format(target_info=target_info)}
        ]

        try:
            response = self._call_api(messages, use_json_mode=True, temperature=0.3)
            data = self._parse_json_response(response)
            return self._build_features(data)
        except Exception as e:
            print(f"  Stage 1 error: {e}")
            return TargetFeatures()

    def _stage_validate(self, target_info: str, features: TargetFeatures) -> TargetFeatures:
        """Stage 2: Validate and correct extraction results"""
        extracted_json = json.dumps(asdict(features), ensure_ascii=False, indent=2)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.VALIDATION_PROMPT.format(
                target_info=target_info,
                extracted_json=extracted_json
            )}
        ]

        try:
            response = self._call_api(messages, use_json_mode=True, temperature=0.2)
            data = self._parse_json_response(response)
            validated = self._build_features(data)

            # Merge: keep non-empty values from validated result, fill gaps from original
            merged = self._merge_features(features, validated)
            return merged
        except Exception as e:
            print(f"  Stage 2 error: {e}, keeping stage 1 results")
            return features

    def _stage_refine(self, target_info: str, features: TargetFeatures) -> TargetFeatures:
        """Stage 3: Deep refinement for overlooked information"""
        extracted_json = json.dumps(asdict(features), ensure_ascii=False, indent=2)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.REFINE_PROMPT.format(
                target_info=target_info,
                extracted_json=extracted_json
            )}
        ]

        try:
            response = self._call_api(messages, use_json_mode=True, temperature=0.5)
            data = self._parse_json_response(response)
            refined = self._build_features(data)

            # Merge refined results
            merged = self._merge_features(features, refined)
            return merged
        except Exception as e:
            print(f"  Stage 3 error: {e}, keeping stage 2 results")
            return features

    @staticmethod
    def _merge_features(base: TargetFeatures, updated: TargetFeatures) -> TargetFeatures:
        """
        Merge two TargetFeatures, preferring non-empty values.

        For list fields, union the two sets.
        """
        def merge_str(a: str, b: str) -> str:
            return b if b and not a else a

        def merge_list(a: List[str], b: List[str]) -> List[str]:
            return list(set(a + b))

        return TargetFeatures(
            full_name=merge_str(base.full_name, updated.full_name),
            first_name=merge_str(base.first_name, updated.first_name),
            last_name=merge_str(base.last_name, updated.last_name),
            nickname=merge_str(base.nickname, updated.nickname),
            birthday=merge_str(base.birthday, updated.birthday),
            birth_year=merge_str(base.birth_year, updated.birth_year),
            birth_month=merge_str(base.birth_month, updated.birth_month),
            birth_day=merge_str(base.birth_day, updated.birth_day),
            anniversary=merge_str(base.anniversary, updated.anniversary),
            phone=merge_str(base.phone, updated.phone),
            email_prefix=merge_str(base.email_prefix, updated.email_prefix),
            hobbies=merge_list(base.hobbies, updated.hobbies),
            favorite_words=merge_list(base.favorite_words, updated.favorite_words),
            favorite_numbers=merge_list(base.favorite_numbers, updated.favorite_numbers),
            sports_teams=merge_list(base.sports_teams, updated.sports_teams),
            pet_names=merge_list(base.pet_names, updated.pet_names),
            city=merge_str(base.city, updated.city),
            country=merge_str(base.country, updated.country),
            known_patterns=merge_list(base.known_patterns, updated.known_patterns),
            keywords=merge_list(base.keywords, updated.keywords),
        )

    def generate_password_hints(self, features: TargetFeatures) -> Dict[str, Any]:
        """Use LLM to generate additional password hints and patterns."""
        hint_prompt = f"""基于以下个人信息，分析可能的密码模式和组合方式：

姓名: {features.full_name or features.first_name or features.last_name}
生日: {features.birthday}
电话: {features.phone}
兴趣爱好: {', '.join(features.hobbies)}
喜爱的数字: {', '.join(features.favorite_numbers)}
关键词: {', '.join(features.keywords)}

请分析：
1. 可能的密码结构模式（如：姓名+数字、首字母+生日等）
2. 常见的组合方式
3. 需要优先尝试的组合

以JSON格式输出：
{{
    "patterns": ["pattern1", "pattern2", ...],
    "combinations": ["combination1", "combination2", ...],
    "priority_order": ["item1", "item2", ...]
}}"""

        messages = [
            {"role": "system", "content": "你是一个密码安全分析专家。请提供专业的安全分析。"},
            {"role": "user", "content": hint_prompt}
        ]

        try:
            response = self._call_api(messages, use_json_mode=True)
            return self._parse_json_response(response)
        except Exception as e:
            print(f"Hint generation error: {e}")
            return {"patterns": [], "combinations": [], "priority_order": []}

    def _call_api_async(
        self,
        messages: list,
        use_json_mode: bool = True,
        temperature: Optional[float] = None
    ) -> str:
        """Thread-compatible wrapper for _call_api (no changes needed since requests is blocking)."""
        return self._call_api(messages, use_json_mode=use_json_mode, temperature=temperature)

    def extract_multistage_parallel(
        self,
        target_info: str,
        max_workers: int = 3,
        verbose: bool = True
    ) -> TargetFeatures:
        """
        Parallel multi-stage extraction: runs all 3 stages concurrently,
        then merges results for maximum coverage.

        Stage 1 (Extract): Initial structured extraction
        Stage 2 (Validate): Independent validation pass
        Stage 3 (Refine): Independent deep-dive pass

        All three stages run in parallel via ThreadPoolExecutor,
        then results are merged into a single TargetFeatures.

        Args:
            target_info: Raw text with target's personal information
            max_workers: Number of parallel API calls
            verbose: Print progress

        Returns:
            Merged TargetFeatures from all stages
        """
        if verbose:
            print("[Parallel] Launching 3 extraction stages concurrently...")

        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Stage 1: Extract
            msg1 = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.EXTRACTION_PROMPT.format(target_info=target_info)}
            ]
            futures['extract'] = executor.submit(
                self._call_api_async, msg1, True, 0.3
            )

            # Stage 2: Validate (independent extraction with lower temperature)
            msg2 = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.VALIDATION_PROMPT.format(
                    target_info=target_info,
                    extracted_json="{}"
                )}
            ]
            futures['validate'] = executor.submit(
                self._call_api_async, msg2, True, 0.2
            )

            # Stage 3: Refine
            msg3 = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.REFINE_PROMPT.format(
                    target_info=target_info,
                    extracted_json="{}"
                )}
            ]
            futures['refine'] = executor.submit(
                self._call_api_async, msg3, True, 0.5
            )

            # Collect results
            results = {}
            for name, future in futures.items():
                try:
                    response = future.result(timeout=90)
                    data = self._parse_json_response(response)
                    results[name] = self._build_features(data)
                    if verbose:
                        print(f"  [{name}] completed successfully")
                except Exception as e:
                    if verbose:
                        print(f"  [{name}] failed: {e}")

        # Merge all results: extract as base, then merge validate and refine
        merged = results.get('extract', TargetFeatures())
        if 'validate' in results:
            merged = self._merge_features(merged, results['validate'])
        if 'refine' in results:
            merged = self._merge_features(merged, results['refine'])

        if verbose:
            print("[Parallel] All stages merged.")

        return merged


def create_extractor_from_config(config_path: str = "config.yaml") -> LLMInfoExtractor:
    """Factory function to create LLM extractor from config file"""
    return LLMInfoExtractor(config_path=config_path)
