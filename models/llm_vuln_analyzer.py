"""
LLM Vulnerability Analyzer

Uses DeepSeek LLM for detailed vulnerability analysis,
exploit assessment, and proof-of-concept generation.
"""

import json
import logging
from typing import Dict, List, Optional

from models.llm_extractor import LLMInfoExtractor, LLMConfig

logger = logging.getLogger(__name__)


class LLMVulnerabilityAnalyzer:
    """
    LLM-powered vulnerability analysis.

    Provides:
    - CVE analysis and context enrichment
    - Exploitability assessment
    - Attack scenario generation
    - Remediation recommendations
    """

    SYSTEM_PROMPT = """你是一个漏洞分析专家。你负责分析安全漏洞的技术细节、
可利用性和影响范围。所有分析仅用于授权安全测试目的。
输出必须是有效的 JSON 格式。"""

    def __init__(self, config: Optional[LLMConfig] = None, provider=None):
        self.extractor = LLMInfoExtractor(config=config, provider=provider)
        self.config = config or self.extractor.config

    def analyze_cve(self, cve_id: str, context: Optional[dict] = None) -> dict:
        """
        Analyze a CVE vulnerability with LLM enrichment.

        Args:
            cve_id: CVE identifier (e.g., CVE-2021-44228)
            context: Additional context about the target environment

        Returns:
            Detailed analysis including:
            - Technical description
            - Attack vectors
            - Impact assessment
            - Exploit difficulty
            - Remediation steps
        """
        context_str = json.dumps(context, ensure_ascii=False, indent=2) if context else "Not provided"

        prompt = f"""Provide a detailed security analysis of {cve_id}.

Target context:
{context_str}

Analyze:
1. Technical vulnerability details
2. Attack vectors and entry points
3. Impact (confidentiality, integrity, availability)
4. Exploitation difficulty (easy/medium/hard)
5. Known exploits and proof-of-concepts
6. Required conditions for exploitation
7. Recommended remediation

JSON format:
{{
    "cve_id": "{cve_id}",
    "technical_details": "...",
    "attack_vectors": ["..."],
    "impact": {{
        "confidentiality": "high|medium|low",
        "integrity": "high|medium|low",
        "availability": "high|medium|low"
    }},
    "exploitation_difficulty": "easy|medium|hard",
    "known_exploits": ["..."],
    "required_conditions": ["..."],
    "remediation": ["..."],
    "related_cves": ["..."]
}}"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.extractor._call_api(messages, use_json_mode=True, temperature=0.3)
            return self.extractor._parse_json_response(response)
        except Exception as e:
            logger.error(f"CVE analysis failed for {cve_id}: {e}")
            return {"cve_id": cve_id, "error": str(e)}

    def assess_exploitability(
        self,
        vuln_data: dict,
        target: dict,
    ) -> float:
        """
        Assess how exploitable a vulnerability is against a specific target.

        Args:
            vuln_data: Vulnerability information (CVE details)
            target: Target host/service configuration

        Returns:
            Exploitability score (0.0 - 1.0)
        """
        prompt = f"""Assess the exploitability of this vulnerability against the target:

Vulnerability:
{json.dumps(vuln_data, ensure_ascii=False, indent=2)}

Target:
{json.dumps(target, ensure_ascii=False, indent=2)}

Rate the exploitability from 0.0 to 1.0 based on:
- Is the vulnerable service running?
- Is the version affected?
- Are there network-level protections?
- Are there required conditions met?

JSON format:
{{"score": 0.85, "reasoning": "...", "blocking_factors": ["..."]}}"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.extractor._call_api(messages, use_json_mode=True, temperature=0.2)
            result = self.extractor._parse_json_response(response)
            return float(result.get("score", 0.0))
        except Exception as e:
            logger.error(f"Exploitability assessment failed: {e}")
            return 0.0

    def generate_attack_scenarios(
        self,
        vulnerabilities: List[dict],
        target: dict,
    ) -> List[dict]:
        """
        Generate possible attack scenarios combining multiple vulnerabilities.

        Args:
            vulnerabilities: List of known vulnerabilities
            target: Target environment description

        Returns:
            List of attack scenarios with steps
        """
        prompt = f"""Generate attack scenarios for this target:

Vulnerabilities:
{json.dumps(vulnerabilities, ensure_ascii=False, indent=2)}

Target:
{json.dumps(target, ensure_ascii=False, indent=2)}

Generate realistic attack scenarios (for authorized testing only):

JSON format:
[
    {{
        "name": "Scenario name",
        "difficulty": "easy|medium|hard",
        "steps": [
            {{"order": 1, "action": "...", "cve_id": "...", "technique": "TXXXX"}}
        ],
        "expected_outcome": "...",
        "detection_risk": "high|medium|low"
    }}
]"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.extractor._call_api(messages, use_json_mode=True, temperature=0.5)
            result = self.extractor._parse_json_response(response)
            return result if isinstance(result, list) else [result]
        except Exception as e:
            logger.error(f"Attack scenario generation failed: {e}")
            return []

    def generate_remediation(
        self,
        vulnerabilities: List[dict],
        priority: str = "critical",
    ) -> List[dict]:
        """
        Generate prioritized remediation recommendations.

        Args:
            vulnerabilities: List of vulnerabilities to remediate
            priority: Priority filter (critical, high, medium, low)

        Returns:
            List of remediation actions
        """
        prompt = f"""Generate remediation recommendations for these vulnerabilities:

{json.dumps(vulnerabilities, ensure_ascii=False, indent=2)}

Priority level: {priority}

JSON format:
[
    {{
        "cve_id": "...",
        "priority": 1,
        "action": "...",
        "effort": "low|medium|high",
        "timeline": "immediate|short-term|long-term",
        "verification_steps": ["..."]
    }}
]"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.extractor._call_api(messages, use_json_mode=True, temperature=0.3)
            result = self.extractor._parse_json_response(response)
            return result if isinstance(result, list) else [result]
        except Exception as e:
            logger.error(f"Remediation generation failed: {e}")
            return []
