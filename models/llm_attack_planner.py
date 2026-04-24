"""
LLM-powered Attack Planner

Uses DeepSeek LLM for strategic attack planning, vulnerability analysis,
and reflective reasoning. Extends LLMInfoExtractor's API calling patterns.
"""

import json
import logging
from typing import Dict, List, Optional, Any

from models.llm_extractor import LLMInfoExtractor, LLMConfig

logger = logging.getLogger(__name__)


class LLMAttackPlanner:
    """
    Uses DeepSeek LLM for penetration test planning and analysis.

    Reuses LLMInfoExtractor's _call_api() for reliable API communication
    with retry, rate limiting, and JSON mode support.
    """

    SYSTEM_PROMPT = """你是一个授权渗透测试专家助手。你的任务是分析目标环境信息，规划最优攻击路径。

重要原则：
1. 只建议已授权的渗透测试操作
2. 优先选择低风险、高成功率的方法
3. 考虑攻击链的依赖关系
4. 提供详细的技术分析和理由
5. 遵循MITRE ATT&CK框架的分类标准

输出必须是有效的 JSON 格式。"""

    ANALYSIS_PROMPT = """分析以下目标环境，识别潜在攻击向量：

## 目标信息
{target_info}

## 已发现漏洞
{vulnerabilities}

## 已获取权限
{current_access}

## ATT&CK 技术
{techniques}

请分析：
1. 可利用的攻击向量（按优先级排序）
2. 推荐的攻击顺序和理由
3. 潜在风险和缓解措施
4. 所需的前置条件

以 JSON 格式输出：
{{
    "attack_vectors": [
        {{
            "type": "exploit|brute_force|misconfig",
            "target": "host:port",
            "cve_id": "CVE-XXXX-XXXXX",
            "technique_id": "TXXXX",
            "priority": 1,
            "probability": 0.8,
            "reasoning": "..."
        }}
    ],
    "recommended_sequence": ["step1", "step2", ...],
    "risk_assessment": {{
        "overall_risk": "high|medium|low",
        "detection_probability": 0.3,
        "potential_impact": "..."
    }},
    "alternative_approaches": [...]
}}"""

    REFLECTION_PROMPT = """反思上一次攻击尝试的结果：

## 执行的动作
{action_taken}

## 结果
{result}

## 错误信息（如有）
{error_info}

## 当前环境状态
{current_state}

请深入分析：
1. 失败的根本原因是什么？
2. 是否有环境因素被忽略？
3. 攻击方式是否最优？
4. 有哪些替代方案？
5. 下一步建议是什么？

以 JSON 格式输出反思结果：
{{
    "failure_analysis": "...",
    "root_cause": "...",
    "missed_factors": ["..."],
    "alternative_approaches": [
        {{
            "approach": "...",
            "reasoning": "...",
            "estimated_success_rate": 0.7
        }}
    ],
    "next_steps": ["..."],
    "lessons_learned": ["..."]
}}"""

    PATH_PLANNING_PROMPT = """基于攻击图信息，规划最优攻击路径：

## 攻击图概要
- 节点数: {node_count}
- 边数: {edge_count}
- 已发现漏洞: {vuln_count}

## 关键节点
{key_nodes}

## 目标
{target_goal}

请规划从初始访问到目标达成的攻击路径：

以 JSON 格式输出：
{{
    "phases": [
        {{
            "name": "reconnaissance",
            "actions": ["..."],
            "expected_outcome": "..."
        }}
    ],
    "critical_path": ["node1", "node2", ...],
    "risk_points": ["..."],
    "contingency_plans": ["..."]
}}"""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        config_path: Optional[str] = None,
        provider=None,
    ):
        """
        Initialize attack planner.

        Args:
            config: LLM configuration
            config_path: Path to YAML config file
            provider: Pre-configured LLM provider instance
        """
        self.extractor = LLMInfoExtractor(
            config=config,
            config_path=config_path,
            provider=provider,
        )
        self.config = config or self.extractor.config

    def analyze_target(self, target_info: dict) -> dict:
        """
        Analyze target environment for attack vectors.

        Args:
            target_info: Dict with keys: hosts, ports, services, vulnerabilities

        Returns:
            Analysis results with prioritized attack vectors
        """
        vulnerabilities = target_info.get("vulnerabilities", [])
        current_access = target_info.get("current_access", "none")
        techniques = target_info.get("techniques", [])

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.ANALYSIS_PROMPT.format(
                target_info=json.dumps(target_info.get("hosts", {}), ensure_ascii=False, indent=2),
                vulnerabilities=json.dumps(vulnerabilities, ensure_ascii=False, indent=2),
                current_access=current_access,
                techniques=json.dumps(techniques, ensure_ascii=False),
            )},
        ]

        try:
            response = self.extractor._call_api(messages, use_json_mode=True, temperature=0.3)
            return self.extractor._parse_json_response(response)
        except Exception as e:
            logger.error(f"Target analysis failed: {e}")
            return {
                "attack_vectors": [],
                "recommended_sequence": [],
                "risk_assessment": {"overall_risk": "unknown"},
                "alternative_approaches": [],
                "error": str(e),
            }

    def plan_attack_path(self, graph_data: dict, target: str) -> List[dict]:
        """
        Plan attack path based on attack graph.

        Args:
            graph_data: Attack graph data (from AttackGraph.to_json())
            target: Target node ID or goal description

        Returns:
            List of attack phases with actions
        """
        nodes = graph_data.get("nodes", {})
        vuln_nodes = [
            n for n in nodes.values()
            if n.get("type") == "vuln"
        ]
        key_nodes = [
            f"- {n['name']} ({n['type']})"
            for n in list(nodes.values())[:10]
        ]

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.PATH_PLANNING_PROMPT.format(
                node_count=len(nodes),
                edge_count=len(graph_data.get("edges", [])),
                vuln_count=len(vuln_nodes),
                key_nodes="\n".join(key_nodes),
                target_goal=target,
            )},
        ]

        try:
            response = self.extractor._call_api(messages, use_json_mode=True, temperature=0.4)
            result = self.extractor._parse_json_response(response)
            return result.get("phases", [])
        except Exception as e:
            logger.error(f"Path planning failed: {e}")
            return []

    def suggest_next_action(
        self,
        state,
        available_actions: List,
    ):
        """
        Suggest the next action based on current state.

        Uses LLM to evaluate the strategic situation and recommend
        the best action from available options.
        """
        from rl_agent.state import PenTestState
        from rl_agent.action import PenTestAction

        if not isinstance(state, PenTestState):
            return None

        if not available_actions:
            return None

        # Format actions for LLM
        action_descriptions = []
        for i, action in enumerate(available_actions[:20]):  # Limit to 20
            action_descriptions.append(
                f"{i}. {action.type.value} -> {action.target}"
                + (f" (CVE: {action.cve_id})" if action.cve_id else "")
                + (f" (Technique: {action.technique_id})" if action.technique_id else "")
            )

        prompt = f"""Based on the current penetration test state:

{state.summary()}

Available actions:
{chr(10).join(action_descriptions)}

Which action should be taken next? Consider:
1. Which action has the highest probability of success?
2. Which action provides the most information gain?
3. Which action advances the attack chain?

Respond with the action number in JSON:
{{"recommended_action": <number>, "reasoning": "..."}}"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.extractor._call_api(messages, use_json_mode=True, temperature=0.2)
            result = self.extractor._parse_json_response(response)
            action_idx = result.get("recommended_action", 0)
            if 0 <= action_idx < len(available_actions):
                return available_actions[action_idx]
        except Exception as e:
            logger.debug(f"Action suggestion failed: {e}")

        return None

    def analyze_failure(self, action, error: str, state=None) -> dict:
        """
        Analyze why an action failed and suggest improvements.

        Used by the reflective RL agent for self-reflection.
        """
        from rl_agent.action import PenTestAction

        action_desc = f"{action.type.value} -> {action.target}" if isinstance(action, PenTestAction) else str(action)
        state_desc = state.summary() if state else "Unknown"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.REFLECTION_PROMPT.format(
                action_taken=action_desc,
                result="failure",
                error_info=error,
                current_state=state_desc,
            )},
        ]

        try:
            response = self.extractor._call_api(messages, use_json_mode=True, temperature=0.3)
            return self.extractor._parse_json_response(response)
        except Exception as e:
            logger.error(f"Failure analysis failed: {e}")
            return {
                "failure_analysis": str(e),
                "root_cause": "unknown",
                "alternative_approaches": [],
                "next_steps": [],
            }

    def assess_vulnerability(
        self,
        cve_id: str,
        target_context: dict,
    ) -> dict:
        """
        Assess vulnerability exploitability for a specific target.

        Args:
            cve_id: CVE identifier
            target_context: Target host/service information

        Returns:
            Exploitability assessment
        """
        prompt = f"""Assess the exploitability of {cve_id} against this target:

Target context:
{json.dumps(target_context, ensure_ascii=False, indent=2)}

Provide:
1. Exploitability score (0-10)
2. Required conditions
3. Available exploits (public/MSF)
4. Detection risk
5. Recommended exploitation method

JSON format:
{{
    "exploitability_score": 8.5,
    "required_conditions": ["..."],
    "available_exploits": ["..."],
    "detection_risk": "medium",
    "recommended_method": "...",
    "prerequisites": ["..."]
}}"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.extractor._call_api(messages, use_json_mode=True, temperature=0.3)
            return self.extractor._parse_json_response(response)
        except Exception as e:
            logger.error(f"Vulnerability assessment failed: {e}")
            return {"exploitability_score": 0, "error": str(e)}
