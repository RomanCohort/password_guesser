"""
Base Expert Class

Abstract base class for all penetration testing experts.
Each expert specializes in a specific domain and can analyze situations,
retrieve relevant knowledge via RAG, and provide actionable advice.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from models.enums import ExpertType

logger = logging.getLogger(__name__)


@dataclass
class ExpertAdvice:
    """Advice from an expert."""
    expert_type: ExpertType
    summary: str
    recommended_actions: List[dict]
    tools_to_use: List[str]
    confidence: float
    reasoning: str
    warnings: List[str] = field(default_factory=list)
    alternatives: List[dict] = field(default_factory=list)
    relevant_cves: List[str] = field(default_factory=list)
    relevant_techniques: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "expert_type": self.expert_type.value,
            "summary": self.summary,
            "recommended_actions": self.recommended_actions,
            "tools_to_use": self.tools_to_use,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "warnings": self.warnings,
            "alternatives": self.alternatives,
            "relevant_cves": self.relevant_cves,
            "relevant_techniques": self.relevant_techniques,
            "timestamp": self.timestamp,
        }


@dataclass
class ExpertCapability:
    """Description of an expert's capabilities."""
    expert_type: ExpertType
    description: str
    tools: List[str]
    techniques: List[str]  # MITRE ATT&CK technique IDs
    required_inputs: List[str]
    outputs: List[str]


class PenTestExpert(ABC):
    """
    Base class for penetration testing experts.

    Each expert:
    - Specializes in a specific penetration testing domain
    - Uses RAG to retrieve relevant knowledge
    - Can analyze situations and provide expert advice
    - Tracks its own success/failure history
    """

    def __init__(
        self,
        expert_type: ExpertType,
        llm_provider=None,
        rag_retriever=None,
        tools: List[str] = None,
    ):
        self.expert_type = expert_type
        self.llm = llm_provider
        self.rag = rag_retriever
        self.tools = tools or []
        self.success_history: List[dict] = []
        self.call_count: int = 0
        self.success_count: int = 0

    @abstractmethod
    def analyze(self, state: dict, context: dict = None) -> ExpertAdvice:
        """
        Analyze the situation and provide expert advice.

        Args:
            state: Current penetration test state
            context: Additional context (previous actions, results, etc.)

        Returns:
            ExpertAdvice with recommendations
        """
        pass

    @abstractmethod
    def get_prompt_template(self) -> str:
        """Get the system prompt template for this expert."""
        pass

    def retrieve_relevant_knowledge(self, query: str) -> str:
        """
        Retrieve relevant knowledge from RAG.

        Args:
            query: Search query

        Returns:
            Assembled context string
        """
        if not self.rag:
            return ""

        result = self.rag.retrieve_for_query(query, top_k=5, strategy="hybrid")
        return result.context

    def suggest_tools(self, scenario: str) -> List[str]:
        """
        Suggest appropriate tools for the scenario.

        Args:
            scenario: Scenario description

        Returns:
            List of suggested tool names
        """
        if not self.rag:
            return self.tools

        tool_doc = self.rag.retrieve_tool_usage("", scenario)
        if tool_doc:
            # Find which of our tools are mentioned
            mentioned = [t for t in self.tools if t.lower() in tool_doc.lower()]
            if mentioned:
                return mentioned

        return self.tools

    def record_outcome(self, advice: ExpertAdvice, was_successful: bool) -> None:
        """
        Record the outcome of following this expert's advice.

        Args:
            advice: The advice that was given
            was_successful: Whether the action succeeded
        """
        self.call_count += 1
        if was_successful:
            self.success_count += 1

        self.success_history.append({
            "advice": advice.to_dict(),
            "successful": was_successful,
            "timestamp": time.time(),
        })

        # Keep history bounded
        if len(self.success_history) > 100:
            self.success_history = self.success_history[-100:]

    def get_success_rate(self) -> float:
        """Get the success rate of this expert's advice."""
        if self.call_count == 0:
            return 0.5  # Default
        return self.success_count / self.call_count

    def get_capabilities(self) -> ExpertCapability:
        """Get capabilities description for this expert."""
        return ExpertCapability(
            expert_type=self.expert_type,
            description=self.get_prompt_template()[:200],
            tools=self.tools,
            techniques=self._get_techniques(),
            required_inputs=self._get_required_inputs(),
            outputs=self._get_outputs(),
        )

    def _get_techniques(self) -> List[str]:
        """Get MITRE ATT&CK techniques for this expert. Override in subclasses."""
        return []

    def _get_required_inputs(self) -> List[str]:
        """Get required inputs for this expert. Override in subclasses."""
        return []

    def _get_outputs(self) -> List[str]:
        """Get outputs from this expert. Override in subclasses."""
        return []

    def _build_analysis_prompt(self, state: dict, context: dict = None) -> str:
        """Build a full analysis prompt combining state and context."""
        prompt = self.get_prompt_template()
        prompt += f"\n\n当前渗透测试状态:\n"

        if state:
            prompt += f"- 目标: {state.get('target', '未知')}\n"
            prompt += f"- 阶段: {state.get('phase', '未知')}\n"

            services = state.get("services", [])
            if services:
                prompt += f"- 已发现服务: {', '.join(services)}\n"

            vulns = state.get("vulnerabilities", [])
            if vulns:
                prompt += f"- 已发现漏洞: {', '.join(str(v) for v in vulns)}\n"

            creds = state.get("credentials", [])
            if creds:
                prompt += f"- 已获取凭据: {len(creds)} 个\n"

            hosts = state.get("hosts", [])
            if hosts:
                prompt += f"- 已发现主机: {len(hosts)} 台\n"

        if context:
            prompt += f"\n附加上下文:\n"
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"

        return prompt
