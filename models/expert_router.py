"""
Expert Router

Routes queries and situations to the most appropriate penetration testing expert(s).
Supports multiple routing strategies: rule-based, LLM-based, and performance-based.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from models.enums import ExpertType
from models.experts.base import PenTestExpert, ExpertAdvice, ExpertCapability

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Decision from expert router."""
    primary_expert: ExpertType
    supporting_experts: List[ExpertType]
    confidence: float
    reasoning: str
    keywords_matched: List[str] = field(default_factory=list)


# Keyword mappings for rule-based routing
ROUTING_KEYWORDS: Dict[ExpertType, List[str]] = {
    ExpertType.RECONNAISSANCE: [
        "扫描", "scan", "侦察", "recon", "发现", "discover", "端口", "port",
        "nmap", "masscan", "服务", "service", "osint", "信息收集",
        "子域名", "subdomain", "dns", "枚举", "enumerate",
    ],
    ExpertType.VULNERABILITY: [
        "漏洞", "vulnerability", "vuln", "cve", "cvss", "漏洞扫描",
        "nikto", "nuclei", "风险评估", "risk", "安全评估",
        "补丁", "patch", "修复", "remediation",
    ],
    ExpertType.EXPLOITATION: [
        "利用", "exploit", "攻击", "attack", "payload", "shellcode",
        "metasploit", "msf", "meterpreter", "poc", "漏洞利用",
        "绕过", "bypass", "编码", "encode", "shell", "反弹",
    ],
    ExpertType.POST_EXPLOITATION: [
        "提权", "privilege", "escalation", "后渗透", "post-exploit",
        "mimikatz", "持久化", "persistence", "信息收集", "seatbelt",
        "winpeas", "linpeas", "数据窃取", "exfiltration",
    ],
    ExpertType.CREDENTIAL: [
        "密码", "password", "凭据", "credential", "hash", "哈希",
        "破解", "crack", "brute", "暴力", "hydra", "hashcat", "john",
        "kerberos", "kerberoast", "认证", "authentication", "ntlm",
        "字典", "wordlist", "spray", "喷洒",
    ],
    ExpertType.LATERAL_MOVEMENT: [
        "横向", "lateral", "移动", "movement", "跳板", "pivot",
        "psexec", "wmi", "winrm", "ssh", "远程", "remote",
        "传递攻击", "pass-the-hash", "pth", "代理", "proxy",
        "隧道", "tunnel",
    ],
}

# Phase-to-expert mapping
PHASE_EXPERT_MAP: Dict[str, ExpertType] = {
    "reconnaissance": ExpertType.RECONNAISSANCE,
    "scanning": ExpertType.RECONNAISSANCE,
    "vulnerability_assessment": ExpertType.VULNERABILITY,
    "exploitation": ExpertType.EXPLOITATION,
    "post_exploitation": ExpertType.POST_EXPLOITATION,
    "credential_attacks": ExpertType.CREDENTIAL,
    "lateral_movement": ExpertType.LATERAL_MOVEMENT,
    "privilege_escalation": ExpertType.POST_EXPLOITATION,
    "persistence": ExpertType.POST_EXPLOITATION,
}


class ExpertRouter:
    """
    Routes queries to the most appropriate expert(s).

    Routing strategies:
    1. Rule-based: Match keywords to expert domains
    2. LLM-based: Use LLM to classify and route
    3. Performance-based: Route to experts with best track record
    """

    def __init__(
        self,
        llm_provider=None,
        rag_retriever=None,
    ):
        self.llm = llm_provider
        self.rag = rag_retriever
        self.experts: Dict[ExpertType, PenTestExpert] = {}
        self.routing_history: List[dict] = []

    def register_expert(self, expert: PenTestExpert) -> None:
        """Register an expert with the router."""
        self.experts[expert.expert_type] = expert
        logger.info(f"Registered expert: {expert.expert_type.value}")

    def unregister_expert(self, expert_type: ExpertType) -> None:
        """Remove an expert from the router."""
        self.experts.pop(expert_type, None)

    def get_registered_experts(self) -> List[ExpertType]:
        """Get list of registered expert types."""
        return list(self.experts.keys())

    def analyze_situation(self, state: dict, query: str = None) -> RoutingDecision:
        """
        Analyze current situation and determine which expert(s) to consult.

        Args:
            state: Current penetration test state
            query: Optional query text

        Returns:
            RoutingDecision with primary and supporting experts
        """
        # Try rule-based routing first
        decision = self._rule_based_routing(state, query)

        # If LLM available and confidence low, try LLM routing
        if self.llm and decision.confidence < 0.7:
            llm_decision = self._llm_based_routing(state, query)
            if llm_decision and llm_decision.confidence > decision.confidence:
                decision = llm_decision

        # Check performance history for adjustment
        decision = self._performance_adjustment(decision)

        # Record routing decision
        self.routing_history.append({
            "decision": decision,
            "state_phase": state.get("phase", ""),
            "timestamp": time.time(),
        })

        return decision

    def route_query(
        self,
        query: str,
        state: dict,
        context: dict = None,
    ) -> Dict[str, any]:
        """
        Route a query to appropriate experts and aggregate responses.

        Args:
            query: The query to route
            state: Current state
            context: Additional context

        Returns:
            Dictionary with routing decision and expert advice
        """
        # Determine routing
        decision = self.analyze_situation(state, query)

        result = {
            "routing_decision": decision,
            "primary_advice": None,
            "supporting_advice": [],
        }

        # Get advice from primary expert
        primary_expert = self.experts.get(decision.primary_expert)
        if primary_expert:
            try:
                advice = primary_expert.analyze(state, context)
                result["primary_advice"] = advice
            except Exception as e:
                logger.error(f"Primary expert {decision.primary_expert} failed: {e}")
                result["primary_error"] = str(e)

        # Get advice from supporting experts
        for expert_type in decision.supporting_experts:
            expert = self.experts.get(expert_type)
            if expert:
                try:
                    advice = expert.analyze(state, context)
                    result["supporting_advice"].append(advice)
                except Exception as e:
                    logger.warning(f"Supporting expert {expert_type} failed: {e}")

        return result

    def get_expert_capabilities(self, expert_type: ExpertType) -> Optional[ExpertCapability]:
        """Get capabilities of a specific expert."""
        expert = self.experts.get(expert_type)
        if expert:
            return expert.get_capabilities()
        return None

    def get_all_capabilities(self) -> Dict[ExpertType, ExpertCapability]:
        """Get capabilities of all registered experts."""
        return {et: e.get_capabilities() for et, e in self.experts.items()}

    def record_outcome(self, expert_type: ExpertType, was_successful: bool) -> None:
        """Record the outcome of an expert's advice for performance tracking."""
        expert = self.experts.get(expert_type)
        if expert:
            # Record generic outcome for performance tracking
            expert.call_count += 1
            if was_successful:
                expert.success_count += 1

    def _rule_based_routing(self, state: dict, query: str = None) -> RoutingDecision:
        """Route based on keywords and state phase."""
        scores: Dict[ExpertType, float] = {}
        matched_keywords: Dict[ExpertType, List[str]] = {}

        # Score from state phase
        phase = state.get("phase", "").lower()
        if phase in PHASE_EXPERT_MAP:
            phase_expert = PHASE_EXPERT_MAP[phase]
            scores[phase_expert] = scores.get(phase_expert, 0) + 0.5

        # Score from state indicators
        state_indicators = self._extract_state_indicators(state)
        for indicator in state_indicators:
            for expert_type, keywords in ROUTING_KEYWORDS.items():
                if indicator.lower() in [k.lower() for k in keywords]:
                    scores[expert_type] = scores.get(expert_type, 0) + 0.3
                    if expert_type not in matched_keywords:
                        matched_keywords[expert_type] = []
                    matched_keywords[expert_type].append(indicator)

        # Score from query keywords
        if query:
            query_lower = query.lower()
            for expert_type, keywords in ROUTING_KEYWORDS.items():
                for keyword in keywords:
                    if keyword.lower() in query_lower:
                        scores[expert_type] = scores.get(expert_type, 0) + 0.2
                        if expert_type not in matched_keywords:
                            matched_keywords[expert_type] = []
                        matched_keywords[expert_type].append(keyword)

        # Score from state-specific conditions
        scores = self._score_from_state_conditions(state, scores)

        # Determine primary and supporting
        if not scores:
            # Default to reconnaissance
            return RoutingDecision(
                primary_expert=ExpertType.RECONNAISSANCE,
                supporting_experts=[],
                confidence=0.3,
                reasoning="无法确定最佳专家，默认选择侦察专家",
                keywords_matched=[],
            )

        sorted_experts = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary = sorted_experts[0][0]
        supporting = [e for e, s in sorted_experts[1:3] if s > 0.2]

        all_matched = []
        for et in [primary] + supporting:
            all_matched.extend(matched_keywords.get(et, []))

        confidence = min(1.0, sorted_experts[0][1] / 2.0 + 0.3)

        reasoning = self._generate_routing_reasoning(primary, supporting, scores)

        return RoutingDecision(
            primary_expert=primary,
            supporting_experts=supporting,
            confidence=confidence,
            reasoning=reasoning,
            keywords_matched=list(set(all_matched)),
        )

    def _llm_based_routing(self, state: dict, query: str = None) -> Optional[RoutingDecision]:
        """Route using LLM classification."""
        if not self.llm:
            return None

        try:
            expert_list = ", ".join([e.value for e in self.experts.keys()])
            if not expert_list:
                return None

            prompt = f"""作为专家路由器，请分析以下渗透测试状态，选择最合适的专家类型。

可用的专家类型: {expert_list}

当前状态:
- 目标: {state.get('target', '未知')}
- 阶段: {state.get('phase', '未知')}
- 服务: {state.get('services', [])}
- 漏洞: {state.get('vulnerabilities', [])}
- 凭据: {state.get('credentials', [])}
- 权限: {'管理员' if state.get('is_admin') else '普通用户'}

查询: {query or '无'}

请返回JSON格式:
{{"primary": "专家类型", "supporting": ["专家类型"], "confidence": 0.0-1.0, "reasoning": "原因"}}"""

            response = self.llm.call(
                [{"role": "user", "content": prompt}],
                use_json_mode=True,
            )

            import json
            result = json.loads(response.content)

            primary = ExpertType(result["primary"])
            supporting = [ExpertType(s) for s in result.get("supporting", []) if s in [e.value for e in self.experts]]

            return RoutingDecision(
                primary_expert=primary,
                supporting_experts=supporting,
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", ""),
            )

        except Exception as e:
            logger.warning(f"LLM routing failed: {e}")
            return None

    def _performance_adjustment(self, decision: RoutingDecision) -> RoutingDecision:
        """Adjust routing based on expert performance history."""
        primary = self.experts.get(decision.primary_expert)
        if not primary:
            return decision

        primary_rate = primary.get_success_rate()

        # If primary expert has low success rate, consider alternatives
        if primary_rate < 0.3 and decision.supporting_experts:
            for alt_type in decision.supporting_experts:
                alt_expert = self.experts.get(alt_type)
                if alt_expert and alt_expert.get_success_rate() > primary_rate + 0.2:
                    # Swap primary with supporting
                    new_supporting = [decision.primary_expert] + [
                        e for e in decision.supporting_experts if e != alt_type
                    ]
                    decision = RoutingDecision(
                        primary_expert=alt_type,
                        supporting_experts=new_supporting,
                        confidence=decision.confidence * 0.9,
                        reasoning=f"{decision.reasoning} (根据历史表现调整为 {alt_type.value})",
                        keywords_matched=decision.keywords_matched,
                    )
                    break

        return decision

    def _extract_state_indicators(self, state: dict) -> List[str]:
        """Extract routing indicators from state."""
        indicators = []

        # Services
        for service in state.get("services", []):
            indicators.append(str(service))

        # Vulnerabilities
        for vuln in state.get("vulnerabilities", []):
            if isinstance(vuln, dict):
                indicators.append(str(vuln.get("type", "")))
                indicators.append(str(vuln.get("id", "")))
            else:
                indicators.append(str(vuln))

        # Credentials
        if state.get("credentials"):
            indicators.append("credential")
            indicators.append("password")

        # Hosts
        if state.get("hosts"):
            indicators.append("discover")

        # Admin status
        if state.get("is_admin"):
            indicators.append("privilege")
            indicators.append("escalation")

        return [i for i in indicators if i]

    def _score_from_state_conditions(self, state: dict, scores: Dict[ExpertType, float]) -> Dict[ExpertType, float]:
        """Add scores based on state conditions."""
        # No shell -> recon/exploit
        if not state.get("has_shell") and not state.get("compromised_hosts"):
            if state.get("services"):
                # Services found but no shell -> exploitation
                scores[ExpertType.EXPLOITATION] = scores.get(ExpertType.EXPLOITATION, 0) + 0.4
            else:
                # No services -> recon
                scores[ExpertType.RECONNAISSANCE] = scores.get(ExpertType.RECONNAISSANCE, 0) + 0.4

        # Has shell but not admin -> post-exploitation (privilege escalation)
        if state.get("has_shell") and not state.get("is_admin"):
            scores[ExpertType.POST_EXPLOITATION] = scores.get(ExpertType.POST_EXPLOITATION, 0) + 0.3

        # Has admin -> credential or lateral
        if state.get("is_admin"):
            if state.get("credentials"):
                scores[ExpertType.LATERAL_MOVEMENT] = scores.get(ExpertType.LATERAL_MOVEMENT, 0) + 0.3
            scores[ExpertType.CREDENTIAL] = scores.get(ExpertType.CREDENTIAL, 0) + 0.2

        # Multiple hosts -> lateral movement
        hosts = state.get("hosts", [])
        compromised = state.get("compromised_hosts", [])
        if len(hosts) > 1 and len(compromised) >= 1:
            scores[ExpertType.LATERAL_MOVEMENT] = scores.get(ExpertType.LATERAL_MOVEMENT, 0) + 0.4

        # Hashes found -> credential
        if state.get("hashes"):
            scores[ExpertType.CREDENTIAL] = scores.get(ExpertType.CREDENTIAL, 0) + 0.3

        # Vulnerabilities found -> vulnerability analysis
        if state.get("vulnerabilities"):
            scores[ExpertType.VULNERABILITY] = scores.get(ExpertType.VULNERABILITY, 0) + 0.2
            scores[ExpertType.EXPLOITATION] = scores.get(ExpertType.EXPLOITATION, 0) + 0.2

        return scores

    def _generate_routing_reasoning(self, primary: ExpertType, supporting: List[ExpertType], scores: Dict[ExpertType, float]) -> str:
        """Generate human-readable routing reasoning."""
        type_labels = {
            ExpertType.RECONNAISSANCE: "侦察",
            ExpertType.VULNERABILITY: "漏洞分析",
            ExpertType.EXPLOITATION: "漏洞利用",
            ExpertType.POST_EXPLOITATION: "后渗透",
            ExpertType.CREDENTIAL: "凭据攻击",
            ExpertType.LATERAL_MOVEMENT: "横向移动",
        }

        primary_label = type_labels.get(primary, primary.value)
        parts = [f"选择{primary_label}专家为主专家 (评分: {scores.get(primary, 0):.2f})"]

        if supporting:
            support_labels = [type_labels.get(e, e.value) for e in supporting]
            parts.append(f"辅助专家: {', '.join(support_labels)}")

        return "。".join(parts)


def create_default_router(llm_provider=None, rag_retriever=None) -> ExpertRouter:
    """Create a router with all default experts registered."""
    from models.experts import (
        ReconnaissanceExpert,
        VulnerabilityExpert,
        ExploitationExpert,
        PostExploitationExpert,
        CredentialExpert,
        LateralMovementExpert,
    )

    router = ExpertRouter(llm_provider=llm_provider, rag_retriever=rag_retriever)

    # Register all experts
    router.register_expert(ReconnaissanceExpert(llm_provider, rag_retriever))
    router.register_expert(VulnerabilityExpert(llm_provider, rag_retriever))
    router.register_expert(ExploitationExpert(llm_provider, rag_retriever))
    router.register_expert(PostExploitationExpert(llm_provider, rag_retriever))
    router.register_expert(CredentialExpert(llm_provider, rag_retriever))
    router.register_expert(LateralMovementExpert(llm_provider, rag_retriever))

    return router
