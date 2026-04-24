"""
Reconnaissance Expert

Expert in information gathering, network mapping, and service identification.
"""

import logging
from typing import List, Dict, Optional

from models.experts.base import PenTestExpert, ExpertAdvice, ExpertType

logger = logging.getLogger(__name__)


class ReconnaissanceExpert(PenTestExpert):
    """Expert in reconnaissance and information gathering."""

    TOOLS = ["nmap", "masscan", "shodan", "dnsrecon", "theharvester", "nuclei"]

    SYSTEM_PROMPT = """你是一位资深的侦察专家。

你的专长包括：
- 网络侦察和信息收集
- 服务识别和版本探测
- 网络拓扑发现
- DNS枚举和子域名发现
- OSINT（开源情报）收集
- 端口扫描策略优化

你可以调用以下工具：
- nmap: 网络扫描和服务识别
- masscan: 高速端口扫描
- shodan: 互联网设备搜索
- dnsrecon: DNS枚举
- theharvester: 邮箱和子域名收集
- nuclei: 漏洞扫描

侦察原则：
1. 从被动侦察开始，逐步转向主动
2. 最小化扫描流量以避免检测
3. 系统化地收集所有可用信息
4. 优先发现高价值目标
"""

    def __init__(self, llm_provider=None, rag_retriever=None):
        super().__init__(
            expert_type=ExpertType.RECONNAISSANCE,
            llm_provider=llm_provider,
            rag_retriever=rag_retriever,
            tools=self.TOOLS,
        )

    def analyze(self, state: dict, context: dict = None) -> ExpertAdvice:
        """Analyze reconnaissance needs and provide advice."""
        self.call_count += 1

        # Build query for RAG
        query = self._build_recon_query(state)
        knowledge = self.retrieve_relevant_knowledge(query)

        # Analyze current state
        hosts = state.get("hosts", [])
        services = state.get("services", [])
        target = state.get("target", "")
        scan_history = state.get("scan_history", [])

        actions = []
        tools = []
        warnings = []
        reasoning = ""
        confidence = 0.5

        if not hosts and not scan_history:
            # No recon done yet - start with port scan
            actions.append({
                "type": "scan",
                "tool": "nmap",
                "params": {
                    "target": target,
                    "scan_type": "service_version",
                    "ports": "common",
                },
                "description": f"对目标 {target} 进行服务和版本探测扫描",
            })
            tools.append("nmap")
            reasoning = "尚未进行任何侦察，首先进行服务版本探测以发现开放端口和运行服务。"
            confidence = 0.9

        elif hosts and not services:
            # Hosts found but no service details
            actions.append({
                "type": "scan",
                "tool": "nmap",
                "params": {
                    "target": target,
                    "scan_type": "detailed",
                    "ports": "all",
                },
                "description": "发现主机但缺少服务信息，进行详细端口扫描",
            })
            tools.append("nmap")
            reasoning = "已发现主机但未识别服务，需要详细扫描以确定攻击面。"
            confidence = 0.85

        elif services:
            # Services found - suggest targeted recon
            actions.append({
                "type": "scan",
                "tool": "nuclei",
                "params": {
                    "target": target,
                    "templates": "vulnerabilities",
                },
                "description": "基于已发现服务进行漏洞扫描",
            })
            tools.append("nuclei")

            # Check for web services
            if any("http" in str(s).lower() for s in services):
                actions.append({
                    "type": "recon",
                    "tool": "nmap",
                    "params": {
                        "target": target,
                        "script": "http-enum,http-headers",
                    },
                    "description": "对Web服务进行目录和头信息枚举",
                })

            reasoning = "已发现服务，建议进行针对性漏洞扫描和Web枚举。"
            confidence = 0.75

        # OSINT suggestions
        if target:
            actions.append({
                "type": "recon",
                "tool": "theharvester",
                "params": {"target": target},
                "description": f"收集目标 {target} 的OSINT信息",
                "optional": True,
            })
            tools.append("theharvester")

        # Add warnings
        if len(scan_history) > 5:
            warnings.append("扫描次数较多，注意避免触发IDS/IPS告警")

        summary = self._generate_summary(actions, services, hosts)

        return ExpertAdvice(
            expert_type=self.expert_type,
            summary=summary,
            recommended_actions=actions,
            tools_to_use=tools,
            confidence=confidence,
            reasoning=reasoning,
            warnings=warnings,
        )

    def get_prompt_template(self) -> str:
        return self.SYSTEM_PROMPT

    def _build_recon_query(self, state: dict) -> str:
        """Build a RAG query for reconnaissance context."""
        parts = ["侦察 网络扫描 信息收集"]
        target = state.get("target", "")
        if target:
            parts.append(target)
        services = state.get("services", [])
        if services:
            parts.extend(str(s) for s in services)
        return " ".join(parts)

    def _generate_summary(self, actions, services, hosts) -> str:
        """Generate summary of reconnaissance recommendations."""
        parts = []
        if not hosts:
            parts.append("建议开始初始侦察阶段")
        else:
            parts.append(f"已发现 {len(hosts)} 台主机")
        if services:
            parts.append(f"已识别 {len(services)} 个服务")
        parts.append(f"推荐 {len(actions)} 个侦察行动")
        return "。".join(parts)

    def _get_techniques(self) -> List[str]:
        return ["T1595", "T1592", "T1589", "T1590", "T1591", "T1046"]

    def _get_required_inputs(self) -> List[str]:
        return ["target"]

    def _get_outputs(self) -> List[str]:
        return ["hosts", "services", "ports", "os_info", "dns_records"]
