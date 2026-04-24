"""
Lateral Movement Expert

Expert in network traversal, session hijacking, and moving through compromised environments.
"""

import logging
from typing import List, Dict, Optional

from models.experts.base import PenTestExpert, ExpertAdvice, ExpertType

logger = logging.getLogger(__name__)


class LateralMovementExpert(PenTestExpert):
    """Expert in lateral movement techniques."""

    TOOLS = ["crackmapexec", "psexec", "wmi", "winrm", "ssh", "evil-winrm", "impacket"]

    SYSTEM_PROMPT = """你是一位资深的横向移动专家。

你的专长包括：
- 网络横向移动技术
- 凭据复用和传递攻击
- 会话劫持
- 远程执行技术
- 跳板主机设置
- 代理链和网络隧道

你可以调用以下工具：
- crackmapexec: 凭据验证和命令执行
- psexec: Windows远程执行
- wmi: WMI远程命令执行
- winrm: WinRM远程管理
- ssh: SSH远程连接
- evil-winrm: WinRM渗透工具
- impacket: Python网络工具集

横向移动原则：
1. 优先使用已获取的凭据
2. 最小化网络流量以避免检测
3. 建立稳定的远程访问
4. 记录移动路径以便回溯
5. 准备备用移动方法
"""

    def __init__(self, llm_provider=None, rag_retriever=None):
        super().__init__(
            expert_type=ExpertType.LATERAL_MOVEMENT,
            llm_provider=llm_provider,
            rag_retriever=rag_retriever,
            tools=self.TOOLS,
        )

    def analyze(self, state: dict, context: dict = None) -> ExpertAdvice:
        """Analyze lateral movement opportunities."""
        self.call_count += 1

        # Get state
        target = state.get("target", "")
        hosts = state.get("hosts", [])
        credentials = state.get("credentials", [])
        domain = state.get("domain", "")
        current_host = state.get("current_host", "")
        compromised_hosts = state.get("compromised_hosts", [])
        is_admin = state.get("is_admin", False)

        # Build RAG query
        query = self._build_lateral_query(state)
        knowledge = self.retrieve_relevant_knowledge(query)

        actions = []
        tools = []
        warnings = []
        alternatives = []
        relevant_techniques = []
        reasoning = ""
        confidence = 0.5

        # Check if we have compromised any host
        if not compromised_hosts and not current_host:
            actions.append({
                "type": "wait",
                "description": "需要先获得初始访问权限才能进行横向移动",
            })
            reasoning = "尚未获得任何主机的访问权限，无法进行横向移动。"
            confidence = 0.7

        else:
            # Identify potential targets
            potential_targets = [h for h in hosts if h not in compromised_hosts]

            if not potential_targets:
                actions.append({
                    "type": "recon",
                    "description": "所有已知主机已被攻陷，建议进行更多网络发现",
                })
                reasoning = "已攻陷所有已知主机。"
                confidence = 0.6

            else:
                # Sort targets by priority
                prioritized_targets = self._prioritize_targets(potential_targets, domain)

                for target_host in prioritized_targets[:3]:
                    target_ip = target_host.get("ip", target_host.get("host", ""))

                    # Try credential reuse
                    for cred in credentials[:3]:
                        username = cred.get("username", "")
                        password = cred.get("password", "")
                        nthash = cred.get("nthash", "")

                        if username and (password or nthash):
                            # WinRM
                            actions.append({
                                "type": "lateral",
                                "tool": "evil-winrm",
                                "params": {
                                    "target": target_ip,
                                    "user": username,
                                    "password": password,
                                },
                                "description": f"使用 {username} 凭据通过WinRM连接 {target_ip}",
                                "priority": 1,
                            })
                            tools.append("evil-winrm")
                            relevant_techniques.append("T1021.006")

                            # SMB/PSExec
                            actions.append({
                                "type": "lateral",
                                "tool": "crackmapexec",
                                "params": {
                                    "target": target_ip,
                                    "cred": f"{username}:{password}",
                                    "module": "psexec",
                                },
                                "description": f"通过PSExec横向移动到 {target_ip}",
                                "priority": 2,
                            })
                            tools.append("crackmapexec")
                            relevant_techniques.append("T1021.002")

                            # WMI
                            actions.append({
                                "type": "lateral",
                                "tool": "wmi",
                                "params": {
                                    "target": target_ip,
                                    "user": username,
                                    "password": password,
                                },
                                "description": f"通过WMI横向移动到 {target_ip}",
                                "alternative": True,
                            })
                            tools.append("wmi")
                            relevant_techniques.append("T1047")

                            break  # One cred per target is enough for initial recommendations

                    # Pass-the-hash if we have hashes
                    for cred in credentials:
                        nthash = cred.get("nthash", "")
                        username = cred.get("username", "")
                        if nthash and username:
                            actions.append({
                                "type": "pth",
                                "tool": "crackmapexec",
                                "params": {
                                    "target": target_ip,
                                    "user": username,
                                    "hash": nthash,
                                },
                                "description": f"通过Pass-the-Hash攻击 {target_ip}",
                                "optional": True,
                            })
                            tools.append("crackmapexec")
                            relevant_techniques.append("T1550.002")
                            break

                # Alternative approaches
                alternatives.append({
                    "approach": "SSH横向移动",
                    "description": "如果目标开放SSH服务，可以使用SSH进行横向移动",
                    "when": "目标开放22端口且为Linux系统",
                })

                alternatives.append({
                    "approach": "远程桌面",
                    "description": "如果获取了RDP凭据，可以通过远程桌面进行横向移动",
                    "when": "目标开放3389端口",
                })

                reasoning = f"发现 {len(potential_targets)} 个潜在横向移动目标，{len(credentials)} 个可用凭据。"
                confidence = 0.75 if credentials else 0.5

        # Setup proxy/pivot suggestions
        if compromised_hosts:
            actions.append({
                "type": "pivot",
                "description": "建议在已攻陷主机上设置跳板或代理",
                "tools": ["chisel", "ligolo", "ssh隧道"],
                "optional": True,
            })
            relevant_techniques.append("T1572")

        # Warnings
        if actions:
            warnings.append("横向移动可能触发EDR/IDS告警")
            warnings.append("建议使用加密通道避免流量分析")

        if domain:
            warnings.append("域环境中的横向移动需要更谨慎以避免触发高级检测")

        summary = self._generate_summary(hosts, compromised_hosts, credentials, actions)

        return ExpertAdvice(
            expert_type=self.expert_type,
            summary=summary,
            recommended_actions=actions,
            tools_to_use=list(set(tools)),
            confidence=confidence,
            reasoning=reasoning,
            warnings=warnings,
            alternatives=alternatives,
            relevant_techniques=relevant_techniques,
        )

    def get_prompt_template(self) -> str:
        return self.SYSTEM_PROMPT

    def _build_lateral_query(self, state: dict) -> str:
        """Build RAG query for lateral movement context."""
        parts = ["横向移动 远程执行"]

        if state.get("domain"):
            parts.append("域环境")

        hosts = state.get("hosts", [])
        if hosts:
            parts.append(f"{len(hosts)}台主机")

        return " ".join(parts)

    def _prioritize_targets(self, hosts: list, domain: str = "") -> list:
        """Prioritize lateral movement targets."""
        # Simple prioritization: prefer domain controllers and file servers
        priority_keywords = ["dc", "domain", "controller", "file", "share", "sql", "db"]

        def score(host):
            host_str = str(host.get("hostname", "")).lower() + str(host.get("ip", "")).lower()
            for kw in priority_keywords:
                if kw in host_str:
                    return 0
            return 1

        return sorted(hosts, key=score)

    def _generate_summary(self, hosts: list, compromised: list, credentials: list, actions: list) -> str:
        """Generate summary of lateral movement recommendations."""
        parts = []

        if compromised:
            parts.append(f"已攻陷 {len(compromised)} 台主机")

        remaining = len(hosts) - len(compromised) if hosts else 0
        if remaining > 0:
            parts.append(f"{remaining} 台主机待攻陷")

        if credentials:
            parts.append(f"{len(credentials)} 个可用凭据")

        parts.append(f"推荐 {len(actions)} 个横向移动行动")

        return "，".join(parts)

    def _get_techniques(self) -> List[str]:
        return ["T1021", "T1047", "T1563", "T1072", "T1080", "T1550"]

    def _get_required_inputs(self) -> List[str]:
        return ["hosts", "credentials", "compromised_hosts"]

    def _get_outputs(self) -> List[str]:
        return ["new_access", "pivot_hosts", "lateral_path"]
