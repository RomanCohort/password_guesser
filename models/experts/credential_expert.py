"""
Credential Expert

Expert in credential attacks, password cracking, and authentication bypass.
"""

import logging
from typing import List, Dict, Optional

from models.experts.base import PenTestExpert, ExpertAdvice, ExpertType

logger = logging.getLogger(__name__)


class CredentialExpert(PenTestExpert):
    """Expert in credential attacks and harvesting."""

    TOOLS = ["hydra", "john", "hashcat", "mimikatz", "kerberoast", "crackmapexec"]

    SYSTEM_PROMPT = """你是一位资深的凭据攻击专家。

你的专长包括：
- 密码暴力破解和字典攻击
- 哈希破解（NTLM, LM, SHA, MD5等）
- Kerberos攻击（Kerberoasting, AS-REP Roasting）
- 凭据复用和密码喷洒
- 密码策略分析
- 认证绕过技术

你可以调用以下工具：
- hydra: 在线密码破解
- john: 离线哈希破解
- hashcat: GPU加速哈希破解
- mimikatz: 凭据窃取
- kerberoast: Kerberos服务票据攻击
- crackmapexec: 凭据验证和横向移动

攻击原则：
1. 优先尝试常见弱密码和默认密码
2. 使用目标相关信息的定制字典
3. 避免账户锁定策略
4. 记录成功的凭据以供复用
5. 结合多种攻击方法提高成功率
"""

    def __init__(self, llm_provider=None, rag_retriever=None):
        super().__init__(
            expert_type=ExpertType.CREDENTIAL,
            llm_provider=llm_provider,
            rag_retriever=rag_retriever,
            tools=self.TOOLS,
        )

    def analyze(self, state: dict, context: dict = None) -> ExpertAdvice:
        """Analyze credential attack opportunities."""
        self.call_count += 1

        # Get state
        target = state.get("target", "")
        services = state.get("services", [])
        credentials = state.get("credentials", [])
        hashes = state.get("hashes", [])
        users = state.get("users", [])
        domain = state.get("domain", "")

        # Build RAG query
        query = self._build_cred_query(state)
        knowledge = self.retrieve_relevant_knowledge(query)

        actions = []
        tools = []
        warnings = []
        relevant_techniques = []
        reasoning = ""
        confidence = 0.5

        # Check for available attack surfaces
        has_smb = any("smb" in str(s).lower() or "445" in str(s) for s in services)
        has_ssh = any("ssh" in str(s).lower() or "22" in str(s) for s in services)
        has_rdp = any("rdp" in str(s).lower() or "3389" in str(s) for s in services)
        has_ftp = any("ftp" in str(s).lower() or "21" in str(s) for s in services)
        has_http = any("http" in str(s).lower() for s in services)

        # Hash cracking
        if hashes:
            for hash_info in hashes[:3]:
                hash_type = hash_info.get("type", "unknown")
                hash_value = hash_info.get("hash", "")

                actions.append({
                    "type": "crack_hash",
                    "tool": "hashcat",
                    "params": {
                        "hash": hash_value,
                        "hash_type": self._identify_hash_type(hash_type),
                        "wordlist": "rockyou.txt",
                    },
                    "description": f"尝试破解 {hash_type} 哈希",
                })
                tools.append("hashcat")

                # Alternative with john
                actions.append({
                    "type": "crack_hash",
                    "tool": "john",
                    "params": {
                        "hash": hash_value,
                        "wordlist": "rockyou.txt",
                    },
                    "description": f"使用John the Ripper破解哈希",
                    "alternative": True,
                })
                tools.append("john")

            relevant_techniques.append("T1110.002")
            reasoning = f"发现 {len(hashes)} 个哈希可尝试破解。"
            confidence = 0.75

        # Online brute force
        if users and (has_ssh or has_rdp or has_smb):
            for service in ["ssh", "rdp", "smb"]:
                if service == "ssh" and has_ssh:
                    actions.append({
                        "type": "brute_force",
                        "tool": "hydra",
                        "params": {
                            "target": target,
                            "service": "ssh",
                            "users": users[:5],
                            "wordlist": "common_passwords.txt",
                        },
                        "description": "SSH密码爆破",
                        "caution": "注意账户锁定策略",
                    })
                    tools.append("hydra")

                elif service == "rdp" and has_rdp:
                    actions.append({
                        "type": "brute_force",
                        "tool": "hydra",
                        "params": {
                            "target": target,
                            "service": "rdp",
                            "users": users[:5],
                        },
                        "description": "RDP密码爆破",
                        "caution": "RDP爆破容易被检测",
                    })
                    tools.append("hydra")

                elif service == "smb" and has_smb:
                    actions.append({
                        "type": "password_spray",
                        "tool": "crackmapexec",
                        "params": {
                            "target": target,
                            "users": users[:5],
                            "passwords": ["Password123!", "Welcome1", "P@ssw0rd"],
                        },
                        "description": "SMB密码喷洒攻击",
                    })
                    tools.append("crackmapexec")

            relevant_techniques.extend(["T1110.001", "T1110.003"])
            reasoning = reasoning or f"发现可尝试在线密码攻击的服务。"
            confidence = max(confidence, 0.7)

        # Kerberos attacks
        if domain and (has_smb or domain):
            actions.append({
                "type": "kerberoast",
                "tool": "kerberoast",
                "params": {
                    "domain": domain,
                },
                "description": "Kerberoasting攻击获取服务票据",
            })
            tools.append("kerberoast")
            relevant_techniques.append("T1558.003")

            actions.append({
                "type": "asrep_roast",
                "description": "AS-REP Roasting攻击（针对禁用预认证的用户）",
                "optional": True,
            })
            relevant_techniques.append("T1558.004")

            reasoning = reasoning or "域环境可尝试Kerberos攻击。"

        # Credential reuse
        if credentials:
            for cred in credentials[:3]:
                username = cred.get("username", "")
                password = cred.get("password", "")

                if username and password:
                    actions.append({
                        "type": "credential_reuse",
                        "tool": "crackmapexec",
                        "params": {
                            "target": target,
                            "cred": f"{username}:{password}",
                        },
                        "description": f"测试凭据 {username} 的复用性",
                    })
                    tools.append("crackmapexec")

            relevant_techniques.append("T1078")
            reasoning = reasoning or f"已获取 {len(credentials)} 个凭据，可测试复用性。"
            confidence = max(confidence, 0.8)

        # Default if no specific attack vector
        if not actions:
            if services:
                actions.append({
                    "type": "info",
                    "description": "发现服务但缺少攻击向量，建议收集更多信息（用户名、哈希等）",
                })
            else:
                actions.append({
                    "type": "wait",
                    "description": "需要先完成侦察阶段",
                })
            reasoning = "缺少凭据攻击所需的输入信息。"
            confidence = 0.5

        # Warnings
        warnings.append("密码攻击可能触发账户锁定或安全告警")
        if len(actions) > 5:
            warnings.append("建议分批次执行攻击以避免被检测")

        summary = self._generate_summary(hashes, credentials, actions)

        return ExpertAdvice(
            expert_type=self.expert_type,
            summary=summary,
            recommended_actions=actions,
            tools_to_use=list(set(tools)),
            confidence=confidence,
            reasoning=reasoning,
            warnings=warnings,
            relevant_techniques=relevant_techniques,
        )

    def get_prompt_template(self) -> str:
        return self.SYSTEM_PROMPT

    def _build_cred_query(self, state: dict) -> str:
        """Build RAG query for credential context."""
        parts = ["凭据攻击 密码破解"]

        if state.get("domain"):
            parts.append("Kerberos 域环境")

        services = state.get("services", [])
        if any("smb" in str(s).lower() for s in services):
            parts.append("SMB")

        return " ".join(parts)

    def _identify_hash_type(self, hash_type: str) -> str:
        """Map hash type to hashcat mode."""
        hash_map = {
            "ntlm": "1000",
            "ntlmv2": "5600",
            "lm": "3000",
            "sha1": "100",
            "sha256": "1400",
            "sha512": "1700",
            "md5": "0",
            "bcrypt": "3200",
            "krb5tgs": "13100",
            "krb5asrep": "18200",
        }
        return hash_map.get(hash_type.lower(), "0")

    def _generate_summary(self, hashes: list, credentials: list, actions: list) -> str:
        """Generate summary of credential attack recommendations."""
        parts = []

        if hashes:
            parts.append(f"发现 {len(hashes)} 个哈希可破解")
        if credentials:
            parts.append(f"已获取 {len(credentials)} 个凭据")

        parts.append(f"推荐 {len(actions)} 个凭据攻击行动")

        return "，".join(parts)

    def _get_techniques(self) -> List[str]:
        return ["T1110", "T1555", "T1003", "T1558", "T1078"]

    def _get_required_inputs(self) -> List[str]:
        return ["target", "services"]

    def _get_outputs(self) -> List[str]:
        return ["credentials", "cracked_hashes", "valid_accounts"]
