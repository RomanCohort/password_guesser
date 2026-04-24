"""
LLM + MOE Expert System + Attack Team Demonstration

Shows how each component participates in the attack decision process:
  Phase 1: LLM Provider initialization and analysis
  Phase 2: MOE Expert Router - route to appropriate experts
  Phase 3: Attack Team - 7-member team collaboration
  Phase 4: Full integration - team-based attack run

Usage:
    python scripts/demo_llm_moe_team.py
    python scripts/demo_llm_moe_team.py --api-key sk-xxx   # with real LLM
"""

import asyncio
import json
import sys
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.llm_provider import (
    BaseLLMProvider, LLMConfig, LLMResponse,
    create_provider, get_provider,
)
from models.vector_store import VectorStore, EmbeddingService
from models.rag_retriever import RAGRetriever
from models.expert_router import ExpertRouter, ExpertType, RoutingDecision
from models.experts import (
    PenTestExpert, ExpertAdvice,
    ReconnaissanceExpert, VulnerabilityExpert, ExploitationExpert,
    PostExploitationExpert, CredentialExpert, LateralMovementExpert,
)
from models.attack_team import (
    AttackTeam, TeamMember, TeamRole, MeetingType,
    TeamMemory, MeetingResult, create_attack_team,
)
from rl_agent.action import ActionType

# Colors
R = '\033[91m'
G = '\033[92m'
Y = '\033[93m'
B = '\033[94m'
M = '\033[95m'
C = '\033[96m'
BOLD = '\033[1m'
DIM = '\033[2m'
END = '\033[0m'


def section(title, color=C):
    print(f"\n{color}{BOLD}{'='*65}\n  {title}\n{'='*65}{END}\n")


def detail(label, value, indent=2):
    print(f"{' '*indent}{B}{label}:{END} {value}")


# =============================================================================
# Phase 1: LLM Provider
# =============================================================================
def demo_llm_provider(api_key: str = None):
    section("PHASE 1: LLM PROVIDER", M)

    print(f"  {B}LLM Provider{END} 是整个系统的大脑。")
    print(f"  支持 DeepSeek / OpenAI / vLLM / Local 模型。\n")

    # Show config
    config = LLMConfig(provider="deepseek", api_key=api_key or "")
    print(f"  {BOLD}当前配置:{END}")
    detail("Provider", config.provider)
    detail("Model", config.model)
    detail("API Base", config.api_base)
    detail("Temperature", config.temperature)
    detail("Max Tokens", config.max_tokens)

    if api_key:
        print(f"\n  {G}[+] API Key 已提供 - 使用真实 LLM{END}")
        try:
            provider = create_provider(provider="deepseek", api_key=api_key)
            available = provider.is_available()
            detail("Available", available)
        except Exception as e:
            print(f"  {R}[-] LLM 初始化失败: {e}{END}")
    else:
        print(f"\n  {Y}[!] 无 API Key - 使用规则引擎 (Rule-based) 模式{END}")
        print(f"  专家系统可以在无 LLM 的情况下工作，使用内置规则分析。")

    # Demonstrate what LLM would analyze
    print(f"\n  {BOLD}LLM 在攻击中的角色:{END}")
    roles = [
        ("1. 情况分析", "分析当前渗透状态，判断攻击阶段"),
        ("2. 专家路由", "当规则引擎置信度 < 0.7 时，LLM 辅助路由"),
        ("3. 方案综合", "综合多个专家的建议，生成最优方案"),
        ("4. 反思学习", "分析失败原因，调整攻击策略"),
    ]
    for title, desc in roles:
        print(f"    {M}{title}{END}: {desc}")

    return config


# =============================================================================
# Phase 2: MOE Expert System
# =============================================================================
def demo_moe_experts(rag_retriever=None, llm_provider=None):
    section("PHASE 2: MOE (Mixture of Experts) 专家系统", Y)

    print(f"  {B}MOE 架构{END} 根据攻击阶段动态选择专家。\n")

    # Create router
    router = ExpertRouter(llm_provider=llm_provider, rag_retriever=rag_retriever)

    # Register all 6 experts
    experts = [
        ReconnaissanceExpert(llm_provider=llm_provider, rag_retriever=rag_retriever),
        VulnerabilityExpert(llm_provider=llm_provider, rag_retriever=rag_retriever),
        ExploitationExpert(llm_provider=llm_provider, rag_retriever=rag_retriever),
        PostExploitationExpert(llm_provider=llm_provider, rag_retriever=rag_retriever),
        CredentialExpert(llm_provider=llm_provider, rag_retriever=rag_retriever),
        LateralMovementExpert(llm_provider=llm_provider, rag_retriever=rag_retriever),
    ]

    print(f"  {BOLD}注册 {len(experts)} 个专家:{END}\n")

    expert_display = {
        ExpertType.RECONNAISSANCE:     ("侦察专家",  "Scout",    "网络发现、端口扫描、服务识别"),
        ExpertType.VULNERABILITY:      ("漏洞专家",  "Analyst",  "CVE分析、漏洞评估、风险排序"),
        ExpertType.EXPLOITATION:       ("利用专家",  "Striker",  "漏洞利用、Payload生成、绕过防护"),
        ExpertType.POST_EXPLOITATION:  ("后渗透专家","Ghost",    "权限提升、持久化、数据收集"),
        ExpertType.CREDENTIAL:         ("凭据专家",  "Hunter",   "密码破解、Hash攻击、票据伪造"),
        ExpertType.LATERAL_MOVEMENT:   ("横向移动专家","Phantom", "网络跳板、凭据复用、会话劫持"),
    }

    for expert in experts:
        router.register_expert(expert)
        display = expert_display.get(expert.expert_type, ("?", "?", "?"))
        print(f"    {G}[+]{END} {display[0]:10s} ({display[1]:8s}) - {display[2]}")
        print(f"       工具: {', '.join(expert.tools[:4])}")

    # Demonstrate routing for different scenarios
    print(f"\n  {BOLD}专家路由演示:{END}\n")

    scenarios = [
        {
            "name": "初始侦察阶段",
            "state": {
                "discovered_hosts": [],
                "open_ports": {},
                "compromised_hosts": [],
                "phase": "reconnaissance",
            },
            "query": "如何发现目标网络中的主机",
        },
        {
            "name": "发现漏洞后的利用",
            "state": {
                "discovered_hosts": ["192.168.1.100"],
                "open_ports": {"192.168.1.100": [22, 80, 445]},
                "vulnerabilities": {"192.168.1.100": ["CVE-2017-0144"]},
                "compromised_hosts": [],
                "phase": "exploitation",
            },
            "query": "如何利用 EternalBlue 漏洞",
        },
        {
            "name": "获取初始访问后",
            "state": {
                "discovered_hosts": ["192.168.1.100"],
                "compromised_hosts": ["192.168.1.100"],
                "privileges": {"192.168.1.100": "user"},
                "phase": "post_exploitation",
            },
            "query": "如何提升权限到管理员",
        },
        {
            "name": "横向移动到其他主机",
            "state": {
                "discovered_hosts": ["192.168.1.100", "192.168.1.101", "192.168.1.102"],
                "compromised_hosts": ["192.168.1.100"],
                "credentials": [{"user": "admin", "password": "P@ss123"}],
                "privileges": {"192.168.1.100": "admin"},
                "phase": "lateral_movement",
            },
            "query": "如何使用已知凭据横向移动",
        },
    ]

    for i, scenario in enumerate(scenarios):
        print(f"  {BOLD}场景 {i+1}: {scenario['name']}{END}")
        detail("攻击阶段", scenario['state'].get('phase', 'unknown'))
        detail("已发现主机", scenario['state'].get('discovered_hosts', []))
        detail("已攻破主机", scenario['state'].get('compromised_hosts', []))
        detail("查询", scenario['query'])

        # Route
        decision = router.analyze_situation(
            state=scenario['state'],
            query=scenario['query'],
        )

        display = expert_display.get(decision.primary_expert, ("?", "?", "?"))
        print(f"\n    {G}路由结果:{END}")
        print(f"      主专家:     {M}{display[0]} ({display[1]}){END}")
        print(f"      置信度:     {decision.confidence:.1%}")

        if decision.supporting_experts:
            supporting = [expert_display.get(e, ("?", "?", "?"))[0] for e in decision.supporting_experts]
            print(f"      辅助专家:   {', '.join(supporting)}")
        print(f"      原因:       {decision.reasoning[:60]}")

        # Get expert advice
        try:
            primary_expert = router.experts.get(decision.primary_expert)
            if primary_expert:
                advice = primary_expert.analyze(scenario['state'])
                print(f"      建议操作:   {', '.join(a.get('action', '?') for a in advice.recommended_actions[:3])}")
                print(f"      推荐工具:   {', '.join(advice.tools_to_use[:3])}")
                print(f"      专家置信度: {advice.confidence:.1%}")
        except Exception as e:
            print(f"      (分析中: {e})")

        print()

    # Show routing stats
    print(f"  {BOLD}路由统计:{END}")
    detail("总路由次数", len(router.routing_history))

    return router


# =============================================================================
# Phase 3: Attack Team Collaboration
# =============================================================================
def demo_attack_team(rag_retriever=None, llm_provider=None):
    section("PHASE 3: ATTACK TEAM (7人攻击小组)", R)

    print(f"  {B}攻击小组{END} 模拟真实红队协作，7名成员各司其职。\n")

    # Create team
    team = create_attack_team(
        llm_provider=llm_provider,
        rag_retriever=rag_retriever,
    )

    # Display team
    print(f"  {BOLD}团队成员:{END}\n")

    role_display = {
        TeamRole.LEADER:      ("Commander", "指挥官", "战略决策、资源分配"),
        TeamRole.RECON:       ("Scout",     "侦察兵", "目标发现、信息收集"),
        TeamRole.VULN_ANALYST:("Analyst",   "分析师", "漏洞分析、风险评估"),
        TeamRole.EXPLOITER:   ("Striker",   "突击手", "漏洞利用、获取访问"),
        TeamRole.POST_EX:     ("Ghost",     "幽灵",   "权限提升、持久化"),
        TeamRole.CRED_HUNTER: ("Hunter",    "猎手",   "凭据获取、密码破解"),
        TeamRole.MOVER:       ("Phantom",   "幻影",   "横向移动、网络穿透"),
    }

    for member in team.members:
        display = role_display.get(member.role, ("?", "?", "?"))
        print(f"    {M}{display[0]:10s}{END} ({display[1]}) - {display[2]}")
        print(f"       角色: {member.role.value:15s} | 专家: {member.expert_type.value}")
        print(f"       初始置信度: {member.confidence:.0%}")
        print()

    # Demonstrate team meetings
    print(f"  {BOLD}团队会议演示:{END}\n")

    # Meeting 1: Briefing
    print(f"  {B}--- 会议 1: BRIEFING (任务简报) ---{END}")
    target_info = {
        "ip": "192.168.1.100",
        "os": "Linux",
        "ports": {22: "ssh", 80: "http", 443: "https"},
        "vulnerabilities": [],
    }

    briefing = team.brief_team(
        target="192.168.1.100",
        initial_info=target_info,
    )

    print(f"\n    参与者: {', '.join(p for p in briefing.participants)}")
    print(f"    讨论摘要: {briefing.discussion[:100]}...")
    print(f"    决策: {len(briefing.decisions)} 项")
    for d in briefing.decisions[:3]:
        print(f"      - {d}")
    print(f"    共识度: {briefing.consensus_level:.1%}")
    print(f"    行动计划: {len(briefing.action_plan)} 步")
    for step in briefing.action_plan[:3]:
        print(f"      {step}")

    # Meeting 2: Planning after scan
    print(f"\n  {B}--- 会议 2: PLANNING (攻击规划) ---{END}")

    state_after_scan = {
        "discovered_hosts": ["192.168.1.100"],
        "open_ports": {"192.168.1.100": [22, 80, 443, 3306]},
        "services": {
            "192.168.1.100": {
                22: "ssh",
                80: "http (Apache/2.4.49)",
                443: "https",
                3306: "mysql",
            }
        },
        "compromised_hosts": [],
        "phase": "scanning",
    }

    planning = team.hold_meeting(
        meeting_type=MeetingType.PLANNING,
        state=state_after_scan,
    )

    print(f"\n    参与者: {', '.join(p for p in planning.participants)}")
    print(f"    讨论摘要: {planning.discussion[:100]}...")
    print(f"    决策: {len(planning.decisions)} 项")
    for d in planning.decisions[:3]:
        print(f"      - {d}")
    print(f"    共识度: {planning.consensus_level:.1%}")

    # Meeting 3: Emergency - found CVE
    print(f"\n  {B}--- 会议 3: EMERGENCY (紧急协商) ---{END}")

    emergency_state = {
        "discovered_hosts": ["192.168.1.100"],
        "vulnerabilities": {"192.168.1.100": ["CVE-2021-41773"]},
        "compromised_hosts": [],
        "phase": "exploitation",
    }

    emergency = team.hold_meeting(
        meeting_type=MeetingType.EMERGENCY,
        state=emergency_state,
    )

    print(f"\n    触发原因: 发现 Apache 路径遍历漏洞 (CVE-2021-41773)")
    print(f"    参与者: {', '.join(p for p in emergency.participants)}")
    print(f"    决策: {len(emergency.decisions)} 项")
    for d in emergency.decisions[:3]:
        print(f"      - {d}")
    print(f"    共识度: {emergency.consensus_level:.1%}")

    # Show team memory
    print(f"\n  {BOLD}团队记忆 (共享知识库):{END}")
    memory = team.memory
    detail("已发现主机", memory.discovered_hosts)
    detail("已发现服务", memory.discovered_services)
    detail("已发现漏洞", memory.discovered_vulnerabilities)
    detail("已获取凭据", memory.obtained_credentials)
    detail("已攻破主机", memory.compromised_hosts)
    detail("攻击历史", f"{len(memory.attack_history)} 条记录")
    detail("经验教训", f"{len(memory.lessons)} 条")

    return team


# =============================================================================
# Phase 4: Full Integration
# =============================================================================
def demo_full_integration():
    section("PHASE 4: 完整集成 - 攻击流程可视化", B)

    print(f"  {B}完整攻击流程{END}: LLM + MOE + Team 协同工作\n")

    # Simulate a full attack timeline
    attack_timeline = [
        {
            "step": 1,
            "phase": "侦察",
            "action": "SCAN_NETWORK",
            "team_action": "Scout 执行网络扫描",
            "expert": "RECONNAISSANCE",
            "llm_role": "分析扫描结果，判断网段",
            "moe_role": "路由到侦察专家",
            "result": "发现 192.168.1.0/24 网段，3 台主机",
            "reward": 0.5,
        },
        {
            "step": 2,
            "phase": "侦察",
            "action": "SCAN_PORT",
            "team_action": "Scout 端口扫描 192.168.1.100",
            "expert": "RECONNAISSANCE",
            "llm_role": "识别服务指纹",
            "moe_role": "保持侦察专家",
            "result": "22(ssh), 80(http), 443(https), 3306(mysql)",
            "reward": 0.3,
        },
        {
            "step": 3,
            "phase": "侦察",
            "action": "ENUMERATE_SERVICE",
            "team_action": "Scout + Analyst 枚举服务",
            "expert": "RECONNAISSANCE + VULNERABILITY",
            "llm_role": "分析服务版本，搜索已知漏洞",
            "moe_role": "双专家协作",
            "result": "Apache/2.4.49, OpenSSH 8.2, MySQL 5.7",
            "reward": 0.2,
        },
        {
            "step": 4,
            "phase": "漏洞分析",
            "action": "EXPLOIT_VULN",
            "team_action": "PLANNING 会议: Analyst 发现 CVE-2021-41773",
            "expert": "VULNERABILITY",
            "llm_role": "检索 CVE 数据库，评估可行性",
            "moe_role": "路由到漏洞专家 → 利用专家",
            "result": "Apache 路径遍历，可 RCE",
            "reward": 0.5,
        },
        {
            "step": 5,
            "phase": "利用",
            "action": "EXPLOIT_VULN",
            "team_action": "Striker 利用 CVE-2021-41773",
            "expert": "EXPLOITATION",
            "llm_role": "生成 exploit payload",
            "moe_role": "路由到利用专家",
            "result": "获得 www-data shell",
            "reward": 5.0,
        },
        {
            "step": 6,
            "phase": "后渗透",
            "action": "PRIV_ESCALATE",
            "team_action": "Ghost 提权到 root",
            "expert": "POST_EXPLOITATION",
            "llm_role": "分析提权路径，推荐方法",
            "moe_role": "路由到后渗透专家",
            "result": "通过内核漏洞提权到 root",
            "reward": 3.0,
        },
        {
            "step": 7,
            "phase": "凭据获取",
            "action": "DUMP_CREDS",
            "team_action": "Hunter 收集凭据",
            "expert": "CREDENTIAL",
            "llm_role": "识别凭据存储位置",
            "moe_role": "路由到凭据专家",
            "result": "获取 3 组用户名/密码",
            "reward": 3.0,
        },
        {
            "step": 8,
            "phase": "横向移动",
            "action": "LATERAL_MOVE",
            "team_action": "Phantom 使用凭据横向移动",
            "expert": "LATERAL_MOVEMENT",
            "llm_role": "规划移动路径",
            "moe_role": "路由到横向移动专家",
            "result": "攻破 192.168.1.101, 192.168.1.102",
            "reward": 4.0,
        },
    ]

    total_reward = 0
    for step_data in attack_timeline:
        step = step_data["step"]
        phase = step_data["phase"]
        reward = step_data["reward"]
        total_reward += reward

        # Print step header
        phase_colors = {
            "侦察": B, "漏洞分析": Y, "利用": R,
            "后渗透": M, "凭据获取": C, "横向移动": G,
        }
        pc = phase_colors.get(phase, "")

        print(f"  {pc}{BOLD}Step {step}: [{phase}]{END} {step_data['action']}")
        print(f"  {'─'*60}")

        # Show component participation
        print(f"    {B}攻击小组:{END} {step_data['team_action']}")
        print(f"    {Y}MOE 专家:{END} {step_data['expert']}")
        print(f"    {M}LLM 角色:{END} {step_data['llm_role']}")
        print(f"    {G}结果:{END}    {step_data['result']}")
        print(f"    {C}奖励:{END}    +{reward:.1f} (累计: {total_reward:.1f})")

        # Show team meeting
        if step in [4, 5, 8]:
            print(f"    {R}{BOLD}[会议]{END} 召开 PLANNING 会议协商行动")

        print()

    # Final summary
    print(f"  {BOLD}{G}攻击完成!{END}")
    print(f"  总步数: {len(attack_timeline)}")
    print(f"  总奖励: {total_reward:.1f}")
    print(f"  攻破主机: 3/3")
    print(f"  获取凭据: 3 组")
    print(f"  目标达成: 完全攻破 (Full Compromise)")


# =============================================================================
# Phase 5: Component Interaction Diagram
# =============================================================================
def show_architecture():
    section("PHASE 5: 系统架构图", DIM)

    print(f"""
  {B}决策流程:{END}

  ┌──────────────────────────────────────────────────────────────┐
  │                    当前渗透状态 (PenTestState)                │
  │  主机列表 / 开放端口 / 漏洞 / 凭据 / 权限 / 攻击阶段        │
  └──────────────────────────┬───────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  {M}Expert Router{END}  │ ← 规则引擎 (默认)
                    │  (MOE 路由器)   │ ← LLM 路由 (置信度<0.7时)
                    └────────┬────────┘
                             │ 路由到最佳专家
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐    ┌──────▼──────┐    ┌─────▼─────┐
    │ {B}Recon{END}    │    │ {Y}Vuln/Exploit{END}│    │ {R}PostExp{END}  │
    │ 侦察专家  │    │  漏洞/利用   │    │ 后渗透    │
    └─────┬─────┘    └──────┬──────┘    └─────┬─────┘
          │                  │                  │
          │          ┌──────▼──────┐           │
          │          │ {M}RAG 知识库{END}  │           │
          │          │ CVE/ATT&CK  │           │
          │          │ 经验/工具文档 │           │
          │          └──────┬──────┘           │
          │                 │                  │
          └─────────────────┼──────────────────┘
                            │
                   ┌────────▼────────┐
                   │ {B}Attack Team{END}   │  7人小组投票
                   │ Commander 统筹   │  → 共识决策
                   │ 6 专家提供分析   │  → 行动计划
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │  {G}执行动作{END}     │
                   │  选择最优行动    │
                   │  环境执行并返回  │
                   │  新状态 + 奖励   │
                   └─────────────────┘

  {B}LLM 参与时机:{END}
    1. Expert Router 置信度 < 0.7 时 → LLM 辅助路由
    2. 专家需要知识检索时 → RAG 检索 + LLM 总结
    3. 团队会议多专家意见冲突 → LLM 综合分析
    4. 攻击失败后 → LLM 反思并调整策略
""")


def main():
    api_key = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--api-key="):
                api_key = arg.split("=", 1)[1]

    print(f"\n{R}{BOLD}")
    print("  ╔═══════════════════════════════════════════════════════════╗")
    print("  ║  LLM + MOE Expert + Attack Team 交互演示               ║")
    print("  ║  展示三大核心组件如何协同工作                           ║")
    print(f"  ║  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                        ║")
    print("  ╚═══════════════════════════════════════════════════════════╝")
    print(END)

    # Phase 5: Architecture overview first
    show_architecture()

    # Phase 1: LLM
    llm_config = demo_llm_provider(api_key)

    # Create provider if key available
    llm_provider = None
    if api_key:
        try:
            llm_provider = create_provider(provider="deepseek", api_key=api_key)
        except Exception:
            pass

    # Phase 2: MOE Experts
    router = demo_moe_experts(rag_retriever=None, llm_provider=llm_provider)

    # Phase 3: Attack Team
    team = demo_attack_team(rag_retriever=None, llm_provider=llm_provider)

    # Phase 4: Full Integration
    demo_full_integration()

    # Final
    print(f"\n{G}{BOLD}{'='*65}")
    print("  演示完成!")
    print(f"{'='*65}{END}\n")
    print(f"  组件参与总结:")
    print(f"    {M}LLM{END}       - 情况分析 / 专家路由 / 方案综合 / 反思学习")
    print(f"    {Y}MOE 专家{END}   - 6个专业领域按需调度")
    print(f"    {R}攻击小组{END}   - 7人团队协作，投票决策")
    print(f"    {B}RAG{END}       - 知识检索增强 (CVE/ATT&CK/经验)")
    print(f"\n  提供 API Key 可启用完整 LLM 功能:")
    print(f"    python scripts/demo_llm_moe_team.py --api-key=sk-xxx\n")


if __name__ == "__main__":
    main()
