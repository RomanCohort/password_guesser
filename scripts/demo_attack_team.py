#!/usr/bin/env python
"""
Attack Team Demo

Demonstrates the multi-expert attack team system for collaborative
penetration testing decision-making.

Usage:
    python scripts/demo_attack_team.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo_team_briefing():
    """Demonstrate team briefing."""
    from models.attack_team import create_attack_team, MeetingType

    print("=" * 60)
    print("攻击小组系统演示")
    print("=" * 60)

    # Create attack team
    team = create_attack_team()

    print(f"\n团队成员 ({len(team.members)} 人):")
    for name, member in team.members.items():
        print(f"  - {name}: {member.role.value} (专长: {', '.join(member.expertise[:2])})")

    # Brief the team on a new target
    print("\n" + "=" * 60)
    print("场景: 目标侦察简报")
    print("=" * 60)

    target = "192.168.1.100"
    initial_info = {
        "services": ["ssh:22", "http:80", "https:443", "smb:445"],
        "os_hint": "Windows Server",
    }

    result = team.brief_team(target, initial_info)

    print(f"\n目标: {target}")
    print(f"会议类型: {result.meeting_type.value}")
    print(f"参与成员: {', '.join(result.participants)}")
    print(f"\n讨论摘要:\n{result.discussion[:500]}...")
    print(f"\n共识水平: {result.consensus_level:.2f}")

    print("\n行动计划:")
    for i, action in enumerate(result.action_plan[:3], 1):
        print(f"  {i}. {action.get('description', action.get('action'))} (优先级: {action.get('priority', 1)})")


def demo_team_planning():
    """Demonstrate team planning meeting."""
    from models.attack_team import create_attack_team, MeetingType

    print("\n" + "=" * 60)
    print("场景: 攻击计划会议")
    print("=" * 60)

    team = create_attack_team()

    # Simulate discovered vulnerabilities
    state = {
        "target": "192.168.1.100",
        "services": ["ssh", "http", "smb"],
        "vulnerabilities": [
            {"id": "CVE-2021-44228", "severity": "critical", "service": "http"},
            {"id": "CVE-2020-1472", "severity": "critical", "service": "smb"},
        ],
        "credentials": [],
        "has_shell": False,
        "is_admin": False,
        "compromised_hosts": [],
    }

    result = team.hold_meeting(MeetingType.PLANNING, state)

    print(f"\n发现漏洞: CVE-2021-44228 (Log4j), CVE-2020-1472 (Zerologon)")
    print(f"\n团队决策:")
    for decision in result.decisions[:3]:
        print(f"  - {decision['action']}: {decision['description']}")
        print(f"    支持者: {', '.join(decision['supporters'])} ({decision['votes']} 票)")

    print(f"\n执行计划:")
    for i, action in enumerate(result.action_plan[:5], 1):
        tool = action.get('tool', 'N/A')
        desc = action.get('description', '')
        print(f"  {i}. [{tool}] {desc}")


def demo_emergency_consult():
    """Demonstrate emergency consultation."""
    from models.attack_team import create_attack_team

    print("\n" + "=" * 60)
    print("场景: 紧急咨询")
    print("=" * 60)

    team = create_attack_team()

    state = {
        "target": "192.168.1.100",
        "has_shell": True,
        "is_admin": False,
        "compromised_hosts": ["192.168.1.100"],
    }

    problem = "获得的shell权限很低，提权尝试失败，IDS已触发告警"

    result = team.emergency_consult(state, problem)

    print(f"\n问题: {problem}")
    print(f"\n紧急会议参与者: {', '.join(result.participants)}")
    print(f"\n建议行动:")
    for action in result.action_plan[:3]:
        print(f"  - {action.get('description', action.get('action'))}")


def demo_task_tracking():
    """Demonstrate task assignment and tracking."""
    from models.attack_team import create_attack_team

    print("\n" + "=" * 60)
    print("场景: 任务分配与跟踪")
    print("=" * 60)

    team = create_attack_team()

    # Assign tasks
    task1 = team.assign_task("扫描目标端口", "Scout", priority=1)
    task2 = team.assign_task("分析发现的漏洞", "Analyst", priority=2)
    task3 = team.assign_task("尝试利用CVE-2021-44228", "Striker", priority=1)

    print(f"\n已分配任务:")
    for task_id, task in team.tasks.items():
        print(f"  - [{task.assigned_to}] {task.description} (状态: {task.status})")

    # Complete tasks
    team.complete_task(task1.task_id, {"ports": [22, 80, 443, 445]}, success=True)
    team.complete_task(task2.task_id, {"vulns": ["CVE-2021-44228"]}, success=True)
    team.complete_task(task3.task_id, {"error": "Connection refused"}, success=False)

    print(f"\n任务完成状态:")
    for task_id, task in team.tasks.items():
        status = "[OK]" if task.status == "completed" and task.result else "[FAIL]"
        print(f"  {status} [{task.assigned_to}] {task.description}")

    print(f"\n团队状态:")
    status = team.get_team_status()
    for name, member_status in status["members"].items():
        success_rate = member_status["success_rate"]
        tasks = member_status["tasks_completed"]
        print(f"  - {name}: {tasks} 任务, 成功率 {success_rate:.1%}")


def demo_with_orchestrator():
    """Demonstrate using attack team with orchestrator."""
    print("\n" + "=" * 60)
    print("场景: 与编排器集成")
    print("=" * 60)

    from pentest.orchestrator import PenTestOrchestrator, PenTestConfig

    config = PenTestConfig(
        max_steps=3,
        enable_attack_team=True,
        enable_self_improvement=False,
    )

    orch = PenTestOrchestrator(config)

    print(f"\n攻击小组已启用: {orch.attack_team is not None}")
    print(f"团队成员数: {len(orch.attack_team.members)}")

    # Team consultation
    result = orch.team_consult("目标开启了WAF，如何绕过")
    print(f"\n咨询结果:")
    print(f"  决策数: {len(result.get('decisions', []))}")

    # Team briefing
    briefing = orch.team_briefing("target.example.com", {"os": "Linux"})
    print(f"\n简报完成:")
    print(f"  参与者: {len(briefing.get('participants', []))} 人")
    print(f"  共识: {briefing.get('consensus_level', 0):.2f}")


def main():
    """Run all demos."""
    demo_team_briefing()
    demo_team_planning()
    demo_emergency_consult()
    demo_task_tracking()
    demo_with_orchestrator()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
