"""
Attack Team - Multi-Expert Collaborative Attack Group

A team of penetration testing experts coordinated by LLM for collaborative
decision-making, strategy planning, and coordinated attacks.

Features:
- Team coordination with role-based specialists
- Collaborative planning meetings
- Shared memory and knowledge
- Consensus-based decision making
- Task delegation and tracking
"""

import logging
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict

from models.enums import ExpertType
from models.experts.base import ExpertAdvice
from models.expert_router import ExpertRouter, RoutingDecision

logger = logging.getLogger(__name__)


class TeamRole(Enum):
    """Roles within the attack team."""
    LEADER = "leader"              # Team leader - coordinates and makes final decisions
    RECON = "recon"                # Reconnaissance specialist
    VULN_ANALYST = "vuln_analyst"  # Vulnerability analyst
    EXPLOITER = "exploiter"        # Exploitation specialist
    POST_EX = "post_ex"            # Post-exploitation specialist
    CRED_HUNTER = "cred_hunter"    # Credential specialist
    MOVER = "mover"                # Lateral movement specialist


class MeetingType(Enum):
    """Types of team meetings."""
    BRIEFING = "briefing"        # Initial situation briefing
    PLANNING = "planning"        # Attack planning
    REVIEW = "review"            # Progress review
    DEBRIEF = "debrief"          # Post-operation debrief
    EMERGENCY = "emergency"      # Emergency consultation


@dataclass
class TeamMember:
    """A member of the attack team."""
    name: str
    role: TeamRole
    expert_type: ExpertType
    expertise: List[str]
    confidence: float = 0.5
    tasks_completed: int = 0
    success_rate: float = 0.5

    def __hash__(self):
        return hash(self.name)


@dataclass
class TeamTask:
    """A task assigned to a team member."""
    task_id: str
    description: str
    assigned_to: str
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Any = None
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class TeamMemory:
    """Shared memory for the attack team."""
    # Shared knowledge
    discovered_hosts: List[dict] = field(default_factory=list)
    discovered_services: List[dict] = field(default_factory=list)
    discovered_vulnerabilities: List[dict] = field(default_factory=list)
    obtained_credentials: List[dict] = field(default_factory=list)
    compromised_hosts: List[str] = field(default_factory=list)

    # Attack history
    attack_history: List[dict] = field(default_factory=list)

    # Lessons learned
    lessons: List[str] = field(default_factory=list)

    # Team decisions
    decisions: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "discovered_hosts": self.discovered_hosts,
            "discovered_services": self.discovered_services,
            "discovered_vulnerabilities": self.discovered_vulnerabilities,
            "obtained_credentials": self.obtained_credentials,
            "compromised_hosts": self.compromised_hosts,
            "attack_history": self.attack_history,
            "lessons": self.lessons,
            "decisions": self.decisions,
        }

    def update_from_state(self, state: dict) -> None:
        """Update memory from current state."""
        if state.get("hosts"):
            for host in state["hosts"]:
                if host not in self.discovered_hosts:
                    self.discovered_hosts.append(host)

        if state.get("services"):
            for service in state["services"]:
                if service not in self.discovered_services:
                    self.discovered_services.append(service)

        if state.get("vulnerabilities"):
            for vuln in state["vulnerabilities"]:
                if vuln not in self.discovered_vulnerabilities:
                    self.discovered_vulnerabilities.append(vuln)

        if state.get("credentials"):
            for cred in state["credentials"]:
                if cred not in self.obtained_credentials:
                    self.obtained_credentials.append(cred)

        if state.get("compromised_hosts"):
            for host in state["compromised_hosts"]:
                if host not in self.compromised_hosts:
                    self.compromised_hosts.append(host)


@dataclass
class MeetingResult:
    """Result of a team meeting."""
    meeting_type: MeetingType
    participants: List[str]
    discussion: str
    decisions: List[dict]
    action_plan: List[dict]
    consensus_level: float
    timestamp: float = field(default_factory=time.time)


class AttackTeam:
    """
    A coordinated team of penetration testing experts.

    The team works together to:
    - Analyze target environments
    - Plan attack strategies
    - Execute coordinated attacks
    - Share knowledge and findings
    - Learn from successes and failures
    """

    # Default team configuration
    DEFAULT_TEAM = [
        {"role": TeamRole.LEADER, "expert_type": ExpertType.VULNERABILITY, "name": "Commander"},
        {"role": TeamRole.RECON, "expert_type": ExpertType.RECONNAISSANCE, "name": "Scout"},
        {"role": TeamRole.VULN_ANALYST, "expert_type": ExpertType.VULNERABILITY, "name": "Analyst"},
        {"role": TeamRole.EXPLOITER, "expert_type": ExpertType.EXPLOITATION, "name": "Striker"},
        {"role": TeamRole.POST_EX, "expert_type": ExpertType.POST_EXPLOITATION, "name": "Ghost"},
        {"role": TeamRole.CRED_HUNTER, "expert_type": ExpertType.CREDENTIAL, "name": "Hunter"},
        {"role": TeamRole.MOVER, "expert_type": ExpertType.LATERAL_MOVEMENT, "name": "Phantom"},
    ]

    def __init__(
        self,
        llm_provider=None,
        rag_retriever=None,
        team_config: List[dict] = None,
    ):
        self.llm = llm_provider
        self.rag = rag_retriever
        self.router = ExpertRouter(llm_provider, rag_retriever)

        # Team members
        self.members: Dict[str, TeamMember] = {}
        self.expert_instances: Dict[str, Any] = {}

        # Shared memory
        self.memory = TeamMemory()

        # Task tracking
        self.tasks: Dict[str, TeamTask] = {}
        self.task_counter = 0

        # Meeting history
        self.meetings: List[MeetingResult] = []

        # Initialize team
        self._init_team(team_config or self.DEFAULT_TEAM)

    def _init_team(self, config: List[dict]) -> None:
        """Initialize team members from configuration."""
        from models.experts import (
            ReconnaissanceExpert,
            VulnerabilityExpert,
            ExploitationExpert,
            PostExploitationExpert,
            CredentialExpert,
            LateralMovementExpert,
        )

        expert_classes = {
            ExpertType.RECONNAISSANCE: ReconnaissanceExpert,
            ExpertType.VULNERABILITY: VulnerabilityExpert,
            ExpertType.EXPLOITATION: ExploitationExpert,
            ExpertType.POST_EXPLOITATION: PostExploitationExpert,
            ExpertType.CREDENTIAL: CredentialExpert,
            ExpertType.LATERAL_MOVEMENT: LateralMovementExpert,
        }

        for member_config in config:
            role = member_config["role"]
            expert_type = member_config["expert_type"]
            name = member_config["name"]

            # Create member
            member = TeamMember(
                name=name,
                role=role,
                expert_type=expert_type,
                expertise=self._get_expertise(expert_type),
            )
            self.members[name] = member

            # Create expert instance
            expert_class = expert_classes.get(expert_type)
            if expert_class:
                expert = expert_class(self.llm, self.rag)
                self.expert_instances[name] = expert
                self.router.register_expert(expert)

        logger.info(f"Initialized attack team with {len(self.members)} members")

    def _get_expertise(self, expert_type: ExpertType) -> List[str]:
        """Get expertise description for an expert type."""
        expertise_map = {
            ExpertType.RECONNAISSANCE: ["scanning", "enumeration", "osint", "network_mapping"],
            ExpertType.VULNERABILITY: ["vulnerability_analysis", "cve_research", "risk_assessment"],
            ExpertType.EXPLOITATION: ["exploit_development", "payload_generation", "bypass_techniques"],
            ExpertType.POST_EXPLOITATION: ["privilege_escalation", "persistence", "data_exfiltration"],
            ExpertType.CREDENTIAL: ["password_cracking", "hash_attack", "credential_harvesting"],
            ExpertType.LATERAL_MOVEMENT: ["network_pivoting", "remote_execution", "session_hijacking"],
        }
        return expertise_map.get(expert_type, [])

    def hold_meeting(
        self,
        meeting_type: MeetingType,
        state: dict,
        context: dict = None,
        specific_question: str = None,
    ) -> MeetingResult:
        """
        Hold a team meeting to discuss situation and make decisions.

        Args:
            meeting_type: Type of meeting
            state: Current penetration test state
            context: Additional context
            specific_question: Optional specific question to discuss

        Returns:
            MeetingResult with decisions and action plan
        """
        # Update memory
        self.memory.update_from_state(state)

        # Determine participants based on meeting type
        participants = self._select_participants(meeting_type, state)

        # Collect input from each participant
        inputs = {}
        for name in participants:
            member = self.members.get(name)
            expert = self.expert_instances.get(name)

            if member and expert:
                try:
                    advice = expert.analyze(state, context)
                    inputs[name] = {
                        "role": member.role.value,
                        "advice": advice,
                        "confidence": advice.confidence,
                    }
                except Exception as e:
                    logger.warning(f"Expert {name} failed to provide input: {e}")

        # Synthesize discussion
        discussion = self._synthesize_discussion(inputs, meeting_type, state, specific_question)

        # Make decisions
        decisions = self._make_decisions(inputs, meeting_type, state)

        # Create action plan
        action_plan = self._create_action_plan(inputs, decisions)

        # Calculate consensus level
        consensus_level = self._calculate_consensus(inputs)

        # Record meeting
        result = MeetingResult(
            meeting_type=meeting_type,
            participants=participants,
            discussion=discussion,
            decisions=decisions,
            action_plan=action_plan,
            consensus_level=consensus_level,
        )
        self.meetings.append(result)

        # Record decisions in memory
        for decision in decisions:
            self.memory.decisions.append(decision)

        return result

    def _select_participants(self, meeting_type: MeetingType, state: dict) -> List[str]:
        """Select appropriate participants for a meeting."""
        # Leader always participates
        participants = ["Commander"]

        # Add relevant experts based on state and meeting type
        phase = state.get("phase", "").lower()

        if meeting_type == MeetingType.BRIEFING:
            # Everyone joins briefing
            participants = list(self.members.keys())

        elif meeting_type == MeetingType.PLANNING:
            # Include experts relevant to current phase
            if phase in ["reconnaissance", "scanning"]:
                participants.extend(["Scout", "Analyst"])
            elif phase in ["exploitation"]:
                participants.extend(["Striker", "Analyst"])
            elif phase in ["post_exploitation"]:
                participants.extend(["Ghost", "Hunter"])
            elif phase in ["lateral_movement"]:
                participants.extend(["Phantom", "Hunter"])

        elif meeting_type == MeetingType.REVIEW:
            # Include active experts
            if state.get("vulnerabilities"):
                participants.append("Analyst")
            if state.get("credentials"):
                participants.append("Hunter")
            if state.get("compromised_hosts"):
                participants.append("Ghost")

        elif meeting_type == MeetingType.EMERGENCY:
            # All hands on deck
            participants = list(self.members.keys())

        else:
            # Default: include most relevant
            decision = self.router.analyze_situation(state)
            for name, member in self.members.items():
                if member.expert_type == decision.primary_expert:
                    participants.append(name)
                    break

        return list(set(participants))

    def _synthesize_discussion(
        self,
        inputs: Dict[str, dict],
        meeting_type: MeetingType,
        state: dict,
        specific_question: str = None,
    ) -> str:
        """Synthesize team discussion into a summary."""
        if not inputs:
            return "No expert input available."

        discussion_parts = [f"=== 团队会议: {meeting_type.value} ===\n"]

        if specific_question:
            discussion_parts.append(f"讨论议题: {specific_question}\n")

        discussion_parts.append("\n各专家意见:\n")

        for name, data in inputs.items():
            role = data["role"]
            advice = data["advice"]
            confidence = data["confidence"]

            discussion_parts.append(f"\n[{name}] ({role}):\n")
            discussion_parts.append(f"  总结: {advice.summary}\n")
            discussion_parts.append(f"  置信度: {confidence:.2f}\n")

            if advice.tools_to_use:
                discussion_parts.append(f"  建议工具: {', '.join(advice.tools_to_use)}\n")

            if advice.warnings:
                discussion_parts.append(f"  警告: {'; '.join(advice.warnings)}\n")

        # LLM synthesis if available
        if self.llm and len(inputs) > 1:
            synthesis = self._get_llm_synthesis(inputs, state, specific_question)
            if synthesis:
                discussion_parts.append(f"\n=== 综合分析 ===\n{synthesis}\n")

        return "".join(discussion_parts)

    def _get_llm_synthesis(self, inputs: Dict[str, dict], state: dict, question: str = None) -> str:
        """Get LLM synthesis of team opinions."""
        try:
            # Build prompt
            prompt = """作为渗透测试团队协调员，请综合以下专家意见并给出建议。

专家意见:
"""
            for name, data in inputs.items():
                advice = data["advice"]
                prompt += f"\n{name} ({data['role']}):\n"
                prompt += f"- 总结: {advice.summary}\n"
                prompt += f"- 推荐行动数: {len(advice.recommended_actions)}\n"
                prompt += f"- 置信度: {data['confidence']:.2f}\n"

            if question:
                prompt += f"\n具体问题: {question}\n"

            prompt += "\n请给出综合建议（200字以内）:"

            response = self.llm.call([{"role": "user", "content": prompt}])
            if response is None:
                logger.warning("LLM synthesis returned None")
                return ""
            content = getattr(response, 'content', None)
            if content is None:
                logger.warning("LLM response has no content attribute")
                return ""
            return content[:500]

        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return ""

    def _make_decisions(
        self,
        inputs: Dict[str, dict],
        meeting_type: MeetingType,
        state: dict,
    ) -> List[dict]:
        """Make team decisions based on expert inputs."""
        decisions = []

        if not inputs:
            return decisions

        # Aggregate recommended actions
        action_votes: Dict[str, int] = defaultdict(int)
        action_details: Dict[str, dict] = {}

        for name, data in inputs.items():
            advice = data["advice"]
            for action in advice.recommended_actions[:3]:  # Top 3 per expert
                action_key = f"{action.get('type', 'unknown')}:{action.get('tool', '')}"
                action_votes[action_key] += 1
                if action_key not in action_details:
                    action_details[action_key] = action
                    action_details[action_key]["supporters"] = []
                action_details[action_key]["supporters"].append(name)

        # Select actions with most votes
        sorted_actions = sorted(action_votes.items(), key=lambda x: x[1], reverse=True)

        for action_key, votes in sorted_actions[:5]:  # Top 5 actions
            action = action_details[action_key]
            decisions.append({
                "action": action_key,
                "type": action.get("type"),
                "tool": action.get("tool"),
                "description": action.get("description", ""),
                "supporters": action["supporters"],
                "votes": votes,
                "params": action.get("params", {}),
                "priority": 5 - sorted_actions.index((action_key, votes)),  # Higher priority for more votes
            })

        # Record high-priority decision
        if decisions:
            self.memory.decisions.append({
                "meeting_type": meeting_type.value,
                "top_decision": decisions[0]["action"],
                "votes": decisions[0]["votes"],
                "timestamp": time.time(),
            })

        return decisions

    def _create_action_plan(self, inputs: Dict[str, dict], decisions: List[dict]) -> List[dict]:
        """Create a prioritized action plan."""
        action_plan = []

        for decision in decisions:
            action_plan.append({
                "action": decision["action"],
                "type": decision.get("type"),
                "tool": decision.get("tool"),
                "description": decision.get("description"),
                "params": decision.get("params", {}),
                "priority": decision.get("priority", 1),
                "assigned_to": decision.get("supporters", ["Unknown"])[0],
            })

        return action_plan

    def _calculate_consensus(self, inputs: Dict[str, dict]) -> float:
        """Calculate consensus level among team members."""
        if len(inputs) <= 1:
            return 1.0

        # Calculate based on tool overlap
        tools = [set(data["advice"].tools_to_use) for data in inputs.values() if data["advice"].tools_to_use]

        if not tools:
            return 0.5

        # Jaccard similarity average
        similarities = []
        for i, tools_i in enumerate(tools):
            for tools_j in tools[i+1:]:
                intersection = len(tools_i & tools_j)
                union = len(tools_i | tools_j)
                if union > 0:
                    similarities.append(intersection / union)

        return sum(similarities) / len(similarities) if similarities else 0.5

    def assign_task(self, description: str, assigned_to: str, priority: int = 1) -> TeamTask:
        """Assign a task to a team member."""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"

        task = TeamTask(
            task_id=task_id,
            description=description,
            assigned_to=assigned_to,
            priority=priority,
        )
        self.tasks[task_id] = task

        logger.info(f"Task {task_id} assigned to {assigned_to}: {description}")
        return task

    def complete_task(self, task_id: str, result: Any, success: bool = True) -> None:
        """Mark a task as completed."""
        task = self.tasks.get(task_id)
        if task:
            task.status = "completed" if success else "failed"
            task.result = result
            task.completed_at = time.time()

            # Update member stats
            member = self.members.get(task.assigned_to)
            if member:
                member.tasks_completed += 1
                if success:
                    member.success_rate = (
                        (member.success_rate * (member.tasks_completed - 1) + 1.0)
                        / member.tasks_completed
                    )
                else:
                    member.success_rate = (
                        (member.success_rate * (member.tasks_completed - 1))
                        / member.tasks_completed
                    )

            # Record in memory
            self.memory.attack_history.append({
                "task_id": task_id,
                "description": task.description,
                "assigned_to": task.assigned_to,
                "success": success,
                "result": result,
            })

    def get_pending_tasks(self) -> List[TeamTask]:
        """Get all pending tasks."""
        return [t for t in self.tasks.values() if t.status == "pending"]

    def get_next_action(self, state: dict) -> Optional[dict]:
        """
        Get the next recommended action based on team consensus.

        Args:
            state: Current state

        Returns:
            Recommended action dict or None
        """
        # Hold a quick planning meeting
        result = self.hold_meeting(MeetingType.PLANNING, state)

        if result.action_plan:
            return result.action_plan[0]

        return None

    def brief_team(self, target: str, initial_info: dict = None) -> MeetingResult:
        """
        Brief the team on a new target.

        Args:
            target: Target identifier
            initial_info: Any initial reconnaissance data

        Returns:
            MeetingResult with initial plan
        """
        state = {
            "target": target,
            "phase": "reconnaissance",
            **(initial_info or {}),
        }

        return self.hold_meeting(MeetingType.BRIEFING, state)

    def debrief(self, state: dict, outcomes: List[dict]) -> MeetingResult:
        """
        Debrief the team after an operation.

        Args:
            state: Final state
            outcomes: List of action outcomes

        Returns:
            MeetingResult with lessons learned
        """
        # Extract lessons from outcomes
        lessons = []
        for outcome in outcomes:
            if outcome.get("success"):
                lessons.append(f"成功: {outcome.get('action', 'unknown')} - {outcome.get('reason', '')}")
            else:
                lessons.append(f"失败: {outcome.get('action', 'unknown')} - {outcome.get('error', '')}")

        for lesson in lessons:
            if lesson not in self.memory.lessons:
                self.memory.lessons.append(lesson)

        return self.hold_meeting(MeetingType.DEBRIEF, state)

    def emergency_consult(self, state: dict, problem: str) -> MeetingResult:
        """
        Emergency consultation when something goes wrong.

        Args:
            state: Current state
            problem: Description of the problem

        Returns:
            MeetingResult with recommendations
        """
        return self.hold_meeting(
            MeetingType.EMERGENCY,
            state,
            specific_question=problem,
        )

    def get_team_status(self) -> dict:
        """Get current team status."""
        return {
            "members": {
                name: {
                    "role": member.role.value,
                    "confidence": member.confidence,
                    "tasks_completed": member.tasks_completed,
                    "success_rate": member.success_rate,
                }
                for name, member in self.members.items()
            },
            "memory_summary": {
                "hosts_discovered": len(self.memory.discovered_hosts),
                "vulnerabilities_found": len(self.memory.discovered_vulnerabilities),
                "credentials_obtained": len(self.memory.obtained_credentials),
                "hosts_compromised": len(self.memory.compromised_hosts),
                "lessons_learned": len(self.memory.lessons),
            },
            "pending_tasks": len(self.get_pending_tasks()),
            "meetings_held": len(self.meetings),
        }


def create_attack_team(llm_provider=None, rag_retriever=None) -> AttackTeam:
    """Create an attack team with default configuration."""
    return AttackTeam(llm_provider=llm_provider, rag_retriever=rag_retriever)
