"""
Curriculum Learning Generator

Generates training scenarios from historical experiences to focus learning
on areas where the agent has struggled or can improve.
"""

import json
import os
import time
import random
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrainingScenario:
    """A training scenario for the RL agent."""
    scenario_id: str
    name: str
    description: str
    difficulty: float  # 0.0 - 1.0
    target_config: dict  # Host configuration for environment
    learning_objectives: List[str]  # What this scenario teaches
    prerequisite_scenarios: List[str]  # Must complete these first
    created_from: str  # 'failure_pattern', 'success_pattern', 'manual'
    attempts: int = 0
    successes: int = 0
    avg_reward: float = 0.0
    mastery_level: float = 0.0  # 0.0 - 1.0
    created_at: float = field(default_factory=time.time)
    last_attempted: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.attempts)

    def update_mastery(self) -> None:
        """Update mastery level based on recent performance."""
        # Mastery increases with success rate and attempts
        if self.attempts >= 3:
            self.mastery_level = min(1.0, self.success_rate * (1 + self.attempts * 0.1))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingScenario':
        return cls(**data)


@dataclass
class MasteryTracker:
    """Tracks mastery across different skill areas."""
    skill_areas: Dict[str, float] = field(default_factory=lambda: {
        "network_scan": 0.0,
        "port_scan": 0.0,
        "exploitation": 0.0,
        "lateral_movement": 0.0,
        "privilege_escalation": 0.0,
        "credential_harvesting": 0.0,
        "persistence": 0.0,
        "evasion": 0.0,
    })

    def update(self, skill: str, reward: float, success: bool) -> None:
        """Update mastery for a skill."""
        if skill in self.skill_areas:
            delta = 0.1 if success else -0.05
            self.skill_areas[skill] = max(0.0, min(1.0, self.skill_areas[skill] + delta))

    def get_weak_areas(self, threshold: float = 0.4) -> List[str]:
        """Get skills below mastery threshold."""
        return [k for k, v in self.skill_areas.items() if v < threshold]

    def get_strong_areas(self, threshold: float = 0.7) -> List[str]:
        """Get skills above mastery threshold."""
        return [k for k, v in self.skill_areas.items() if v >= threshold]


class CurriculumGenerator:
    """
    Generates and manages training curriculum for the RL agent.

    The curriculum:
    - Analyzes failure patterns from experience store
    - Creates targeted scenarios to address weaknesses
    - Tracks mastery progression
    - Adjusts difficulty based on performance
    """

    # Template configurations for different scenario types
    SCENARIO_TEMPLATES = {
        "basic_scan": {
            "name": "Basic Network Scanning",
            "difficulty": 0.2,
            "hosts": [{"ip": "192.168.1.1", "ports": [22, 80, 443]}],
            "objectives": ["network_scan", "port_scan"],
        },
        "single_exploit": {
            "name": "Single Host Exploitation",
            "difficulty": 0.4,
            "hosts": [{"ip": "192.168.1.10", "ports": [22, 80], "vulns": ["CVE-2021-44228"]}],
            "objectives": ["exploitation"],
        },
        "lateral_movement": {
            "name": "Lateral Movement Exercise",
            "difficulty": 0.6,
            "hosts": [
                {"ip": "192.168.1.10", "ports": [22], "compromised": True},
                {"ip": "192.168.1.20", "ports": [445, 3389]},
            ],
            "objectives": ["lateral_movement", "credential_harvesting"],
        },
        "privilege_escalation": {
            "name": "Privilege Escalation Challenge",
            "difficulty": 0.5,
            "hosts": [{"ip": "10.0.0.5", "ports": [22], "has_shell": True, "priv": "user"}],
            "objectives": ["privilege_escalation"],
        },
        "full_chain": {
            "name": "Full Attack Chain",
            "difficulty": 0.8,
            "hosts": [
                {"ip": "192.168.1.0/24", "scan_target": True},
                {"ip": "192.168.1.100", "ports": [80, 443], "vulns": ["CVE-2021-41773"]},
                {"ip": "192.168.1.200", "ports": [445], "requires_creds": True},
            ],
            "objectives": ["network_scan", "exploitation", "lateral_movement", "privilege_escalation"],
        },
        "adversarial": {
            "name": "Adversarial Scenario",
            "difficulty": 0.9,
            "hosts": [],  # Dynamically generated
            "objectives": ["all"],
        },
    }

    def __init__(self, experience_store=None, path: str = "data/curriculum/scenarios.json"):
        self.experience_store = experience_store
        self.path = path
        self.scenarios: Dict[str, TrainingScenario] = {}
        self.mastery = MasteryTracker()
        self.curriculum_history: List[dict] = []

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Initialize with base scenarios
        self._init_base_scenarios()

    def _init_base_scenarios(self) -> None:
        """Create base scenarios from templates."""
        for template_id, template in self.SCENARIO_TEMPLATES.items():
            scenario_id = f"base_{template_id}"
            if scenario_id not in self.scenarios:
                scenario = TrainingScenario(
                    scenario_id=scenario_id,
                    name=template["name"],
                    description=f"Base training scenario for {template['name']}",
                    difficulty=template["difficulty"],
                    target_config={"hosts": template["hosts"]},
                    learning_objectives=template["objectives"],
                    prerequisite_scenarios=[],
                    created_from="manual",
                )
                self.scenarios[scenario_id] = scenario

    def generate_from_failures(self, failure_experiences: List[dict] = None) -> List[TrainingScenario]:
        """
        Generate scenarios targeting failure patterns.

        Args:
            failure_experiences: List of failure experiences to analyze

        Returns:
            List of new scenarios
        """
        if failure_experiences is None and self.experience_store:
            failures = self.experience_store.get_failed_episodes(limit=100)
            failure_experiences = [f.to_dict() for f in failures]

        if not failure_experiences:
            return []

        new_scenarios = []

        # Analyze failure patterns
        failure_by_action = {}
        for exp in failure_experiences:
            action_idx = exp.get("action_index", 0)
            if action_idx not in failure_by_action:
                failure_by_action[action_idx] = []
            failure_by_action[action_idx].append(exp)

        # Create targeted scenarios for high-failure actions
        for action_idx, failures in sorted(failure_by_action.items(), key=lambda x: -len(x[1])):
            if len(failures) < 3:  # Need minimum failures to create scenario
                continue

            scenario_id = f"failure_targeted_{action_idx}_{int(time.time())}"
            avg_reward = sum(f.get("reward", 0) for f in failures) / len(failures)

            # Determine objective from action
            objective = self._action_to_skill(action_idx)

            scenario = TrainingScenario(
                scenario_id=scenario_id,
                name=f"Targeted: {objective.replace('_', ' ').title()}",
                description=f"Scenario generated from {len(failures)} failures in {objective}",
                difficulty=min(0.9, 0.5 + abs(avg_reward) * 0.3),
                target_config=self._generate_config_for_skill(objective, failures),
                learning_objectives=[objective],
                prerequisite_scenarios=self._get_prerequisites(objective),
                created_from="failure_pattern",
                avg_reward=avg_reward,
            )

            self.scenarios[scenario_id] = scenario
            new_scenarios.append(scenario)

        if new_scenarios:
            logger.info(f"Generated {len(new_scenarios)} failure-targeted scenarios")
        return new_scenarios

    def generate_from_successes(self, success_experiences: List[dict] = None) -> List[TrainingScenario]:
        """
        Generate scenarios to reinforce successful patterns.

        Args:
            success_experiences: List of successful experiences

        Returns:
            List of new scenarios
        """
        if success_experiences is None and self.experience_store:
            successes = self.experience_store.get_successful_episodes(limit=50)
            success_experiences = [s.to_dict() for s in successes]

        if not success_experiences:
            return []

        new_scenarios = []

        # Find successful attack chains
        chains = self._extract_attack_chains(success_experiences)

        for i, chain in enumerate(chains[:5]):  # Limit to top 5
            scenario_id = f"success_chain_{i}_{int(time.time())}"

            scenario = TrainingScenario(
                scenario_id=scenario_id,
                name=f"Success Pattern {i+1}",
                description=f"Reinforce successful attack chain: {' -> '.join(chain['actions'][:3])}",
                difficulty=0.4 + len(chain['actions']) * 0.1,
                target_config=chain.get('config', {}),
                learning_objectives=list(set(chain.get('objectives', []))),
                prerequisite_scenarios=[],
                created_from="success_pattern",
            )

            self.scenarios[scenario_id] = scenario
            new_scenarios.append(scenario)

        if new_scenarios:
            logger.info(f"Generated {len(new_scenarios)} success-pattern scenarios")
        return new_scenarios

    def create_adversarial_scenario(self, weak_areas: List[str] = None) -> TrainingScenario:
        """
        Create an adversarial scenario targeting weak areas.

        Args:
            weak_areas: Skills to target (from mastery tracker)

        Returns:
            New adversarial scenario
        """
        if weak_areas is None:
            weak_areas = self.mastery.get_weak_areas()

        if not weak_areas:
            weak_areas = ["exploitation", "privilege_escalation"]

        scenario_id = f"adversarial_{int(time.time())}"

        # Generate challenging configuration
        hosts = self._generate_challenging_hosts(weak_areas)

        scenario = TrainingScenario(
            scenario_id=scenario_id,
            name="Adversarial Challenge",
            description=f"Challenging scenario targeting weak areas: {', '.join(weak_areas)}",
            difficulty=0.9,
            target_config={"hosts": hosts, "adversarial": True},
            learning_objectives=weak_areas,
            prerequisite_scenarios=[],
            created_from="failure_pattern",
        )

        self.scenarios[scenario_id] = scenario
        logger.info(f"Created adversarial scenario targeting: {weak_areas}")
        return scenario

    def assess_mastery(self, scenario_id: str, recent_episodes: List[dict]) -> float:
        """
        Assess mastery level for a scenario.

        Args:
            scenario_id: Scenario to assess
            recent_episodes: Recent training episodes

        Returns:
            Mastery level (0-1)
        """
        if scenario_id not in self.scenarios:
            return 0.0

        scenario = self.scenarios[scenario_id]
        scenario.update_mastery()

        return scenario.mastery_level

    def get_next_scenario(self, current_mastery: float = None) -> TrainingScenario:
        """
        Get the next appropriate scenario based on current mastery.

        Args:
            current_mastery: Overall mastery level

        Returns:
            Best next scenario
        """
        if current_mastery is None:
            current_mastery = sum(self.mastery.skill_areas.values()) / len(self.mastery.skill_areas)

        # Find scenarios with appropriate difficulty
        target_difficulty = min(0.9, max(0.3, current_mastery + 0.1))

        candidates = []
        for scenario in self.scenarios.values():
            # Check prerequisites
            prereqs_met = all(
                self.scenarios.get(p, TrainingScenario("", "", "", 0, {}, [], [])).mastery_level >= 0.5
                for p in scenario.prerequisite_scenarios
                if p in self.scenarios
            )

            if not prereqs_met:
                continue

            # Check difficulty match
            diff_distance = abs(scenario.difficulty - target_difficulty)
            candidates.append((scenario, diff_distance))

        if candidates:
            # Sort by difficulty match and return best
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        # Fallback to basic scenario
        return self.scenarios.get("base_basic_scan", list(self.scenarios.values())[0])

    def record_attempt(self, scenario_id: str, reward: float, success: bool) -> None:
        """Record an attempt at a scenario."""
        if scenario_id in self.scenarios:
            scenario = self.scenarios[scenario_id]
            scenario.attempts += 1
            scenario.avg_reward = (scenario.avg_reward * (scenario.attempts - 1) + reward) / scenario.attempts
            if success:
                scenario.successes += 1
            scenario.last_attempted = time.time()
            scenario.update_mastery()

            # Update mastery tracker
            for objective in scenario.learning_objectives:
                self.mastery.update(objective, reward, success)

        self.curriculum_history.append({
            "scenario_id": scenario_id,
            "reward": reward,
            "success": success,
            "timestamp": time.time(),
        })

    def get_curriculum_progress(self) -> dict:
        """Get overall curriculum progress."""
        total = len(self.scenarios)
        mastered = sum(1 for s in self.scenarios.values() if s.mastery_level >= 0.7)
        in_progress = sum(1 for s in self.scenarios.values() if 0.3 <= s.mastery_level < 0.7)

        return {
            "total_scenarios": total,
            "mastered": mastered,
            "in_progress": in_progress,
            "not_started": total - mastered - in_progress,
            "overall_mastery": sum(self.mastery.skill_areas.values()) / len(self.mastery.skill_areas),
            "weak_areas": self.mastery.get_weak_areas(),
            "strong_areas": self.mastery.get_strong_areas(),
        }

    def save(self) -> None:
        """Save curriculum state."""
        data = {
            "scenarios": {sid: s.to_dict() for sid, s in self.scenarios.items()},
            "mastery": self.mastery.skill_areas,
            "history": self.curriculum_history[-100:],  # Keep last 100
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load curriculum state."""
        if not os.path.exists(self.path):
            return

        with open(self.path, 'r') as f:
            data = json.load(f)

        self.scenarios = {
            sid: TrainingScenario.from_dict(s)
            for sid, s in data.get("scenarios", {}).items()
        }
        self.mastery.skill_areas = data.get("mastery", self.mastery.skill_areas)
        self.curriculum_history = data.get("history", [])

    def _action_to_skill(self, action_index: int) -> str:
        """Map action index to skill area."""
        skill_map = {
            0: "network_scan",
            1: "port_scan",
            2: "port_scan",
            3: "exploitation",
            4: "credential_harvesting",
            5: "lateral_movement",
            6: "privilege_escalation",
            7: "credential_harvesting",
            8: "persistence",
        }
        return skill_map.get(action_index % 9, "exploitation")

    def _generate_config_for_skill(self, skill: str, failures: List[dict]) -> dict:
        """Generate target config for a specific skill."""
        base_config = {
            "network_scan": {"hosts": [{"ip": "10.0.0.0/24", "scan_target": True}]},
            "port_scan": {"hosts": [{"ip": "192.168.1.10", "ports": "range(1-1000)"}]},
            "exploitation": {"hosts": [{"ip": "192.168.1.10", "ports": [80, 443], "vulns": ["unknown"]}]},
            "lateral_movement": {
                "hosts": [
                    {"ip": "192.168.1.10", "compromised": True},
                    {"ip": "192.168.1.20", "ports": [445]},
                ]
            },
            "privilege_escalation": {"hosts": [{"ip": "192.168.1.10", "has_shell": True}]},
            "credential_harvesting": {"hosts": [{"ip": "192.168.1.10", "compromised": True, "harvest": True}]},
        }
        return base_config.get(skill, {"hosts": [{"ip": "192.168.1.10"}]})

    def _get_prerequisites(self, skill: str) -> List[str]:
        """Get prerequisite scenarios for a skill."""
        prereq_map = {
            "exploitation": ["base_basic_scan", "base_single_exploit"],
            "lateral_movement": ["base_single_exploit", "base_lateral_movement"],
            "privilege_escalation": ["base_single_exploit", "base_privilege_escalation"],
            "credential_harvesting": ["base_single_exploit"],
            "persistence": ["base_privilege_escalation"],
        }
        return prereq_map.get(skill, [])

    def _extract_attack_chains(self, experiences: List[dict]) -> List[dict]:
        """Extract successful attack chains from experiences."""
        chains = []
        session_exps = {}

        for exp in experiences:
            sid = exp.get("session_id", "unknown")
            if sid not in session_exps:
                session_exps[sid] = []
            session_exps[sid].append(exp)

        for sid, exps in session_exps.items():
            if any(e.get("success") for e in exps):
                chains.append({
                    "session_id": sid,
                    "actions": [str(e.get("action_index")) for e in exps],
                    "objectives": ["exploitation", "lateral_movement"],
                    "config": {},
                })

        return chains[:5]

    def _generate_challenging_hosts(self, weak_areas: List[str]) -> List[dict]:
        """Generate challenging host configurations for weak areas."""
        hosts = []

        if "network_scan" in weak_areas or "port_scan" in weak_areas:
            hosts.append({"ip": "10.10.10.0/24", "scan_target": True, "firewall": True})

        if "exploitation" in weak_areas:
            hosts.append({
                "ip": "10.10.10.100",
                "ports": [80, 443, 8080],
                "vulns": ["CVE-2021-44228"],  # Log4Shell
                "patched_common": True,  # Common exploits patched
            })

        if "lateral_movement" in weak_areas:
            hosts.extend([
                {"ip": "10.10.10.10", "compromised": True, "limited_access": True},
                {"ip": "10.10.10.20", "ports": [445], "lateral_blocked": True},
            ])

        if "privilege_escalation" in weak_areas:
            hosts.append({
                "ip": "10.10.10.50",
                "has_shell": True,
                "priv": "user",
                "sudo_logged": True,  # Sudo commands logged
                "no_kernel_exploits": True,  # Kernel hardened
            })

        return hosts if hosts else [{"ip": "10.10.10.1", "ports": [22, 80]}]


# Global instance
_global_curriculum: Optional[CurriculumGenerator] = None


def get_curriculum() -> CurriculumGenerator:
    """Get or create global curriculum generator."""
    global _global_curriculum
    if _global_curriculum is None:
        _global_curriculum = CurriculumGenerator()
        _global_curriculum.load()
    return _global_curriculum
