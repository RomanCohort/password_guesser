"""
Adaptive Reward Shaper

Automatically adjusts reward signals based on accumulated experience and lessons.
Uses learned rules to provide shaped rewards that guide the agent toward better strategies.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RewardShapingRule:
    """A rule that adjusts rewards based on state/action context."""
    rule_id: str
    name: str
    description: str
    condition_type: str  # 'action_type', 'target_service', 'vuln_severity', 'state_progress', 'custom'
    condition_params: dict
    bonus: float  # Reward adjustment
    decay: float  # Decay factor per application (0-1, 1 = no decay)
    current_decay: float = 1.0  # Current decay multiplier
    applications: int = 0
    successes: int = 0
    effectiveness: float = 0.5  # Running estimate of rule effectiveness
    created_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.applications)

    def apply(self) -> float:
        """Get the current adjusted bonus."""
        return self.bonus * self.current_decay

    def record_result(self, was_successful: bool) -> None:
        """Record application result and update effectiveness."""
        self.applications += 1
        if was_successful:
            self.successes += 1

        # Update effectiveness with exponential moving average
        self.effectiveness = 0.7 * self.effectiveness + 0.3 * (1.0 if was_successful else 0.0)

        # Apply decay
        self.current_decay *= self.decay


class AdaptiveRewardShaper:
    """
    Adaptive reward shaping based on accumulated lessons and experience.

    The shaper maintains a set of rules that modify the base reward signal.
    Rules are:
    - Created from lessons learned
    - Updated based on their effectiveness
    - Pruned if consistently ineffective
    """

    def __init__(self, lessons_db=None):
        self.lessons_db = lessons_db
        self.rules: Dict[str, RewardShapingRule] = {}
        self.shaping_history: List[dict] = []

        # Initialize with default heuristic rules
        self._init_default_rules()

    def _init_default_rules(self) -> None:
        """Initialize with common-sense reward shaping rules."""
        defaults = [
            RewardShapingRule(
                rule_id="exploit_high_severity",
                name="Prioritize High Severity Exploits",
                description="Give bonus for exploiting high severity vulnerabilities",
                condition_type="vuln_severity",
                condition_params={"min_severity": 8.0},
                bonus=0.3,
                decay=0.99,
            ),
            RewardShapingRule(
                rule_id="avoid_repeated_failures",
                name="Penalize Repeated Failures",
                description="Penalize attempting the same failed action repeatedly",
                condition_type="action_type",
                condition_params={"action": "exploit_vuln", "max_retries": 3},
                bonus=-0.2,
                decay=0.95,
            ),
            RewardShapingRule(
                rule_id="credential_reuse",
                name="Reward Credential Reuse",
                description="Bonus for using found credentials on new hosts",
                condition_type="state_progress",
                condition_params={"uses_credentials": True},
                bonus=0.4,
                decay=0.98,
            ),
            RewardShapingRule(
                rule_id="scan_before_exploit",
                name="Scan Before Exploit",
                description="Bonus for scanning before attempting exploitation",
                condition_type="action_sequence",
                condition_params={"requires_prior": "scan_port"},
                bonus=0.2,
                decay=0.99,
            ),
            RewardShapingRule(
                rule_id="lateral_after_compromise",
                name="Lateral Movement After Compromise",
                description="Bonus for lateral movement after gaining initial access",
                condition_type="state_progress",
                condition_params={"has_compromised_host": True, "action": "lateral_move"},
                bonus=0.3,
                decay=0.98,
            ),
            RewardShapingRule(
                rule_id="priv_escalation_bonus",
                name="Privilege Escalation Bonus",
                description="Extra reward for successful privilege escalation",
                condition_type="action_type",
                condition_params={"action": "priv_escalate"},
                bonus=0.5,
                decay=0.97,
            ),
        ]

        for rule in defaults:
            self.rules[rule.rule_id] = rule

    def shape_reward(
        self,
        base_reward: float,
        state=None,
        action=None,
        next_state=None,
        action_history: List[str] = None,
    ) -> float:
        """
        Apply reward shaping rules to a base reward.

        Args:
            base_reward: Original reward from environment
            state: Current PenTestState
            action: Action taken
            next_state: Resulting state
            action_history: List of recent action types

        Returns:
            Shaped reward
        """
        shaped_reward = base_reward
        applied_rules = []

        for rule in self.rules.values():
            if rule.effectiveness < 0.15:
                continue  # Skip ineffective rules

            matches = self._check_rule_condition(rule, state, action, next_state, action_history)
            if matches:
                bonus = rule.apply()
                shaped_reward += bonus
                applied_rules.append({
                    "rule_id": rule.rule_id,
                    "bonus": bonus,
                    "effectiveness": rule.effectiveness,
                })

        # Record shaping
        if applied_rules:
            self.shaping_history.append({
                "timestamp": time.time(),
                "base_reward": base_reward,
                "shaped_reward": shaped_reward,
                "applied_rules": applied_rules,
            })

        return shaped_reward

    def learn_rule_from_lesson(self, lesson) -> Optional[RewardShapingRule]:
        """
        Create a new reward shaping rule from a learned lesson.

        Args:
            lesson: Lesson object from LessonsLearnedDB

        Returns:
            Newly created rule or None
        """
        from rl_agent.lessons_db import Lesson

        if not isinstance(lesson, Lesson):
            return None

        # Generate rule based on lesson category
        rule_id = f"lesson_{lesson.lesson_id}"

        if rule_id in self.rules:
            # Update existing rule confidence
            existing = self.rules[rule_id]
            existing.bonus *= 1.0 + lesson.confidence * 0.1
            return existing

        # Determine bonus from lesson category
        if lesson.category == "success_pattern":
            bonus = 0.2 * lesson.confidence
            condition_type = "action_type"
        elif lesson.category == "failure_pattern":
            bonus = -0.2 * lesson.confidence
            condition_type = "action_type"
        elif lesson.category == "avoidance":
            bonus = -0.3 * lesson.confidence
            condition_type = "action_type"
        elif lesson.category == "optimization":
            bonus = 0.15 * lesson.confidence
            condition_type = "state_progress"
        else:
            bonus = 0.1 * lesson.confidence
            condition_type = "custom"

        rule = RewardShapingRule(
            rule_id=rule_id,
            name=f"Learned: {lesson.description[:50]}",
            description=lesson.description,
            condition_type=condition_type,
            condition_params={"lesson_context": lesson.context[:200]},
            bonus=bonus,
            decay=0.98,
        )

        self.rules[rule_id] = rule
        logger.info(f"Learned new reward rule: {rule.name} (bonus={bonus:.3f})")
        return rule

    def update_rule_effectiveness(self, rule_id: str, was_successful: bool) -> None:
        """Update a rule's effectiveness tracking."""
        if rule_id in self.rules:
            self.rules[rule_id].record_result(was_successful)

    def prune_ineffective_rules(self, threshold: float = 0.2, min_applications: int = 5) -> int:
        """
        Remove rules that have consistently underperformed.

        Args:
            threshold: Minimum effectiveness to keep
            min_applications: Minimum applications before pruning

        Returns:
            Number of rules pruned
        """
        to_remove = [
            rid for rid, rule in self.rules.items()
            if rule.applications >= min_applications and rule.effectiveness < threshold
            and not rid.startswith("lesson_")  # Keep learned rules longer
        ]

        # Also prune learned rules with very low effectiveness
        learned_to_remove = [
            rid for rid, rule in self.rules.items()
            if rid.startswith("lesson_") and rule.applications >= 10 and rule.effectiveness < 0.1
        ]
        to_remove.extend(learned_to_remove)

        for rid in to_remove:
            del self.rules[rid]

        if to_remove:
            logger.info(f"Pruned {len(to_remove)} ineffective reward rules")
        return len(to_remove)

    def sync_with_lessons(self) -> int:
        """Synchronize rules with lessons database."""
        if not self.lessons_db:
            return 0

        new_rules = 0
        for lesson in self.lessons_db.lessons.values():
            if lesson.confidence >= 0.4 and lesson.occurrences >= 2:
                rule = self.learn_rule_from_lesson(lesson)
                if rule:
                    new_rules += 1

        return new_rules

    def get_statistics(self) -> dict:
        """Get reward shaper statistics."""
        return {
            "total_rules": len(self.rules),
            "active_rules": sum(1 for r in self.rules.values() if r.effectiveness >= 0.15),
            "total_applications": sum(r.applications for r in self.rules.values()),
            "avg_effectiveness": (
                sum(r.effectiveness for r in self.rules.values()) / max(1, len(self.rules))
            ),
            "shaping_events": len(self.shaping_history),
        }

    def _check_rule_condition(
        self,
        rule: RewardShapingRule,
        state,
        action,
        next_state,
        action_history: List[str],
    ) -> bool:
        """Check if a rule's condition matches the current situation."""
        params = rule.condition_params

        if rule.condition_type == "action_type":
            if action is None:
                return False
            action_type = getattr(action, 'type', None)
            action_str = action_type.value if hasattr(action_type, 'value') else str(action_type)
            return params.get("action", "") == action_str

        elif rule.condition_type == "vuln_severity":
            if action is None:
                return False
            params_action = getattr(action, 'parameters', {})
            severity = params_action.get('severity', 0)
            return severity >= params.get("min_severity", 0)

        elif rule.condition_type == "state_progress":
            if state is None:
                return False
            if params.get("has_compromised_host"):
                compromised = getattr(state, 'compromised_hosts', set())
                if not compromised:
                    return False
            if params.get("uses_credentials"):
                credentials = getattr(state, 'credentials', {})
                if not credentials:
                    return False
            if action and params.get("action"):
                action_type = getattr(action, 'type', None)
                action_str = action_type.value if hasattr(action_type, 'value') else str(action_type)
                if params["action"] != action_str:
                    return False
            return True

        elif rule.condition_type == "action_sequence":
            if not action_history:
                return False
            required_prior = params.get("requires_prior", "")
            return required_prior in action_history

        elif rule.condition_type == "custom":
            # Custom rules match based on lesson context
            return True  # Will be refined by the lesson context

        return False

    def save_state(self, path: str) -> None:
        """Save shaper state."""
        import json
        data = {
            "rules": {rid: {
                "rule_id": r.rule_id,
                "name": r.name,
                "description": r.description,
                "condition_type": r.condition_type,
                "condition_params": r.condition_params,
                "bonus": r.bonus,
                "decay": r.decay,
                "current_decay": r.current_decay,
                "applications": r.applications,
                "successes": r.successes,
                "effectiveness": r.effectiveness,
            } for rid, r in self.rules.items()},
            "last_sync": datetime.now().isoformat(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_state(self, path: str) -> None:
        """Load shaper state."""
        import json
        if not os.path.exists(path):
            return

        with open(path, 'r') as f:
            data = json.load(f)

        self.rules.clear()
        for rid, rdata in data.get("rules", {}).items():
            self.rules[rid] = RewardShapingRule(**rdata)


import os  # needed for load_state
