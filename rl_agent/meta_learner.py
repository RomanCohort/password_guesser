"""
Meta Learner

Meta-learning controller that monitors learning progress and adjusts
hyperparameters, exploration strategy, and curriculum to optimize
the learning process itself.
"""

import time
import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math

logger = logging.getLogger(__name__)


@dataclass
class LearningMetrics:
    """Metrics for tracking learning progress."""
    episode: int
    total_reward: float
    success: bool
    steps: int
    avg_step_reward: float
    timestamp: float
    scenario_id: str = ""
    hyperparams: dict = field(default_factory=dict)


@dataclass
class HyperparameterState:
    """Trackable hyperparameter state."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    exploration_rate: float = 0.3  # epsilon-greedy or temperature factor
    reflection_frequency: int = 5
    batch_size: int = 64

    # Adjustment history
    lr_adjustments: List[float] = field(default_factory=list)
    exploration_adjustments: List[float] = field(default_factory=list)


class MetaLearner:
    """
    Meta-learning controller for continuous self-improvement.

    Monitors learning progress across sessions and:
    - Detects plateaus in learning
    - Adjusts hyperparameters adaptively
    - Suggests exploration strategies
    - Triggers curriculum difficulty shifts
    - Coordinates between all learning components
    """

    # Plateau detection window
    WINDOW_SIZE = 10

    def __init__(
        self,
        agent=None,
        reward_shaper=None,
        curriculum=None,
        lessons_db=None,
        experience_store=None,
    ):
        self.agent = agent
        self.reward_shaper = reward_shaper
        self.curriculum = curriculum
        self.lessons_db = lessons_db
        self.experience_store = experience_store

        self.hyperparams = HyperparameterState()
        self.metrics_history: List[LearningMetrics] = []
        self.adjustments_log: List[dict] = []

        # Learning curve tracking
        self.reward_window: List[float] = []
        self.success_window: List[bool] = []

        # State
        self.total_episodes = 0
        self.best_avg_reward = float('-inf')
        self.plateau_count = 0
        self.last_adjustment_time = 0
        self.adjustment_cooldown = 5  # Episodes between adjustments

    def record_episode(self, metrics: LearningMetrics) -> None:
        """Record metrics from a completed episode."""
        self.metrics_history.append(metrics)
        self.total_episodes += 1

        # Update sliding windows
        self.reward_window.append(metrics.total_reward)
        self.success_window.append(metrics.success)

        if len(self.reward_window) > self.WINDOW_SIZE:
            self.reward_window.pop(0)
            self.success_window.pop(0)

        # Track best performance
        current_avg = self._window_avg_reward()
        if current_avg > self.best_avg_reward:
            self.best_avg_reward = current_avg

    def analyze_learning_curve(self) -> dict:
        """
        Analyze the learning curve to identify trends.

        Returns:
            Analysis result with trend, plateau status, recommendations
        """
        if len(self.reward_window) < 5:
            return {
                "trend": "insufficient_data",
                "is_plateau": False,
                "recommendations": ["continue_training"],
                "recent_avg_reward": self._window_avg_reward(),
                "recent_success_rate": self._window_success_rate(),
            }

        recent_avg = self._window_avg_reward()
        recent_success_rate = self._window_success_rate()

        # Calculate trend using linear regression on recent rewards
        trend = self._calculate_trend()

        # Detect plateau
        is_plateau = self.detect_plateau()

        # Calculate improvement rate
        improvement_rate = 0.0
        if len(self.metrics_history) >= 20:
            old_avg = sum(m.total_reward for m in self.metrics_history[-20:-10]) / 10
            new_avg = sum(m.total_reward for m in self.metrics_history[-10:]) / 10
            improvement_rate = new_avg - old_avg

        analysis = {
            "trend": trend,  # 'improving', 'plateau', 'declining', 'volatile'
            "is_plateau": is_plateau,
            "recent_avg_reward": recent_avg,
            "recent_success_rate": recent_success_rate,
            "best_avg_reward": self.best_avg_reward,
            "improvement_rate": improvement_rate,
            "total_episodes": self.total_episodes,
            "current_exploration": self.hyperparams.exploration_rate,
            "current_lr": self.hyperparams.learning_rate,
        }

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)

        return analysis

    def detect_plateau(self) -> bool:
        """
        Detect if learning has plateaued.

        A plateau is detected when:
        - Recent reward variance is low (stable but not improving)
        - Average reward hasn't improved significantly over recent window
        """
        if len(self.reward_window) < self.WINDOW_SIZE:
            return False

        recent = self.reward_window
        variance = sum((r - sum(recent) / len(recent)) ** 2 for r in recent) / len(recent)
        std_dev = math.sqrt(variance)

        # Low variance = stable performance
        low_variance = std_dev < 0.1

        # Not improving compared to best
        not_improving = self._window_avg_reward() < self.best_avg_reward * 0.95

        # Both conditions = plateau
        is_plateau = low_variance and not_improving

        if is_plateau:
            self.plateau_count += 1
        else:
            self.plateau_count = max(0, self.plateau_count - 1)

        return is_plateau

    def adjust_hyperparameters(self) -> dict:
        """
        Adjust hyperparameters based on learning analysis.

        Returns:
            Dictionary of adjustments made
        """
        # Cooldown check
        if self.total_episodes - self.last_adjustment_time < self.adjustment_cooldown:
            return {"status": "cooldown", "episodes_until_next": self.adjustment_cooldown - (self.total_episodes - self.last_adjustment_time)}

        analysis = self.analyze_learning_curve()
        adjustments = {}

        trend = analysis["trend"]
        is_plateau = analysis["is_plateau"]

        if is_plateau or trend == "plateau":
            # Plateau: increase exploration, adjust learning rate
            adjustments["exploration_rate"] = self._boost_exploration()
            adjustments["learning_rate"] = self._adjust_lr_plateau()
            adjustments["action"] = "plateau_break"

        elif trend == "declining":
            # Declining: reduce lr, increase stability
            new_lr = self.hyperparams.learning_rate * 0.8
            adjustments["learning_rate"] = new_lr
            self.hyperparams.learning_rate = new_lr
            adjustments["action"] = "stabilize"

        elif trend == "improving":
            # Improving: maintain or slightly reduce exploration
            new_exploration = self.hyperparams.exploration_rate * 0.95
            adjustments["exploration_rate"] = new_exploration
            self.hyperparams.exploration_rate = max(0.05, new_exploration)
            adjustments["action"] = "exploit"

        elif trend == "volatile":
            # Volatile: reduce lr for stability
            new_lr = self.hyperparams.learning_rate * 0.9
            adjustments["learning_rate"] = new_lr
            self.hyperparams.learning_rate = new_lr
            adjustments["action"] = "stabilize"

        # Apply adjustments to agent if available
        if self.agent and adjustments:
            self._apply_to_agent(adjustments)

        # Log adjustment
        if adjustments:
            adjustments["episode"] = self.total_episodes
            adjustments["timestamp"] = time.time()
            self.adjustments_log.append(adjustments)
            self.last_adjustment_time = self.total_episodes

            logger.info(f"Meta adjustment: {adjustments.get('action', 'unknown')} | LR={self.hyperparams.learning_rate:.6f} | Exploit={self.hyperparams.exploration_rate:.3f}")

        return adjustments

    def suggest_exploration_strategy(self) -> str:
        """
        Suggest an exploration strategy based on current state.

        Returns:
            Strategy name
        """
        if len(self.reward_window) < 5:
            return "high_exploration"  # Early training: explore a lot

        success_rate = self._window_success_rate()

        if success_rate < 0.2:
            return "high_exploration"
        elif success_rate < 0.5:
            return "balanced"
        elif success_rate < 0.8:
            return "focused_exploitation"
        else:
            return "conservative_exploitation"

    def trigger_curriculum_shift(self) -> Optional[str]:
        """
        Check if curriculum difficulty should shift and trigger it.

        Returns:
            Shift direction ('up', 'down', None) if triggered
        """
        if not self.curriculum:
            return None

        if len(self.reward_window) < self.WINDOW_SIZE:
            return None

        success_rate = self._window_success_rate()
        avg_reward = self._window_avg_reward()

        # Shift up if doing well
        if success_rate >= 0.8 and avg_reward > 0:
            progress = self.curriculum.get_curriculum_progress()
            weak_areas = progress.get("weak_areas", [])

            if not weak_areas or all(v >= 0.6 for v in self.curriculum.mastery.skill_areas.values()):
                logger.info("Performance high, triggering curriculum difficulty increase")
                return "up"

        # Shift down if struggling
        if success_rate < 0.2 and avg_reward < -0.5:
            logger.info("Performance low, suggesting easier scenarios")
            return "down"

        return None

    def coordinate_improvement_cycle(self) -> dict:
        """
        Run one complete improvement coordination cycle.

        This is the main self-improvement loop that ties everything together.

        Returns:
            Summary of what was done
        """
        results = {
            "episode": self.total_episodes,
            "timestamp": time.time(),
            "actions_taken": [],
        }

        # 1. Analyze learning curve
        analysis = self.analyze_learning_curve()
        results["analysis"] = analysis

        # 2. Adjust hyperparameters if needed
        adjustments = self.adjust_hyperparameters()
        if adjustments.get("action"):
            results["actions_taken"].append(f"hyperparam_adjust: {adjustments['action']}")

        # 3. Sync reward shaper with lessons
        if self.reward_shaper and self.lessons_db:
            new_rules = self.reward_shaper.sync_with_lessons()
            if new_rules > 0:
                results["actions_taken"].append(f"synced_{new_rules}_reward_rules")

            # Prune ineffective rules periodically
            if self.total_episodes % 20 == 0:
                pruned = self.reward_shaper.prune_ineffective_rules()
                if pruned > 0:
                    results["actions_taken"].append(f"pruned_{pruned}_rules")

        # 4. Check for curriculum shift
        shift = self.trigger_curriculum_shift()
        if shift:
            results["actions_taken"].append(f"curriculum_shift_{shift}")

        # 5. Generate new scenarios from experience periodically
        if self.curriculum and self.experience_store and self.total_episodes % 10 == 0:
            failures = self.curriculum.generate_from_failures()
            if failures:
                results["actions_taken"].append(f"generated_{len(failures)}_scenarios")

        # 6. Prune low-confidence lessons periodically
        if self.lessons_db and self.total_episodes % 25 == 0:
            pruned = self.lessons_db.prune_low_confidence()
            if pruned > 0:
                results["actions_taken"].append(f"pruned_{pruned}_lessons")

        return results

    def get_state(self) -> dict:
        """Get serializable state for persistence."""
        return {
            "hyperparams": asdict(self.hyperparams),
            "total_episodes": self.total_episodes,
            "best_avg_reward": self.best_avg_reward,
            "plateau_count": self.plateau_count,
            "recent_metrics": [asdict(m) for m in self.metrics_history[-50:]],
            "adjustments_log": self.adjustments_log[-20:],
        }

    def load_state(self, state: dict) -> None:
        """Load state from persistence."""
        hp = state.get("hyperparams", {})
        for key, value in hp.items():
            if hasattr(self.hyperparams, key) and key not in ('lr_adjustments', 'exploration_adjustments'):
                setattr(self.hyperparams, key, value)

        self.total_episodes = state.get("total_episodes", 0)
        self.best_avg_reward = state.get("best_avg_reward", float('-inf'))
        self.plateau_count = state.get("plateau_count", 0)

    def save(self, path: str = "data/curriculum/meta_learner.json") -> None:
        """Save meta learner state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.get_state(), f, indent=2)

    def load(self, path: str = "data/curriculum/meta_learner.json") -> None:
        """Load meta learner state."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.load_state(json.load(f))

    # ---- Internal helpers ----

    def _generate_recommendations(self, analysis: dict) -> List[str]:
        """Generate actionable recommendations from analysis."""
        recommendations = []

        trend = analysis.get("trend", "")
        is_plateau = analysis.get("is_plateau", False)
        success_rate = analysis.get("recent_success_rate", 0)

        if is_plateau:
            recommendations.append("increase_exploration")
            recommendations.append("try_curriculum_shift")

        if trend == "declining":
            recommendations.append("reduce_learning_rate")
            recommendations.append("review_recent_failures")

        if trend == "improving":
            recommendations.append("continue_current_strategy")

        if success_rate < 0.2:
            recommendations.append("try_easier_scenarios")
            recommendations.append("increase_reflection_frequency")

        if success_rate > 0.8:
            recommendations.append("increase_difficulty")
            recommendations.append("reduce_exploration")

        if not recommendations:
            recommendations.append("continue_training")

        return recommendations

    def _window_avg_reward(self) -> float:
        """Average reward over the sliding window."""
        if not self.reward_window:
            return 0.0
        return sum(self.reward_window) / len(self.reward_window)

    def _window_success_rate(self) -> float:
        """Success rate over the sliding window."""
        if not self.success_window:
            return 0.0
        return sum(1 for s in self.success_window if s) / len(self.success_window)

    def _calculate_trend(self) -> str:
        """Calculate trend direction from recent rewards."""
        if len(self.reward_window) < 5:
            return "insufficient_data"

        # Split window into first half and second half
        mid = len(self.reward_window) // 2
        first_half = self.reward_window[:mid]
        second_half = self.reward_window[mid:]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        diff = second_avg - first_avg
        variance = sum((r - second_avg) ** 2 for r in second_half) / len(second_half)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        if std_dev > 1.0:
            return "volatile"
        elif diff > 0.2:
            return "improving"
        elif diff < -0.2:
            return "declining"
        else:
            return "plateau"

    def _boost_exploration(self) -> float:
        """Increase exploration rate to break plateau."""
        boost = 0.15 * min(3, self.plateau_count)
        new_rate = min(0.8, self.hyperparams.exploration_rate + boost)
        self.hyperparams.exploration_rate = new_rate
        self.hyperparams.exploration_adjustments.append(new_rate)
        return new_rate

    def _adjust_lr_plateau(self) -> float:
        """Adjust learning rate during plateau."""
        # Try both increasing and decreasing based on plateau count
        if self.plateau_count % 2 == 0:
            # Decrease lr for fine-tuning
            new_lr = self.hyperparams.learning_rate * 0.7
        else:
            # Increase lr to escape local minimum
            new_lr = min(1e-3, self.hyperparams.learning_rate * 1.5)

        self.hyperparams.learning_rate = new_lr
        self.hyperparams.lr_adjustments.append(new_lr)
        return new_lr

    def _apply_to_agent(self, adjustments: dict) -> None:
        """Apply hyperparameter adjustments to the agent."""
        if not self.agent:
            return

        if "learning_rate" in adjustments and hasattr(self.agent, 'optimizer'):
            import torch.optim as optim
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = adjustments["learning_rate"]

        if "clip_epsilon" in adjustments and hasattr(self.agent, 'clip_epsilon'):
            self.agent.clip_epsilon = adjustments["clip_epsilon"]

        if "reflection_frequency" in adjustments and hasattr(self.agent, 'reflection_frequency'):
            self.agent.reflection_frequency = int(adjustments["reflection_frequency"])
