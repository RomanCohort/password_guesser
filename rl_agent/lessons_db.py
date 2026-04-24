"""
Lessons Learned Database

Accumulates and applies lessons from reflective analysis across sessions.
Automatically extracts structured lessons from reflections and uses them
to improve action selection and strategy.
"""

import json
import os
import time
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Lesson:
    """A structured lesson learned from reflective analysis."""
    lesson_id: str
    category: str  # 'failure_pattern', 'success_pattern', 'optimization', 'avoidance'
    description: str
    context: str  # Trigger condition
    action_suggestion: str  # Suggested action adjustment
    confidence: float  # 0-1, increases with more evidence
    occurrences: int  # How many times this pattern appeared
    success_correlation: float  # Correlation with successful outcomes
    created_at: float
    last_seen: float
    source_session: str = ""
    applied_count: int = 0  # How many times this lesson was applied
    applied_success: int = 0  # How many times application led to success

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Lesson':
        return cls(**data)

    def update_confidence(self, delta: float) -> None:
        """Adjust confidence based on new evidence."""
        self.confidence = max(0.0, min(1.0, self.confidence + delta))

    def record_application(self, was_successful: bool) -> None:
        """Record the result of applying this lesson."""
        self.applied_count += 1
        if was_successful:
            self.applied_success += 1
        # Adjust confidence based on application results
        if self.applied_count >= 3:
            application_rate = self.applied_success / self.applied_count
            self.confidence = self.confidence * 0.7 + application_rate * 0.3


@dataclass
class LessonPattern:
    """Pattern matcher for applying lessons to new situations."""
    pattern_id: str
    lesson_id: str
    state_keywords: List[str]  # Keywords to match in state
    action_types: List[str]  # Action types this applies to
    target_patterns: List[str]  # Target patterns (e.g., "ssh", "80")
    weight_adjustment: float  # How much to adjust action weight (+ or -)

    def matches(self, state_summary: str, action_type: str, target: str) -> bool:
        """Check if this pattern matches the current situation."""
        # Check action type
        if self.action_types and action_type not in self.action_types:
            return False

        # Check target patterns
        if self.target_patterns:
            target_match = any(p in target.lower() for p in self.target_patterns)
            if not target_match:
                return False

        # Check state keywords
        if self.state_keywords:
            state_lower = state_summary.lower()
            keyword_match = any(kw in state_lower for kw in self.state_keywords)
            if not keyword_match:
                return False

        return True


class LessonsLearnedDB:
    """
    Cross-session accumulated lessons learned database.

    Automatically extracts structured lessons from reflections,
    tracks their effectiveness, and applies them to action selection.
    """

    # Categories for lessons
    FAILURE_PATTERN = "failure_pattern"
    SUCCESS_PATTERN = "success_pattern"
    OPTIMIZATION = "optimization"
    AVOIDANCE = "avoidance"

    def __init__(self, path: str = "data/lessons/lessons.json"):
        self.path = path
        self.lessons: Dict[str, Lesson] = {}
        self.patterns: List[LessonPattern] = []

        os.makedirs(os.path.dirname(path), exist_ok=True)

    def extract_from_reflection(self, reflection, session_id: str = "") -> List[Lesson]:
        """
        Extract structured lessons from a reflection.

        Args:
            reflection: Reflection object with lessons_learned and suggested_modifications
            session_id: Source session identifier

        Returns:
            List of extracted Lesson objects
        """
        new_lessons = []

        # Extract from lessons_learned
        lessons_text = getattr(reflection, 'lessons_learned', []) or []
        suggestions = getattr(reflection, 'suggested_modifications', []) or []
        alternatives = getattr(reflection, 'alternative_actions', []) or []

        total_reward = getattr(reflection, 'total_reward', 0)
        observation = getattr(reflection, 'observation', '') or ''

        # Determine category based on reward
        category = self.SUCCESS_PATTERN if total_reward > 0 else self.FAILURE_PATTERN

        for lesson_text in lessons_text:
            lesson_id = self._generate_id(lesson_text)
            if lesson_id in self.lessons:
                # Existing lesson - update
                existing = self.lessons[lesson_id]
                existing.occurrences += 1
                existing.last_seen = time.time()
                existing.source_session = session_id
                if total_reward > 0:
                    existing.update_confidence(0.05)
                else:
                    existing.update_confidence(-0.02)
                new_lessons.append(existing)
            else:
                # New lesson
                lesson = Lesson(
                    lesson_id=lesson_id,
                    category=category,
                    description=lesson_text,
                    context=observation[:500],
                    action_suggestion="",
                    confidence=0.5,
                    occurrences=1,
                    success_correlation=1.0 if total_reward > 0 else 0.0,
                    created_at=time.time(),
                    last_seen=time.time(),
                    source_session=session_id,
                )
                self.lessons[lesson_id] = lesson
                new_lessons.append(lesson)

        # Extract from suggestions (optimization type)
        for suggestion in suggestions:
            lesson_id = self._generate_id(suggestion)
            if lesson_id not in self.lessons:
                lesson = Lesson(
                    lesson_id=lesson_id,
                    category=self.OPTIMIZATION,
                    description=suggestion,
                    context=observation[:500],
                    action_suggestion=suggestion,
                    confidence=0.4,
                    occurrences=1,
                    success_correlation=0.5,
                    created_at=time.time(),
                    last_seen=time.time(),
                    source_session=session_id,
                )
                self.lessons[lesson_id] = lesson
                new_lessons.append(lesson)

        # Extract avoidance patterns from failures
        if total_reward < -1.0 and observation:
            # High failure - create avoidance lesson
            actions_taken = getattr(reflection, 'actions_taken', []) or []
            for action in actions_taken:
                action_str = str(action)
                lesson_id = self._generate_id(f"avoid_{action_str}")
                if lesson_id not in self.lessons:
                    lesson = Lesson(
                        lesson_id=lesson_id,
                        category=self.AVOIDANCE,
                        description=f"Avoid action in similar failure context: {action_str[:200]}",
                        context=observation[:500],
                        action_suggestion="Try alternative approach",
                        confidence=0.3,
                        occurrences=1,
                        success_correlation=0.0,
                        created_at=time.time(),
                        last_seen=time.time(),
                        source_session=session_id,
                    )
                    self.lessons[lesson_id] = lesson
                    new_lessons.append(lesson)

        # Create action patterns from alternatives
        for alt in alternatives:
            if isinstance(alt, dict):
                alt_type = alt.get('type', '')
                alt_target = alt.get('target', '')
                if alt_type and alt_target:
                    pattern = LessonPattern(
                        pattern_id=self._generate_id(f"pattern_{alt_type}_{alt_target}"),
                        lesson_id=self._generate_id(alt.get('reason', '')),
                        state_keywords=observation.split()[:10] if observation else [],
                        action_types=[alt_type],
                        target_patterns=[alt_target],
                        weight_adjustment=0.2,
                    )
                    self.patterns.append(pattern)

        return new_lessons

    def add_lesson(self, lesson: Lesson) -> None:
        """Add or update a lesson."""
        if lesson.lesson_id in self.lessons:
            existing = self.lessons[lesson.lesson_id]
            existing.occurrences += lesson.occurrences
            existing.last_seen = time.time()
            existing.confidence = max(existing.confidence, lesson.confidence)
        else:
            self.lessons[lesson.lesson_id] = lesson

    def get_relevant_lessons(
        self,
        state_summary: str = "",
        action_type: str = "",
        target: str = "",
        category: str = None,
        min_confidence: float = 0.3,
        limit: int = 10,
    ) -> List[Lesson]:
        """
        Get lessons relevant to the current situation.

        Args:
            state_summary: Text summary of current state
            action_type: Current action type being considered
            target: Current target
            category: Filter by category
            min_confidence: Minimum confidence threshold
            limit: Maximum lessons to return

        Returns:
            List of relevant lessons, sorted by relevance
        """
        relevant = []

        for lesson in self.lessons.values():
            if lesson.confidence < min_confidence:
                continue

            if category and lesson.category != category:
                continue

            relevance = lesson.confidence

            # Boost relevance for matching context keywords
            if state_summary:
                state_lower = state_summary.lower()
                context_words = lesson.context.lower().split()
                matching_words = sum(1 for w in context_words if w in state_lower)
                relevance += matching_words * 0.05

            relevant.append((lesson, relevance))

        # Sort by relevance and return
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [lesson for lesson, _ in relevant[:limit]]

    def apply_to_action_selection(
        self,
        state_summary: str,
        actions: List[Tuple],
    ) -> List[Tuple]:
        """
        Adjust action weights based on learned lessons.

        Args:
            state_summary: Current state description
            actions: List of (action, base_weight) tuples

        Returns:
            List of (action, adjusted_weight) tuples
        """
        adjusted = []

        for action, base_weight in actions:
            weight = base_weight

            # Apply pattern matching
            action_type = getattr(action, 'type', None)
            action_type_str = action_type.value if hasattr(action_type, 'value') else str(action_type)
            target = getattr(action, 'target', '')

            for pattern in self.patterns:
                if pattern.matches(state_summary, action_type_str, target):
                    lesson = self.lessons.get(pattern.lesson_id)
                    if lesson and lesson.confidence > 0.3:
                        weight += pattern.weight_adjustment * lesson.confidence
                        lesson.record_application(False)  # Will be updated with actual result

            # Apply lesson-based adjustments
            relevant_lessons = self.get_relevant_lessons(
                state_summary=state_summary,
                action_type=action_type_str,
                target=target,
                limit=5,
            )

            for lesson in relevant_lessons:
                if lesson.category == self.AVOIDANCE:
                    weight -= 0.2 * lesson.confidence
                elif lesson.category == self.SUCCESS_PATTERN:
                    weight += 0.15 * lesson.confidence
                elif lesson.category == self.OPTIMIZATION:
                    weight += 0.1 * lesson.confidence
                elif lesson.category == self.FAILURE_PATTERN:
                    weight -= 0.1 * lesson.confidence

            adjusted.append((action, max(0.01, weight)))

        return adjusted

    def record_application_result(self, lesson_id: str, was_successful: bool) -> None:
        """Record the outcome of applying a lesson."""
        if lesson_id in self.lessons:
            self.lessons[lesson_id].record_application(was_successful)

    def get_statistics(self) -> dict:
        """Get database statistics."""
        categories = {}
        for lesson in self.lessons.values():
            cat = lesson.category
            if cat not in categories:
                categories[cat] = {"count": 0, "avg_confidence": 0.0}
            categories[cat]["count"] += 1
            categories[cat]["avg_confidence"] += lesson.confidence

        for cat in categories:
            count = categories[cat]["count"]
            if count > 0:
                categories[cat]["avg_confidence"] /= count

        return {
            "total_lessons": len(self.lessons),
            "total_patterns": len(self.patterns),
            "categories": categories,
            "high_confidence_lessons": sum(
                1 for l in self.lessons.values() if l.confidence >= 0.7
            ),
        }

    def prune_low_confidence(self, min_confidence: float = 0.1, min_occurrences: int = 1) -> int:
        """Remove lessons with very low confidence."""
        to_remove = [
            lid for lid, lesson in self.lessons.items()
            if lesson.confidence < min_confidence and lesson.occurrences <= min_occurrences
        ]
        for lid in to_remove:
            del self.lessons[lid]

        # Also remove patterns referencing removed lessons
        self.patterns = [
            p for p in self.patterns
            if p.lesson_id in self.lessons
        ]

        if to_remove:
            logger.info(f"Pruned {len(to_remove)} low-confidence lessons")
        return len(to_remove)

    def save(self) -> None:
        """Persist lessons and patterns to disk."""
        data = {
            "lessons": {lid: l.to_dict() for lid, l in self.lessons.items()},
            "patterns": [asdict(p) for p in self.patterns],
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "total_lessons": len(self.lessons),
            }
        }

        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.lessons)} lessons to {self.path}")

    def load(self) -> None:
        """Load lessons and patterns from disk."""
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, 'r') as f:
                data = json.load(f)

            self.lessons = {
                lid: Lesson.from_dict(l) for lid, l in data.get("lessons", {}).items()
            }

            self.patterns = [
                LessonPattern(**p) for p in data.get("patterns", [])
            ]

            logger.info(f"Loaded {len(self.lessons)} lessons from {self.path}")
        except Exception as e:
            logger.warning(f"Failed to load lessons: {e}")

    @staticmethod
    def _generate_id(text: str) -> str:
        """Generate a stable ID from text content."""
        return hashlib.md5(text.encode()).hexdigest()[:12]


# Global instance
_global_db: Optional[LessonsLearnedDB] = None


def get_lessons_db() -> LessonsLearnedDB:
    """Get or create global lessons database."""
    global _global_db
    if _global_db is None:
        _global_db = LessonsLearnedDB()
        _global_db.load()
    return _global_db
