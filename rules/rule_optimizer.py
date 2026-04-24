"""
Rule optimisation based on historical success rates.

Tracks how often each rule produces a successful guess and uses that
information to reorder, prune, and compose rules for future cracking
sessions.
"""

import itertools
from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass, field


@dataclass
class RuleStatistics:
    """Running statistics for a single rule."""

    rule_name: str
    applications: int = 0
    successes: int = 0
    success_rate: float = 0.0

    def record(self, success: bool) -> None:
        self.applications += 1
        if success:
            self.successes += 1
        if self.applications > 0:
            self.success_rate = self.successes / self.applications


# ======================================================================
# Rule Optimizer
# ======================================================================

class RuleOptimizer:
    """Order and compose rules based on observed success rates."""

    def __init__(self):
        self.stats: Dict[str, RuleStatistics] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_application(self, rule_name: str, success: bool) -> None:
        """Record the outcome of applying *rule_name*."""
        if rule_name not in self.stats:
            self.stats[rule_name] = RuleStatistics(rule_name=rule_name)
        self.stats[rule_name].record(success)

    def record_batch(self, results: List[Tuple[str, bool]]) -> None:
        """Record multiple ``(rule_name, success)`` pairs at once."""
        for rule_name, success in results:
            self.record_application(rule_name, success)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_stats(self, rule_name: str) -> Optional[RuleStatistics]:
        return self.stats.get(rule_name)

    def get_top_rules(self, n: int = 20) -> List[RuleStatistics]:
        """Return the top *n* rules sorted by success rate (descending).

        Rules that have never been applied are excluded.  Ties are broken
        by total number of successes (more is better).
        """
        applied = [s for s in self.stats.values() if s.applications > 0]
        applied.sort(key=lambda s: (s.success_rate, s.successes), reverse=True)
        return applied[:n]

    def get_bottom_rules(self, n: int = 20) -> List[RuleStatistics]:
        """Return the worst-performing *n* rules."""
        applied = [s for s in self.stats.values() if s.applications > 0]
        applied.sort(key=lambda s: (s.success_rate, s.successes))
        return applied[:n]

    # ------------------------------------------------------------------
    # Rule ordering
    # ------------------------------------------------------------------

    def optimize_rule_order(self, rules: List[str]) -> List[str]:
        """Reorder *rules* so that historically successful rules come first.

        Rules without any recorded statistics are placed after rules with
        stats but above rules with zero success rate.
        """
        scored: List[Tuple[str, float, int]] = []
        for rn in rules:
            st = self.stats.get(rn)
            if st and st.applications > 0:
                scored.append((rn, st.success_rate, st.successes))
            else:
                # Unknown rules get a neutral score
                scored.append((rn, 0.5, 0))

        scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
        return [t[0] for t in scored]

    # ------------------------------------------------------------------
    # Rule combination generation
    # ------------------------------------------------------------------

    def generate_rule_combinations(
        self,
        base_rules: List[str],
        max_depth: int = 2,
    ) -> List[str]:
        """Generate composite rule strings by combining *base_rules*.

        At depth 1 each base rule is returned as-is.  At depth 2 every
        ordered pair is combined into a two-step rule string separated by
        whitespace (Hashcat convention).  Higher depths follow the same
        pattern.

        Returns a flat list of composite rule strings, deduplicated.
        """
        # Depth 1: base rules themselves
        combos: List[str] = list(base_rules)

        for depth in range(2, max_depth + 1):
            for combo in itertools.product(base_rules, repeat=depth):
                combos.append(" ".join(combo))

        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for c in combos:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_rules(
        self,
        rules: List[str],
        min_applications: int = 10,
        min_success_rate: float = 0.01,
    ) -> List[str]:
        """Remove consistently under-performing rules.

        A rule is pruned when it has been applied at least
        *min_applications* times and its success rate is below
        *min_success_rate*.
        """
        pruned: List[str] = []
        for rn in rules:
            st = self.stats.get(rn)
            if st and st.applications >= min_applications:
                if st.success_rate < min_success_rate:
                    continue  # drop
            pruned.append(rn)
        return pruned

    # ------------------------------------------------------------------
    # Adaptive rule set
    # ------------------------------------------------------------------

    def get_adaptive_ruleset(
        self,
        all_rules: List[str],
        max_rules: int = 50,
        min_applications: int = 5,
    ) -> List[str]:
        """Return a pruned, ordered subset of *all_rules*.

        1. Prune rules with low success rates.
        2. Optimise the remaining order.
        3. Cap at *max_rules*.
        """
        pruned = self.prune_rules(all_rules, min_applications=min_applications)
        ordered = self.optimize_rule_order(pruned)
        return ordered[:max_rules]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Dict]:
        """Export all statistics as a plain dict (JSON-serialisable)."""
        return {
            name: {
                "applications": st.applications,
                "successes": st.successes,
                "success_rate": st.success_rate,
            }
            for name, st in self.stats.items()
        }

    def from_dict(self, data: Dict[str, Dict]) -> None:
        """Load statistics from a dict produced by :meth:`to_dict`."""
        for name, info in data.items():
            st = RuleStatistics(
                rule_name=name,
                applications=info.get("applications", 0),
                successes=info.get("successes", 0),
                success_rate=info.get("success_rate", 0.0),
            )
            self.stats[name] = st

    def merge(self, other: "RuleOptimizer") -> None:
        """Merge statistics from *other* into this optimizer."""
        for name, other_st in other.stats.items():
            if name not in self.stats:
                self.stats[name] = RuleStatistics(rule_name=name)
            my = self.stats[name]
            my.applications += other_st.applications
            my.successes += other_st.successes
            if my.applications > 0:
                my.success_rate = my.successes / my.applications
