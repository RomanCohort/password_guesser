"""Password generation evaluation metrics."""

from dataclasses import dataclass
from typing import List, Set, Optional
import math
import time


@dataclass
class EvaluationResult:
    """Complete evaluation result for a password generation method."""
    coverage: float          # Coverage rate: how many generated passwords hit targets
    hit_rate: float          # Hit rate: what fraction of targets were generated
    mrr: float               # Mean Reciprocal Rank
    diversity: float         # Average edit distance diversity
    avg_entropy: float       # Average entropy of generated passwords
    avg_strength: float      # Average strength score (0-1)
    unique_ratio: float      # Fraction of unique passwords
    generation_time: float   # Time taken to generate (seconds)


class EvaluationMetrics:
    """Evaluation metrics for password generation systems.

    Provides standard metrics for comparing password generation methods:
    - Coverage: how well generated passwords cover the target set
    - Hit rate: fraction of targets successfully generated
    - MRR: Mean Reciprocal Rank for ranking quality
    - Diversity: how diverse the generated passwords are
    """

    @staticmethod
    def coverage(generated: Set[str], targets: Set[str]) -> float:
        """Calculate coverage rate.

        Coverage = |generated intersect targets| / |generated|

        Measures what fraction of generated passwords are valid targets.
        """
        if not generated:
            return 0.0

        hits = generated & targets
        return len(hits) / len(generated)

    @staticmethod
    def hit_rate(generated: List[str], targets: Set[str]) -> float:
        """Calculate hit rate.

        Hit rate = |generated intersect targets| / |targets|

        Measures what fraction of target passwords were successfully generated.
        """
        if not targets:
            return 0.0

        generated_set = set(generated)
        hits = generated_set & targets
        return len(hits) / len(targets)

    @staticmethod
    def mrr(generated: List[str], targets: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank.

        MRR = average of 1/rank for each target found in generated list.

        Measures how high in the ranking the target passwords appear.
        Higher MRR means targets appear earlier in the generated list.
        """
        if not targets or not generated:
            return 0.0

        reciprocal_ranks = []
        for target in targets:
            try:
                rank = generated.index(target) + 1  # 1-based ranking
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                # Target not in generated list
                reciprocal_ranks.append(0.0)

        if not reciprocal_ranks:
            return 0.0

        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    @staticmethod
    def diversity(passwords: List[str]) -> float:
        """Calculate password diversity based on average pairwise edit distance.

        Edit distance (Levenshtein) measures how different two strings are.
        Higher diversity indicates more varied password generation.

        For efficiency, samples a subset for large lists.
        """
        if len(passwords) < 2:
            return 0.0

        # For large lists, sample to keep computation reasonable
        sample_size = min(len(passwords), 200)
        if len(passwords) > sample_size:
            import random
            passwords = random.sample(passwords, sample_size)

        total_distance = 0.0
        comparisons = 0

        for i in range(len(passwords)):
            for j in range(i + 1, len(passwords)):
                dist = EvaluationMetrics._edit_distance(passwords[i], passwords[j])
                total_distance += dist
                comparisons += 1

        if comparisons == 0:
            return 0.0

        avg_distance = total_distance / comparisons
        # Normalize by average password length
        avg_len = sum(len(p) for p in passwords) / len(passwords) if passwords else 1
        return avg_distance / avg_len if avg_len > 0 else 0.0

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings.

        The minimum number of single-character edits (insert, delete, replace)
        needed to transform s1 into s2.
        """
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def unique_ratio(passwords: List[str]) -> float:
        """Calculate the ratio of unique passwords.

        unique_ratio = number of unique passwords / total count

        Lower ratio suggests redundant generation.
        """
        if not passwords:
            return 0.0

        unique_count = len(set(passwords))
        return unique_count / len(passwords)

    @staticmethod
    def evaluate_ranking(
        generated: List[str],
        targets: Set[str],
        generation_time: float = 0.0,
        entropy_fn=None,
        strength_fn=None
    ) -> EvaluationResult:
        """Complete evaluation of a ranked password list.

        Args:
            generated: List of generated passwords (ranked by preference)
            targets: Set of target passwords to find
            generation_time: Time taken to generate (seconds)
            entropy_fn: Optional function to calculate entropy for each password
            strength_fn: Optional function to calculate strength score (0-1)

        Returns:
            EvaluationResult with all metrics computed.
        """
        generated_set = set(generated)

        coverage = EvaluationMetrics.coverage(generated_set, targets)
        hit_rate = EvaluationMetrics.hit_rate(generated, targets)
        mrr = EvaluationMetrics.mrr(generated, targets)
        diversity = EvaluationMetrics.diversity(generated)
        unique_ratio = EvaluationMetrics.unique_ratio(generated)

        # Calculate average entropy if function provided
        avg_entropy = 0.0
        if entropy_fn and generated:
            entropies = [entropy_fn(p) for p in generated[:500]]  # limit for speed
            avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        # Calculate average strength if function provided
        avg_strength = 0.0
        if strength_fn and generated:
            strengths = [strength_fn(p) for p in generated[:500]]
            avg_strength = sum(strengths) / len(strengths) if strengths else 0.0

        return EvaluationResult(
            coverage=coverage,
            hit_rate=hit_rate,
            mrr=mrr,
            diversity=diversity,
            avg_entropy=avg_entropy,
            avg_strength=avg_strength,
            unique_ratio=unique_ratio,
            generation_time=generation_time,
        )

    @staticmethod
    def calculate_accuracy_at_k(
        generated: List[str],
        targets: Set[str],
        k_values: List[int] = [10, 100, 1000]
    ) -> dict:
        """Calculate accuracy at different cutoff points.

        Accuracy@K = 1 if any target is in top K, else 0.
        Returns dict mapping K value to accuracy rate.

        Useful for evaluating ranking quality at different depths.
        """
        results = {}
        for k in k_values:
            top_k = set(generated[:k])
            hits = top_k & targets
            # Accuracy: at least one hit in top k
            results[f'acc@{k}'] = 1.0 if hits else 0.0
            # Hit count in top k
            results[f'hits@{k}'] = len(hits)
        return results

    @staticmethod
    def calculate_ndcg(
        generated: List[str],
        targets: Set[str],
        k: int = 100
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain.

        NDCG measures ranking quality, giving higher scores when targets
        appear earlier in the list.

        Args:
            generated: Ranked list of generated passwords
            targets: Set of target passwords
            k: Cutoff position (evaluate top k only)

        Returns:
            NDCG@k score from 0.0 to 1.0
        """
        # DCG = sum of 1/log2(i+1) for each target at position i
        dcg = 0.0
        for i, pwd in enumerate(generated[:k]):
            if pwd in targets:
                # Position is 1-indexed for DCG formula
                dcg += 1.0 / math.log2(i + 2)

        # Ideal DCG: all targets at the top positions
        ideal_dcg = 0.0
        num_targets_in_top_k = min(len(targets), k)
        for i in range(num_targets_in_top_k):
            ideal_dcg += 1.0 / math.log2(i + 2)

        if ideal_dcg == 0:
            return 0.0

        return dcg / ideal_dcg
