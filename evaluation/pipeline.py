"""Evaluation pipeline: batch evaluation and end-to-end assessment."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import time

from evaluation.strength import PasswordStrengthEvaluator, StrengthLevel, StrengthReport
from evaluation.metrics import EvaluationMetrics, EvaluationResult


class BatchEvaluator:
    """Batch evaluator for processing multiple passwords.

    Provides methods for batch strength evaluation, filtering,
    and generating aggregate reports.
    """

    def __init__(self, strength_evaluator: PasswordStrengthEvaluator = None):
        self.evaluator = strength_evaluator or PasswordStrengthEvaluator()

    def evaluate_batch(self, passwords: List[str]) -> List[StrengthReport]:
        """Evaluate a batch of passwords.

        Returns a list of StrengthReport objects in the same order as input.
        """
        return [self.evaluator.evaluate(pwd) for pwd in passwords]

    def rank_by_strength(self, passwords: List[str]) -> List[Tuple[str, StrengthReport]]:
        """Rank passwords by strength (strongest first).

        Returns list of (password, report) tuples sorted by strength score descending.
        """
        scored = []
        for pwd in passwords:
            report = self.evaluator.evaluate(pwd)
            scored.append((pwd, report))

        scored.sort(key=lambda x: x[1].score, reverse=True)
        return scored

    def filter_weak(
        self, passwords: List[str], min_score: StrengthLevel = StrengthLevel.FAIR
    ) -> List[str]:
        """Filter out weak passwords below the minimum score.

        Args:
            passwords: List of passwords to filter
            min_score: Minimum acceptable strength level

        Returns:
            List of passwords meeting the minimum strength requirement.
        """
        filtered = []
        for pwd in passwords:
            report = self.evaluator.evaluate(pwd)
            if report.score >= min_score:
                filtered.append(pwd)
        return filtered

    def generate_report(self, passwords: List[str]) -> dict:
        """Generate an aggregate report for a batch of passwords.

        Returns a dictionary with:
        - total_count: total number of passwords
        - strength_distribution: count per strength level
        - average_entropy: mean entropy
        - average_guess_number: geometric mean of guess numbers
        - common_warnings: frequency of warnings
        - common_suggestions: frequency of suggestions
        """
        if not passwords:
            return {
                'total_count': 0,
                'strength_distribution': {level.name: 0 for level in StrengthLevel},
                'average_entropy': 0.0,
                'average_guess_number': 0.0,
                'common_warnings': {},
                'common_suggestions': {},
            }

        reports = self.evaluate_batch(passwords)

        # Strength distribution
        strength_dist = {level.name: 0 for level in StrengthLevel}
        for report in reports:
            strength_dist[report.score.name] += 1

        # Average entropy
        avg_entropy = sum(r.entropy for r in reports) / len(reports)

        # Average guess number (use log to handle large values)
        log_guesses = []
        for r in reports:
            if r.guess_number > 0:
                log_guesses.append(0 if r.guess_number < 1 else
                                   sum(x * 10**x for x in [r.guess_number]))
        # Simpler: use log-scale mean
        import math
        log_guess_sum = sum(math.log10(max(r.guess_number, 1)) for r in reports)
        avg_log_guess = log_guess_sum / len(reports) if reports else 0
        avg_guess_number = 10 ** avg_log_guess

        # Common warnings
        warning_counts: Dict[str, int] = {}
        for report in reports:
            for warning in report.warnings:
                warning_counts[warning] = warning_counts.get(warning, 0) + 1

        # Common suggestions
        suggestion_counts: Dict[str, int] = {}
        for report in reports:
            for suggestion in report.suggestions:
                suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1

        return {
            'total_count': len(passwords),
            'strength_distribution': strength_dist,
            'average_entropy': avg_entropy,
            'average_guess_number': avg_guess_number,
            'common_warnings': dict(sorted(warning_counts.items(), key=lambda x: -x[1])[:10]),
            'common_suggestions': dict(sorted(suggestion_counts.items(), key=lambda x: -x[1])[:10]),
        }


class EvaluationPipeline:
    """End-to-end evaluation pipeline for password generation systems.

    Combines strength evaluation with generation metrics to provide
    comprehensive assessment of password generation methods.
    """

    def __init__(self):
        self.strength_eval = PasswordStrengthEvaluator()
        self.metrics = EvaluationMetrics()
        self.batch_eval = BatchEvaluator(self.strength_eval)

    def evaluate_generation(
        self,
        generated: List[str],
        targets: List[str],
        generation_time: float = 0.0
    ) -> dict:
        """Evaluate a generation result against target passwords.

        Args:
            generated: List of generated passwords (ranked by preference)
            targets: List of target passwords to find
            generation_time: Time taken to generate (seconds)

        Returns:
            Complete evaluation report with metrics and strength statistics.
        """
        target_set = set(targets)

        # Compute core metrics
        eval_result = EvaluationMetrics.evaluate_ranking(
            generated=generated,
            targets=target_set,
            generation_time=generation_time,
            entropy_fn=lambda p: self.strength_eval.evaluate(p).entropy,
            strength_fn=self.strength_eval.score,
        )

        # Compute additional ranking metrics
        accuracy_metrics = EvaluationMetrics.calculate_accuracy_at_k(
            generated, target_set, k_values=[10, 50, 100, 500, 1000]
        )

        ndcg_100 = EvaluationMetrics.calculate_ndcg(generated, target_set, k=100)
        ndcg_500 = EvaluationMetrics.calculate_ndcg(generated, target_set, k=500)

        # Strength statistics for generated passwords
        strength_stats = self.batch_eval.generate_report(generated[:1000])

        # Combine results
        return {
            'core_metrics': {
                'coverage': eval_result.coverage,
                'hit_rate': eval_result.hit_rate,
                'mrr': eval_result.mrr,
                'diversity': eval_result.diversity,
                'unique_ratio': eval_result.unique_ratio,
                'generation_time': eval_result.generation_time,
            },
            'ranking_metrics': {
                'ndcg@100': ndcg_100,
                'ndcg@500': ndcg_500,
                **accuracy_metrics,
            },
            'strength_statistics': strength_stats,
            'average_entropy': eval_result.avg_entropy,
            'average_strength': eval_result.avg_strength,
            'summary': self._generate_summary(eval_result, target_set),
        }

    def _generate_summary(self, result: EvaluationResult, targets: set) -> str:
        """Generate a human-readable summary of evaluation results."""
        lines = []
        lines.append(f"Hit Rate: {result.hit_rate:.2%} of targets found")
        lines.append(f"Coverage: {result.coverage:.2%} of generated are targets")
        lines.append(f"MRR: {result.mrr:.4f} (higher is better)")
        lines.append(f"Diversity: {result.diversity:.2%} (normalized edit distance)")
        lines.append(f"Unique Ratio: {result.unique_ratio:.2%}")
        lines.append(f"Generation Time: {result.generation_time:.2f}s")

        if result.hit_rate > 0.5:
            lines.append("Overall: GOOD - More than half of targets recovered.")
        elif result.hit_rate > 0.1:
            lines.append("Overall: FAIR - Some targets recovered.")
        else:
            lines.append("Overall: POOR - Few targets recovered.")

        return "\n".join(lines)

    def compare_methods(
        self,
        results: Dict[str, List[str]],
        targets: List[str]
    ) -> Dict[str, dict]:
        """Compare multiple password generation methods.

        Args:
            results: Dict mapping method name to list of generated passwords
            targets: List of target passwords

        Returns:
            Dict mapping method name to its evaluation report.
        """
        comparison = {}
        for method_name, generated in results.items():
            comparison[method_name] = self.evaluate_generation(
                generated=generated,
                targets=targets,
            )
        return comparison

    def benchmark_generation(
        self,
        generator_fn,
        targets: List[str],
        n_candidates: int = 1000,
        repetitions: int = 1
    ) -> dict:
        """Benchmark a password generation function.

        Args:
            generator_fn: Function that takes n_candidates and returns list of passwords
            targets: Target passwords for evaluation
            n_candidates: Number of candidates to generate
            repetitions: Number of times to run for timing

        Returns:
            Benchmark results including timing and quality metrics.
        """
        times = []
        all_generated = []

        for _ in range(repetitions):
            start = time.time()
            generated = generator_fn(n_candidates)
            elapsed = time.time() - start
            times.append(elapsed)
            all_generated.extend(generated)

        avg_time = sum(times) / len(times)

        # Evaluate the last generation result
        evaluation = self.evaluate_generation(
            generated=generated,
            targets=targets,
            generation_time=avg_time,
        )

        return {
            'avg_generation_time': avg_time,
            'times': times,
            'candidates_generated': n_candidates,
            'repetitions': repetitions,
            'candidates_per_second': n_candidates / avg_time if avg_time > 0 else 0,
            **evaluation,
        }

    def generate_comparison_report(
        self,
        comparison: Dict[str, dict]
    ) -> str:
        """Generate a formatted comparison report for multiple methods.

        Args:
            comparison: Dict from compare_methods()

        Returns:
            Formatted string report.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PASSWORD GENERATION METHOD COMPARISON")
        lines.append("=" * 80)

        # Header
        methods = list(comparison.keys())
        header = f"{'Metric':<25}" + "".join(f"{m[:15]:>15}" for m in methods)
        lines.append(header)
        lines.append("-" * len(header))

        # Core metrics
        metrics_to_show = [
            ('Hit Rate', 'core_metrics', 'hit_rate', '{:.4f}'),
            ('Coverage', 'core_metrics', 'coverage', '{:.4f}'),
            ('MRR', 'core_metrics', 'mrr', '{:.4f}'),
            ('Diversity', 'core_metrics', 'diversity', '{:.4f}'),
            ('Unique Ratio', 'core_metrics', 'unique_ratio', '{:.4f}'),
            ('NDCG@100', 'ranking_metrics', 'ndcg@100', '{:.4f}'),
            ('Avg Entropy', None, 'average_entropy', '{:.2f}'),
            ('Avg Strength', None, 'average_strength', '{:.4f}'),
        ]

        for display_name, category, key, fmt in metrics_to_show:
            row = f"{display_name:<25}"
            for method in methods:
                data = comparison[method]
                if category:
                    val = data.get(category, {}).get(key, 0)
                else:
                    val = data.get(key, 0)
                row += f"{fmt.format(val):>15}"
            lines.append(row)

        # Accuracy metrics
        lines.append("")
        lines.append("Accuracy @K:")
        for k in [10, 50, 100, 500, 1000]:
            row = f"Acc@{k:<22}"
            for method in methods:
                val = comparison[method].get('ranking_metrics', {}).get(f'acc@{k}', 0)
                row += f"{val:>15.4f}"
            lines.append(row)

        lines.append("")
        lines.append("=" * 80)

        # Best method summary
        best_by_hitrate = max(methods, key=lambda m: comparison[m]['core_metrics']['hit_rate'])
        best_by_mrr = max(methods, key=lambda m: comparison[m]['core_metrics']['mrr'])

        lines.append(f"Best by Hit Rate: {best_by_hitrate}")
        lines.append(f"Best by MRR: {best_by_mrr}")

        return "\n".join(lines)
