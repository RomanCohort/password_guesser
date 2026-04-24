"""Comprehensive password strength evaluation combining entropy and zxcvbn analysis."""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import IntEnum

from evaluation.entropy import EntropyCalculator, EntropyReport
from evaluation.zxcvbn_lite import ZxcvbnLite, PatternMatch


class StrengthLevel(IntEnum):
    """Password strength levels."""
    VERY_WEAK = 0
    WEAK = 1
    FAIR = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class StrengthReport:
    """Comprehensive password strength report."""
    password: str
    score: StrengthLevel
    entropy: float
    charset_entropy: float
    pattern_entropy: float
    guess_number: float
    crack_time: str
    crack_time_seconds: float
    patterns: List[str]
    warnings: List[str]
    suggestions: List[str]
    details: dict = field(default_factory=dict)


class PasswordStrengthEvaluator:
    """Multi-dimensional password strength evaluator.

    Combines entropy analysis and zxcvbn-style pattern detection
    for comprehensive password assessment.
    """

    def __init__(self):
        self.zxcvbn = ZxcvbnLite()
        self.entropy_calc = EntropyCalculator()

    def evaluate(self, password: str) -> StrengthReport:
        """Comprehensively evaluate password strength.

        Combines results from entropy calculation and zxcvbn analysis
        to produce a unified strength report.
        """
        # Entropy analysis
        entropy_report = self.entropy_calc.evaluate(password)

        # Zxcvbn analysis
        zxcvbn_result = self.zxcvbn.evaluate(password)

        # Determine overall strength level
        # Combine zxcvbn score with entropy-based assessment
        zxcvbn_score = zxcvbn_result['score']
        entropy_bits = entropy_report.bits

        # Map entropy bits to a score
        if entropy_bits < 10:
            entropy_score = 0
        elif entropy_bits < 20:
            entropy_score = 1
        elif entropy_bits < 35:
            entropy_score = 2
        elif entropy_bits < 55:
            entropy_score = 3
        else:
            entropy_score = 4

        # Take the lower of the two scores (more conservative)
        overall_score_val = min(zxcvbn_score, entropy_score)

        # Adjust based on password length
        if len(password) < 6:
            overall_score_val = min(overall_score_val, 1)
        elif len(password) >= 16 and overall_score_val < 3:
            overall_score_val = min(overall_score_val + 1, 4)

        overall_score = StrengthLevel(overall_score_val)

        # Collect patterns as string descriptions
        pattern_descriptions = []
        for p in zxcvbn_result['patterns']:
            if isinstance(p, PatternMatch):
                pattern_descriptions.append(f"{p.pattern}: '{p.token}' at position {p.start}-{p.end}")
            else:
                pattern_descriptions.append(str(p))

        # Collect warnings
        warnings = []
        if zxcvbn_result.get('warning'):
            warnings.append(zxcvbn_result['warning'])

        # Add entropy-based warnings
        if entropy_report.shannon_entropy < len(password) * 0.5:
            warnings.append('Low character diversity detected.')

        if entropy_report.charset_size <= 10:
            warnings.append('Password uses only digits.')

        # Collect suggestions
        suggestions = zxcvbn_result.get('suggestions', [])

        # Build details dict
        details = {
            'length': len(password),
            'shannon_entropy': entropy_report.shannon_entropy,
            'charset_size': entropy_report.charset_size,
            'zxcvbn_score': zxcvbn_score,
            'entropy_score': entropy_score,
            'final_score': overall_score_val,
        }

        return StrengthReport(
            password=password,
            score=overall_score,
            entropy=entropy_report.bits,
            charset_entropy=entropy_report.charset_entropy,
            pattern_entropy=entropy_report.pattern_entropy,
            guess_number=zxcvbn_result.get('guesses', 0),
            crack_time=zxcvbn_result.get('crack_time', 'unknown'),
            crack_time_seconds=zxcvbn_result.get('crack_time_seconds', 0),
            patterns=pattern_descriptions,
            warnings=warnings,
            suggestions=suggestions,
            details=details,
        )

    def batch_evaluate(self, passwords: List[str]) -> List[StrengthReport]:
        """Evaluate a batch of passwords.

        Returns a list of StrengthReport objects in the same order.
        """
        return [self.evaluate(pwd) for pwd in passwords]

    def score(self, password: str) -> float:
        """Quick strength score from 0.0 to 1.0.

        Useful for fast comparison without generating a full report.
        """
        report = self.evaluate(password)
        return float(report.score) / 4.0

    def is_weak(self, password: str) -> bool:
        """Check if a password is weak (score <= 1)."""
        report = self.evaluate(password)
        return report.score <= StrengthLevel.WEAK

    def compare(self, password1: str, password2: str) -> int:
        """Compare the strength of two passwords.

        Returns:
            -1 if password1 is weaker than password2
             0 if they are roughly equal
             1 if password1 is stronger than password2
        """
        s1 = self.score(password1)
        s2 = self.score(password2)

        if s1 < s2 - 0.05:
            return -1
        elif s1 > s2 + 0.05:
            return 1
        else:
            return 0
