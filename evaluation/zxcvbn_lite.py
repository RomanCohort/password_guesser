"""Lightweight zxcvbn-style password strength evaluation."""

from dataclasses import dataclass
from typing import List, Optional
import math
import re


@dataclass
class PatternMatch:
    """A detected pattern within a password."""
    pattern: str  # 'dictionary', 'date', 'digits', 'keyboard', 'repeat', 'sequence', 'regex'
    token: str
    start: int
    end: int
    entropy: float


class ZxcvbnLite:
    """Lightweight zxcvbn password strength evaluator.

    Detects common patterns in passwords and estimates crack resistance:
    - Dictionary words (common passwords, names)
    - Date patterns
    - Digit sequences
    - Keyboard patterns
    - Repeated characters
    - Sequential characters (abc, 123)
    """

    # Common passwords (top ~100)
    COMMON_PASSWORDS = {
        'password', '123456', '12345678', 'qwerty', 'abc123',
        'monkey', '1234567', 'letmein', 'trustno1', 'dragon',
        'baseball', 'iloveyou', 'master', 'sunshine', 'ashley',
        'bailey', 'passw0rd', 'shadow', '123123', '654321',
        'superman', 'qazwsx', 'michael', 'football', 'password1',
        'password123', 'batman', 'admin', 'admin123', 'welcome',
        'hello', 'charlie', 'donald', 'login', 'qwerty123',
        'mustang', 'access', 'joshua', 'jesse', '1234567890',
        'computer', 'starwars', '121212', 'george', 'andrea',
        'amanda', 'nicole', 'jessica', 'hannah', 'daniel',
        'whatever', 'tigger', 'thomas', 'robert', 'pepper',
        'hunter', 'silver', 'test', 'test123', 'root',
        'pass', 'guest', 'master', 'changeme', 'default',
    }

    # Keyboard layouts for pattern detection
    KEYBOARD_ROWS = [
        'qwertyuiop',
        'asdfghjkl',
        'zxcvbnm',
        '1234567890',
        '!@#$%^&*()',
    ]

    # Common names for dictionary matching
    COMMON_NAMES = {
        'john', 'james', 'robert', 'michael', 'david', 'william',
        'mary', 'patricia', 'jennifer', 'linda', 'elizabeth',
        'zhang', 'wang', 'li', 'chen', 'liu', 'yang', 'zhao',
        'richard', 'thomas', 'charles', 'christopher', 'daniel',
        'matthew', 'anthony', 'mark', 'donald', 'steven', 'paul',
        'andrew', 'joshua', 'kenneth', 'kevin', 'brian', 'george',
        'barbara', 'susan', 'jessica', 'sarah', 'karen', 'lisa',
        'nancy', 'betty', 'margaret', 'sandra', 'ashley', 'dorothy',
    }

    def evaluate(self, password: str) -> dict:
        """Evaluate password strength.

        Returns a dictionary with:
        - score: 0-4 (weak to strong)
        - guesses: estimated number of guesses needed
        - crack_time: human-readable crack time description
        - crack_time_seconds: crack time in seconds
        - patterns: list of detected PatternMatch objects
        - entropy: total entropy estimate (bits)
        - warning: warning message if applicable
        - suggestions: list of improvement suggestions
        """
        if not password:
            return {
                'score': 0,
                'guesses': 0,
                'crack_time': 'instant',
                'crack_time_seconds': 0,
                'patterns': [],
                'entropy': 0,
                'warning': 'Empty password',
                'suggestions': ['Use a password with at least 8 characters.'],
            }

        # Collect all pattern matches
        patterns: List[PatternMatch] = []
        patterns.extend(self._match_dictionary(password))
        patterns.extend(self._match_date(password))
        patterns.extend(self._match_digits(password))
        patterns.extend(self._match_keyboard(password))
        patterns.extend(self._match_repeat(password))
        patterns.extend(self._match_sequence(password))

        # Sort patterns by start position
        patterns.sort(key=lambda p: p.start)

        # Remove overlapping patterns (keep higher entropy ones)
        patterns = self._remove_overlaps(patterns)

        # Estimate total guesses
        guesses = self._estimate_guesses(patterns, len(password))

        # Calculate entropy from guesses
        entropy = math.log2(max(guesses, 1))

        # Calculate crack time (assume 10 billion guesses per second for offline attack)
        guesses_per_second = 10_000_000_000
        crack_time_seconds = guesses / guesses_per_second
        crack_time = self._crack_time_display(crack_time_seconds)

        # Determine score (0-4)
        score = self._entropy_to_score(entropy)

        # Generate warnings and suggestions
        warning = self._generate_warning(password, patterns)
        suggestions = self._generate_suggestions(password, patterns, score)

        return {
            'score': score,
            'guesses': guesses,
            'crack_time': crack_time,
            'crack_time_seconds': crack_time_seconds,
            'patterns': patterns,
            'entropy': entropy,
            'warning': warning,
            'suggestions': suggestions,
        }

    def _match_dictionary(self, password: str) -> List[PatternMatch]:
        """Match dictionary words and common passwords in the password."""
        matches = []
        pwd_lower = password.lower()

        # Check if entire password is a common password
        if pwd_lower in self.COMMON_PASSWORDS:
            matches.append(PatternMatch(
                pattern='dictionary',
                token=password,
                start=0,
                end=len(password),
                entropy=math.log2(len(self.COMMON_PASSWORDS)),
            ))
            return matches

        # Check for common password substrings
        for common in self.COMMON_PASSWORDS:
            if len(common) < 3:
                continue
            idx = pwd_lower.find(common)
            while idx != -1:
                matches.append(PatternMatch(
                    pattern='dictionary',
                    token=password[idx:idx + len(common)],
                    start=idx,
                    end=idx + len(common),
                    entropy=math.log2(len(self.COMMON_PASSWORDS)),
                ))
                idx = pwd_lower.find(common, idx + 1)

        # Check for name matches
        for name in self.COMMON_NAMES:
            if len(name) < 3:
                continue
            idx = pwd_lower.find(name)
            while idx != -1:
                matches.append(PatternMatch(
                    pattern='dictionary',
                    token=password[idx:idx + len(name)],
                    start=idx,
                    end=idx + len(name),
                    entropy=math.log2(len(self.COMMON_NAMES)),
                ))
                idx = pwd_lower.find(name, idx + 1)

        return matches

    def _match_date(self, password: str) -> List[PatternMatch]:
        """Match date patterns (YYYY, MMDD, DDMM, MMYY, etc.)."""
        matches = []

        # Match full years: 19xx, 20xx
        for m in re.finditer(r'(?:19|20)\d{2}', password):
            matches.append(PatternMatch(
                pattern='date',
                token=m.group(),
                start=m.start(),
                end=m.end(),
                entropy=math.log2(200),  # ~200 years
            ))

        # Match MMDD or DDMM patterns: 0101-1231
        for m in re.finditer(r'(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])', password):
            token = m.group()
            # Avoid re-matching year patterns
            already_matched = any(
                p.start == m.start() and p.pattern == 'date'
                for p in matches
            )
            if not already_matched:
                matches.append(PatternMatch(
                    pattern='date',
                    token=token,
                    start=m.start(),
                    end=m.end(),
                    entropy=math.log2(366),
                ))

        # Match DDMM patterns
        for m in re.finditer(r'(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])', password):
            token = m.group()
            already_matched = any(
                p.start == m.start() and p.pattern == 'date'
                for p in matches
            )
            if not already_matched:
                matches.append(PatternMatch(
                    pattern='date',
                    token=token,
                    start=m.start(),
                    end=m.end(),
                    entropy=math.log2(366),
                ))

        return matches

    def _match_digits(self, password: str) -> List[PatternMatch]:
        """Match long digit sequences."""
        matches = []
        for m in re.finditer(r'\d{3,}', password):
            token = m.group()
            entropy = math.log2(10 ** len(token))
            matches.append(PatternMatch(
                pattern='digits',
                token=token,
                start=m.start(),
                end=m.end(),
                entropy=entropy,
            ))
        return matches

    def _match_keyboard(self, password: str) -> List[PatternMatch]:
        """Match keyboard walk patterns (qwerty, asdf, etc.)."""
        matches = []
        pwd_lower = password.lower()

        for row in self.KEYBOARD_ROWS:
            # Check forward and backward keyboard walks of length 4+
            for length in range(len(row), 3, -1):
                for start in range(len(row) - length + 1):
                    pattern = row[start:start + length]
                    idx = pwd_lower.find(pattern)
                    while idx != -1:
                        matches.append(PatternMatch(
                            pattern='keyboard',
                            token=password[idx:idx + length],
                            start=idx,
                            end=idx + length,
                            entropy=math.log2(max(len(row) * 2, 1)),
                        ))
                        idx = pwd_lower.find(pattern, idx + 1)

                    # Reverse pattern
                    rev_pattern = pattern[::-1]
                    idx = pwd_lower.find(rev_pattern)
                    while idx != -1:
                        matches.append(PatternMatch(
                            pattern='keyboard',
                            token=password[idx:idx + length],
                            start=idx,
                            end=idx + length,
                            entropy=math.log2(max(len(row) * 2, 1)),
                        ))
                        idx = pwd_lower.find(rev_pattern, idx + 1)

        return matches

    def _match_repeat(self, password: str) -> List[PatternMatch]:
        """Match repeated characters (aaa, 111, etc.)."""
        matches = []
        if not password:
            return matches

        i = 0
        while i < len(password):
            char = password[i]
            j = i + 1
            while j < len(password) and password[j] == char:
                j += 1
            repeat_len = j - i
            if repeat_len >= 3:
                entropy = math.log2(max(94 * repeat_len, 1))  # 94 printable ASCII chars
                matches.append(PatternMatch(
                    pattern='repeat',
                    token=password[i:j],
                    start=i,
                    end=j,
                    entropy=entropy,
                ))
            i = j

        return matches

    def _match_sequence(self, password: str) -> List[PatternMatch]:
        """Match sequential characters (abc, 123, xyz, etc.)."""
        matches = []
        if len(password) < 3:
            return matches

        i = 0
        while i < len(password) - 2:
            # Check forward sequence
            if self._is_sequential(password[i], password[i + 1], password[i + 2]):
                j = i + 2
                while j < len(password) - 1 and self._is_sequential(password[j], password[j + 1]):
                    j += 1
                # j is the last char in sequence
                seq_len = j - i + 1
                if seq_len >= 3:
                    entropy = math.log2(max(26 * 2, 1))  # forward or backward
                    matches.append(PatternMatch(
                        pattern='sequence',
                        token=password[i:j + 1],
                        start=i,
                        end=j + 1,
                        entropy=entropy,
                    ))
                i = j + 1
                continue

            # Check backward sequence
            if self._is_sequential_reverse(password[i], password[i + 1], password[i + 2]):
                j = i + 2
                while j < len(password) - 1 and self._is_sequential_reverse(password[j], password[j + 1]):
                    j += 1
                seq_len = j - i + 1
                if seq_len >= 3:
                    entropy = math.log2(max(26 * 2, 1))
                    matches.append(PatternMatch(
                        pattern='sequence',
                        token=password[i:j + 1],
                        start=i,
                        end=j + 1,
                        entropy=entropy,
                    ))
                i = j + 1
                continue

            i += 1

        return matches

    @staticmethod
    def _is_sequential(c1: str, c2: str, c3: str = None) -> bool:
        """Check if characters are in ascending sequence."""
        if c3 is not None:
            return ord(c2) - ord(c1) == 1 and ord(c3) - ord(c2) == 1
        return ord(c2) - ord(c1) == 1

    @staticmethod
    def _is_sequential_reverse(c1: str, c2: str, c3: str = None) -> bool:
        """Check if characters are in descending sequence."""
        if c3 is not None:
            return ord(c1) - ord(c2) == 1 and ord(c2) - ord(c3) == 1
        return ord(c1) - ord(c2) == 1

    def _remove_overlaps(self, patterns: List[PatternMatch]) -> List[PatternMatch]:
        """Remove overlapping patterns, keeping the ones with lower entropy (stronger pattern)."""
        if not patterns:
            return patterns

        # Sort by start, then by length (longer first)
        patterns.sort(key=lambda p: (p.start, -(p.end - p.start)))

        result = [patterns[0]]
        for p in patterns[1:]:
            last = result[-1]
            if p.start >= last.end:
                # No overlap
                result.append(p)
            else:
                # Overlap: keep the one with lower entropy (more predictable = stronger match)
                if p.entropy < last.entropy:
                    result[-1] = p

        return result

    def _estimate_guesses(self, patterns: List[PatternMatch], length: int) -> float:
        """Estimate the total number of guesses needed to crack the password.

        Combines pattern entropies and adds brute-force component for unmatched parts.
        """
        if not patterns:
            # No patterns detected: pure brute force
            charset_size = self._detect_charset_from_password_of_length(length)
            return charset_size ** length

        # Find covered positions
        covered = set()
        total_pattern_entropy = 0.0

        for p in patterns:
            total_pattern_entropy += p.entropy
            for i in range(p.start, p.end):
                covered.add(i)

        # Uncovered positions need brute-force estimation
        uncovered_count = length - len(covered)
        brute_force_entropy = 0.0
        if uncovered_count > 0:
            charset_size = 95  # assume full printable ASCII
            brute_force_entropy = uncovered_count * math.log2(charset_size)

        # Total guesses = 2^total_entropy
        total_entropy = total_pattern_entropy + brute_force_entropy

        # Minimum entropy based on password length (sanity check)
        min_entropy = length * 1.0
        total_entropy = max(total_entropy, min_entropy)

        return 2 ** total_entropy

    @staticmethod
    def _detect_charset_from_password_of_length(length: int) -> int:
        """Return a default charset size for brute-force estimation."""
        return 95  # printable ASCII

    def _entropy_to_score(self, entropy: float) -> int:
        """Convert entropy (bits) to a 0-4 score."""
        if entropy < 10:
            return 0
        elif entropy < 20:
            return 1
        elif entropy < 35:
            return 2
        elif entropy < 55:
            return 3
        else:
            return 4

    def _crack_time_display(self, seconds: float) -> str:
        """Convert seconds to a human-readable time description."""
        if seconds < 0.001:
            return 'instant'
        if seconds < 1:
            return 'less than a second'
        if seconds < 60:
            return f'{int(seconds)} seconds'
        if seconds < 3600:
            return f'{int(seconds / 60)} minutes'
        if seconds < 86400:
            return f'{int(seconds / 3600)} hours'
        if seconds < 86400 * 30:
            return f'{int(seconds / 86400)} days'
        if seconds < 86400 * 365:
            return f'{int(seconds / (86400 * 30))} months'
        if seconds < 86400 * 365 * 100:
            return f'{int(seconds / (86400 * 365))} years'
        if seconds < 86400 * 365 * 1e6:
            return f'{int(seconds / (86400 * 365 * 100))} centuries'
        if seconds < 86400 * 365 * 1e9:
            return f'{int(seconds / (86400 * 365 * 1e6))} million years'
        return 'centuries'

    def _generate_warning(self, password: str, patterns: List[PatternMatch]) -> str:
        """Generate a warning message based on detected patterns."""
        if not password:
            return ''

        # Check for very common passwords
        if password.lower() in self.COMMON_PASSWORDS:
            return 'This is a commonly used password.'

        for p in patterns:
            if p.pattern == 'keyboard':
                return 'Keyboard patterns are easy to guess.'
            if p.pattern == 'repeat':
                return 'Repeated characters are easy to guess.'
            if p.pattern == 'sequence':
                return 'Character sequences are easy to guess.'

        if len(password) < 6:
            return 'This password is very short.'

        for p in patterns:
            if p.pattern == 'dictionary':
                return 'This contains a common word or name.'
            if p.pattern == 'date':
                return 'This contains a date pattern.'

        return ''

    def _generate_suggestions(
        self, password: str, patterns: List[PatternMatch], score: int
    ) -> List[str]:
        """Generate suggestions for improving password strength."""
        suggestions = []

        if len(password) < 8:
            suggestions.append('Use at least 8 characters.')
        if len(password) < 12 and score < 3:
            suggestions.append('Use 12 or more characters for better security.')

        # Check character diversity
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)

        char_classes = sum([has_lower, has_upper, has_digit, has_special])
        if char_classes < 3:
            suggestions.append('Mix uppercase, lowercase, digits, and special characters.')

        # Check for dictionary words
        has_dictionary = any(p.pattern == 'dictionary' for p in patterns)
        if has_dictionary:
            suggestions.append('Avoid common words and names.')

        # Check for patterns
        has_pattern = any(p.pattern in ('keyboard', 'repeat', 'sequence') for p in patterns)
        if has_pattern:
            suggestions.append('Avoid keyboard patterns, repeated characters, and sequences.')

        # Check for dates
        has_date = any(p.pattern == 'date' for p in patterns)
        if has_date:
            suggestions.append('Avoid using dates.')

        if score <= 1 and not suggestions:
            suggestions.append('Consider using a passphrase or a randomly generated password.')

        if not suggestions and score >= 3:
            suggestions.append('Good password strength.')

        return suggestions
