"""Password entropy calculation: Shannon entropy, charset entropy, pattern entropy."""

from dataclasses import dataclass
import math
from typing import List
from collections import Counter


@dataclass
class EntropyReport:
    """Complete entropy report for a password."""
    password: str
    shannon_entropy: float
    charset_entropy: float
    pattern_entropy: float
    charset_size: int
    bits: float


class EntropyCalculator:
    """Password information entropy calculator.

    Provides multiple entropy measures:
    - Shannon entropy: based on character frequency distribution
    - Charset entropy: based on character set size and password length
    - Pattern entropy: adjusted for common patterns
    """

    CHARSET_SIZES = {
        'digits': 10,
        'lowercase': 26,
        'uppercase': 26,
        'letters': 52,
        'alphanumeric': 62,
        'ascii_printable': 95,
        'common_special': 33,
    }

    @staticmethod
    def shannon_entropy(password: str) -> float:
        """Calculate Shannon information entropy.

        H = -sum(p_i * log2(p_i)) where p_i is the frequency of each character.

        Higher values indicate more randomness in character distribution.
        """
        if not password:
            return 0.0

        length = len(password)
        char_counts = Counter(password)

        entropy = 0.0
        for count in char_counts.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)

        # Scale to total bits: entropy per char * length
        return entropy * length

    @staticmethod
    def charset_entropy(password: str) -> float:
        """Calculate entropy based on character set size.

        entropy = length * log2(charset_size)

        This assumes characters are uniformly distributed within the detected charset.
        """
        if not password:
            return 0.0

        charset_size = EntropyCalculator.detect_charset_size(password)
        if charset_size <= 0:
            return 0.0

        return len(password) * math.log2(charset_size)

    @staticmethod
    def detect_charset_size(password: str) -> int:
        """Detect the character set size used in the password.

        Determines which character classes are present (lowercase, uppercase,
        digits, specials) and returns the total alphabet size.
        """
        if not password:
            return 0

        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)

        charset_size = 0
        if has_lower:
            charset_size += 26
        if has_upper:
            charset_size += 26
        if has_digit:
            charset_size += 10
        if has_special:
            charset_size += 33  # common special characters

        return charset_size

    @staticmethod
    def pattern_entropy(password: str) -> float:
        """Calculate entropy adjusted for common patterns.

        Starts with charset entropy and applies penalties for:
        - Repeated characters (aaa, 111)
        - Sequential characters (abc, 123)
        - Common keyboard patterns (qwerty, asdf)
        - Simple substitutions (p@ssw0rd)
        """
        if not password:
            return 0.0

        base_entropy = EntropyCalculator.charset_entropy(password)
        penalty = 0.0
        length = len(password)

        # Penalty for repeated characters (3+ consecutive identical)
        repeat_penalty = 0.0
        for i in range(length - 2):
            if password[i] == password[i + 1] == password[i + 2]:
                # Penalty proportional to how many repeated chars
                j = i + 2
                while j < length and password[j] == password[i]:
                    j += 1
                repeat_len = j - i
                repeat_penalty += min(repeat_len * 2.0, base_entropy * 0.3)
                i = j  # skip ahead (note: loop will still increment i)

        penalty += min(repeat_penalty, base_entropy * 0.4)

        # Penalty for sequential characters
        seq_penalty = 0.0
        for i in range(length - 2):
            c1, c2, c3 = ord(password[i]), ord(password[i + 1]), ord(password[i + 2])
            if c2 - c1 == 1 and c3 - c2 == 1:
                seq_penalty += 3.0  # 3 bits penalty per sequential triplet
            elif c1 - c2 == 1 and c2 - c3 == 1:
                seq_penalty += 3.0

        penalty += min(seq_penalty, base_entropy * 0.3)

        # Penalty for common keyboard patterns
        keyboard_patterns = [
            'qwerty', 'qwert', 'werty', 'asdf', 'asdfg', 'zxcv', 'zxcvb',
            'qazwsx', '1qaz', '2wsx', '!qaz',
            'yuiop', 'hjkl', 'bnm',
        ]
        pwd_lower = password.lower()
        for pattern in keyboard_patterns:
            if pattern in pwd_lower:
                penalty += min(len(pattern) * 1.5, base_entropy * 0.25)
                break

        # Penalty for date-like patterns (4+ consecutive digits that look like a year)
        import re
        year_matches = re.findall(r'(?:19|20)\d{2}', password)
        if year_matches:
            penalty += min(len(year_matches) * 4.0, base_entropy * 0.2)

        # Penalty for simple substitution patterns
        substitutions = {'@': 'a', '3': 'e', '1': 'i', '0': 'o', '$': 's', '5': 's', '7': 't'}
        sub_count = sum(1 for c in password if c in substitutions)
        if sub_count > 0:
            penalty += min(sub_count * 1.5, base_entropy * 0.15)

        # Ensure pattern entropy doesn't go below a minimum based on length
        adjusted = base_entropy - penalty
        minimum_entropy = length * 1.0  # at least 1 bit per character
        return max(adjusted, minimum_entropy)

    def evaluate(self, password: str) -> EntropyReport:
        """Produce a complete entropy report for a password.

        Returns an EntropyReport with all entropy measures.
        """
        shannon = self.shannon_entropy(password)
        charset_e = self.charset_entropy(password)
        pattern_e = self.pattern_entropy(password)
        charset_size = self.detect_charset_size(password)

        # Overall bits: use the minimum of the entropy measures as a conservative estimate
        bits = min(shannon, charset_e, pattern_e) if password else 0.0
        # But if we have a reasonable password, use pattern_entropy as the main metric
        if password and len(password) >= 4:
            bits = pattern_e

        return EntropyReport(
            password=password,
            shannon_entropy=shannon,
            charset_entropy=charset_e,
            pattern_entropy=pattern_e,
            charset_size=charset_size,
            bits=bits,
        )
