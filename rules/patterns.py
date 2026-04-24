"""
Password pattern matching and generation.

Provides regex-based pattern recognition, structural analysis, and
template-driven password generation from personal-information features.
"""

import re
import string
import itertools
from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class PasswordPattern:
    """A recognisable password pattern."""

    name: str
    regex: str
    description: str
    template: str  # e.g. "{name}{digits}"


# ======================================================================
# Pattern Matcher
# ======================================================================

class PatternMatcher:
    """Match passwords against common structural patterns."""

    COMMON_PATTERNS: List[PasswordPattern] = [
        PasswordPattern(
            name="name_digits",
            regex=r"^[a-zA-Z]+\d+$",
            description="Letters followed by digits",
            template="{name}{digits}",
        ),
        PasswordPattern(
            name="digits_name",
            regex=r"^\d+[a-zA-Z]+$",
            description="Digits followed by letters",
            template="{digits}{name}",
        ),
        PasswordPattern(
            name="capitalized_digits",
            regex=r"^[A-Z][a-z]+\d+$",
            description="Capitalized word followed by digits",
            template="{cap_name}{digits}",
        ),
        PasswordPattern(
            name="name_digits_special",
            regex=r"^[a-zA-Z]+\d+[!@#$%^&*]+$",
            description="Letters, digits, then special characters",
            template="{name}{digits}{special}",
        ),
        PasswordPattern(
            name="capitalized_digits_special",
            regex=r"^[A-Z][a-z]+\d+[!@#$%^&*]+$",
            description="Capitalized word, digits, special",
            template="{cap_name}{digits}{special}",
        ),
        PasswordPattern(
            name="word_word_digits",
            regex=r"^[a-zA-Z]+[_.][a-zA-Z]+\d+$",
            description="Two words separated by . or _ then digits",
            template="{word1}{sep}{word2}{digits}",
        ),
        PasswordPattern(
            name="digits_only",
            regex=r"^\d{4,}$",
            description="Numeric-only password",
            template="{digits}",
        ),
        PasswordPattern(
            name="lower_only",
            regex=r"^[a-z]{4,}$",
            description="Lowercase-only password",
            template="{lower}",
        ),
        PasswordPattern(
            name="leet_speak",
            regex=r"[4@3€1!|0$5]",
            description="Contains leet-speak substitutions",
            template="{leet}",
        ),
        PasswordPattern(
            name="keyboard_walk",
            regex=r"(qwerty|asdf|zxcv|qaz|1qaz|1234)",
            description="Contains keyboard-walk sequences",
            template="{keyboard}",
        ),
        PasswordPattern(
            name="date_name",
            regex=r"^\d{4,8}[a-zA-Z]+$",
            description="Date-like digits followed by letters",
            template="{date}{name}",
        ),
        PasswordPattern(
            name="name_date",
            regex=r"^[a-zA-Z]+\d{4,8}$",
            description="Letters followed by date-like digits",
            template="{name}{date}",
        ),
    ]

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    @staticmethod
    def match(password: str) -> List[PasswordPattern]:
        """Return all patterns that *password* matches."""
        results: List[PasswordPattern] = []
        for pat in PatternMatcher.COMMON_PATTERNS:
            if re.search(pat.regex, password):
                results.append(pat)
        return results

    # ------------------------------------------------------------------
    # Structural analysis
    # ------------------------------------------------------------------

    @staticmethod
    def extract_structure(password: str) -> str:
        """Extract the high-level structure of a password.

        Uses compact run-length encoding:
        - **L** for letters
        - **D** for digits
        - **S** for special characters

        Example::

            >>> PatternMatcher.extract_structure("hello123!")
            'L5D3S1'
        """
        if not password:
            return ""

        structure: List[str] = []
        current_type: Optional[str] = None
        count = 0

        for ch in password:
            if ch.isalpha():
                t = "L"
            elif ch.isdigit():
                t = "D"
            else:
                t = "S"

            if t == current_type:
                count += 1
            else:
                if current_type is not None:
                    structure.append(f"{current_type}{count}")
                current_type = t
                count = 1

        if current_type is not None:
            structure.append(f"{current_type}{count}")

        return "".join(structure)

    # ------------------------------------------------------------------
    # Component extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_components(password: str) -> Dict[str, str]:
        """Split a password into its typed components.

        Returns an ordered dict like
        ``{"L": "hello", "D": "123", "S": "!"}``
        """
        if not password:
            return {}

        components: Dict[str, List[str]] = {}
        current_type: Optional[str] = None
        current_chars: List[str] = []

        for ch in password:
            if ch.isalpha():
                t = "L"
            elif ch.isdigit():
                t = "D"
            else:
                t = "S"

            if t != current_type:
                if current_type is not None and current_chars:
                    components[current_type] = components.get(current_type, [])
                    components[current_type].append("".join(current_chars))
                current_type = t
                current_chars = [ch]
            else:
                current_chars.append(ch)

        if current_type is not None and current_chars:
            components[current_type] = components.get(current_type, [])
            components[current_type].append("".join(current_chars))

        # Flatten to single strings (take last run of each type)
        flat: Dict[str, str] = {}
        for key, runs in components.items():
            flat[key] = runs[-1]
        return flat


# ======================================================================
# Pattern Generator
# ======================================================================

class PatternGenerator:
    """Generate password guesses from structural patterns and templates."""

    def __init__(self):
        self.matcher = PatternMatcher()

    # ------------------------------------------------------------------
    # Pattern-based generation
    # ------------------------------------------------------------------

    def generate_from_pattern(
        self,
        pattern: str,
        components: Dict[str, List[str]],
    ) -> List[str]:
        """Generate passwords matching a structural *pattern*.

        Parameters
        ----------
        pattern:
            Structural specification like ``"L4D3"`` (4 letters + 3 digits).
        components:
            Keys ``"L"``, ``"D"``, ``"S"`` map to lists of candidate strings.
            For letters the strings are used as-is (possibly truncated).
            For digits and specials, the strings are used directly.
        """
        # Parse pattern into segments, e.g. "L4D3S1" -> [("L",4),("D",3),("S",1)]
        segments: List[Tuple[str, int]] = []
        for m in re.finditer(r"([LDS])(\d+)", pattern):
            seg_type = m.group(1)
            seg_len = int(m.group(2))
            segments.append((seg_type, seg_len))

        if not segments:
            return []

        # Collect candidate lists for each segment
        segment_candidates: List[List[str]] = []
        for seg_type, seg_len in segments:
            cands = components.get(seg_type, [])
            if not cands:
                # Fall back to defaults
                if seg_type == "D":
                    cands = [str(i).zfill(seg_len) for i in range(10 ** min(seg_len, 4))]
                    cands = cands[:50]  # cap
                elif seg_type == "S":
                    cands = list("!@#$%&*")
                else:
                    cands = ["aaaa"]
            # Trim candidates to the requested segment length
            trimmed = [c[:seg_len].ljust(seg_len, c[-1] if c else "a") for c in cands]
            segment_candidates.append(trimmed[:100])  # cap per segment

        # Cartesian product
        results: List[str] = []
        for combo in itertools.islice(itertools.product(*segment_candidates), 500):
            results.append("".join(combo))
        return results

    # ------------------------------------------------------------------
    # Template-based generation
    # ------------------------------------------------------------------

    def generate_from_template(
        self,
        template: str,
        values: Dict[str, List[str]],
    ) -> List[str]:
        """Expand a *template* by substituting ``{key}`` placeholders.

        Parameters
        ----------
        template:
            e.g. ``"{name}{year}!"``
        values:
            e.g. ``{"name": ["john", "John"], "year": ["1990", "90"]}``
        """
        # Find all placeholders
        placeholders = re.findall(r"\{(\w+)\}", template)
        if not placeholders:
            return [template]

        # Build candidate lists for each placeholder
        placeholder_cands: List[List[str]] = []
        for ph in placeholders:
            cands = values.get(ph, [])
            if not cands:
                placeholder_cands.append([""])
            else:
                placeholder_cands.append(cands)

        results: List[str] = []
        for combo in itertools.islice(itertools.product(*placeholder_cands), 500):
            subs = dict(zip(placeholders, combo))
            results.append(template.format(**subs))
        return results

    # ------------------------------------------------------------------
    # Personal-information combination generation
    # ------------------------------------------------------------------

    def generate_common_combinations(self, features) -> List[str]:
        """Generate common password guesses from a :class:`TargetFeatures` object.

        This uses a fixed set of templates and the personal information encoded
        in *features* to produce likely password candidates.
        """
        from utils.feature_utils import FeatureVectorizer

        vectorizer = FeatureVectorizer()
        comps = vectorizer.generate_password_components(features)

        passwords: List[str] = []

        # ---- Helper lambdas ----
        names = comps.get("names", [])
        dates = comps.get("dates", [])
        numbers = comps.get("numbers", [])
        words = comps.get("words", [])
        specials = comps.get("special_chars", ["!", "@", "#", "$"])

        # 1) name + date
        for name in names:
            for date in dates:
                passwords.append(f"{name}{date}")

        # 2) name + common numbers
        common_suffixes = ["123", "1234", "12345", "123456", "1", "12", "321", "000"]
        for name in names:
            for suffix in common_suffixes:
                passwords.append(f"{name}{suffix}")

        # 3) name + special
        for name in names:
            for sp in specials[:5]:
                passwords.append(f"{name}{sp}")

        # 4) name + date + special
        for name in names[:10]:
            for date in dates[:5]:
                for sp in specials[:3]:
                    passwords.append(f"{name}{date}{sp}")

        # 5) word + digits
        for word in words[:20]:
            for num in numbers[:10]:
                passwords.append(f"{word}{num}")
            for suffix in common_suffixes[:4]:
                passwords.append(f"{word}{suffix}")

        # 6) capitalized variants of all above (applied post-hoc)
        base_set = set(passwords)
        for pw in list(base_set):
            cap = pw.capitalize()
            if cap != pw:
                passwords.append(cap)
            upper = pw.upper()
            if upper != pw:
                passwords.append(upper)

        # 7) Reverse
        for pw in list(passwords[:200]):
            rev = pw[::-1]
            if rev != pw:
                passwords.append(rev)

        # 8) Leet substitutions on a subset
        leet_map = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
        for pw in passwords[:100]:
            leet = []
            for ch in pw:
                if ch.lower() in leet_map:
                    leet.append(leet_map[ch.lower()])
                else:
                    leet.append(ch)
            leet_pw = "".join(leet)
            if leet_pw != pw:
                passwords.append(leet_pw)

        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for pw in passwords:
            if pw and pw not in seen:
                seen.add(pw)
                unique.append(pw)

        return unique
