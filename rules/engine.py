"""
Core password rule engine.

Provides a composable, extensible system for generating password variants
by applying transformation rules to base strings.
"""

import re
import string
from typing import Callable, Dict, List, Optional, Tuple

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """A single transformation rule consisting of a name and a callable."""

    name: str
    transform: Callable[[str], str]

    def __call__(self, password: str) -> str:
        return self.transform(password)


@dataclass
class RuleResult:
    """Result of applying a rule to a password."""

    original: str
    transformed: str
    rule_name: str
    applied: bool  # True when transformed != original


# ---------------------------------------------------------------------------
# Built-in rule helpers
# ---------------------------------------------------------------------------

_LEET_BASIC: Dict[str, str] = {
    "a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "l": "1",
}

_LEET_ADVANCED: Dict[str, str] = {
    **_LEET_BASIC,
    "a": "@", "b": "8", "g": "9", "i": "!", "s": "$", "t": "+", "z": "2",
}


def _leet_substitute(password: str, table: Dict[str, str]) -> str:
    result = []
    for ch in password:
        low = ch.lower()
        if low in table:
            result.append(table[low])
        else:
            result.append(ch)
    return "".join(result)


def _make_repeat(n: int) -> Callable[[str], str]:
    def repeat(password: str) -> str:
        return password * n
    return repeat


def _make_delete_at(n: int) -> Callable[[str], str]:
    def delete_at(password: str) -> str:
        if 0 <= n < len(password):
            return password[:n] + password[n + 1:]
        return password
    return delete_at


def _make_insert(n: int, ch: str) -> Callable[[str], str]:
    def insert(password: str) -> str:
        pos = min(n, len(password))
        return password[:pos] + ch + password[pos:]
    return insert


def _make_replace(n: int, ch: str) -> Callable[[str], str]:
    def replace(password: str) -> str:
        if 0 <= n < len(password):
            return password[:n] + ch + password[n + 1:]
        return password
    return replace


def _make_truncate(n: int) -> Callable[[str], str]:
    def truncate(password: str) -> str:
        if n < len(password):
            return password[:n]
        return password
    return truncate


def _make_extract(start: int, end: int) -> Callable[[str], str]:
    def extract(password: str) -> str:
        s = max(0, start)
        e = min(len(password), end)
        return password[s:e]
    return extract


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PasswordRuleEngine:
    """Manage and execute password transformation rules."""

    # Numeric / special suffixes used by multiple helpers
    _YEAR_RANGE = [str(y) for y in range(1970, 2027)]
    _COMMON_NUMBERS = ["123", "1234", "12345", "123456", "321", "111", "000", "007"]
    _SPECIAL_CHARS = ["!", "@", "#", "$", "%", "!!", "!@#"]

    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self._register_builtin_rules()

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def _add(self, name: str, fn: Callable[[str], str]) -> None:
        self.rules[name] = Rule(name=name, transform=fn)

    def register_rule(self, rule: Rule) -> None:
        """Register a custom rule."""
        self.rules[rule.name] = rule

    def register(self, name: str, fn: Callable[[str], str]) -> None:
        """Register a callable as a rule under *name*."""
        self._add(name, fn)

    # ------------------------------------------------------------------
    # Built-in rules
    # ------------------------------------------------------------------

    def _register_builtin_rules(self) -> None:  # noqa: C901 (acceptable complexity)
        # -- Case transformations --
        self._add("lowercase", lambda p: p.lower())
        self._add("uppercase", lambda p: p.upper())
        self._add("capitalize", lambda p: p.capitalize() if p else p)
        self._add("invert", lambda p: p.swapcase())
        self._add("toggle_case", lambda p: "".join(
            c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(p)
        ))

        # -- Reverse --
        self._add("reverse", lambda p: p[::-1])

        # -- Duplicate / repeat --
        self._add("duplicate", lambda p: p + p)
        for _n in range(2, 6):
            self._add(f"repeat({_n})", _make_repeat(_n))

        # -- Mirror (append reversed) --
        self._add("mirror", lambda p: p + p[::-1])

        # -- Rotate --
        self._add("rotate_left", lambda p: p[1:] + p[:1] if p else p)
        self._add("rotate_right", lambda p: p[-1:] + p[:-1] if p else p)

        # -- Append / prepend characters --
        for ch in string.ascii_lowercase + string.digits + "!@#$":
            self._add(f"append({ch})", lambda p, c=ch: p + c)
            self._add(f"prepend({ch})", lambda p, c=ch: c + p)

        # -- Delete --
        self._add("delete_first", lambda p: p[1:] if p else p)
        self._add("delete_last", lambda p: p[:-1] if p else p)
        for _n in range(10):
            self._add(f"delete_at({_n})", _make_delete_at(_n))

        # -- Insert --
        for _n in range(10):
            for _ch in "aeiou1234":
                self._add(f"insert({_n},{_ch})", _make_insert(_n, _ch))

        # -- Replace at position --
        for _n in range(10):
            for _ch in "aeiou1234!@":
                self._add(f"replace({_n},{_ch})", _make_replace(_n, _ch))

        # -- Swap case (alias for invert) --
        self._add("swap_case", lambda p: p.swapcase())

        # -- Truncate / extract --
        for _n in [4, 6, 8, 10]:
            self._add(f"truncate({_n})", _make_truncate(_n))
        for _s in range(1, 6):
            for _e in range(_s + 1, 11):
                self._add(f"extract({_s},{_e})", _make_extract(_s, _e))

        # -- Leet speak --
        self._add("leet_basic", lambda p: _leet_substitute(p, _LEET_BASIC))
        self._add("leet_advanced", lambda p: _leet_substitute(p, _LEET_ADVANCED))

        # -- Numeric / special suffixes --
        self._add("append_year",
                  lambda p: p + str(__import__("random").choice(self._YEAR_RANGE)))
        for _num in self._COMMON_NUMBERS:
            self._add(f"append_{_num}", lambda p, n=_num: p + n)
        for _sp in self._SPECIAL_CHARS:
            self._add(f"append_special_{_sp}", lambda p, s=_sp: p + s)

        # -- Case styles --
        self._add("camel_case", lambda p: self._to_camel(p))
        self._add("snake_case", lambda p: self._to_snake(p))
        self._add("kebab_case", lambda p: self._to_kebab(p))

    # ------------------------------------------------------------------
    # Case-style helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_words(text: str) -> List[str]:
        """Split a string into word tokens on separators/case boundaries."""
        # Split on common separators
        parts = re.split(r"[ _\-.,;:!]", text)
        words: List[str] = []
        for part in parts:
            if not part:
                continue
            # Split on camelCase boundaries
            sub = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", part)
            if sub:
                words.extend(sub)
            else:
                words.append(part)
        return words

    @classmethod
    def _to_camel(cls, text: str) -> str:
        words = cls._split_words(text)
        if not words:
            return text
        return words[0].lower() + "".join(w.capitalize() for w in words[1:])

    @classmethod
    def _to_snake(cls, text: str) -> str:
        return "_".join(w.lower() for w in cls._split_words(text))

    @classmethod
    def _to_kebab(cls, text: str) -> str:
        return "-".join(w.lower() for w in cls._split_words(text))

    # ------------------------------------------------------------------
    # Apply rules
    # ------------------------------------------------------------------

    def apply_rule(self, password: str, rule_name: str, **kwargs) -> RuleResult:
        """Apply a single named rule to *password*."""
        rule = self.rules.get(rule_name)
        if rule is None:
            return RuleResult(
                original=password,
                transformed=password,
                rule_name=rule_name,
                applied=False,
            )
        try:
            transformed = rule.transform(password)
        except Exception:
            transformed = password
        return RuleResult(
            original=password,
            transformed=transformed,
            rule_name=rule_name,
            applied=(transformed != password),
        )

    def apply_rules(self, password: str, rule_names: List[str]) -> List[RuleResult]:
        """Apply each rule in *rule_names* independently to *password*."""
        return [self.apply_rule(password, rn) for rn in rule_names]

    def apply_all_rules(self, password: str) -> List[RuleResult]:
        """Apply every registered rule to *password*."""
        return [self.apply_rule(password, rn) for rn in self.rules]

    # ------------------------------------------------------------------
    # Variant generation
    # ------------------------------------------------------------------

    def generate_variants(self, password: str, max_variants: int = 50) -> List[str]:
        """Apply all rules and return up to *max_variants* unique transformed strings."""
        seen = {password}
        results: List[str] = []
        for rule_name in self.rules:
            res = self.apply_rule(password, rule_name)
            if res.applied and res.transformed not in seen:
                seen.add(res.transformed)
                results.append(res.transformed)
                if len(results) >= max_variants:
                    break
        return results

    def generate_combinations(
        self,
        passwords: List[str],
        rules: List[str],
        max_combinations: int = 100,
    ) -> List[str]:
        """Apply every rule in *rules* to every password in *passwords*.

        Returns up to *max_combinations* unique strings.
        """
        seen: set = set()
        results: List[str] = []
        for pw in passwords:
            for rn in rules:
                res = self.apply_rule(pw, rn)
                if res.transformed not in seen:
                    seen.add(res.transformed)
                    results.append(res.transformed)
                    if len(results) >= max_combinations:
                        return results
        return results

    # ------------------------------------------------------------------
    # Chained application
    # ------------------------------------------------------------------

    def apply_chain(self, password: str, rule_names: List[str]) -> str:
        """Apply rules sequentially, piping each output into the next rule."""
        result = password
        for rn in rule_names:
            res = self.apply_rule(result, rn)
            result = res.transformed
        return result

    def generate_chained_variants(
        self,
        password: str,
        chains: List[List[str]],
        max_variants: int = 200,
    ) -> List[str]:
        """Apply multiple rule chains and collect unique results."""
        seen: set = {password}
        results: List[str] = []
        for chain in chains:
            transformed = self.apply_chain(password, chain)
            if transformed not in seen:
                seen.add(transformed)
                results.append(transformed)
                if len(results) >= max_variants:
                    break
        return results

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def list_rules(self) -> List[str]:
        """Return sorted list of all registered rule names."""
        return sorted(self.rules.keys())

    def has_rule(self, name: str) -> bool:
        return name in self.rules
