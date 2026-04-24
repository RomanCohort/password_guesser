"""PCFG Grammar definition: terminals, production rules, and grammar operations."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
import math
import random
from collections import defaultdict


@dataclass
class Terminal:
    """Terminal symbol with associated probability."""
    value: str
    probability: float


@dataclass
class ProductionRule:
    """Production rule: lhs -> rhs with probability."""
    lhs: str  # Left-hand side nonterminal, e.g. "S", "L", "D", "S"
    rhs: List[str]  # Right-hand side symbols, e.g. ["L3", "D2"] or ["password"]
    probability: float


class Grammar:
    """Probabilistic Context-Free Grammar for password modeling."""

    def __init__(self):
        self.rules: Dict[str, List[ProductionRule]] = {}
        self.terminals: Dict[str, List[Terminal]] = {}

    def add_rule(self, lhs: str, rhs: List[str], probability: float):
        """Add a production rule."""
        if lhs not in self.rules:
            self.rules[lhs] = []
        self.rules[lhs].append(ProductionRule(lhs=lhs, rhs=rhs, probability=probability))

    def add_terminal(self, category: str, value: str, probability: float):
        """Add a terminal symbol under a category."""
        if category not in self.terminals:
            self.terminals[category] = []
        self.terminals[category].append(Terminal(value=value, probability=probability))

    def get_productions(self, nonterminal: str) -> List[ProductionRule]:
        """Get all production rules for a nonterminal."""
        return self.rules.get(nonterminal, [])

    def get_terminals(self, category: str) -> List[Terminal]:
        """Get all terminals for a category."""
        return self.terminals.get(category, [])

    def sample(self, symbol: str = "S") -> str:
        """Sample a password from the grammar starting from the given symbol."""
        # If symbol is a terminal category, sample from terminals
        if symbol in self.terminals and symbol not in self.rules:
            terminals = self.terminals[symbol]
            if not terminals:
                return ""
            # Weighted random choice
            probs = [t.probability for t in terminals]
            total = sum(probs)
            if total <= 0:
                return random.choice(terminals).value
            r = random.random() * total
            cumulative = 0.0
            for t in terminals:
                cumulative += t.probability
                if r <= cumulative:
                    return t.value
            return terminals[-1].value

        # If symbol has production rules, expand
        productions = self.get_productions(symbol)
        if not productions:
            # Might be a terminal category not in terminals dict
            if symbol in self.terminals:
                terminals = self.terminals[symbol]
                if not terminals:
                    return ""
                probs = [t.probability for t in terminals]
                total = sum(probs)
                if total <= 0:
                    return random.choice(terminals).value
                r = random.random() * total
                cumulative = 0.0
                for t in terminals:
                    cumulative += t.probability
                    if r <= cumulative:
                        return t.value
                return terminals[-1].value
            return ""

        # Weighted random choice among productions
        probs = [p.probability for p in productions]
        total = sum(probs)
        if total <= 0:
            chosen = random.choice(productions)
        else:
            r = random.random() * total
            cumulative = 0.0
            chosen = productions[-1]
            for p in productions:
                cumulative += p.probability
                if r <= cumulative:
                    chosen = p
                    break

        # Recursively expand each symbol in rhs
        result_parts = []
        for sym in chosen.rhs:
            result_parts.append(self.sample(sym))
        return "".join(result_parts)

    def sample_top_k(self, symbol: str = "S", k: int = 10) -> List[Tuple[str, float]]:
        """Enumerate the highest probability passwords up to k results.

        Uses iterative expansion: generates all combinations from rules/terminals
        and sorts by probability.
        """
        # Build candidates iteratively using beam search style expansion
        # Start with a single candidate: (accumulated_string, remaining_symbols, accumulated_prob)
        candidates = [("", [symbol], 1.0)]
        results: List[Tuple[str, float]] = []
        max_expansions = 50000  # safety limit
        expansions = 0

        while candidates and expansions < max_expansions:
            next_candidates = []
            for acc_str, remaining, acc_prob in candidates:
                if not remaining:
                    # Fully expanded
                    results.append((acc_str, acc_prob))
                    continue

                sym = remaining[0]
                rest = remaining[1:]

                # Try to expand as production rules
                productions = self.get_productions(sym)
                terminals = self.get_terminals(sym)

                if productions:
                    for prod in productions:
                        new_remaining = prod.rhs + rest
                        new_prob = acc_prob * prod.probability
                        if new_prob > 0:
                            next_candidates.append((acc_str, new_remaining, new_prob))
                            expansions += 1
                elif terminals:
                    for term in terminals:
                        new_str = acc_str + term.value
                        new_prob = acc_prob * term.probability
                        if new_prob > 0:
                            next_candidates.append((new_str, rest, new_prob))
                            expansions += 1
                else:
                    # Unknown symbol, treat as literal
                    next_candidates.append((acc_str + sym, rest, acc_prob))
                    expansions += 1

            # Prune: keep top candidates by probability to avoid explosion
            if len(next_candidates) > 10000:
                next_candidates.sort(key=lambda x: x[2], reverse=True)
                next_candidates = next_candidates[:5000]

            candidates = next_candidates

            # Early termination if we have enough completed results
            completed = [c for c in candidates if not c[1]]
            if len(completed) >= k * 10:
                break

        # Sort results by probability descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def probability(self, password: str) -> float:
        """Compute the probability of a password under this grammar.

        Uses a recursive algorithm: tries to parse the password by expanding
        rules from the start symbol and matching terminals against the password.
        """
        return self._parse_probability("S", password)

    def _parse_probability(self, symbol: str, text: str) -> float:
        """Recursively compute probability of text given a starting symbol."""
        if not text:
            return 0.0

        # Check if symbol is a terminal category
        terminals = self.get_terminals(symbol)
        if terminals and symbol not in self.rules:
            for t in terminals:
                if t.value == text:
                    return t.probability
            return 0.0

        # Try to expand via production rules
        productions = self.get_productions(symbol)
        if not productions:
            # Check terminals as fallback
            for t in terminals:
                if t.value == text:
                    return t.probability
            return 0.0

        total_prob = 0.0
        for prod in productions:
            prob = prod.probability * self._parse_rhs(prod.rhs, text)
            total_prob += prob

        return total_prob

    def _parse_rhs(self, rhs_symbols: List[str], text: str) -> float:
        """Try to match a sequence of RHS symbols against text. Return probability."""
        if not rhs_symbols:
            return 1.0 if not text else 0.0

        if not text:
            return 0.0

        total = 0.0
        sym = rhs_symbols[0]
        rest_symbols = rhs_symbols[1:]

        # Try all possible splits of text between current symbol and the rest
        terminals = self.get_terminals(sym)
        productions = self.get_productions(sym)

        # If this symbol has terminals, try matching them
        if terminals and not productions:
            for t in terminals:
                if text.startswith(t.value):
                    remaining = text[len(t.value):]
                    sub_prob = self._parse_rhs(rest_symbols, remaining)
                    total += t.probability * sub_prob

        # If this symbol has production rules, expand
        if productions:
            for prod in productions:
                for split_point in range(len(text) + 1):
                    prefix = text[:split_point]
                    suffix = text[split_point:]
                    sub_prob = self._parse_rhs(prod.rhs, prefix)
                    if sub_prob > 0:
                        rest_prob = self._parse_rhs(rest_symbols, suffix)
                        total += prod.probability * sub_prob * rest_prob

        # Also try as both terminals and rules
        if terminals and productions:
            for t in terminals:
                if text.startswith(t.value):
                    remaining = text[len(t.value):]
                    rest_prob = self._parse_rhs(rest_symbols, remaining)
                    total += t.probability * rest_prob

        return total

    def save(self, path: str):
        """Save grammar to a JSON file."""
        data = {
            'rules': {},
            'terminals': {},
        }
        for lhs, rules in self.rules.items():
            data['rules'][lhs] = [
                {'lhs': r.lhs, 'rhs': r.rhs, 'probability': r.probability}
                for r in rules
            ]
        for cat, terms in self.terminals.items():
            data['terminals'][cat] = [
                {'value': t.value, 'probability': t.probability}
                for t in terms
            ]

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'Grammar':
        """Load grammar from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        grammar = cls()
        for lhs, rules in data.get('rules', {}).items():
            for r in rules:
                grammar.add_rule(r['lhs'], r['rhs'], r['probability'])

        for cat, terms in data.get('terminals', {}).items():
            for t in terms:
                grammar.add_terminal(cat, t['value'], t['probability'])

        return grammar
