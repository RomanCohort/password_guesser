"""PCFG training: learn grammar from password datasets."""

from collections import Counter, defaultdict
import re
import math
import os
from typing import Dict, List, Tuple, Optional

from pcfg.grammar import Grammar, Terminal, ProductionRule


class PCFGTrainer:
    """Train a PCFG from a password dataset."""

    # Character category labels
    LETTER = 'L'
    DIGIT = 'D'
    SPECIAL = 'S'  # special characters

    def __init__(self, max_length: int = 32):
        self.max_length = max_length
        self.structures: Counter = Counter()  # structure string -> frequency
        self.terminals: Dict[str, Counter] = defaultdict(Counter)  # category -> value -> frequency

    @staticmethod
    def classify_char(c: str) -> str:
        """Classify a single character into L (letter), D (digit), or S (special)."""
        if c.isalpha():
            return PCFGTrainer.LETTER
        elif c.isdigit():
            return PCFGTrainer.DIGIT
        else:
            return PCFGTrainer.SPECIAL

    def extract_structure(self, password: str) -> str:
        """Extract the structural representation of a password.

        Example: "hello123!" -> "L5D3S1"
        The result is a concatenation of (category, run_length) pairs.
        """
        if not password:
            return ""

        structure_parts = []
        current_type = self.classify_char(password[0])
        current_len = 1

        for ch in password[1:]:
            ch_type = self.classify_char(ch)
            if ch_type == current_type:
                current_len += 1
            else:
                structure_parts.append(f"{current_type}{current_len}")
                current_type = ch_type
                current_len = 1

        structure_parts.append(f"{current_type}{current_len}")
        return "".join(structure_parts)

    def _parse_structure_to_segments(self, password: str, structure: str) -> List[Tuple[str, str]]:
        """Parse a password into segments matching its structure.

        Returns list of (category_with_length, value) pairs.
        Example: ("hello123!", "L5D3S1") -> [("L5", "hello"), ("D3", "123"), ("S1", "!")]
        """
        segments = []
        pos = 0
        # Parse structure tokens like "L5", "D3", "S1"
        tokens = re.findall(r'([LDS])(\d+)', structure)
        for cat, length_str in tokens:
            length = int(length_str)
            value = password[pos:pos + length]
            category_key = f"{cat}{length}"  # e.g. "L5"
            segments.append((category_key, value))
            pos += length
        return segments

    def train(self, passwords: List[str]) -> 'PCFGModel':
        """Train a PCFG from a list of passwords.

        Steps:
        1. Extract structure of each password.
        2. Count structure frequencies.
        3. Count terminal values within each category.
        4. Convert counts to probabilities.
        5. Build a Grammar object.
        """
        self.structures.clear()
        self.terminals.clear()

        total_passwords = 0

        for pwd in passwords:
            if not pwd or len(pwd) > self.max_length:
                continue

            # Normalize: strip whitespace
            pwd = pwd.strip()
            if not pwd:
                continue

            structure = self.extract_structure(pwd)
            if not structure:
                continue

            self.structures[structure] += 1

            # Parse into segments and count terminal values
            segments = self._parse_structure_to_segments(pwd, structure)
            for category_key, value in segments:
                self.terminals[category_key][value] += 1

            total_passwords += 1

        if total_passwords == 0:
            # Return empty model
            return PCFGModel(Grammar())

        # Build grammar
        grammar = Grammar()

        # Add production rules: S -> structure with probability
        for structure, count in self.structures.items():
            prob = count / total_passwords
            # Parse structure tokens to build rhs
            tokens = re.findall(r'([LDS]\d+)', structure)
            grammar.add_rule("S", tokens, prob)

        # Add terminals for each category with probabilities
        for category_key, value_counts in self.terminals.items():
            total_in_category = sum(value_counts.values())
            if total_in_category == 0:
                continue
            for value, count in value_counts.items():
                prob = count / total_in_category
                grammar.add_terminal(category_key, value, prob)

        return PCFGModel(grammar)

    def train_from_file(self, filepath: str, max_samples: int = 100000) -> 'PCFGModel':
        """Train from a password file (one password per line)."""
        passwords = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    passwords.append(line.strip())
        except FileNotFoundError:
            raise FileNotFoundError(f"Password file not found: {filepath}")

        return self.train(passwords)


class PCFGModel:
    """A trained PCFG model that can generate and rank passwords."""

    def __init__(self, grammar: Grammar):
        self.grammar = grammar

    def generate(self, n: int = 100) -> List[Tuple[str, float]]:
        """Generate n password candidates with probabilities.

        Uses sampling from the grammar. Deduplicates results.
        """
        results: Dict[str, float] = {}

        # Over-sample to account for duplicates
        attempts = n * 5
        for _ in range(attempts):
            if len(results) >= n:
                break
            pwd = self.grammar.sample("S")
            if pwd and pwd not in results:
                prob = self.grammar.probability(pwd)
                results[pwd] = prob

        # Sort by probability descending
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:n]

    def generate_by_structure(self, structure: str, n: int = 10) -> List[Tuple[str, float]]:
        """Generate passwords matching a specific structure.

        Args:
            structure: e.g. "L4D3" means 4 letters followed by 3 digits.
            n: Number of candidates to generate.

        Returns:
            List of (password, probability) tuples.
        """
        import re as _re
        tokens = _re.findall(r'([LDS])(\d+)', structure)
        if not tokens:
            return []

        results: Dict[str, float] = {}

        # Try generating by sampling each terminal category
        attempts = n * 20
        for _ in range(attempts):
            if len(results) >= n:
                break

            parts = []
            prob = 1.0
            valid = True

            for cat, length_str in tokens:
                category_key = f"{cat}{length_str}"
                terminals = self.grammar.get_terminals(category_key)

                if not terminals:
                    # If no exact match, try generating random chars of the right type
                    import random
                    if cat == self.grammar.__class__.__name__:  # won't match, just use fallback
                        pass
                    # Fallback: generate random characters
                    chars = []
                    for _ in range(int(length_str)):
                        if cat == 'L':
                            chars.append(chr(ord('a') + random.randint(0, 25)))
                        elif cat == 'D':
                            chars.append(str(random.randint(0, 9)))
                        else:
                            specials = '!@#$%^&*()_+-='
                            chars.append(random.choice(specials))
                    parts.append("".join(chars))
                    prob *= 1e-6  # Very low probability for random fallback
                    continue

                # Weighted sample from terminals
                import random
                total_prob = sum(t.probability for t in terminals)
                if total_prob <= 0:
                    valid = False
                    break

                r = random.random() * total_prob
                cumulative = 0.0
                chosen = terminals[-1]
                for t in terminals:
                    cumulative += t.probability
                    if r <= cumulative:
                        chosen = t
                        break

                parts.append(chosen.value)
                prob *= chosen.probability

            if valid and parts:
                pwd = "".join(parts)
                if pwd not in results:
                    # Calculate structure probability
                    struct_prob = 0.0
                    for rule in self.grammar.get_productions("S"):
                        rule_structure = "".join(rule.rhs)
                        if rule_structure == structure:
                            struct_prob += rule.probability
                    full_prob = prob * struct_prob if struct_prob > 0 else prob
                    results[pwd] = full_prob

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:n]

    def rank_passwords(self, passwords: List[str]) -> List[Tuple[str, float]]:
        """Rank passwords by model probability (descending)."""
        scored = []
        for pwd in passwords:
            prob = self.grammar.probability(pwd)
            scored.append((pwd, prob))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def save(self, path: str):
        """Save model to file."""
        self.grammar.save(path)

    @classmethod
    def load(cls, path: str) -> 'PCFGModel':
        """Load model from file."""
        grammar = Grammar.load(path)
        return cls(grammar)
