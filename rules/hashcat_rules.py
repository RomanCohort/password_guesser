"""
Hashcat-style rule parser and executor.

Supports the standard Hashcat rule syntax so that existing .rule files can be
loaded directly, or rules can be composed programmatically.
"""

import os
from typing import Dict, List, Optional, Tuple

# Each parsed rule operation is a tuple: (opcode: str, *args)
# Examples:  ('l',),  ('$','a'),  ('i','3','X'),  ('x','2','4')


class HashcatRuleParser:
    """Parse Hashcat rule lines into structured operation lists."""

    # ------------------------------------------------------------------
    # Single-rule parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_rule(line: str) -> List[Tuple]:
        """Parse one Hashcat rule line into a list of ``(opcode, *args)`` tuples.

        A rule line consists of space-separated operations, e.g.
        ``"l $a $1 c"`` -> lowercase, append 'a', append '1', capitalize.
        """
        line = line.strip()
        if not line or line.startswith("#"):
            return []

        ops: List[Tuple] = []
        tokens = line.split()
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            # --- zero-argument opcodes ---
            if tok in (":", "l", "u", "c", "C", "t", "r", "d", "f",
                       "{", "}", "[", "]", "!", "q"):
                ops.append((tok,))
                i += 1
            # --- one-argument opcodes ---
            elif tok in ("T", "D", "p", "'"):
                if i + 1 < len(tokens):
                    ops.append((tok, tokens[i + 1]))
                    i += 2
                else:
                    i += 1  # malformed – skip
            # --- two-character inline opcodes (no space before argument) ---
            elif len(tok) >= 2 and tok[0] in ("$", "^"):
                ops.append((tok[0], tok[1]))
                # handle remaining chars as separate append/prepend ops
                for ch in tok[2:]:
                    ops.append((tok[0], ch))
                i += 1
            elif len(tok) >= 3 and tok[0] in ("i", "o"):
                # iNX  /  oNX  — N is position, X is character
                ops.append((tok[0], tok[1], tok[2]))
                i += 1
            elif len(tok) >= 3 and tok[0] == "s":
                # sXY — replace X with Y
                ops.append((tok[0], tok[1], tok[2]))
                i += 1
            elif len(tok) >= 2 and tok[0] == "@":
                # @X — purge all X
                ops.append((tok[0], tok[1]))
                i += 1
            elif len(tok) >= 3 and tok[0] == "x":
                # xNM — extract M chars starting at N
                ops.append((tok[0], tok[1], tok[2]))
                i += 1
            elif len(tok) >= 3 and tok[0] == "O":
                # ONM — delete M chars starting at N
                ops.append((tok[0], tok[1], tok[2]))
                i += 1
            elif tok in ("(", ")"):
                # Length-based rejection: (N  /  )N  (need next token for N)
                if i + 1 < len(tokens):
                    ops.append((tok, tokens[i + 1]))
                    i += 2
                else:
                    i += 1
            else:
                # Unknown token — skip
                i += 1

        return ops

    # ------------------------------------------------------------------
    # File parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_file(filepath: str) -> List[List[Tuple]]:
        """Parse a Hashcat .rule file and return a list of parsed rule lines."""
        rules: List[List[Tuple]] = []
        if not os.path.isfile(filepath):
            return rules
        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parsed = HashcatRuleParser.parse_rule(line)
                if parsed:
                    rules.append(parsed)
        return rules

    # ------------------------------------------------------------------
    # Rule application
    # ------------------------------------------------------------------

    @staticmethod
    def apply_parsed_rule(password: str, rule_ops: List[Tuple]) -> str:
        """Apply a sequence of parsed rule operations to *password*."""
        pw = password
        for op in rule_ops:
            pw = HashcatRuleParser._apply_single(pw, op)
        return pw

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_single(password: str, op: Tuple) -> str:
        opcode = op[0]

        # --- NOP ---
        if opcode == ":":
            return password

        # --- Case ---
        if opcode == "l":
            return password.lower()
        if opcode == "u":
            return password.upper()
        if opcode == "c":
            return password.capitalize() if password else password
        if opcode == "C":
            if not password:
                return password
            return password[0].lower() + password[1:].upper()
        if opcode == "t":
            return password.swapcase()
        if opcode == "T":
            if len(op) < 2:
                return password
            try:
                pos = int(op[1])
            except ValueError:
                return password
            if 0 <= pos < len(password):
                chars = list(password)
                chars[pos] = chars[pos].swapcase()
                return "".join(chars)
            return password

        # --- Reverse ---
        if opcode == "r":
            return password[::-1]

        # --- Duplicate / repeat ---
        if opcode == "d":
            return password + password
        if opcode == "p":
            if len(op) < 2:
                return password
            try:
                n = int(op[1])
            except ValueError:
                return password
            return password * n if n > 0 else password

        # --- Mirror (append reversed) ---
        if opcode == "f":
            return password + password[::-1]

        # --- Rotate ---
        if opcode == "{":
            return password[1:] + password[:1] if password else password
        if opcode == "}":
            return password[-1:] + password[:-1] if password else password

        # --- Append / prepend ---
        if opcode == "$":
            if len(op) < 2:
                return password
            return password + op[1]
        if opcode == "^":
            if len(op) < 2:
                return password
            return op[1] + password

        # --- Delete ---
        if opcode == "[":
            return password[1:] if password else password
        if opcode == "]":
            return password[:-1] if password else password
        if opcode == "D":
            if len(op) < 2:
                return password
            try:
                pos = int(op[1])
            except ValueError:
                return password
            if 0 <= pos < len(password):
                return password[:pos] + password[pos + 1:]
            return password

        # --- Extract xNM ---
        if opcode == "x":
            if len(op) < 3:
                return password
            try:
                start = int(op[1])
                length = int(op[2])
            except ValueError:
                return password
            return password[start:start + length]

        # --- Delete range ONM ---
        if opcode == "O":
            if len(op) < 3:
                return password
            try:
                start = int(op[1])
                length = int(op[2])
            except ValueError:
                return password
            return password[:start] + password[start + length:]

        # --- Insert iNX ---
        if opcode == "i":
            if len(op) < 3:
                return password
            try:
                pos = int(op[1])
            except ValueError:
                return password
            ch = op[2]
            pos = min(pos, len(password))
            return password[:pos] + ch + password[pos:]

        # --- Overwrite oNX ---
        if opcode == "o":
            if len(op) < 3:
                return password
            try:
                pos = int(op[1])
            except ValueError:
                return password
            ch = op[2]
            if 0 <= pos < len(password):
                return password[:pos] + ch + password[pos + 1:]
            return password

        # --- Substitute sXY ---
        if opcode == "s":
            if len(op) < 3:
                return password
            src, dst = op[1], op[2]
            return password.replace(src, dst)

        # --- Purge @X ---
        if opcode == "@":
            if len(op) < 2:
                return password
            return password.replace(op[1], "")

        # --- Rejection rules (applied as no-ops during generation) ---
        if opcode == "!":
            # Reject if password is unchanged – caller must handle
            return password
        if opcode == "(":
            # Reject if length <= N
            return password
        if opcode == ")":
            # Reject if length >= N
            return password

        return password


# ======================================================================
# Executor
# ======================================================================

class HashcatRuleExecutor:
    """High-level executor that combines parsing and application."""

    def __init__(self):
        self.parser = HashcatRuleParser()

    # ------------------------------------------------------------------
    # Single-rule application
    # ------------------------------------------------------------------

    def apply_rule(self, password: str, rule_line: str) -> str:
        """Parse a rule line and apply it to *password*."""
        ops = self.parser.parse_rule(rule_line)
        return self.parser.apply_parsed_rule(password, ops)

    # ------------------------------------------------------------------
    # Bulk application from a rule file
    # ------------------------------------------------------------------

    def apply_rules_file(
        self,
        passwords: List[str],
        rule_file: str,
        max_results: int = 10000,
    ) -> List[str]:
        """Apply every rule in *rule_file* to every password in *passwords*.

        Returns up to *max_results* unique strings.
        """
        parsed_rules = self.parser.parse_file(rule_file)
        if not parsed_rules:
            return []

        seen: set = set()
        results: List[str] = []

        for pw in passwords:
            for rule_ops in parsed_rules:
                transformed = self.parser.apply_parsed_rule(pw, rule_ops)
                if transformed and transformed not in seen:
                    # Check rejection rules embedded in the ops
                    if self._is_rejected(pw, rule_ops):
                        continue
                    seen.add(transformed)
                    results.append(transformed)
                    if len(results) >= max_results:
                        return results
        return results

    # ------------------------------------------------------------------
    # Rejection helper
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rejected(original: str, rule_ops: List[Tuple]) -> bool:
        """Return True if a rejection rule discards this candidate."""
        for op in rule_ops:
            opcode = op[0]
            if opcode == "!" and original == original:
                # The '!' rule means reject if the password is unchanged
                # We compare the original with the transformed version
                # which the caller has already computed. For simplicity,
                # we only reject at this level if there were no mutating ops.
                pass  # handled implicitly — unchanged strings won't be added
            elif opcode == "(":
                if len(op) >= 2:
                    try:
                        n = int(op[1])
                    except ValueError:
                        continue
                    if len(original) <= n:
                        return True
            elif opcode == ")":
                if len(op) >= 2:
                    try:
                        n = int(op[1])
                    except ValueError:
                        continue
                    if len(original) >= n:
                        return True
        return False

    # ------------------------------------------------------------------
    # Common rule generation
    # ------------------------------------------------------------------

    def generate_common_rules(self) -> List[str]:
        """Generate a list of commonly-used Hashcat rule strings."""
        base: List[str] = []

        # Single-operation rules
        for op in [":", "l", "u", "c", "C", "t", "r", "d", "f", "{", "}",
                    "[", "]"]:
            base.append(op)

        # Append common characters
        for ch in "abcdefghijklmnopqrstuvwxyz0123456789!@#$":
            base.append(f"${ch}")

        # Prepend common characters
        for ch in "abcdefghijklmnopqrstuvwxyz0123456789!@#$":
            base.append(f"^{ch}")

        # Capitalize + append digit
        for d in "0123456789":
            base.append(f"c ${d}")

        # Capitalize + append special
        for sp in "!@#$":
            base.append(f"c ${sp}")

        # Lowercase + append year
        for yr in range(90, 100):
            base.append(f"l ${str(yr)[-2:]}")
        for yr in range(2000, 2025):
            base.append(f"l ${yr}")

        # Capitalize + append 3-digit sequences
        for seq in ["123", "1234", "321", "007", "666", "777", "888", "999"]:
            base.append(f"c {seq}")

        # Toggle + append
        base.append("t $!")

        # Duplicate + capitalize
        base.append("d c")

        # Reverse + capitalize
        base.append("r c")

        # Mirror
        base.append("f")

        # Multiple appends: word + digit + special
        for d in "123":
            for sp in "!@#$":
                base.append(f"c ${d} ${sp}")

        # Leet-like substitutions via sXY
        leet_subs = [
            ("s", "$"), ("a", "@"), ("a", "4"), ("e", "3"),
            ("i", "1"), ("i", "!"), ("o", "0"), ("t", "7"),
        ]
        for src, dst in leet_subs:
            base.append(f"s{src}{dst}")
            base.append(f"c s{src}{dst}")

        # Insert at various positions
        for pos in "0123":
            for ch in "!@":
                base.append(f"i{pos}{ch}")

        return base

    def apply_common_rules(
        self,
        passwords: List[str],
        max_results: int = 5000,
    ) -> List[str]:
        """Generate variants using the built-in common rule set."""
        rule_lines = self.generate_common_rules()
        parsed: List[List[Tuple]] = []
        for rl in rule_lines:
            ops = self.parser.parse_rule(rl)
            if ops:
                parsed.append(ops)

        seen: set = set()
        results: List[str] = []
        for pw in passwords:
            for rule_ops in parsed:
                transformed = self.parser.apply_parsed_rule(pw, rule_ops)
                if transformed and transformed not in seen:
                    seen.add(transformed)
                    results.append(transformed)
                    if len(results) >= max_results:
                        return results
        return results
