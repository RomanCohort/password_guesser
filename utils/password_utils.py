"""
Password tokenization and decoding utilities
"""

from typing import List, Optional
import numpy as np


class PasswordTokenizer:
    """Tokenize passwords at character level"""

    # ASCII printable characters (32-126)
    VOCAB = ''.join(chr(i) for i in range(32, 127))
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<SOS>': 2,  # Start of sequence
        '<EOS>': 3,  # End of sequence
    }

    def __init__(self):
        self.char_to_idx = {char: i + len(self.SPECIAL_TOKENS)
                           for i, char in enumerate(self.VOCAB)}
        self.idx_to_char = {i + len(self.SPECIAL_TOKENS): char
                           for i, char in enumerate(self.VOCAB)}

        # Add special tokens
        for token, idx in self.SPECIAL_TOKENS.items():
            self.idx_to_char[idx] = token

        self.vocab_size = len(self.VOCAB) + len(self.SPECIAL_TOKENS)
        self.pad_idx = self.SPECIAL_TOKENS['<PAD>']
        self.unk_idx = self.SPECIAL_TOKENS['<UNK>']
        self.sos_idx = self.SPECIAL_TOKENS['<SOS>']
        self.eos_idx = self.SPECIAL_TOKENS['<EOS>']

    def encode(self, password: str, max_length: Optional[int] = None) -> List[int]:
        """Convert password to token indices"""
        tokens = [self.sos_idx]

        for char in password:
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
            else:
                tokens.append(self.unk_idx)

        tokens.append(self.eos_idx)

        if max_length is not None:
            # Pad or truncate
            if len(tokens) < max_length:
                tokens.extend([self.pad_idx] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]

        return tokens

    def decode(self, tokens: List[int], remove_special: bool = True) -> str:
        """Convert token indices back to password string"""
        chars = []

        for idx in tokens:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if remove_special and char in self.SPECIAL_TOKENS:
                    continue
                chars.append(char)

        return ''.join(chars)

    def encode_batch(self, passwords: List[str], max_length: Optional[int] = None) -> np.ndarray:
        """Encode a batch of passwords"""
        if max_length is None:
            max_length = max(len(p) for p in passwords) + 2  # +2 for SOS and EOS

        return np.array([self.encode(p, max_length) for p in passwords])


class PasswordDecoder:
    """Decode model outputs to password strings"""

    def __init__(self, tokenizer: PasswordTokenizer):
        self.tokenizer = tokenizer

    def greedy_decode(self, logits: np.ndarray) -> str:
        """Greedy decoding from logits"""
        tokens = np.argmax(logits, axis=-1)
        return self.tokenizer.decode(tokens.tolist())

    def sample_decode(self, logits: np.ndarray, temperature: float = 1.0,
                     top_k: int = 0, top_p: float = 1.0) -> str:
        """Sampling with temperature, top-k, and top-p"""
        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = np.argsort(logits)[:-top_k]
            logits[indices_to_remove] = -float('inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            cumulative_probs = np.cumsum(self._softmax(sorted_logits))

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            logits[sorted_indices[sorted_indices_to_remove]] = -float('inf')

        # Sample
        probs = self._softmax(logits)
        token = np.random.choice(len(probs), p=probs)

        return self.tokenizer.decode([token])

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def beam_search(self, logits_sequence: np.ndarray, beam_width: int = 5) -> List[str]:
        """Beam search decoding"""
        batch_size, seq_len, vocab_size = logits_sequence.shape

        # Initialize beams
        beams = [([], 0.0)]  # (tokens, log_prob)

        for t in range(seq_len):
            new_beams = []

            for tokens, log_prob in beams:
                if tokens and tokens[-1] == self.tokenizer.eos_idx:
                    new_beams.append((tokens, log_prob))
                    continue

                # Get top-k tokens
                log_probs_t = logits_sequence[0, t]  # Assuming batch_size=1 for simplicity
                top_k_indices = np.argsort(log_probs_t)[-beam_width:]

                for idx in top_k_indices:
                    new_tokens = tokens + [idx]
                    new_log_prob = log_prob + log_probs_t[idx]
                    new_beams.append((new_tokens, new_log_prob))

            # Keep top beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        # Decode beams
        passwords = []
        for tokens, _ in beams:
            password = self.tokenizer.decode(tokens)
            passwords.append(password)

        return passwords


class PasswordPatternAnalyzer:
    """Analyze common password patterns"""

    PATTERNS = {
        'lower': r'^[a-z]+$',
        'upper': r'^[A-Z]+$',
        'digit': r'^\d+$',
        'alpha': r'^[a-zA-Z]+$',
        'alphanumeric': r'^[a-zA-Z0-9]+$',
        'name_digit': r'^[a-zA-Z]+\d+$',
        'digit_name': r'^\d+[a-zA-Z]+$',
        'name_special': r'^[a-zA-Z]+[!@#$%^&*]+$',
        'name_digit_special': r'^[a-zA-Z]+\d+[!@#$%^&*]+$',
        'capitalized': r'^[A-Z][a-z]+',
        'leet': r'[4@3€1!|0$]',  # Common leet speak substitutions
    }

    @staticmethod
    def analyze_structure(password: str) -> dict:
        """Analyze the structure of a password"""
        structure = {
            'length': len(password),
            'has_lower': any(c.islower() for c in password),
            'has_upper': any(c.isupper() for c in password),
            'has_digit': any(c.isdigit() for c in password),
            'has_special': any(not c.isalnum() for c in password),
            'digit_ratio': sum(c.isdigit() for c in password) / len(password) if password else 0,
            'upper_ratio': sum(c.isupper() for c in password) / len(password) if password else 0,
        }

        # Identify structural pattern
        pattern = ""
        for c in password:
            if c.islower():
                pattern += 'l'
            elif c.isupper():
                pattern += 'u'
            elif c.isdigit():
                pattern += 'd'
            else:
                pattern += 's'

        structure['pattern'] = pattern

        return structure

    @staticmethod
    def generate_pattern_variations(base: str, pattern_type: str) -> List[str]:
        """Generate variations based on common patterns"""
        variations = []

        if pattern_type == 'name_digit':
            # John123, john123, JOHN123
            import re
            match = re.match(r'^([a-zA-Z]+)(\d+)$', base)
            if match:
                name, digits = match.groups()
                variations = [
                    name + digits,
                    name.lower() + digits,
                    name.upper() + digits,
                    name.capitalize() + digits,
                    name + str(int(digits) + 1),
                    name + str(int(digits) - 1) if int(digits) > 0 else None,
                ]
                variations = [v for v in variations if v]

        elif pattern_type == 'leet':
            # Common leet substitutions
            leet_map = {
                'a': ['4', '@'],
                'e': ['3'],
                'i': ['1', '!'],
                'o': ['0'],
                's': ['$', '5'],
                't': ['7'],
                'l': ['1'],
            }

            variations = [base]
            for orig, leets in leet_map.items():
                new_vars = []
                for var in variations:
                    if orig in var.lower():
                        for leet in leets:
                            new_vars.append(var.replace(orig, leet).replace(orig.upper(), leet))
                variations.extend(new_vars)

        return list(set(variations))
