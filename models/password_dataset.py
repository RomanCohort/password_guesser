"""
Password dataset for training the MAMBA model

Features:
- Frequency-weighted sampling (common passwords appear more)
- Password-aware augmentation strategies
- Efficient batch loading
"""

import os
import sys
import random
from typing import List, Optional, Tuple, Dict
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.password_utils import PasswordTokenizer


class PasswordDataset(Dataset):
    """
    Dataset for password sequences.

    Supports loading from:
    - Text files (one password per line)
    - RockYou-style datasets (password <tab> frequency)
    - Custom labeled datasets with target information
    """

    def __init__(
        self,
        passwords: List[str],
        tokenizer: PasswordTokenizer,
        max_length: int = 32,
        features: Optional[List[np.ndarray]] = None,
        frequencies: Optional[List[int]] = None
    ):
        self.passwords = passwords
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = features
        self.frequencies = frequencies

        # Pre-tokenize
        self.encoded = [tokenizer.encode(p, max_length) for p in passwords]

    def __len__(self) -> int:
        return len(self.passwords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        input_ids = torch.tensor(self.encoded[idx], dtype=torch.long)
        labels = input_ids.clone()

        if self.features is not None:
            feature = torch.tensor(self.features[idx], dtype=torch.float32)
            return input_ids, labels, feature
        else:
            feature = torch.zeros(64, dtype=torch.float32)
            return input_ids, labels, feature

    def get_sample_weights(self) -> torch.Tensor:
        """
        Compute sampling weights based on password frequency.

        Passwords with higher frequency get higher weight.
        Uses log-smoothed frequencies to avoid extreme imbalance.
        """
        if self.frequencies is not None:
            freqs = np.array(self.frequencies, dtype=np.float64)
            # Log-smooth to avoid extreme weights
            weights = np.log1p(freqs)
        else:
            weights = np.ones(len(self.passwords), dtype=np.float64)

        # Normalize
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float64)


class FrequencyWeightedPasswordDataset(PasswordDataset):
    """
    Dataset that automatically counts password frequencies
    and provides weighted sampling.
    """

    def __init__(
        self,
        passwords: List[str],
        tokenizer: PasswordTokenizer,
        max_length: int = 32,
        features: Optional[List[np.ndarray]] = None
    ):
        # Deduplicate and count frequencies
        counter = Counter(passwords)
        unique_passwords = list(counter.keys())
        frequencies = [counter[p] for p in unique_passwords]

        # Map features if provided
        if features is not None:
            pwd_to_feat = {p: f for p, f in zip(passwords, features)}
            unique_features = [pwd_to_feat.get(p, np.zeros(64)) for p in unique_passwords]
        else:
            unique_features = None

        super().__init__(
            passwords=unique_passwords,
            tokenizer=tokenizer,
            max_length=max_length,
            features=unique_features,
            frequencies=frequencies
        )

        self.password_counter = counter


def load_password_file(
    filepath: str,
    max_samples: int = 100000,
    has_frequency: bool = False
) -> Tuple[List[str], Optional[List[int]]]:
    """
    Load passwords from a text file.

    Args:
        filepath: Path to password file
        max_samples: Maximum number of samples to load
        has_frequency: Whether file has frequency column (RockYou format)

    Returns:
        Tuple of (passwords, frequencies or None)
    """
    passwords = []
    frequencies = [] if has_frequency else None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if len(passwords) >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            if has_frequency:
                parts = line.split('\t')
                if len(parts) >= 2:
                    password = parts[0]
                    freq = int(parts[1])
                elif len(parts) == 1:
                    password = parts[0]
                    freq = 1
                else:
                    continue
            else:
                password = line
                freq = 1

            if len(password) > 32:
                continue

            # Filter non-printable characters
            password = ''.join(c for c in password if 32 <= ord(c) <= 126)
            if not password:
                continue

            passwords.append(password)
            if frequencies is not None:
                frequencies.append(freq)

    return passwords, frequencies


class PasswordAugmentor:
    """
    Advanced password augmentation strategies.

    Generates realistic password variations for training.
    """

    # Common leet speak substitutions
    LEET_MAP = {
        'a': ['4', '@'],
        'e': ['3'],
        'i': ['1', '!'],
        'o': ['0'],
        's': ['$', '5'],
        't': ['7'],
        'l': ['1'],
    }

    # Common keyboard walks
    KEYBOARD_WALKS = [
        'qwerty', 'asdf', 'zxcv', '1234', 'qazwsx',
        '1qaz', '2wsx', '3edc', '!@#$'
    ]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def augment(self, password: str, n_variants: int = 5) -> List[str]:
        """Generate multiple augmented variants of a password"""
        strategies = [
            self.capitalize_random,
            self.add_digit_suffix,
            self.add_digit_prefix,
            self.leet_speak,
            self.reverse_password,
            self.swap_case,
            self.add_special_suffix,
            self.add_special_prefix,
            self.keyboard_walk_variation,
            self.double_characters,
        ]

        variants = set()
        for _ in range(n_variants):
            strategy = self.rng.choice(strategies)
            variant = strategy(password)
            if variant != password:  # Only keep actual changes
                variants.add(variant)

        return list(variants)

    def capitalize_random(self, pwd: str) -> str:
        chars = list(pwd)
        if chars:
            idx = self.rng.randint(0, len(chars) - 1)
            chars[idx] = chars[idx].swapcase()
        return ''.join(chars)

    def add_digit_suffix(self, pwd: str) -> str:
        return pwd + str(self.rng.randint(0, 999))

    def add_digit_prefix(self, pwd: str) -> str:
        return str(self.rng.randint(0, 99)) + pwd

    def leet_speak(self, pwd: str) -> str:
        result = []
        for c in pwd:
            if c.lower() in self.LEET_MAP and self.rng.random() > 0.5:
                result.append(self.rng.choice(self.LEET_MAP[c.lower()]))
            else:
                result.append(c)
        return ''.join(result)

    def reverse_password(self, pwd: str) -> str:
        return pwd[::-1]

    def swap_case(self, pwd: str) -> str:
        return pwd.swapcase()

    def add_special_suffix(self, pwd: str) -> str:
        specials = ['!', '@', '#', '$', '%', '!!', '@@', '123']
        return pwd + self.rng.choice(specials)

    def add_special_prefix(self, pwd: str) -> str:
        specials = ['!', '@', '#', '$']
        return self.rng.choice(specials) + pwd

    def keyboard_walk_variation(self, pwd: str) -> str:
        """Replace a substring with a keyboard walk pattern"""
        if len(pwd) < 4:
            return pwd
        walk = self.rng.choice(self.KEYBOARD_WALKS)
        # Replace a random portion
        start = self.rng.randint(0, max(0, len(pwd) - len(walk)))
        return pwd[:start] + walk + pwd[start + len(walk):]

    def double_characters(self, pwd: str) -> str:
        """Double a random character"""
        if not pwd:
            return pwd
        idx = self.rng.randint(0, len(pwd) - 1)
        return pwd[:idx] + pwd[idx] * 2 + pwd[idx + 1:]


def generate_training_data(
    base_passwords: List[str],
    target_info: dict,
    n_augment: int = 5
) -> List[str]:
    """
    Generate augmented training data from base passwords and target info.
    """
    augmentor = PasswordAugmentor()
    augmented = list(base_passwords)

    # Augment existing passwords
    for pwd in base_passwords:
        variants = augmentor.augment(pwd, n_augment=n_augment)
        augmented.extend(variants)

    # Generate target-specific passwords
    if target_info:
        names = target_info.get('names', [])
        dates = target_info.get('dates', [])
        numbers = target_info.get('numbers', [])
        words = target_info.get('words', [])

        for name in names:
            for date in dates:
                augmented.append(f"{name}{date}")
                augmented.append(f"{name}{date}!")
                augmented.append(f"{name.capitalize()}{date}")

            for num in numbers[:3]:
                augmented.append(f"{name}{num}")
                augmented.append(f"{name.capitalize()}{num}")

        for word in words:
            for date in dates[:2]:
                augmented.append(f"{word}{date}")
                augmented.append(f"{word.capitalize()}{date}")

    return augmented


def create_dataloader(
    passwords: List[str],
    tokenizer: PasswordTokenizer,
    batch_size: int = 64,
    shuffle: bool = True,
    features: Optional[List[np.ndarray]] = None,
    frequency_weighted: bool = False
) -> DataLoader:
    """
    Create a DataLoader for password data.

    Args:
        passwords: List of password strings
        tokenizer: Tokenizer instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        features: Optional feature vectors
        frequency_weighted: Use frequency-weighted sampling
    """
    if frequency_weighted:
        dataset = FrequencyWeightedPasswordDataset(
            passwords, tokenizer, features=features
        )
        weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True
        )
    else:
        dataset = PasswordDataset(passwords, tokenizer, features=features)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True
        )
