"""
Feature extraction and vectorization utilities
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TargetFeatures:
    """Structured features extracted from target information"""
    # Personal names
    full_name: str = ""
    first_name: str = ""
    last_name: str = ""
    nickname: str = ""

    # Important dates
    birthday: str = ""          # Format: YYYYMMDD
    birth_year: str = ""
    birth_month: str = ""
    birth_day: str = ""
    anniversary: str = ""

    # Contact information
    phone: str = ""
    email_prefix: str = ""

    # Interests and hobbies
    hobbies: List[str] = field(default_factory=list)
    favorite_words: List[str] = field(default_factory=list)
    favorite_numbers: List[str] = field(default_factory=list)
    sports_teams: List[str] = field(default_factory=list)

    # Pet names
    pet_names: List[str] = field(default_factory=list)

    # Location
    city: str = ""
    country: str = ""

    # Known patterns (if any history available)
    known_patterns: List[str] = field(default_factory=list)

    # Additional keywords
    keywords: List[str] = field(default_factory=list)


class FeatureExtractor:
    """Extract features from raw text using regex patterns"""

    # Regex patterns
    DATE_PATTERNS = [
        r'\b(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?\b',  # 1990-01-15, 1990年1月15日
        r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b',        # 01-15-1990
        r'\b(\d{4})(\d{2})(\d{2})\b',                    # 19900115
    ]

    PHONE_PATTERNS = [
        r'\b(\d{11})\b',                                 # Chinese mobile
        r'\b(\d{3}[-.]?\d{3}[-.]?\d{4})\b',             # US format
    ]

    EMAIL_PATTERN = r'\b([a-zA-Z0-9_.+-]+)@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b'

    @classmethod
    def extract_date(cls, text: str) -> Optional[str]:
        """Extract date and return as YYYYMMDD format"""
        for pattern in cls.DATE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    # Determine year position
                    if len(groups[0]) == 4:
                        year, month, day = groups
                    else:
                        day, month, year = groups if len(groups[2]) == 4 else (groups[2], groups[0], groups[1])

                    return f"{year}{month.zfill(2)}{day.zfill(2)}"
        return None

    @classmethod
    def extract_phone(cls, text: str) -> Optional[str]:
        """Extract phone number"""
        for pattern in cls.PHONE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return re.sub(r'[-.]', '', match.group(1))
        return None

    @classmethod
    def extract_email_prefix(cls, text: str) -> Optional[str]:
        """Extract email prefix (before @)"""
        match = re.search(cls.EMAIL_PATTERN, text)
        if match:
            return match.group(1)
        return None


class FeatureVectorizer:
    """Convert TargetFeatures to numerical vectors for MLP input"""

    # Character set for encoding strings
    CHAR_SET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"

    def __init__(self, vector_dim: int = 256, use_embedding: bool = True):
        self.vector_dim = vector_dim
        self.use_embedding = use_embedding
        self.feature_names = [
            'name_encoding', 'date_encoding', 'hobby_encoding',
            'number_encoding', 'keyword_encoding', 'pattern_encoding'
        ]

        # Character to index mapping for learned embeddings
        self.char_to_idx = {c: i for i, c in enumerate(self.CHAR_SET)}
        self.char_to_idx['<UNK>'] = len(self.CHAR_SET)
        self.embedding_dim = 16  # Learned embedding dimension

    def encode_string_embedding(self, s: str, max_len: int = 32) -> np.ndarray:
        """
        Encode string using character indices for learned embeddings.
        Returns indices that can be used with nn.Embedding, not one-hot.
        """
        indices = np.zeros(max_len, dtype=np.int64)
        for i, char in enumerate(s[:max_len]):
            indices[i] = self.char_to_idx.get(char, self.char_to_idx['<UNK>'])
        return indices

    def encode_string_dense(self, s: str, max_len: int = 32) -> np.ndarray:
        """
        Dense encoding using character properties.
        More informative than one-hot, less sparse.
        """
        vec = np.zeros(max_len * 8)  # 8 features per character position

        for i, char in enumerate(s[:max_len]):
            base = i * 8
            # Character class features
            vec[base + 0] = 1.0 if char.islower() else 0.0
            vec[base + 1] = 1.0 if char.isupper() else 0.0
            vec[base + 2] = 1.0 if char.isdigit() else 0.0
            vec[base + 3] = 1.0 if not char.isalnum() else 0.0
            # Normalized character value
            vec[base + 4] = ord(char) / 127.0 if ord(char) < 128 else 0.0
            # Position in common character sets
            vec[base + 5] = (ord(char.lower()) - ord('a')) / 26.0 if char.isalpha() else 0.0
            vec[base + 6] = (ord(char) - ord('0')) / 10.0 if char.isdigit() else 0.0
            # Is ASCII
            vec[base + 7] = 1.0 if ord(char) < 128 else 0.0

        return vec

    def encode_string(self, s: str, max_len: int = 32) -> np.ndarray:
        """Encode a string to a fixed-length vector"""
        if self.use_embedding:
            return self.encode_string_dense(s, max_len)
        # Legacy one-hot (kept for backward compatibility)
        vec = np.zeros(max_len * len(self.CHAR_SET))
        for i, char in enumerate(s[:max_len]):
            if char in self.CHAR_SET:
                idx = i * len(self.CHAR_SET) + self.CHAR_SET.index(char)
                vec[idx] = 1.0
        return vec

    def encode_date_features(self, date: str) -> np.ndarray:
        """Encode date features with multiple representations"""
        features = np.zeros(32)

        if len(date) == 8 and date.isdigit():
            year = int(date[:4])
            month = int(date[4:6])
            day = int(date[6:8])

            # Normalize
            features[0] = (year - 1950) / 100  # Year relative to 1950
            features[1] = month / 12
            features[2] = day / 31

            # Common date patterns
            features[3] = int(date) / 1e8  # Full date normalized
            features[4] = int(date[2:]) / 1e6  # YYMMDD
            features[5] = int(date[4:]) / 1e4  # MMDD
            features[6] = year % 100 / 100  # Last two digits of year

        return features

    def encode_numbers(self, numbers: List[str]) -> np.ndarray:
        """Encode favorite numbers"""
        features = np.zeros(16)

        for i, num in enumerate(numbers[:8]):
            if num.isdigit():
                features[i * 2] = int(num) / 1e4
                features[i * 2 + 1] = len(num) / 4  # Length feature

        return features

    def vectorize(self, features: TargetFeatures) -> np.ndarray:
        """Convert TargetFeatures to a fixed-size vector"""
        vectors = []

        # Name encoding (64 dims)
        name_parts = []
        if features.full_name:
            name_parts.append(features.full_name)
        if features.first_name:
            name_parts.append(features.first_name)
        if features.last_name:
            name_parts.append(features.last_name)
        if features.nickname:
            name_parts.append(features.nickname)

        name_vec = self.encode_string(" ".join(name_parts), max_len=16)
        vectors.append(name_vec[:64])

        # Date encoding (32 dims)
        date_vec = self.encode_date_features(features.birthday)
        vectors.append(date_vec)

        # Hobby encoding (64 dims)
        hobby_str = " ".join(features.hobbies[:5])
        hobby_vec = self.encode_string(hobby_str, max_len=16)
        vectors.append(hobby_vec[:64])

        # Number encoding (16 dims)
        all_numbers = features.favorite_numbers + [features.phone] if features.phone else features.favorite_numbers
        num_vec = self.encode_numbers(all_numbers)
        vectors.append(num_vec)

        # Keyword encoding (64 dims)
        keyword_str = " ".join(features.keywords[:5] + features.favorite_words[:5])
        keyword_vec = self.encode_string(keyword_str, max_len=16)
        vectors.append(keyword_vec[:64])

        # Pattern encoding (16 dims)
        pattern_vec = np.zeros(16)
        for i, pattern in enumerate(features.known_patterns[:4]):
            pattern_vec[i * 4:i * 4 + 4] = self.encode_string(pattern, max_len=1)[:4]
        vectors.append(pattern_vec)

        # Concatenate all vectors
        result = np.concatenate(vectors)

        # Pad or truncate to vector_dim
        if len(result) < self.vector_dim:
            result = np.pad(result, (0, self.vector_dim - len(result)))
        else:
            result = result[:self.vector_dim]

        return result

    def generate_password_components(self, features: TargetFeatures) -> Dict[str, List[str]]:
        """Generate possible password components from features"""
        components = {
            'names': [],
            'dates': [],
            'numbers': [],
            'words': [],
            'special_chars': ['!', '@', '#', '$', '%', '&', '*']
        }

        # Names and variants
        for name in [features.first_name, features.last_name, features.nickname]:
            if name:
                components['names'].extend([
                    name,
                    name.lower(),
                    name.upper(),
                    name.capitalize(),
                    name.lower() + name.upper(),  # e.g., johnJOHN
                    name[:3],  # Abbreviation
                    name[0].upper(),  # Initial
                ])

        # Date variants
        if features.birthday:
            bd = features.birthday
            components['dates'].extend([
                bd,              # 19900115
                bd[2:],          # 900115
                bd[4:],          # 0115
                bd[:4],          # 1990
                bd[2:4],         # 90
                f"{bd[4:6]}{bd[6:8]}",  # MMDD
                bd[6:8] + bd[4:6],  # DDMM
            ])

        # Numbers
        if features.phone:
            components['numbers'].extend([
                features.phone,
                features.phone[-4:],  # Last 4 digits
                features.phone[-6:],  # Last 6 digits
            ])

        components['numbers'].extend(features.favorite_numbers)

        # Words
        components['words'].extend(features.hobbies)
        components['words'].extend(features.favorite_words)
        components['words'].extend(features.pet_names)
        components['words'].extend(features.keywords)

        # Remove duplicates and empty strings
        for key in components:
            components[key] = list(set(filter(None, components[key])))

        return components
