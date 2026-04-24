"""PCFG Generator: main entry point for PCFG-based password generation."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from pcfg.training import PCFGTrainer, PCFGModel
from pcfg.grammar import Grammar


@dataclass
class PCFGConfig:
    """Configuration for the PCFG generator."""
    max_length: int = 32
    min_length: int = 4
    n_gram_order: int = 2
    smoothing: float = 1e-10


class PCFGGenerator:
    """PCFG-based password generator.

    Provides a high-level interface for training a PCFG from passwords
    and generating new password candidates.
    """

    def __init__(self, config: Optional[PCFGConfig] = None):
        self.config = config or PCFGConfig()
        self.model: Optional[PCFGModel] = None
        self._trainer = PCFGTrainer(max_length=self.config.max_length)

    def train(self, passwords: List[str]):
        """Train the PCFG model from a list of passwords.

        Args:
            passwords: List of password strings to learn from.
        """
        # Filter by length constraints
        filtered = [
            pwd.strip() for pwd in passwords
            if pwd and self.config.min_length <= len(pwd.strip()) <= self.config.max_length
        ]
        self._trainer = PCFGTrainer(max_length=self.config.max_length)
        self.model = self._trainer.train(filtered)

    def train_from_file(self, filepath: str, max_samples: int = 100000):
        """Train the PCFG model from a password file.

        Args:
            filepath: Path to a file with one password per line.
            max_samples: Maximum number of samples to read.
        """
        self._trainer = PCFGTrainer(max_length=self.config.max_length)
        self.model = self._trainer.train_from_file(filepath, max_samples=max_samples)

    def generate(self, n: int = 100) -> List[str]:
        """Generate n password candidates.

        Args:
            n: Number of unique passwords to generate.

        Returns:
            List of password strings, ordered by descending probability.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or train_from_file() first.")

        results = self.model.generate(n=n)
        return [pwd for pwd, _ in results]

    def generate_with_scores(self, n: int = 100) -> List[Tuple[str, float]]:
        """Generate n password candidates with their probabilities.

        Args:
            n: Number of unique passwords to generate.

        Returns:
            List of (password, probability) tuples, ordered by descending probability.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or train_from_file() first.")

        return self.model.generate(n=n)

    def score_password(self, password: str) -> float:
        """Score a password using the trained PCFG model.

        Args:
            password: The password to score.

        Returns:
            Log-probability of the password under the model. Higher is more likely.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or train_from_file() first.")

        prob = self.model.grammar.probability(password)
        if prob <= 0:
            return float('-inf')
        import math
        return math.log(prob)

    def load_model(self, path: str):
        """Load a previously saved model.

        Args:
            path: Path to the saved model JSON file.
        """
        self.model = PCFGModel.load(path)

    def save_model(self, path: str):
        """Save the current model to a file.

        Args:
            path: Output file path.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train or load a model first.")
        self.model.save(path)
