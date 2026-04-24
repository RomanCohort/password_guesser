"""
Differential Evolution Optimizer for Password Candidate Search

Implements multiple mutation strategies with adaptive selection:
- DE/rand/1, DE/best/1, DE/current-to-best/1
- DE/rand/2, DE/best/2
- Strategy adaptation based on success rates

Uses DE to search the password space efficiently, guided by
the probability landscape from the MAMBA model.
"""

import numpy as np
from typing import List, Optional, Callable, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import random


class MutationStrategy(Enum):
    """DE mutation strategies"""
    RAND_1 = "rand/1"           # v = x_r1 + F * (x_r2 - x_r3)
    BEST_1 = "best/1"           # v = x_best + F * (x_r1 - x_r2)
    CURRENT_TO_BEST_1 = "current-to-best/1"  # v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
    RAND_2 = "rand/2"           # v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
    BEST_2 = "best/2"           # v = x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)


@dataclass
class PasswordCandidate:
    """A password candidate with its fitness score"""
    password: str
    score: float
    generation: int
    strategy: Optional[str] = None
    parent_indices: Optional[Tuple[int, int]] = None

    def __lt__(self, other):
        return self.score < other.score


class PasswordDEOptimizer:
    """
    Differential Evolution optimizer for password candidate generation.

    Operates in a continuous latent space and maps back to discrete
    password characters through the model's scoring function.

    The DE optimizes over a representation that encodes password structure:
    - Each dimension represents a character position
    - Continuous values are mapped to character probabilities
    - Fitness is evaluated by the MAMBA model's probability score
    """

    def __init__(
        self,
        fitness_fn: Callable[[str], float],
        vocab_size: int = 96,       # Printable ASCII range
        max_length: int = 20,
        population_size: int = 100,
        mutation_rate: float = 0.8,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.1
    ):
        """
        Initialize DE optimizer.

        Args:
            fitness_fn: Function that scores a password (higher = better)
            vocab_size: Number of possible characters
            max_length: Maximum password length to generate
            population_size: Size of the population
            mutation_rate: DE scaling factor F
            crossover_rate: DE crossover probability CR
            elite_ratio: Fraction of elite individuals to preserve
        """
        self.fitness_fn = fitness_fn
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = max(1, int(population_size * elite_ratio))

        # Printable ASCII characters (space to ~)
        self.vocab = [chr(i) for i in range(32, 127)]

        # Population: continuous vectors in [0, 1]
        # Each vector has max_length dimensions
        self.population: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None
        self.generation = 0
        self.best_candidates: List[PasswordCandidate] = []
        self.history: List[dict] = []

    def _continuous_to_password(self, vector: np.ndarray, length: int) -> str:
        """Convert continuous vector to password string"""
        chars = []
        for i in range(length):
            # Ensure idx is within valid range [0, vocab_size-1]
            idx = int(vector[i] * self.vocab_size) % self.vocab_size
            if idx >= len(self.vocab):
                idx = idx % len(self.vocab)
            chars.append(self.vocab[idx])
        return ''.join(chars)

    def _password_to_continuous(self, password: str) -> np.ndarray:
        """Convert password string to continuous vector"""
        vector = np.zeros(self.max_length)
        for i, char in enumerate(password):
            if char in self.vocab:
                vector[i] = self.vocab.index(char) / self.vocab_size
            else:
                vector[i] = np.random.random()
        return vector

    def _encode_length(self, length: int) -> float:
        """Encode password length as a continuous value"""
        return length / self.max_length

    def _decode_length(self, value: float) -> int:
        """Decode continuous value to password length"""
        return max(4, min(self.max_length, int(value * self.max_length)))

    def initialize_population(self, seeds: Optional[List[str]] = None) -> None:
        """
        Initialize population, optionally seeded with known passwords.

        Args:
            seeds: Optional list of seed passwords to include
        """
        self.population = np.random.random((self.population_size, self.max_length + 1))

        # The last dimension encodes the length
        self.population[:, -1] = np.random.uniform(4 / self.max_length, 1.0, self.population_size)

        # Inject seed passwords
        if seeds:
            for i, seed in enumerate(seeds[:self.population_size]):
                if len(seed) > self.max_length:
                    continue
                self.population[i, :-1] = self._password_to_continuous(seed)
                self.population[i, -1] = self._encode_length(len(seed))

        # Evaluate initial fitness
        self.fitness = np.array([self._evaluate_individual(ind) for ind in self.population])
        self.generation = 0

        # Track best
        self._update_best()

    def _evaluate_individual(self, individual: np.ndarray) -> float:
        """Evaluate fitness of an individual"""
        length = self._decode_length(individual[-1])
        password = self._continuous_to_password(individual[:-1], length)
        return self.fitness_fn(password)

    def _mutation(self, target_idx: int) -> np.ndarray:
        """
        DE/rand/1 mutation: v = x_r1 + F * (x_r2 - x_r3)

        Args:
            target_idx: Index of the target individual

        Returns:
            Mutant vector
        """
        # Select three random distinct individuals (different from target)
        candidates = list(range(self.population_size))
        candidates.remove(target_idx)
        r1, r2, r3 = random.sample(candidates, 3)

        mutant = (self.population[r1] +
                 self.mutation_rate * (self.population[r2] - self.population[r3]))

        # Clip to [0, 1]
        mutant = np.clip(mutant, 0, 1)

        return mutant

    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Binomial crossover.

        Args:
            target: Target vector
            mutant: Mutant vector

        Returns:
            Trial vector
        """
        trial = target.copy()
        j_rand = random.randint(0, len(target) - 1)  # Ensure at least one dimension from mutant

        for j in range(len(target)):
            if random.random() < self.crossover_rate or j == j_rand:
                trial[j] = mutant[j]

        return trial

    def _selection(self, target_idx: int, trial: np.ndarray) -> None:
        """
        Greedy selection: keep the better of target and trial.

        Args:
            target_idx: Index of the target individual
            trial: Trial vector
        """
        trial_fitness = self._evaluate_individual(trial)

        if trial_fitness >= self.fitness[target_idx]:
            self.population[target_idx] = trial
            self.fitness[target_idx] = trial_fitness

    def _update_best(self) -> None:
        """Update the list of best candidates"""
        sorted_indices = np.argsort(self.fitness)[::-1]

        self.best_candidates = []
        seen_passwords = set()

        for idx in sorted_indices:
            length = self._decode_length(self.population[idx, -1])
            password = self._continuous_to_password(self.population[idx, :-1], length)

            if password not in seen_passwords:
                seen_passwords.add(password)
                self.best_candidates.append(PasswordCandidate(
                    password=password,
                    score=self.fitness[idx],
                    generation=self.generation
                ))

            if len(self.best_candidates) >= 50:
                break

    def evolve_one_generation(self) -> dict:
        """
        Run one generation of differential evolution.

        Returns:
            Statistics for this generation
        """
        if self.population is None:
            raise RuntimeError("Population not initialized. Call initialize_population() first.")

        for i in range(self.population_size):
            # Mutation
            mutant = self._mutation(i)

            # Crossover
            trial = self._crossover(self.population[i], mutant)

            # Selection
            self._selection(i, trial)

        self.generation += 1
        self._update_best()

        # Statistics
        stats = {
            'generation': self.generation,
            'best_fitness': float(np.max(self.fitness)),
            'mean_fitness': float(np.mean(self.fitness)),
            'std_fitness': float(np.std(self.fitness)),
            'best_password': self.best_candidates[0].password if self.best_candidates else "",
        }

        self.history.append(stats)
        return stats

    def run(
        self,
        max_generations: int = 50,
        convergence_threshold: float = 1e-6,
        verbose: bool = True
    ) -> List[PasswordCandidate]:
        """
        Run the DE optimization.

        Args:
            max_generations: Maximum number of generations
            convergence_threshold: Stop if best fitness doesn't improve
            verbose: Print progress

        Returns:
            List of best candidates found
        """
        for gen in range(max_generations):
            stats = self.evolve_one_generation()

            if verbose and gen % 10 == 0:
                print(f"Gen {gen:3d} | Best: {stats['best_fitness']:.4f} "
                      f"| Mean: {stats['mean_fitness']:.4f} "
                      f"| Best Password: {stats['best_password']}")

            # Check convergence
            if gen > 10:
                recent = [h['best_fitness'] for h in self.history[-10:]]
                if abs(recent[-1] - recent[0]) < convergence_threshold:
                    if verbose:
                        print(f"Converged at generation {gen}")
                    break

        return self.best_candidates

    def get_top_candidates(self, n: int = 20) -> List[PasswordCandidate]:
        """Get the top N password candidates"""
        return self.best_candidates[:n]


class HybridDEOptimizer(PasswordDEOptimizer):
    """
    Hybrid DE optimizer that combines DE with model-guided mutations.

    Uses the MAMBA model to guide mutation direction towards higher
    probability regions of the password space.
    """

    def __init__(self, *args, model_score_fn=None, **kwargs):
        """
        Args:
            model_score_fn: Function from MAMBA model that scores passwords
        """
        super().__init__(*args, **kwargs)
        self.model_score_fn = model_score_fn

    def _model_guided_mutation(self, individual: np.ndarray) -> np.ndarray:
        """
        Use model probabilities to bias mutations.

        For each position, the model suggests the most likely next character,
        which biases the mutation towards higher-probability characters.
        """
        mutant = individual.copy()
        length = self._decode_length(individual[-1])

        if self.model_score_fn is not None:
            # Get current password
            password = self._continuous_to_password(individual[:-1], length)

            # Get character probabilities from model
            try:
                char_probs = self.model_score_fn(password)
                # char_probs: dict mapping characters to probabilities

                # Bias mutation towards high-probability characters
                for i in range(length):
                    if random.random() < 0.3:  # 30% chance of model-guided mutation
                        if i < len(password) and password[i] in char_probs:
                            # Sample from model distribution
                            chars = list(char_probs.keys())
                            probs = list(char_probs.values())
                            if probs and sum(probs) > 0:
                                chosen = np.random.choice(chars, p=probs)
                                mutant[i] = self._password_to_continuous(chosen)[0] if chosen in self.vocab else mutant[i]
            except Exception as e:
                logger.debug(f"Guided mutation fallback to standard: {e}")

        return mutant

    def _mutation(self, target_idx: int) -> np.ndarray:
        """Enhanced mutation with model guidance"""
        # Standard DE mutation
        standard_mutant = super()._mutation(target_idx)

        # Model-guided mutation (50% of the time)
        if random.random() < 0.5 and self.model_score_fn is not None:
            guided_mutant = self._model_guided_mutation(self.population[target_idx])
            # Blend the two
            alpha = random.random()
            return alpha * standard_mutant + (1 - alpha) * guided_mutant

        return standard_mutant


class AdaptiveDEOptimizer(PasswordDEOptimizer):
    """
    Adaptive DE with self-adjusting F and CR parameters.

    Implements the jDE adaptive parameter control strategy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Individual-level parameters
        self.F = np.full(self.population_size, self.mutation_rate)
        self.CR = np.full(self.population_size, self.crossover_rate)

        # Parameter ranges
        self.F_bounds = (0.1, 1.0)
        self.CR_bounds = (0.1, 1.0)
        self.tau1 = 0.1  # Probability of adapting F
        self.tau2 = 0.1  # Probability of adapting CR

    def _adapt_parameters(self, idx: int):
        """Adapt F and CR for individual idx"""
        if random.random() < self.tau1:
            self.F[idx] = self.F_bounds[0] + random.random() * (self.F_bounds[1] - self.F_bounds[0])

        if random.random() < self.tau2:
            self.CR[idx] = self.CR_bounds[0] + random.random() * (self.CR_bounds[1] - self.CR_bounds[0])

    def _mutation(self, target_idx: int) -> np.ndarray:
        """Adaptive mutation"""
        self._adapt_parameters(target_idx)

        candidates = list(range(self.population_size))
        candidates.remove(target_idx)
        r1, r2, r3 = random.sample(candidates, 3)

        mutant = (self.population[r1] +
                 self.F[target_idx] * (self.population[r2] - self.population[r3]))

        return np.clip(mutant, 0, 1)

    def _crossover(self, target: np.ndarray, mutant: np.ndarray, target_idx: int = 0) -> np.ndarray:
        """Adaptive crossover with correct CR indexing"""
        trial = target.copy()
        j_rand = random.randint(0, len(target) - 1)

        for j in range(len(target)):
            if random.random() < self.CR[target_idx] or j == j_rand:
                trial[j] = mutant[j]

        return trial

    def evolve_one_generation(self) -> dict:
        """
        Run one generation of differential evolution.

        Returns:
            Statistics for this generation
        """
        if self.population is None:
            raise RuntimeError("Population not initialized. Call initialize_population() first.")

        for i in range(self.population_size):
            # Mutation
            mutant = self._mutation(i)

            # Crossover
            trial = self._crossover(self.population[i], mutant, target_idx=i)

            # Selection
            self._selection(i, trial)

        self.generation += 1
        self._update_best()

        # Statistics
        stats = {
            'generation': self.generation,
            'best_fitness': float(np.max(self.fitness)),
            'mean_fitness': float(np.mean(self.fitness)),
            'std_fitness': float(np.std(self.fitness)),
            'best_password': self.best_candidates[0].password if self.best_candidates else "",
        }

        self.history.append(stats)
        return stats


class StructuredPasswordDEOptimizer(PasswordDEOptimizer):
    """
    Structured DE that operates on password patterns rather than raw characters.

    The encoding is:
    - Pattern type (categorical, one-hot)
    - Length (continuous)
    - For each slot: character probabilities (softmax over vocab subset)

    This creates a smoother fitness landscape for DE to navigate.
    """

    PATTERNS = ['name_digit', 'name_date', 'word_digit', 'leet', 'capitalized', 'random']

    def __init__(self, fitness_fn: Callable[[str], float],
                 pattern_components: Optional[dict] = None, **kwargs):
        super().__init__(fitness_fn, **kwargs)
        self.pattern_components = pattern_components or {}
        self.num_patterns = len(self.PATTERNS)

    def _decode_structured_individual(self, individual: np.ndarray) -> str:
        """Decode structured representation to password"""
        # First num_patterns dimensions: pattern type (one-hot -> argmax)
        pattern_idx = np.argmax(individual[:self.num_patterns])
        pattern = self.PATTERNS[pattern_idx]

        # Next dimension: length
        length = max(4, min(self.max_length, int(individual[self.num_patterns] * self.max_length)))

        # Remaining: character slots (softmax over each)
        chars = []
        slot_start = self.num_patterns + 1

        for i in range(length):
            slot_start_idx = slot_start + i * self.vocab_size
            slot_end_idx = min(slot_start_idx + self.vocab_size, len(individual))

            if slot_end_idx > slot_start_idx:
                # Softmax over this slot
                slot_probs = individual[slot_start_idx:slot_end_idx]
                # Convert to proper softmax
                slot_probs = np.exp(slot_probs - np.max(slot_probs))
                slot_probs = slot_probs / slot_probs.sum()

                char_idx = np.random.choice(len(slot_probs), p=slot_probs)
                if char_idx < len(self.vocab):
                    chars.append(self.vocab[char_idx])

        return ''.join(chars)


class MultiStrategyDEOptimizer(PasswordDEOptimizer):
    """
    Multi-strategy Adaptive Differential Evolution (MSADE).

    Features:
    - Multiple mutation strategies with adaptive selection
    - Strategy success rate tracking
    - Automatic best strategy promotion

    Mutation strategies:
    - DE/rand/1: Good for exploration
    - DE/best/1: Fast convergence
    - DE/current-to-best/1: Balanced exploration/exploitation
    - DE/rand/2: Strong exploration
    - DE/best/2: Fine-tuning near optimum
    """

    STRATEGIES = [
        MutationStrategy.RAND_1,
        MutationStrategy.BEST_1,
        MutationStrategy.CURRENT_TO_BEST_1,
        MutationStrategy.RAND_2,
        MutationStrategy.BEST_2,
    ]

    def __init__(
        self,
        fitness_fn: Callable[[str], float],
        strategy_weights: Optional[Dict[str, float]] = None,
        adaptation_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(fitness_fn, **kwargs)

        # Strategy weights (probability of selecting each strategy)
        if strategy_weights is None:
            self.strategy_weights = {s.value: 1.0 / len(self.STRATEGIES) for s in self.STRATEGIES}
        else:
            self.strategy_weights = strategy_weights

        # Track success counts for each strategy
        self.strategy_success: Dict[str, int] = {s.value: 0 for s in self.STRATEGIES}
        self.strategy_trials: Dict[str, int] = {s.value: 0 for s in self.STRATEGIES}

        self.adaptation_rate = adaptation_rate
        self._best_idx = 0

    def _select_strategy(self) -> MutationStrategy:
        """Select mutation strategy based on weights (roulette wheel)"""
        strategies = list(self.strategy_weights.keys())
        weights = [self.strategy_weights[s] for s in strategies]
        total = sum(weights)

        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(strategies)] * len(strategies)

        r = random.random()
        cumsum = 0.0
        for strategy, weight in zip(strategies, weights):
            cumsum += weight
            if r <= cumsum:
                return MutationStrategy(strategy)

        return MutationStrategy(strategies[-1])

    def _mutation_with_strategy(
        self,
        target_idx: int,
        strategy: MutationStrategy
    ) -> np.ndarray:
        """Apply mutation according to selected strategy."""
        candidates = list(range(self.population_size))
        candidates.remove(target_idx)

        best_idx = np.argmax(self.fitness)
        x_best = self.population[best_idx]

        if strategy == MutationStrategy.RAND_1:
            r1, r2, r3 = random.sample(candidates, 3)
            mutant = (self.population[r1] +
                     self.mutation_rate * (self.population[r2] - self.population[r3]))

        elif strategy == MutationStrategy.BEST_1:
            r1, r2 = random.sample(candidates, 2)
            mutant = (x_best +
                     self.mutation_rate * (self.population[r1] - self.population[r2]))

        elif strategy == MutationStrategy.CURRENT_TO_BEST_1:
            r1, r2 = random.sample(candidates, 2)
            x_i = self.population[target_idx]
            mutant = (x_i +
                     self.mutation_rate * (x_best - x_i) +
                     self.mutation_rate * (self.population[r1] - self.population[r2]))

        elif strategy == MutationStrategy.RAND_2:
            r1, r2, r3, r4, r5 = random.sample(candidates, 5)
            mutant = (self.population[r1] +
                     self.mutation_rate * (self.population[r2] - self.population[r3]) +
                     self.mutation_rate * (self.population[r4] - self.population[r5]))

        elif strategy == MutationStrategy.BEST_2:
            r1, r2, r3, r4 = random.sample(candidates, 4)
            mutant = (x_best +
                     self.mutation_rate * (self.population[r1] - self.population[r2]) +
                     self.mutation_rate * (self.population[r3] - self.population[r4]))
        else:
            r1, r2, r3 = random.sample(candidates, 3)
            mutant = (self.population[r1] +
                     self.mutation_rate * (self.population[r2] - self.population[r3]))

        return np.clip(mutant, 0, 1)

    def _adapt_strategy_weights(self):
        """Update strategy weights based on success rates"""
        for strategy in self.strategy_weights:
            trials = self.strategy_trials[strategy]
            if trials > 0:
                success_rate = self.strategy_success[strategy] / trials
                self.strategy_weights[strategy] = (
                    (1 - self.adaptation_rate) * self.strategy_weights[strategy] +
                    self.adaptation_rate * success_rate
                )

        total = sum(self.strategy_weights.values())
        if total > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total

    def evolve_one_generation(self) -> dict:
        """Run one generation with multi-strategy adaptation."""
        if self.population is None:
            raise RuntimeError("Population not initialized.")

        generation_success = {s.value: 0 for s in self.STRATEGIES}
        generation_trials = {s.value: 0 for s in self.STRATEGIES}

        for i in range(self.population_size):
            strategy = self._select_strategy()
            strategy_name = strategy.value

            self.strategy_trials[strategy_name] += 1
            generation_trials[strategy_name] += 1

            mutant = self._mutation_with_strategy(i, strategy)
            trial = self._crossover(self.population[i], mutant)
            trial_fitness = self._evaluate_individual(trial)

            if trial_fitness >= self.fitness[i]:
                self.population[i] = trial
                self.fitness[i] = trial_fitness
                self.strategy_success[strategy_name] += 1
                generation_success[strategy_name] += 1

        self.generation += 1

        if self.generation % 10 == 0:
            self._adapt_strategy_weights()

        self._update_best()

        stats = {
            'generation': self.generation,
            'best_fitness': float(np.max(self.fitness)),
            'mean_fitness': float(np.mean(self.fitness)),
            'std_fitness': float(np.std(self.fitness)),
            'best_password': self.best_candidates[0].password if self.best_candidates else "",
            'strategy_weights': dict(self.strategy_weights),
            'generation_success': generation_success,
        }

        self.history.append(stats)
        return stats

    def run(
        self,
        max_generations: int = 50,
        convergence_threshold: float = 1e-6,
        verbose: bool = True
    ) -> List[PasswordCandidate]:
        """Run multi-strategy DE optimization."""
        for gen in range(max_generations):
            stats = self.evolve_one_generation()

            if verbose:
                best_strategy = max(self.strategy_weights.keys(),
                                   key=lambda s: self.strategy_weights[s])
                print(f"Gen {gen:3d} | Best: {stats['best_fitness']:.4f} "
                      f"| Mean: {stats['mean_fitness']:.4f} "
                      f"| Strategy: {best_strategy[:15]} "
                      f"| Password: {stats['best_password'][:20]}")

            if gen > 10:
                recent = [h['best_fitness'] for h in self.history[-10:]]
                if abs(recent[-1] - recent[0]) < convergence_threshold:
                    if verbose:
                        print(f"Converged at generation {gen}")
                    break

        return self.best_candidates

    def get_strategy_stats(self) -> Dict[str, dict]:
        """Get statistics for each strategy"""
        stats = {}
        for strategy in self.STRATEGIES:
            name = strategy.value
            trials = self.strategy_trials[name]
            success = self.strategy_success[name]
            stats[name] = {
                'trials': trials,
                'successes': success,
                'success_rate': success / trials if trials > 0 else 0.0,
                'weight': self.strategy_weights[name],
            }
        return stats


class SHADE(MultiStrategyDEOptimizer):
    """
    Success-History Based Adaptive Differential Evolution (SHADE).

    Advanced variant that maintains a history of successful F and CR values,
    and adapts parameters based on this history.
    """

    def __init__(
        self,
        fitness_fn: Callable[[str], float],
        history_size: int = 100,
        **kwargs
    ):
        super().__init__(fitness_fn, **kwargs)

        self.history_size = history_size
        self.F_history = [0.5] * history_size
        self.CR_history = [0.5] * history_size
        self.history_idx = 0

        self.F = np.full(self.population_size, 0.5)
        self.CR = np.full(self.population_size, 0.5)

    def _adapt_parameters(self, idx: int):
        """Sample F and CR from historical memory"""
        r = random.randint(0, self.history_size - 1)
        self.F[idx] = np.clip(np.random.standard_cauchy() * 0.1 + self.F_history[r], 0, 1)
        self.CR[idx] = np.clip(np.random.normal(self.CR_history[r], 0.1), 0, 1)

    def _update_history(self, successful_F: List[float], successful_CR: List[float]):
        """Update historical memory with successful parameters"""
        if successful_F:
            mean_F = sum(f * f for f in successful_F) / sum(f for f in successful_F)
            mean_CR = sum(successful_CR) / len(successful_CR)

            self.F_history[self.history_idx] = mean_F
            self.CR_history[self.history_idx] = mean_CR
            self.history_idx = (self.history_idx + 1) % self.history_size

    def evolve_one_generation(self) -> dict:
        """Run one generation with SHADE parameter adaptation"""
        if self.population is None:
            raise RuntimeError("Population not initialized.")

        successful_F = []
        successful_CR = []

        for i in range(self.population_size):
            self._adapt_parameters(i)

            strategy = self._select_strategy()
            strategy_name = strategy.value
            self.strategy_trials[strategy_name] += 1

            current_F = self.F[i]
            current_CR = self.CR[i]

            original_rate = self.mutation_rate
            self.mutation_rate = current_F

            mutant = self._mutation_with_strategy(i, strategy)
            self.mutation_rate = original_rate

            trial = self._crossover_adaptive(self.population[i], mutant, current_CR)
            trial_fitness = self._evaluate_individual(trial)

            if trial_fitness >= self.fitness[i]:
                self.population[i] = trial
                self.fitness[i] = trial_fitness

                self.strategy_success[strategy_name] += 1
                successful_F.append(current_F)
                successful_CR.append(current_CR)

        self.generation += 1
        self._update_history(successful_F, successful_CR)

        if self.generation % 10 == 0:
            self._adapt_strategy_weights()

        self._update_best()

        stats = {
            'generation': self.generation,
            'best_fitness': float(np.max(self.fitness)),
            'mean_fitness': float(np.mean(self.fitness)),
            'best_password': self.best_candidates[0].password if self.best_candidates else "",
            'mean_F': float(np.mean(self.F)),
            'mean_CR': float(np.mean(self.CR)),
        }

        self.history.append(stats)
        return stats

    def _crossover_adaptive(self, target: np.ndarray, mutant: np.ndarray, CR: float) -> np.ndarray:
        """Crossover with individual-specific CR"""
        trial = target.copy()
        j_rand = random.randint(0, len(target) - 1)

        for j in range(len(target)):
            if random.random() < CR or j == j_rand:
                trial[j] = mutant[j]

        return trial


class ParallelDEOptimizer(MultiStrategyDEOptimizer):
    """
    Multi-process DE optimizer with parallel fitness evaluation.

    Uses multiprocessing.Pool to evaluate fitness in parallel,
    significantly speeding up the optimization for expensive fitness functions.
    """

    def __init__(
        self,
        fitness_fn: Callable[[str], float],
        n_workers: int = 4,
        chunk_size: int = 10,
        **kwargs
    ):
        super().__init__(fitness_fn, **kwargs)
        self.n_workers = n_workers
        self.chunk_size = chunk_size

    def _evaluate_batch(self, individuals: List[np.ndarray]) -> List[float]:
        """Evaluate a batch of individuals in parallel"""
        # Convert individuals to passwords
        passwords = []
        for ind in individuals:
            length = self._decode_length(ind[-1])
            password = self._continuous_to_password(ind[:-1], length)
            passwords.append(password)

        # Use multiprocessing for expensive fitness functions
        try:
            from multiprocessing import Pool
            with Pool(self.n_workers) as pool:
                fitness_values = pool.map(self.fitness_fn, passwords, chunksize=self.chunk_size)
            return list(fitness_values)
        except Exception:
            # Fallback to sequential evaluation
            return [self.fitness_fn(p) for p in passwords]

    def initialize_population(self, seeds: Optional[List[str]] = None) -> None:
        """Initialize with parallel evaluation"""
        self.population = np.random.random((self.population_size, self.max_length + 1))
        self.population[:, -1] = np.random.uniform(4 / self.max_length, 1.0, self.population_size)

        if seeds:
            for i, seed in enumerate(seeds[:self.population_size]):
                if len(seed) > self.max_length:
                    continue
                self.population[i, :-1] = self._password_to_continuous(seed)
                self.population[i, -1] = self._encode_length(len(seed))

        # Parallel initial evaluation
        self.fitness = np.array(
            self._evaluate_batch([self.population[i] for i in range(self.population_size)])
        )
        self.generation = 0
        self._update_best()

    def evolve_one_generation(self) -> dict:
        """Run one generation with parallel evaluation"""
        if self.population is None:
            raise RuntimeError("Population not initialized.")

        generation_success = {s.value: 0 for s in self.STRATEGIES}
        generation_trials = {s.value: 0 for s in self.STRATEGIES}

        # Generate all trial vectors first
        trials = []
        strategies_used = []

        for i in range(self.population_size):
            strategy = self._select_strategy()
            strategies_used.append(strategy)

            self.strategy_trials[strategy.value] += 1
            generation_trials[strategy.value] += 1

            mutant = self._mutation_with_strategy(i, strategy)
            trial = self._crossover(self.population[i], mutant)
            trials.append(trial)

        # Evaluate all trials in parallel
        trial_fitnesses = self._evaluate_batch(trials)

        # Selection
        for i in range(self.population_size):
            strategy_name = strategies_used[i].value
            if trial_fitnesses[i] >= self.fitness[i]:
                self.population[i] = trials[i]
                self.fitness[i] = trial_fitnesses[i]
                self.strategy_success[strategy_name] += 1
                generation_success[strategy_name] += 1

        self.generation += 1

        if self.generation % 10 == 0:
            self._adapt_strategy_weights()

        self._update_best()

        stats = {
            'generation': self.generation,
            'best_fitness': float(np.max(self.fitness)),
            'mean_fitness': float(np.mean(self.fitness)),
            'std_fitness': float(np.std(self.fitness)),
            'best_password': self.best_candidates[0].password if self.best_candidates else "",
            'strategy_weights': dict(self.strategy_weights),
        }

        self.history.append(stats)
        return stats
