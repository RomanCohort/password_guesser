"""
Persistent Experience Store

Cross-session experience persistence for reinforcement learning.
Stores experiences in JSONL format for incremental appending and efficient sampling.
"""

import json
import os
import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StoredExperience:
    """Experience with metadata for persistent storage."""
    state_vector: List[float]
    action_index: int
    reward: float
    next_state_vector: List[float]
    done: bool
    action_mask: List[float]
    reflection_weight: float
    session_id: str
    timestamp: float
    success: bool  # Whether this step led to eventual success

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'StoredExperience':
        return cls(**data)


class PersistentExperienceStore:
    """
    Cross-session persistent experience storage.

    Features:
    - JSONL format for incremental appending
    - Prioritized sampling based on reward and reflection_weight
    - Session tracking and statistics
    - Automatic pruning of old low-value experiences
    """

    def __init__(self, path: str = "data/experience_store"):
        self.path = path
        self.buffer_file = os.path.join(path, "buffer.jsonl")
        self.metadata_file = os.path.join(path, "metadata.json")
        self.capacity = 100000  # Maximum experiences to store

        # In-memory buffer for fast access
        self.buffer: List[StoredExperience] = []

        # Metadata
        self.metadata = {
            "total_experiences": 0,
            "sessions_contributed": set(),
            "last_updated": None,
            "statistics": {
                "total_reward": 0.0,
                "success_count": 0,
                "failure_count": 0,
                "avg_reward": 0.0,
            }
        }

        # Ensure directory exists
        os.makedirs(path, exist_ok=True)

    def add_experience(
        self,
        state_vector: np.ndarray,
        action_index: int,
        reward: float,
        next_state_vector: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        reflection_weight: float = 1.0,
        session_id: str = None,
        success: bool = False,
    ) -> None:
        """Add a single experience to the store."""
        experience = StoredExperience(
            state_vector=state_vector.tolist() if isinstance(state_vector, np.ndarray) else state_vector,
            action_index=action_index,
            reward=reward,
            next_state_vector=next_state_vector.tolist() if isinstance(next_state_vector, np.ndarray) else next_state_vector,
            done=done,
            action_mask=action_mask.tolist() if isinstance(action_mask, np.ndarray) else action_mask,
            reflection_weight=reflection_weight,
            session_id=session_id or f"session_{int(time.time())}",
            timestamp=time.time(),
            success=success,
        )

        self.buffer.append(experience)
        self._update_metadata(experience)

        # Prune if over capacity
        if len(self.buffer) > self.capacity:
            self._prune_buffer()

    def add_episode(
        self,
        experiences: List[Tuple],
        session_id: str = None,
        final_success: bool = False,
    ) -> None:
        """
        Add a complete episode of experiences.

        Args:
            experiences: List of (state, action, reward, next_state, done, mask, weight) tuples
            session_id: Session identifier
            final_success: Whether the episode ended in success
        """
        session_id = session_id or f"session_{int(time.time())}"

        for i, (state, action, reward, next_state, done, mask, weight) in enumerate(experiences):
            # Mark success for experiences that led to successful outcomes
            is_success = final_success and done
            self.add_experience(
                state_vector=state,
                action_index=action,
                reward=reward,
                next_state_vector=next_state,
                done=done,
                action_mask=mask,
                reflection_weight=weight,
                session_id=session_id,
                success=is_success,
            )

        self.metadata["sessions_contributed"].add(session_id)

    def sample_batch(
        self,
        batch_size: int,
        prioritize_success: bool = True,
        prioritize_reflection: bool = True,
        min_reward: float = None,
    ) -> List[StoredExperience]:
        """
        Sample a batch of experiences with optional prioritization.

        Args:
            batch_size: Number of experiences to sample
            prioritize_success: Weight successful experiences higher
            prioritize_reflection: Weight high reflection_weight experiences higher
            min_reward: Minimum reward threshold for sampling

        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return self.buffer

        # Filter by min_reward if specified
        candidates = self.buffer
        if min_reward is not None:
            candidates = [e for e in self.buffer if e.reward >= min_reward]

        if len(candidates) < batch_size:
            candidates = self.buffer

        # Calculate sampling weights
        weights = np.ones(len(candidates))

        if prioritize_reflection:
            reflection_weights = np.array([e.reflection_weight for e in candidates])
            weights *= reflection_weights

        if prioritize_success:
            success_weights = np.array([2.0 if e.success else 1.0 for e in candidates])
            weights *= success_weights

        # Add reward-based weight
        reward_weights = np.array([max(0.1, e.reward + 1) for e in candidates])
        weights *= reward_weights

        # Normalize
        weights = weights / weights.sum()

        # Sample
        indices = np.random.choice(
            len(candidates),
            size=min(batch_size, len(candidates)),
            replace=False,
            p=weights,
        )

        return [candidates[i] for i in indices]

    def get_successful_episodes(self, limit: int = 100) -> List[StoredExperience]:
        """Get experiences from successful episodes."""
        return [e for e in self.buffer if e.success][:limit]

    def get_failed_episodes(self, limit: int = 100) -> List[StoredExperience]:
        """Get experiences from failed episodes (high negative reward)."""
        return [e for e in self.buffer if not e.success and e.reward < -0.5][:limit]

    def get_statistics(self) -> dict:
        """Get experience store statistics."""
        if not self.buffer:
            return self.metadata["statistics"]

        rewards = [e.reward for e in self.buffer]
        return {
            "total_experiences": len(self.buffer),
            "sessions_contributed": len(self.metadata["sessions_contributed"]),
            "success_rate": self.metadata["statistics"]["success_count"] / max(1, len(self.buffer)),
            "avg_reward": np.mean(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "avg_reflection_weight": np.mean([e.reflection_weight for e in self.buffer]),
            "last_updated": self.metadata["last_updated"],
        }

    def save(self) -> None:
        """Save buffer and metadata to disk."""
        # Save buffer incrementally (append mode for new experiences)
        existing_timestamps = set()
        if os.path.exists(self.buffer_file):
            with open(self.buffer_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        existing_timestamps.add(data.get('timestamp'))
                    except:
                        pass

        # Append only new experiences
        with open(self.buffer_file, 'a') as f:
            for exp in self.buffer:
                if exp.timestamp not in existing_timestamps:
                    f.write(json.dumps(exp.to_dict()) + '\n')

        # Save metadata
        meta_to_save = self.metadata.copy()
        meta_to_save["sessions_contributed"] = list(meta_to_save["sessions_contributed"])
        meta_to_save["last_updated"] = datetime.now().isoformat()

        with open(self.metadata_file, 'w') as f:
            json.dump(meta_to_save, f, indent=2)

        logger.info(f"Saved {len(self.buffer)} experiences to {self.buffer_file}")

    def load(self) -> bool:
        """
        Load buffer and metadata from disk.

        Returns:
            True if loaded successfully, False if file missing or corrupted.
        """
        # Load buffer
        if os.path.exists(self.buffer_file):
            self.buffer = []
            loaded_count = 0
            failed_count = 0
            with open(self.buffer_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        self.buffer.append(StoredExperience.from_dict(data))
                        loaded_count += 1
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"Failed to load experience: {e}")

            if failed_count > 0:
                logger.warning(f"Loaded {loaded_count} experiences, {failed_count} failed")
            logger.info(f"Loaded {len(self.buffer)} experiences from {self.buffer_file}")

        # Load metadata
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    loaded_meta = json.load(f)
                    self.metadata.update(loaded_meta)
                    if isinstance(self.metadata.get("sessions_contributed"), list):
                        self.metadata["sessions_contributed"] = set(self.metadata["sessions_contributed"])
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                return False

        return True

    def clear(self) -> None:
        """Clear all experiences."""
        self.buffer = []
        self.metadata = {
            "total_experiences": 0,
            "sessions_contributed": set(),
            "last_updated": None,
            "statistics": {
                "total_reward": 0.0,
                "success_count": 0,
                "failure_count": 0,
                "avg_reward": 0.0,
            }
        }

        # Remove files
        if os.path.exists(self.buffer_file):
            os.remove(self.buffer_file)
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)

    def _update_metadata(self, experience: StoredExperience) -> None:
        """Update metadata with new experience."""
        self.metadata["total_experiences"] += 1
        self.metadata["last_updated"] = time.time()
        self.metadata["statistics"]["total_reward"] += experience.reward

        if experience.success:
            self.metadata["statistics"]["success_count"] += 1
        else:
            self.metadata["statistics"]["failure_count"] += 1

        total = self.metadata["statistics"]["success_count"] + self.metadata["statistics"]["failure_count"]
        if total > 0:
            self.metadata["statistics"]["avg_reward"] = (
                self.metadata["statistics"]["total_reward"] / total
            )

    def _prune_buffer(self) -> None:
        """Remove low-value experiences when over capacity."""
        if len(self.buffer) <= self.capacity:
            return

        # Sort by value (reward + reflection_weight)
        def value_score(e: StoredExperience) -> float:
            return e.reward + e.reflection_weight * 0.5 + (1.0 if e.success else 0.0)

        self.buffer.sort(key=value_score, reverse=True)
        removed_count = len(self.buffer) - self.capacity
        self.buffer = self.buffer[:self.capacity]

        logger.info(f"Pruned {removed_count} low-value experiences")


# Global instance for convenience
_global_store: Optional[PersistentExperienceStore] = None


def get_experience_store() -> PersistentExperienceStore:
    """Get or create global experience store."""
    global _global_store
    if _global_store is None:
        _global_store = PersistentExperienceStore()
        _global_store.load()
    return _global_store
