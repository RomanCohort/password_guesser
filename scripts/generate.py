"""
Password Generation Script

Generate password candidates for a target using the trained model.

Usage:
    python scripts/generate.py --config config.yaml --checkpoint best_model.pt --target "target_info.txt"
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from typing import List, Optional
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LLMInfoExtractor, MambaPasswordModel, MLPEncoder, create_mamba_model, create_mlp_encoder
from utils import PasswordTokenizer, FeatureVectorizer, TargetFeatures
from optimization import PasswordDEOptimizer, PasswordCandidate


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Password Candidates')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--target', type=str, help='Target info file')
    parser.add_argument('--text', type=str, help='Target info as text')
    parser.add_argument('--output', type=str, default='candidates.json', help='Output file')
    parser.add_argument('--n_candidates', type=int, default=100, help='Number of candidates')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=10, help='Top-k sampling')
    parser.add_argument('--use_de', action='store_true', help='Use differential evolution')
    parser.add_argument('--de_generations', type=int, default=50, help='DE generations')
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


class PasswordGenerator:
    """Main password generation pipeline"""

    def __init__(
        self,
        model: MambaPasswordModel,
        mlp_encoder: MLPEncoder,
        llm_extractor: LLMInfoExtractor,
        tokenizer: PasswordTokenizer,
        vectorizer: FeatureVectorizer,
        device: torch.device
    ):
        self.model = model.to(device)
        self.mlp_encoder = mlp_encoder.to(device)
        self.llm_extractor = llm_extractor
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.device = device

    def extract_features(self, target_info: str) -> TargetFeatures:
        """Extract features from target info using LLM"""
        print("Extracting features with LLM...")
        features = self.llm_extractor.extract(target_info)
        print(f"Extracted features: {features}")
        return features

    def encode_features(self, features: TargetFeatures) -> torch.Tensor:
        """Encode features to latent vector"""
        feature_vec = self.vectorizer.vectorize(features)
        feature_tensor = torch.tensor(feature_vec, dtype=torch.float32).unsqueeze(0)
        feature_tensor = feature_tensor.to(self.device)

        with torch.no_grad():
            latent = self.mlp_encoder(feature_tensor)

        return latent

    def generate_candidates(
        self,
        latent: torch.Tensor,
        n_samples: int = 100,
        temperature: float = 1.0,
        top_k: int = 10
    ) -> List[str]:
        """Generate password candidates using the model"""
        print(f"Generating {n_samples} candidates...")
        passwords = self.model.generate_batch(
            latent,
            self.tokenizer,
            n_samples=n_samples,
            temperature=temperature,
            top_k=top_k
        )
        return list(set(passwords))  # Remove duplicates

    def generate_with_de(
        self,
        latent: torch.Tensor,
        n_candidates: int = 100,
        generations: int = 50,
        seed_passwords: Optional[List[str]] = None
    ) -> List[PasswordCandidate]:
        """Generate candidates using differential evolution"""
        print(f"Running DE optimization for {generations} generations...")

        def fitness_fn(password: str) -> float:
            """Fitness function based on model probability"""
            return self.model.score_password(password, latent, self.tokenizer)

        optimizer = PasswordDEOptimizer(
            fitness_fn=fitness_fn,
            max_length=self.model.max_length,
            population_size=100,
            max_generations=generations
        )

        optimizer.initialize_population(seeds=seed_passwords)
        candidates = optimizer.run(max_generations=generations, verbose=True)

        return candidates[:n_candidates]

    def run_pipeline(
        self,
        target_info: str,
        n_candidates: int = 100,
        temperature: float = 1.0,
        top_k: int = 10,
        use_de: bool = False,
        de_generations: int = 50
    ) -> dict:
        """Run the complete generation pipeline"""
        # Step 1: Extract features
        features = self.extract_features(target_info)

        # Step 2: Encode to latent
        latent = self.encode_features(features)

        # Step 3: Get seed passwords from components
        components = self.vectorizer.generate_password_components(features)
        seed_passwords = []

        # Generate simple combinations
        for name in components['names'][:5]:
            for date in components['dates'][:3]:
                seed_passwords.append(f"{name}{date}")
                seed_passwords.append(f"{name.capitalize()}{date}")
                seed_passwords.append(f"{name}{date}!")
            for num in components['numbers'][:3]:
                seed_passwords.append(f"{name}{num}")

        for word in components['words'][:5]:
            for date in components['dates'][:2]:
                seed_passwords.append(f"{word}{date}")

        # Step 4: Generate candidates
        if use_de:
            # Use differential evolution
            de_candidates = self.generate_with_de(
                latent,
                n_candidates=n_candidates,
                generations=de_generations,
                seed_passwords=seed_passwords
            )

            # Also generate some direct samples
            direct_passwords = self.generate_candidates(
                latent,
                n_samples=n_candidates // 2,
                temperature=temperature,
                top_k=top_k
            )

            # Combine and rank
            all_candidates = {}
            for c in de_candidates:
                all_candidates[c.password] = c.score

            # Score direct passwords
            for pwd in direct_passwords:
                if pwd not in all_candidates:
                    all_candidates[pwd] = self.model.score_password(pwd, latent, self.tokenizer)

            # Sort by score
            sorted_candidates = sorted(
                [{'password': k, 'score': v} for k, v in all_candidates.items()],
                key=lambda x: x['score'],
                reverse=True
            )[:n_candidates]

        else:
            # Direct generation only
            passwords = self.generate_candidates(
                latent,
                n_samples=n_candidates,
                temperature=temperature,
                top_k=top_k
            )

            # Score them
            scored = []
            for pwd in passwords:
                score = self.model.score_password(pwd, latent, self.tokenizer)
                scored.append({'password': pwd, 'score': score})

            sorted_candidates = sorted(scored, key=lambda x: x['score'], reverse=True)

        return {
            'target_features': {
                'full_name': features.full_name,
                'first_name': features.first_name,
                'last_name': features.last_name,
                'nickname': features.nickname,
                'birthday': features.birthday,
                'phone': features.phone,
                'hobbies': features.hobbies,
                'favorite_numbers': features.favorite_numbers,
                'keywords': features.keywords,
            },
            'candidates': sorted_candidates,
            'seed_passwords': seed_passwords[:20],  # Include some seeds for reference
        }


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load models
    print("Loading models...")

    model = create_mamba_model(config)
    mlp_encoder = create_mlp_encoder(config)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    mlp_encoder.load_state_dict(checkpoint['mlp_state_dict'])

    model.eval()
    mlp_encoder.eval()

    # Create components
    llm_extractor = LLMInfoExtractor(config_path=args.config)
    tokenizer = PasswordTokenizer()
    vectorizer = FeatureVectorizer()

    # Create generator
    generator = PasswordGenerator(
        model=model,
        mlp_encoder=mlp_encoder,
        llm_extractor=llm_extractor,
        tokenizer=tokenizer,
        vectorizer=vectorizer,
        device=device
    )

    # Get target info
    if args.target:
        with open(args.target, 'r', encoding='utf-8') as f:
            target_info = f.read()
    elif args.text:
        target_info = args.text
    else:
        # Interactive input
        print("Enter target information (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)
        target_info = "\n".join(lines)

    # Run pipeline
    result = generator.run_pipeline(
        target_info=target_info,
        n_candidates=args.n_candidates,
        temperature=args.temperature,
        top_k=args.top_k,
        use_de=args.use_de,
        de_generations=args.de_generations
    )

    # Output results
    print("\n" + "=" * 60)
    print("TOP PASSWORD CANDIDATES")
    print("=" * 60)

    for i, candidate in enumerate(result['candidates'][:20], 1):
        print(f"{i:3d}. {candidate['password']:<30} (score: {candidate['score']:.4f})")

    # Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
