"""
Quick test to verify all modules work correctly.
Run without training data to check basic functionality.
"""

import sys
import os
import torch
import numpy as np

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.feature_utils import TargetFeatures, FeatureVectorizer, FeatureExtractor
from utils.password_utils import PasswordTokenizer, PasswordDecoder, PasswordPatternAnalyzer
from models.mlp_encoder import MLPEncoder, ConditionalMLPEncoder
from models.mamba_password import MambaPasswordModel, MambaConfig
from optimization.differential_evolution import PasswordDEOptimizer


def test_feature_utils():
    print("\n[1/6] Testing feature_utils...")

    # Test FeatureExtractor
    text = "张三，1995年3月15日出生，手机号13812345678，邮箱zhangsan@gmail.com"
    date = FeatureExtractor.extract_date(text)
    phone = FeatureExtractor.extract_phone(text)
    email = FeatureExtractor.extract_email_prefix(text)
    print(f"  Date: {date}")  # Expected: 19950315
    print(f"  Phone: {phone}")  # Expected: 13812345678
    print(f"  Email prefix: {email}")  # Expected: zhangsan

    # Test FeatureVectorizer
    features = TargetFeatures(
        full_name="ZhangSan",
        first_name="San",
        last_name="Zhang",
        birthday="19950315",
        phone="13812345678",
        hobbies=["basketball", "gaming"],
        favorite_numbers=["7", "88"],
        keywords=["dragon", "blue"]
    )

    vectorizer = FeatureVectorizer(vector_dim=256)
    vec = vectorizer.vectorize(features)
    print(f"  Feature vector shape: {vec.shape}")
    assert vec.shape == (256,), f"Expected (256,), got {vec.shape}"

    # Test password component generation
    components = vectorizer.generate_password_components(features)
    print(f"  Generated components: {list(components.keys())}")
    print(f"  Name variants: {components['names'][:3]}")
    print(f"  Date variants: {components['dates'][:3]}")

    print("  PASSED")


def test_password_utils():
    print("\n[2/6] Testing password_utils...")

    # Test tokenizer
    tokenizer = PasswordTokenizer()
    password = "Hello123!"

    tokens = tokenizer.encode(password)
    decoded = tokenizer.decode(tokens)
    print(f"  Original: {password}")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: {decoded}")
    assert decoded == password, f"Round-trip failed: {decoded} != {password}"

    # Test batch encoding
    passwords = ["abc", "Hello123!", "test@#$"]
    batch = tokenizer.encode_batch(passwords)
    print(f"  Batch shape: {batch.shape}")

    # Test pattern analysis
    analyzer = PasswordPatternAnalyzer()
    for pwd in ["john1995", "Hello123!", "P@ssw0rd"]:
        structure = analyzer.analyze_structure(pwd)
        print(f"  {pwd}: pattern={structure['pattern']}, "
              f"lower={structure['has_lower']}, upper={structure['has_upper']}, "
              f"digit={structure['has_digit']}, special={structure['has_special']}")

    print("  PASSED")


def test_mlp_encoder():
    print("\n[3/6] Testing MLP encoder...")

    device = torch.device('cpu')

    # Basic MLP
    mlp = MLPEncoder(
        input_dim=256,
        hidden_dims=[128, 64],
        output_dim=64
    ).to(device)

    x = torch.randn(4, 256)  # Batch of 4
    out = mlp(x)
    print(f"  Basic MLP: input {x.shape} -> output {out.shape}")
    assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"

    # Conditional MLP
    cond_mlp = ConditionalMLPEncoder(
        input_dim=256,
        hidden_dims=[128, 64],
        output_dim=64
    ).to(device)

    result = cond_mlp(x)
    print(f"  Conditional MLP: latent {result['latent'].shape}, "
          f"length_logits {result['length_logits'].shape}, "
          f"pattern_logits {result['pattern_logits'].shape}")

    # Test length prediction
    length = cond_mlp.predict_length(x[0:1])
    pattern = cond_mlp.predict_pattern(x[0:1])
    print(f"  Predicted length: {length.item()}, pattern: {pattern}")

    print("  PASSED")


def test_mamba_model():
    print("\n[4/6] Testing MAMBA model...")

    device = torch.device('cpu')

    # Create small model for testing
    config = MambaConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        d_state=8,
        d_conv=4,
        max_length=16
    )

    model = MambaPasswordModel(config).to(device)
    tokenizer = PasswordTokenizer()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Forward pass
    input_ids = torch.tensor([tokenizer.encode("Hello123", max_length=16)]).to(device)
    latent = torch.randn(1, 64).to(device)

    logits = model(input_ids, latent)
    print(f"  Forward: input_ids {input_ids.shape} -> logits {logits.shape}")
    assert logits.shape == (1, 16, 128), f"Expected (1, 16, 128), got {logits.shape}"

    # Compute loss
    loss = model.compute_loss(input_ids, latent, input_ids)
    print(f"  Loss: {loss.item():.4f}")

    # Generate
    password = model.generate(latent, tokenizer, max_length=12, temperature=0.8)
    print(f"  Generated password: {password}")

    # Score a password
    score = model.score_password("test123", latent, tokenizer)
    print(f"  Score for 'test123': {score:.4f}")

    print("  PASSED")


def test_differential_evolution():
    print("\n[5/6] Testing differential evolution...")

    # Simple fitness function: prefer passwords with mixed characters
    def fitness(password: str) -> float:
        score = 0.0
        if any(c.islower() for c in password):
            score += 1.0
        if any(c.isdigit() for c in password):
            score += 1.0
        if len(password) >= 6:
            score += 1.0
        return score

    optimizer = PasswordDEOptimizer(
        fitness_fn=fitness,
        max_length=12,
        population_size=50
    )

    # Initialize with seeds
    seeds = ["hello", "test123", "password"]
    optimizer.initialize_population(seeds=seeds)

    # Run for a few generations
    candidates = optimizer.run(max_generations=10, verbose=False)

    print(f"  Best fitness: {candidates[0].score:.4f}")
    print(f"  Best password: {candidates[0].password}")
    print(f"  Top 5 candidates:")
    for i, c in enumerate(candidates[:5], 1):
        print(f"    {i}. {c.password} (score: {c.score:.4f})")

    print("  PASSED")


def test_pipeline():
    print("\n[6/10] Testing full pipeline (without LLM)...")

    device = torch.device('cpu')

    # Create components
    config = MambaConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        d_state=8,
        d_conv=4,
        max_length=16
    )

    model = MambaPasswordModel(config).to(device)
    mlp = MLPEncoder(input_dim=256, hidden_dims=[128, 64], output_dim=64).to(device)
    tokenizer = PasswordTokenizer()
    vectorizer = FeatureVectorizer(vector_dim=256)

    model.eval()
    mlp.eval()

    # Create target features
    features = TargetFeatures(
        full_name="JohnDoe",
        first_name="John",
        last_name="Doe",
        birthday="19950315",
        hobbies=["gaming"],
        favorite_numbers=["42"],
    )

    # Vectorize
    feature_vec = vectorizer.vectorize(features)
    feature_tensor = torch.tensor(feature_vec, dtype=torch.float32).unsqueeze(0)

    # Encode
    with torch.no_grad():
        latent = mlp(feature_tensor)

    # Generate candidates
    passwords = model.generate_batch(
        latent,
        tokenizer,
        n_samples=5,
        temperature=0.8,
        top_k=10
    )

    print(f"  Generated {len(passwords)} candidates:")
    for i, pwd in enumerate(passwords, 1):
        score = model.score_password(pwd, latent, tokenizer)
        print(f"    {i}. {pwd} (score: {score:.4f})")

    print("  PASSED")


def test_rules_engine():
    print("\n[7/10] Testing rules engine...")

    from rules.engine import PasswordRuleEngine

    engine = PasswordRuleEngine()

    # Test single rule
    result = engine.apply_rule("password", "lowercase")
    print(f"  lowercase('password') -> {result.transformed}")
    assert result.transformed == "password"

    result = engine.apply_rule("password", "uppercase")
    print(f"  uppercase('password') -> {result.transformed}")
    assert result.transformed == "PASSWORD"

    result = engine.apply_rule("password", "reverse")
    print(f"  reverse('password') -> {result.transformed}")
    assert result.transformed == "drowssap"

    # Test variants generation
    variants = engine.generate_variants("hello", max_variants=10)
    print(f"  Generated {len(variants)} variants for 'hello'")
    assert len(variants) > 0

    print("  PASSED")


def test_password_strength():
    print("\n[8/10] Testing password strength evaluation...")

    from evaluation.strength import PasswordStrengthEvaluator
    from evaluation.entropy import EntropyCalculator

    evaluator = PasswordStrengthEvaluator()
    entropy_calc = EntropyCalculator()

    passwords = ["password", "P@ssw0rd!", "Xk9#mP2$vL"]

    for pwd in passwords:
        report = evaluator.evaluate(pwd)
        entropy = entropy_calc.charset_entropy(pwd)
        print(f"  '{pwd}': score={report.score.name}, entropy={entropy:.1f} bits")

    print("  PASSED")


def test_pcfg():
    print("\n[9/10] Testing PCFG generator...")

    from pcfg.pcfg import PCFGGenerator
    from pcfg.training import PCFGTrainer

    # Create trainer
    trainer = PCFGTrainer()

    # Extract structure
    structure = trainer.extract_structure("hello123!")
    print(f"  Structure of 'hello123!': {structure}")
    assert structure == "L5D3S1"

    # Train on small sample
    passwords = ["password", "password123", "letmein", "qwerty", "hello", "hello1"]
    model = trainer.train(passwords)

    # Generate
    samples = model.generate(n=5)
    print(f"  Generated {len(samples)} samples")
    for pwd, prob in samples:
        print(f"    '{pwd}' (prob: {prob:.4f})")

    print("  PASSED")


def test_data_pipeline():
    print("\n[10/10] Testing data pipeline...")

    from data.pipeline import DataPipeline
    from data.augmentation import PasswordAugmentor

    # Test augmentor
    augmentor = PasswordAugmentor()
    variants = augmentor.augment("password")
    print(f"  Generated {len(variants)} augmented variants for 'password'")

    # Test pipeline
    pipeline = DataPipeline()
    pipeline.deduplicate().filter_by_length(min_length=4, max_length=16)

    passwords = ["password", "123456", "password", "qwerty", "   ", "a" * 100]
    result = pipeline.run(passwords)
    print(f"  Pipeline: {len(passwords)} -> {len(result)} passwords")
    assert len(result) < len(passwords)  # Should have removed duplicates and invalid

    print("  PASSED")


if __name__ == '__main__':
    print("=" * 60)
    print("Password Guessing System - Module Tests")
    print("=" * 60)

    try:
        test_feature_utils()
        test_password_utils()
        test_mlp_encoder()
        test_mamba_model()
        test_differential_evolution()
        test_pipeline()
        test_rules_engine()
        test_password_strength()
        test_pcfg()
        test_data_pipeline()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
