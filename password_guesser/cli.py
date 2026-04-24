"""
Password Guesser - CLI Entry Point

Usage:
    password-guesser train --config config.yaml --data passwords.txt --amp
    password-guesser generate --checkpoint best_model.pt --target "目标信息"
    password-guesser web --port 8000
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_train(args):
    """Run training"""
    from train import main as train_main
    sys.argv = [
        "train.py",
        "--config", args.config,
        "--data", args.data,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--output", args.output,
        "--warmup_steps", str(args.warmup_steps),
        "--gradient_accumulation_steps", str(args.grad_accum),
    ]
    if args.amp:
        sys.argv.append("--amp")
    if args.scheduler:
        sys.argv.extend(["--scheduler", args.scheduler])
    if args.gradient_checkpointing:
        sys.argv.append("--gradient_checkpointing")
    if args.early_stopping:
        sys.argv.extend(["--early_stopping_patience", str(args.early_stopping)])
    if args.resume:
        sys.argv.extend(["--resume", args.resume])

    train_main()


def cmd_generate(args):
    """Run password generation"""
    import torch
    import yaml
    from models import MambaPasswordModel, MambaConfig, MLPEncoder, LLMInfoExtractor
    from utils import PasswordTokenizer
    from utils.feature_utils import TargetFeatures

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = MambaConfig()
    model = MambaPasswordModel(model_config)
    mlp_encoder = MLPEncoder(input_dim=64, hidden_dims=[128, 128], output_dim=128)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    mlp_encoder.load_state_dict(checkpoint['mlp_state_dict'])
    model = model.to(device).eval()
    mlp_encoder = mlp_encoder.to(device).eval()

    tokenizer = PasswordTokenizer()

    # Read target info
    if args.target_file:
        with open(args.target_file, 'r', encoding='utf-8') as f:
            target_text = f.read()
    else:
        target_text = args.text or ""

    # Extract features
    features = None
    if target_text and config.get('llm', {}).get('api_key'):
        print("Extracting features with LLM...")
        extractor = LLMInfoExtractor(config_path=args.config)
        features = extractor.extract_multistage(target_text, stages=3, verbose=True)

    # Generate passwords
    print(f"\nGenerating {args.n_samples} passwords using {args.method}...")
    latent = torch.randn(1, 64, device=device)

    with torch.no_grad():
        if args.method == "beam":
            results = model.generate_beam_search(latent, tokenizer, beam_width=args.beam_width)
            for pwd, score in results[:args.n_samples]:
                print(f"  {pwd}  (score: {score:.4f})")
        elif args.method == "diverse_beam":
            results = model.generate_diverse_beam(latent, tokenizer, num_groups=3)
            for pwd, score in results[:args.n_samples]:
                print(f"  {pwd}  (score: {score:.4f})")
        elif args.method == "typical":
            for _ in range(args.n_samples):
                pwd = model.generate_typical(latent, tokenizer)
                print(f"  {pwd}")
        elif args.method == "contrastive":
            for _ in range(args.n_samples):
                pwd = model.generate_contrastive(latent, tokenizer)
                print(f"  {pwd}")
        else:
            for _ in range(args.n_samples):
                pwd = model.generate(latent, tokenizer, temperature=args.temperature)
                print(f"  {pwd}")

    print(f"\nDone! Generated {args.n_samples} passwords.")


def cmd_web(args):
    """Start web server"""
    import uvicorn
    from web.app import app

    print(f"Starting web server on http://0.0.0.0:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def cmd_pentest(args):
    """Run penetration test"""
    import json
    from pentest.orchestrator import PenTestOrchestrator, PenTestConfig
    from models.llm_provider import LLMConfig

    # Load targets
    if args.target_file:
        with open(args.target_file, 'r', encoding='utf-8') as f:
            targets = json.load(f)
    elif args.targets:
        targets = json.loads(args.targets)
    else:
        print("Error: Please provide --targets or --target_file")
        return

    # Create config
    config = PenTestConfig(
        max_steps=args.max_steps,
        auto_mode=not args.interactive,
        reflection_frequency=args.reflection_freq,
        enable_attack_team=getattr(args, 'team', False),
        enable_self_improvement=not getattr(args, 'no_self_improvement', False),
    )

    # Add LLM config if provided
    if args.llm_api_key:
        config.llm_config = LLMConfig(
            provider=args.llm_provider,
            api_key=args.llm_api_key,
            api_base=args.llm_api_base,
            model=args.llm_model,
        )

    # Create orchestrator
    orch = PenTestOrchestrator(config)

    # Initialize from targets
    orch.initialize_from_scan({
        "format": "manual",
        "data": targets if isinstance(targets, list) else [targets],
    })

    print(f"\nStarting penetration test...")
    print(f"  Targets: {len(targets)}")
    print(f"  Goal: {args.goal}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Mode: {'Interactive' if args.interactive else 'Autonomous'}")
    print()

    if args.interactive:
        orch.run_interactive()
    elif getattr(args, 'team', False):
        print("  Mode: Team-based")
        results = orch.run_team_based(
            target_goal=args.goal,
            max_steps=args.max_steps,
            verbose=True,
        )
    else:
        results = orch.run_autonomous(
            target_goal=args.goal,
            max_steps=args.max_steps,
            verbose=True,
        )

        print(f"\n{'='*60}")
        print("PENETRATION TEST COMPLETE")
        print(f"{'='*60}")
        print(results["summary"])

        # Save report
        if args.output:
            from pentest.report import PenTestReport, PenTestSession
            session = PenTestSession(
                target_goal=results.get("goal", ""),
                total_steps=results.get("total_steps", 0),
                total_reward=results.get("total_reward", 0),
                duration=results.get("duration", 0),
                state=results.get("state", {}),
                steps=results.get("steps", []),
                attack_graph=results.get("attack_graph", {}),
                knowledge_stats=results.get("knowledge_stats", {}),
                reflections_count=results.get("reflection_count", 0),
            )
            report = PenTestReport()

            output_path = args.output
            if output_path.endswith('.json'):
                with open(output_path, 'w') as f:
                    json.dump(report.generate_json(session), f, indent=2)
            elif output_path.endswith('.md'):
                with open(output_path, 'w') as f:
                    f.write(report.generate_markdown(session))
            else:
                with open(output_path + '.json', 'w') as f:
                    json.dump(report.generate_json(session), f, indent=2)

            print(f"Report saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog="password-guesser",
        description="AI-powered targeted password guessing system"
    )

    # Global options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output (DEBUG level)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output (WARNING level)")
    parser.add_argument("--config", default="config.yaml", help="Global config file")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", default="config.yaml", help="Config file")
    train_parser.add_argument("--data", required=True, help="Password data file")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--output", default="checkpoints", help="Output directory")
    train_parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    train_parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    train_parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation")
    train_parser.add_argument("--scheduler", choices=["cosine", "onecycle"], default="cosine")
    train_parser.add_argument("--gradient_checkpointing", action="store_true")
    train_parser.add_argument("--early_stopping", type=int, default=10, help="Early stopping patience")
    train_parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    train_parser.set_defaults(func=cmd_train)

    # --- generate ---
    gen_parser = subparsers.add_parser("generate", help="Generate password candidates")
    gen_parser.add_argument("--config", default="config.yaml", help="Config file")
    gen_parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    gen_parser.add_argument("--text", default=None, help="Target information text")
    gen_parser.add_argument("--target_file", default=None, help="Target information file")
    gen_parser.add_argument("--method", default="sampling",
                           choices=["sampling", "beam", "diverse_beam", "typical", "contrastive"])
    gen_parser.add_argument("--n_samples", type=int, default=20, help="Number of samples")
    gen_parser.add_argument("--temperature", type=float, default=1.0)
    gen_parser.add_argument("--beam_width", type=int, default=5)
    gen_parser.set_defaults(func=cmd_generate)

    # --- web ---
    web_parser = subparsers.add_parser("web", help="Start web interface")
    web_parser.add_argument("--host", default="0.0.0.0", help="Host")
    web_parser.add_argument("--port", type=int, default=8000, help="Port")
    web_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    web_parser.set_defaults(func=cmd_web)

    # --- pentest ---
    pt_parser = subparsers.add_parser("pentest", help="Run penetration test")
    pt_parser.add_argument("--targets", default=None, help="Target JSON string")
    pt_parser.add_argument("--target_file", default=None, help="Target JSON file")
    pt_parser.add_argument("--goal", default="full_compromise",
                           choices=["full_compromise", "get_shell", "escalate_priv", "exfiltrate_data"])
    pt_parser.add_argument("--max_steps", type=int, default=50)
    pt_parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    pt_parser.add_argument("--reflection_freq", type=int, default=5)
    pt_parser.add_argument("--output", default=None, help="Save report to file")
    pt_parser.add_argument("--llm_provider", default="deepseek")
    pt_parser.add_argument("--llm_api_key", default=None)
    pt_parser.add_argument("--llm_api_base", default="https://api.deepseek.com/v1")
    pt_parser.add_argument("--llm_model", default="deepseek-chat")
    pt_parser.add_argument("--team", action="store_true", help="Use team-based mode")
    pt_parser.add_argument("--no_self_improvement", action="store_true", help="Disable self-improvement")
    pt_parser.set_defaults(func=cmd_pentest)

    # --- status ---
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.set_defaults(func=cmd_status)

    # Parse and run
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    # Initialize logging
    _init_logging(args)

    args.func(args)


def _init_logging(args):
    """Initialize logging based on CLI flags."""
    try:
        from utils.logging import setup_logging
        if getattr(args, 'verbose', False):
            level = "DEBUG"
        elif getattr(args, 'quiet', False):
            level = "WARNING"
        else:
            level = "INFO"
        setup_logging(level=level, json_format=False)
    except ImportError:
        import logging
        if getattr(args, 'verbose', False):
            logging.basicConfig(level=logging.DEBUG)
        elif getattr(args, 'quiet', False):
            logging.basicConfig(level=logging.WARNING)
        else:
            logging.basicConfig(level=logging.INFO)


def cmd_status(args):
    """Show system status."""
    from pentest.orchestrator import PenTestOrchestrator, PenTestConfig

    config = PenTestConfig(enable_attack_team=True, enable_self_improvement=True)
    orch = PenTestOrchestrator(config)

    status = orch.get_system_status()

    print("\n" + "=" * 60)
    print("SYSTEM STATUS")
    print("=" * 60)

    print(f"\nSession: {status.get('session_id', 'N/A')}")
    print(f"Agent: {'Initialized' if status.get('agent_initialized') else 'Not initialized'}")
    print(f"Environment: {'Initialized' if status.get('environment_initialized') else 'Not initialized'}")
    print(f"LLM Planner: {'Configured' if status.get('llm_planner') else 'Not configured'}")

    components = status.get("components", {})

    print("\n--- Components ---")
    rag = components.get("rag", {})
    print(f"RAG System: {'Active' if rag.get('initialized') else 'Inactive'}")
    print(f"Vector Store: {'Active' if rag.get('vector_store') else 'Inactive'}")

    experts = components.get("experts", {})
    print(f"Expert Router: {'Active' if experts.get('initialized') else 'Inactive'}")
    print(f"Registered Experts: {experts.get('registered', 0)}")

    tools = components.get("tools", {})
    print(f"Tool Orchestrator: {'Active' if tools.get('initialized') else 'Inactive'}")
    print(f"Registered Tools: {tools.get('registered', 0)}")
    print(f"Tool Chains: {tools.get('chains', 0)}")

    team = components.get("attack_team", {})
    print(f"Attack Team: {'Active' if team.get('enabled') else 'Inactive'}")
    print(f"Team Members: {team.get('members', 0)}")

    si = components.get("self_improvement", {})
    print(f"\nSelf-Improvement: {'Active' if si.get('enabled') else 'Inactive'}")
    print(f"Experience Store: {'Active' if si.get('experience_store') else 'Inactive'} ({si.get('experiences', 0)} experiences)")
    print(f"Lessons DB: {'Active' if si.get('lessons_db') else 'Inactive'} ({si.get('lessons', 0)} lessons)")
    print(f"Curriculum: {'Active' if si.get('curriculum') else 'Inactive'}")
    print(f"Meta Learner: {'Active' if si.get('meta_learner') else 'Inactive'}")

    knowledge = status.get("knowledge", {})
    print(f"\nKnowledge: {knowledge.get('cve_count', 0)} CVEs, {knowledge.get('technique_count', 0)} techniques")
    print()


if __name__ == "__main__":
    main()
