"""
FastAPI Web Server for Password Guesser

Provides REST API and web interface for the password guessing system.

Usage:
    uvicorn web.app:app --reload --port 8000
"""

import os
import sys
import json
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
import torch
import time as _time

from models import MambaPasswordModel, MambaConfig, MLPEncoder
from models import LLMInfoExtractor, LLMConfig
from optimization import EvaluationCache, CheckpointManager
from utils import PasswordTokenizer
from utils.feature_utils import TargetFeatures

# New imports
from web.auth import APIKeyAuth, JWTAuth
from web.rate_limit import InMemoryRateLimiter, RateLimitMiddleware
from web.tasks import TaskManager, TaskStatus
from web.websocket import websocket_endpoint, manager as ws_manager


# ============== Pydantic Models ==============

class TargetInfoRequest(BaseModel):
    """Request model for target information"""
    raw_text: str = Field(..., description="Raw text with target's personal information")
    use_llm_extraction: bool = Field(True, description="Use LLM for structured extraction")
    extraction_stages: int = Field(3, description="Number of extraction stages (1-3)")
    parallel_extraction: bool = Field(False, description="Use parallel extraction")


class GenerationConfig(BaseModel):
    """Configuration for password generation"""
    method: str = Field("sampling", description="Generation method: sampling, beam, diverse_beam, typical, contrastive")
    n_samples: int = Field(20, description="Number of passwords to generate")
    temperature: float = Field(1.0, description="Sampling temperature")
    temperature_schedule: str = Field("constant", description="Temperature schedule: constant, linear_decay, cosine")
    top_k: int = Field(0, description="Top-k filtering (0 to disable)")
    top_p: float = Field(0.9, description="Nucleus sampling threshold")
    beam_width: int = Field(5, description="Beam width for beam search")
    diversity_penalty: float = Field(0.5, description="Diversity penalty for diverse beam")
    typical_mass: float = Field(0.9, description="Typical mass for typical sampling")
    contrastive_alpha: float = Field(0.5, description="Alpha for contrastive search")


class GenerateRequest(BaseModel):
    """Request model for password generation"""
    target_info: TargetInfoRequest
    generation: GenerationConfig


class PasswordResult(BaseModel):
    """Result model for generated password"""
    password: str
    score: Optional[float] = None
    method: str


class GenerateResponse(BaseModel):
    """Response model for password generation"""
    passwords: List[PasswordResult]
    extracted_features: Dict[str, Any]
    generation_time: float
    extraction_time: float


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    parameters: int
    is_loaded: bool
    checkpoint: Optional[str] = None


# ============== Global State ==============

class AppState:
    """Application state"""
    def __init__(self):
        self.model: Optional[MambaPasswordModel] = None
        self.mlp_encoder: Optional[MLPEncoder] = None
        self.tokenizer: Optional[PasswordTokenizer] = None
        self.llm_extractor: Optional[LLMInfoExtractor] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path: Optional[str] = None
        self.cache = EvaluationCache(max_size=100000)
        self.model_config = MambaConfig()
        # New: Task manager and auth
        self.task_manager = TaskManager(max_concurrent=4)
        self.auth = APIKeyAuth()
        self.rate_limiter = InMemoryRateLimiter(requests_per_minute=60)

    def is_loaded(self) -> bool:
        return self.model is not None


state = AppState()


# ============== FastAPI App ==============

app = FastAPI(
    title="Password Guesser & Penetration Testing Framework",
    description="AI-powered targeted password guessing and knowledge-enhanced automated penetration testing",
    version="2.0.0"
)

# Rate limiting middleware (disabled for now - has compatibility issue)
# app.add_middleware(
#     RateLimitMiddleware,
#     rate_limiter=state.rate_limiter,
# )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include penetration testing router
try:
    from web.pentest_api import router as pentest_router
    app.include_router(pentest_router)
except ImportError as e:
    print(f"Warning: Could not load pentest router: {e}")

# Include attack demo router
try:
    from web.attack_demo_api import router as attack_demo_router
    app.include_router(attack_demo_router)
except ImportError as e:
    print(f"Warning: Could not load attack demo router: {e}")


# ============== Attack Demo Pages ==============

@app.get("/attack-demo")
async def attack_demo_page():
    """Serve the attack demo HTML page."""
    from fastapi.responses import FileResponse
    html_path = os.path.join(os.path.dirname(__file__), "static", "attack_demo.html")
    return FileResponse(html_path)


@app.get("/api/attack-demo")
async def attack_demo_api_root():
    """API root for attack demo."""
    return {"message": "Attack Demo API", "endpoints": ["/init", "/experts", "/team", "/route", "/meeting", "/attack-step", "/simulate-full-attack"]}


# ============== Helper Functions ==============

def default_features() -> TargetFeatures:
    """Create default features when LLM is not available"""
    return TargetFeatures(
        full_name="",
        first_name="",
        last_name="",
        nickname="",
        birthday="",
        birth_year="",
        birth_month="",
        birth_day="",
        anniversary="",
        phone="",
        email_prefix="",
        hobbies=[],
        favorite_words=[],
        favorite_numbers=[],
        sports_teams=[],
        pet_names=[],
        city="",
        country="",
        keywords=[]
    )


def features_to_dict(features: TargetFeatures) -> Dict[str, Any]:
    """Convert TargetFeatures to dictionary"""
    return {
        "full_name": features.full_name,
        "first_name": features.first_name,
        "last_name": features.last_name,
        "nickname": features.nickname,
        "birthday": features.birthday,
        "phone": features.phone,
        "email_prefix": features.email_prefix,
        "hobbies": features.hobbies,
        "favorite_words": features.favorite_words,
        "favorite_numbers": features.favorite_numbers,
        "pet_names": features.pet_names,
        "city": features.city,
        "country": features.country,
        "keywords": features.keywords
    }


# ============== Health & Monitoring Endpoints ==============

@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes/load balancers."""
    return {"status": "healthy", "timestamp": _time.time()}


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    checks = {
        "model_loaded": state.is_loaded(),
        "device": state.device,
    }
    return {"ready": True, "checks": checks}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    try:
        from prometheus_client import generate_latest
        return Response(content=generate_latest(), media_type="text/plain")
    except ImportError:
        # prometheus_client not installed, return basic stats
        return Response(
            content=(
                f"# HELP app_info Application info\n"
                f"# TYPE app_info gauge\n"
                f'app_info{{version="2.0.0"}} 1\n'
                f"# HELP model_loaded Whether model is loaded\n"
                f"# TYPE model_loaded gauge\n"
                f"model_loaded {1 if state.is_loaded() else 0}\n"
            ),
            media_type="text/plain",
        )


# ============== API Routes ==============

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page"""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Password Guesser</h1><p>Please create templates/index.html</p>"


@app.get("/pentest", response_class=HTMLResponse)
async def pentest_page():
    """Serve the penetration testing dashboard"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "pentest.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>PenTest Framework</h1><p>pentest.html not found</p>"


@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "model_loaded": state.is_loaded(),
        "device": state.device,
        "cuda_available": torch.cuda.is_available(),
        "cache_stats": state.cache.stats(),
        "checkpoint": state.checkpoint_path
    }


@app.post("/api/load_model")
async def load_model(checkpoint_path: Optional[str] = None):
    """Load model from checkpoint"""
    try:
        # Initialize model
        state.model = MambaPasswordModel(state.model_config)
        state.mlp_encoder = MLPEncoder(input_dim=64, hidden_dims=[128, 128], output_dim=128)
        state.tokenizer = PasswordTokenizer()

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=state.device)
            state.model.load_state_dict(checkpoint['model_state_dict'])
            state.mlp_encoder.load_state_dict(checkpoint['mlp_state_dict'])
            state.checkpoint_path = checkpoint_path
        else:
            state.checkpoint_path = None

        state.model = state.model.to(state.device)
        state.mlp_encoder = state.mlp_encoder.to(state.device)
        state.model.eval()
        state.mlp_encoder.eval()

        params = sum(p.numel() for p in state.model.parameters())
        params += sum(p.numel() for p in state.mlp_encoder.parameters())

        return {
            "success": True,
            "message": f"Model loaded on {state.device}",
            "parameters": params
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/configure_llm")
async def configure_llm(api_key: str, api_base: str = "https://api.deepseek.com/v1"):
    """Configure LLM extractor"""
    try:
        config = LLMConfig(
            api_key=api_key,
            api_base=api_base,
            model="deepseek-chat"
        )
        state.llm_extractor = LLMInfoExtractor(config=config)
        return {"success": True, "message": "LLM configured successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract", response_model=Dict[str, Any])
async def extract_features(request: TargetInfoRequest):
    """Extract structured features from target information"""
    import time
    start_time = time.time()

    try:
        if request.use_llm_extraction and state.llm_extractor:
            if request.parallel_extraction:
                features = state.llm_extractor.extract_multistage_parallel(
                    request.raw_text,
                    max_workers=3,
                    verbose=True
                )
            else:
                features = state.llm_extractor.extract_multistage(
                    request.raw_text,
                    stages=request.extraction_stages,
                    verbose=True
                )
        else:
            # Simple keyword extraction without LLM
            features = default_features()
            # Basic extraction logic
            text = request.raw_text.lower()
            words = request.raw_text.split()

            for word in words:
                if word.isalpha() and len(word) > 2:
                    features.keywords.append(word)

        extraction_time = time.time() - start_time

        return {
            "features": features_to_dict(features),
            "extraction_time": extraction_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_passwords(request: GenerateRequest):
    """Generate password candidates"""
    import time

    if not state.is_loaded():
        raise HTTPException(status_code=400, detail="Model not loaded. Please load a model first.")

    try:
        # Extract features
        extraction_start = time.time()

        if request.target_info.use_llm_extraction and state.llm_extractor:
            if request.target_info.parallel_extraction:
                features = state.llm_extractor.extract_multistage_parallel(
                    request.target_info.raw_text
                )
            else:
                features = state.llm_extractor.extract_multistage(
                    request.target_info.raw_text,
                    stages=request.target_info.extraction_stages
                )
        else:
            features = default_features()

        extraction_time = time.time() - extraction_start

        # Prepare latent vector
        feature_vector = torch.randn(1, 64).to(state.device)  # Placeholder

        # Generate passwords
        generation_start = time.time()
        passwords = []

        with torch.no_grad():
            if request.generation.method == "beam":
                results = state.model.generate_beam_search(
                    feature_vector,
                    state.tokenizer,
                    beam_width=request.generation.beam_width
                )
                for pwd, score in results[:request.generation.n_samples]:
                    passwords.append(PasswordResult(password=pwd, score=score, method="beam"))

            elif request.generation.method == "diverse_beam":
                results = state.model.generate_diverse_beam(
                    feature_vector,
                    state.tokenizer,
                    num_groups=3,
                    diversity_penalty=request.generation.diversity_penalty
                )
                for pwd, score in results[:request.generation.n_samples]:
                    passwords.append(PasswordResult(password=pwd, score=score, method="diverse_beam"))

            elif request.generation.method == "typical":
                for _ in range(request.generation.n_samples):
                    pwd = state.model.generate_typical(
                        feature_vector,
                        state.tokenizer,
                        typical_mass=request.generation.typical_mass
                    )
                    passwords.append(PasswordResult(password=pwd, method="typical"))

            elif request.generation.method == "contrastive":
                for _ in range(request.generation.n_samples):
                    pwd = state.model.generate_contrastive(
                        feature_vector,
                        state.tokenizer,
                        alpha=request.generation.contrastive_alpha
                    )
                    passwords.append(PasswordResult(password=pwd, method="contrastive"))

            else:  # sampling
                # Use temperature schedule
                temp_schedule = request.generation.temperature_schedule
                if temp_schedule == "constant":
                    temp_schedule = request.generation.temperature

                for _ in range(request.generation.n_samples):
                    pwd = state.model.generate_with_temperature_schedule(
                        feature_vector,
                        state.tokenizer,
                        temperature_schedule=temp_schedule,
                        top_k=request.generation.top_k,
                        top_p=request.generation.top_p
                    )
                    passwords.append(PasswordResult(password=pwd, method="sampling"))

        generation_time = time.time() - generation_start

        return GenerateResponse(
            passwords=passwords,
            extracted_features=features_to_dict(features),
            generation_time=generation_time,
            extraction_time=extraction_time
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/score")
async def score_password(password: str):
    """Score a single password"""
    if not state.is_loaded():
        raise HTTPException(status_code=400, detail="Model not loaded")

    try:
        feature_vector = torch.randn(1, 64).to(state.device)
        score = state.model.score_password(password, feature_vector, state.tokenizer)
        return {"password": password, "score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/methods")
async def get_generation_methods():
    """Get available generation methods"""
    return {
        "methods": [
            {
                "id": "sampling",
                "name": "Sampling",
                "description": "Standard sampling with temperature",
                "params": ["temperature", "temperature_schedule", "top_k", "top_p"]
            },
            {
                "id": "beam",
                "name": "Beam Search",
                "description": "Search multiple paths for best results",
                "params": ["beam_width"]
            },
            {
                "id": "diverse_beam",
                "name": "Diverse Beam Search",
                "description": "Multiple beam groups with diversity penalty",
                "params": ["beam_width", "diversity_penalty"]
            },
            {
                "id": "typical",
                "name": "Typical Sampling",
                "description": "Entropy-based sampling for coherent outputs",
                "params": ["typical_mass"]
            },
            {
                "id": "contrastive",
                "name": "Contrastive Search",
                "description": "Penalize repetitive outputs",
                "params": ["contrastive_alpha"]
            }
        ]
    }


# ============== New API Routes ==============

@app.websocket("/ws/{task_id}")
async def ws_endpoint(websocket, task_id: str = None):
    """WebSocket endpoint for real-time updates"""
    await websocket_endpoint(websocket, task_id)


# --- Task Management ---

class TaskSubmitRequest(BaseModel):
    """Request for async task submission"""
    target_info: TargetInfoRequest
    generation: GenerationConfig


@app.post("/api/tasks/generate")
async def submit_generation_task(request: TaskSubmitRequest):
    """Submit an async password generation task"""
    if not state.is_loaded():
        raise HTTPException(status_code=400, detail="Model not loaded")

    async def generate_task(target_info: dict, generation: dict):
        import time
        # Extract features
        features = default_features()
        if target_info.get("use_llm_extraction") and state.llm_extractor:
            features = state.llm_extractor.extract_multistage(
                target_info["raw_text"],
                stages=target_info.get("extraction_stages", 3)
            )

        feature_vector = torch.randn(1, 64).to(state.device)
        passwords = []

        with torch.no_grad():
            for _ in range(generation.get("n_samples", 20)):
                pwd = state.model.generate(feature_vector, state.tokenizer, temperature=0.8)
                passwords.append(pwd)

        return {
            "passwords": [{"password": p, "method": "async_sampling"} for p in passwords],
            "features": features_to_dict(features)
        }

    task_id = await state.task_manager.submit(
        generate_task,
        target_info=request.target_info.model_dump(),
        generation=request.generation.model_dump()
    )
    return {"task_id": task_id, "status": "submitted"}


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    task = await state.task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_dict()


@app.delete("/api/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a task"""
    success = await state.task_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel task")
    return {"task_id": task_id, "status": "cancelled"}


@app.get("/api/tasks")
async def list_tasks(status: Optional[str] = None):
    """List all tasks"""
    task_status = TaskStatus(status) if status else None
    tasks = await state.task_manager.list_tasks(status=task_status)
    return {"tasks": [t.to_dict() for t in tasks]}


@app.get("/api/tasks/stats")
async def task_stats():
    """Get task queue statistics"""
    return await state.task_manager.get_stats()


# --- Password Evaluation ---

class EvaluateRequest(BaseModel):
    """Password evaluation request"""
    passwords: List[str] = Field(..., description="List of passwords to evaluate")


@app.post("/api/evaluate")
async def evaluate_passwords(request: EvaluateRequest):
    """Evaluate password strength"""
    try:
        from evaluation.strength import PasswordStrengthEvaluator
        evaluator = PasswordStrengthEvaluator()
        results = []
        for pwd in request.passwords:
            report = evaluator.evaluate(pwd)
            results.append({
                "password": pwd,
                "score": report.score.name if hasattr(report.score, 'name') else str(report.score),
                "entropy": report.entropy,
                "guess_number": report.guess_number,
                "crack_time": report.crack_time,
                "warnings": report.warnings,
                "suggestions": report.suggestions,
            })
        return {"results": results}
    except ImportError:
        raise HTTPException(status_code=500, detail="Evaluation module not available")


# --- Rules Engine ---

class RulesRequest(BaseModel):
    """Password rules request"""
    passwords: List[str] = Field(..., description="Passwords to apply rules to")
    max_variants: int = Field(20, description="Maximum variants per password")


@app.post("/api/rules/apply")
async def apply_rules(request: RulesRequest):
    """Apply password transformation rules"""
    try:
        from rules.engine import PasswordRuleEngine
        engine = PasswordRuleEngine()
        results = {}
        for pwd in request.passwords[:10]:  # Limit for performance
            variants = engine.generate_variants(pwd, max_variants=request.max_variants)
            results[pwd] = variants
        return {"results": results}
    except ImportError:
        raise HTTPException(status_code=500, detail="Rules module not available")


# --- Auth ---

@app.post("/api/auth/token")
async def create_token(user_id: str = "default"):
    """Create a JWT token"""
    jwt_auth = JWTAuth()
    token = jwt_auth.create_token(user_id)
    return {"access_token": token, "token_type": "bearer"}


# --- Enhanced Status ---

@app.get("/api/status/enhanced")
async def get_enhanced_status():
    """Get enhanced system status with all component info"""
    return {
        "model_loaded": state.is_loaded(),
        "device": state.device,
        "cuda_available": torch.cuda.is_available(),
        "cache_stats": state.cache.stats(),
        "checkpoint": state.checkpoint_path,
        "task_stats": await state.task_manager.get_stats(),
        "websocket_connections": ws_manager.get_connection_count(),
        "api_key_configured": len(state.auth.api_keys) > 0,
    }


# ============== Startup ==============

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print(f"Starting Password Guesser on {state.device}")

    # Load env config
    try:
        from config.env import load_env_config
        env_config = load_env_config()
        if env_config.get("api_keys"):
            state.auth = APIKeyAuth(api_keys=set(env_config["api_keys"]))
    except Exception as e:
        logger.debug(f"Environment config loading skipped: {e}")

    # Try to load default model
    default_checkpoint = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pt")
    if os.path.exists(default_checkpoint):
        print(f"Loading default model from {default_checkpoint}")
        await load_model(default_checkpoint)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
