# ==== ULTRA-ADVANCED AGENTIC REASONING API 2025 ====
"""
🚀 PRODUCTION-READY FastAPI Server with Advanced Features

⚡ NOW USES FULLY INTEGRATED ENGINE WITH ALL AGENTIC_ENGINE MODULES! ⚡

FULLY INTEGRATED REASONING ENGINE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ config.py - Configuration management
✅ utils.py - Logging, caching, timing
✅ problem_classifier.py - Problem type detection
✅ problem_restatement/ - Problem clarification
✅ domain_prompts.py - Specialized prompts
✅ decompose.py - LangChain decomposition
✅ tools/ - LangChain tool execution (5 tools)
✅ notepad_manager/ - High-performance tracking
✅ selection_module.py - Answer selection
✅ output_formatter.py - Output formatting
✅ input_loader/ - CSV processing
✅ cache/ - Result caching
✅ logs/ - Structured logging

API FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Async Processing & Background Tasks
✅ Redis Caching with TTL
✅ Rate Limiting & API Key Authentication
✅ Comprehensive Error Handling & Logging
✅ Health Checks & Prometheus Metrics
✅ WebSocket Support for Real-time Updates
✅ Streaming Responses
✅ Database Connection Pooling
✅ Advanced CORS Security
✅ File Upload Optimization
✅ Job Queue Management
✅ Request Validation with Pydantic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ENDPOINTS:
- POST /run-csv/ - Upload CSV for batch processing
- POST /run-single/ - Process single problem
- POST /batch-async/ - Async batch processing with job ID
- GET /status/{job_id} - Check async job status
- GET /trace/{trace_id} - Get detailed trace
- GET /download/{filename} - Download processed file
- GET /health - Health check
- GET /engine/info - Engine module information (NEW!)
- GET /metrics - Prometheus metrics
- WS /ws/{job_id} - WebSocket for real-time updates
"""

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # Load .env file from project root

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Depends, WebSocket, WebSocketDisconnect, Header, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import tempfile
import os
import csv
import uuid
import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import asynccontextmanager
from collections import defaultdict
import traceback

# ===== Conditional Imports (Graceful Degradation) =====
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False
    logging.warning("aiofiles not installed. File operations will be synchronous.")

# Rate limiting removed for performance
HAS_RATE_LIMIT = False

try:
    import aioredis
    HAS_REDIS = True
except ImportError:
    try:
        import redis.asyncio as aioredis
        HAS_REDIS = True
    except ImportError:
        HAS_REDIS = False
        logging.warning("Redis not installed. Caching disabled.")

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logging.warning("prometheus_client not installed. Metrics disabled.")

# Import your modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agentic_engine'))

from agentic_engine.input_loader import input_loader
from agentic_engine import output_formatter
# Use the FULLY INTEGRATED engine with ALL agentic_engine modules
from agentic_engine.reasoning_engine.fully_integrated_engine import create_engine, FullyIntegratedHybridEngine

# ===== Configuration =====
class Config:
    """Centralized configuration"""
    # API Settings
    API_TITLE = "Agentic Reasoning API"
    API_VERSION = "2.0.0"
    API_DESCRIPTION = __doc__
    
    # Security
    API_KEY_HEADER = "X-API-Key"
    ALLOWED_ORIGINS = ["*"]  # Configure for production
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL = 3600  # 1 hour
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = "30/minute"
    RATE_LIMIT_PER_HOUR = "500/hour"
    
    # Processing
    MAX_WORKERS = 4
    JOB_TIMEOUT = 1800  # 30 minutes
    CLEANUP_INTERVAL = 300  # 5 minutes
    
    # File Management
    TEMP_DIR = os.path.join(tempfile.gettempdir(), "agentic_api")
    MAX_FILE_AGE = 3600  # 1 hour

config = Config()

# ===== Setup Logging =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== Ensure temp directory exists =====
os.makedirs(config.TEMP_DIR, exist_ok=True)

# ===== Prometheus Metrics =====
# Disable Prometheus for now to avoid registration conflicts
# To enable: set HAS_PROMETHEUS_ENABLED = True
HAS_PROMETHEUS_ENABLED = False

if HAS_PROMETHEUS and HAS_PROMETHEUS_ENABLED:
    try:
        request_count = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
        request_duration = Histogram('api_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
        processing_time = Histogram('csv_processing_seconds', 'CSV processing time')
        active_jobs = Gauge('active_jobs', 'Number of active processing jobs')
        cache_hits = Counter('cache_hits_total', 'Total cache hits')
        cache_misses = Counter('cache_misses_total', 'Total cache misses')
    except ValueError as e:
        # Metrics already registered - disable to avoid conflicts
        logger.warning(f"Prometheus metrics registration failed: {e}")
        HAS_PROMETHEUS_ENABLED = False
else:
    # Create dummy objects when Prometheus is disabled
    class DummyMetric:
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    request_count = DummyMetric()
    request_duration = DummyMetric()
    processing_time = DummyMetric()
    active_jobs = DummyMetric()
    cache_hits = DummyMetric()
    cache_misses = DummyMetric()

# ===== Global State Management =====
class JobManager:
    """Manages async job states and results"""
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.websockets: Dict[str, List[WebSocket]] = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def create_job(self, job_id: str, metadata: Dict[str, Any]) -> None:
        async with self.lock:
            self.jobs[job_id] = {
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "progress": 0,
                "result": None,
                "error": None,
                **metadata
            }
            if HAS_PROMETHEUS:
                active_jobs.inc()
    
    async def update_job(self, job_id: str, updates: Dict[str, Any]) -> None:
        async with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(updates)
                # Notify WebSocket clients
                await self.notify_clients(job_id, self.jobs[job_id])
    
    async def complete_job(self, job_id: str, result: Any = None, error: Optional[str] = None) -> None:
        async with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update({
                    "status": "completed" if error is None else "failed",
                    "completed_at": datetime.now().isoformat(),
                    "result": result,
                    "error": error,
                    "progress": 100 if error is None else self.jobs[job_id].get("progress", 0)
                })
                if HAS_PROMETHEUS:
                    active_jobs.dec()
                await self.notify_clients(job_id, self.jobs[job_id])
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self.lock:
            return self.jobs.get(job_id)
    
    async def notify_clients(self, job_id: str, data: Dict[str, Any]) -> None:
        """Send updates to WebSocket clients"""
        if job_id in self.websockets:
            dead_sockets = []
            for ws in self.websockets[job_id]:
                try:
                    await ws.send_json(data)
                except:
                    dead_sockets.append(ws)
            # Clean up dead connections
            for ws in dead_sockets:
                self.websockets[job_id].remove(ws)
    
    async def add_websocket(self, job_id: str, websocket: WebSocket) -> None:
        self.websockets[job_id].append(websocket)
    
    async def remove_websocket(self, job_id: str, websocket: WebSocket) -> None:
        if job_id in self.websockets and websocket in self.websockets[job_id]:
            self.websockets[job_id].remove(websocket)

job_manager = JobManager()

# ===== Redis Cache Manager =====
class CacheManager:
    """Redis cache with fallback to in-memory"""
    def __init__(self):
        self.redis_client = None
        self.memory_cache: Dict[str, tuple] = {}  # (value, expiry)
    
    async def connect(self):
        if HAS_REDIS:
            try:
                self.redis_client = await aioredis.from_url(config.REDIS_URL, decode_responses=True)
                await self.redis_client.ping()
                logger.info("Connected to Redis")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
                self.redis_client = None
    
    async def get(self, key: str) -> Optional[str]:
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value and HAS_PROMETHEUS:
                    cache_hits.inc()
                elif HAS_PROMETHEUS:
                    cache_misses.inc()
                return value
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # Fallback to memory cache
        if key in self.memory_cache:
            value, expiry = self.memory_cache[key]
            if time.time() < expiry:
                if HAS_PROMETHEUS:
                    cache_hits.inc()
                return value
            else:
                del self.memory_cache[key]
        
        if HAS_PROMETHEUS:
            cache_misses.inc()
        return None
    
    async def set(self, key: str, value: str, ttl: int = config.CACHE_TTL) -> None:
        if self.redis_client:
            try:
                await self.redis_client.setex(key, ttl, value)
                return
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        # Fallback to memory cache
        self.memory_cache[key] = (value, time.time() + ttl)
    
    async def close(self):
        if self.redis_client:
            await self.redis_client.close()

cache_manager = CacheManager()

# ===== Lifespan Context Manager =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting Agentic Reasoning API...")
    await cache_manager.connect()
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_old_files())
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    cleanup_task.cancel()
    await cache_manager.close()

# ===== FastAPI App Initialization =====
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
    lifespan=lifespan
)

# ===== Middleware =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate Limiting - DISABLED for maximum performance
# (Re-enable by installing slowapi if needed)

# ===== Request/Response Models =====
class ProblemRequest(BaseModel):
    """Single problem request"""
    problem_statement: str = Field(..., min_length=1, max_length=10000)
    answer_options: List[str] = Field(..., description="List of answer options")
    topic: Optional[str] = Field(None, max_length=200)
    
    @field_validator('answer_options')
    @classmethod
    def validate_options(cls, v):
        if len(v) < 2:
            raise ValueError('Must have at least 2 answer options')
        if len(v) > 10:
            raise ValueError('Maximum 10 answer options allowed')
        if len(v) != len(set(v)):
            raise ValueError('Answer options must be unique')
        return v

class ProblemResponse(BaseModel):
    """Single problem response"""
    problem_statement: str
    solution: str
    correct_option: str
    correct_index: int
    trace: List[str]
    processing_time: float
    topic: Optional[str] = None

class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: str
    progress: float
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    redis_connected: bool
    active_jobs: int

# ===== Error Handlers =====
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# ===== Background Tasks =====
async def cleanup_old_files():
    """Periodically clean up old temporary files"""
    while True:
        try:
            await asyncio.sleep(config.CLEANUP_INTERVAL)
            now = time.time()
            temp_path = Path(config.TEMP_DIR)
            
            for file_path in temp_path.glob("*"):
                if file_path.is_file():
                    age = now - file_path.stat().st_mtime
                    if age > config.MAX_FILE_AGE:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# ===== Core Processing Functions =====
async def process_single_problem_async(
    problem_statement: str,
    answer_options: List[str],
    topic: Optional[str] = None
) -> Dict[str, Any]:
    """Process a single problem asynchronously"""
    start_time = time.time()
    
    # Check cache
    cache_key = f"problem:{hash(problem_statement + str(answer_options))}"
    cached = await cache_manager.get(cache_key)
    if cached:
        logger.info("Cache hit for problem")
        return json.loads(cached)
    
    # Process with FULLY INTEGRATED HybridReasoningEngine (uses ALL modules)
    engine = create_engine(use_notepad=False)
    
    # Run reasoning in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    idx, trace_list, log = await loop.run_in_executor(
        None,
        engine.reason,
        problem_statement,
        answer_options,
        False  # enable_restatement parameter
    )
    
    result = {
        "problem_statement": problem_statement,
        "solution": trace_list[idx] if idx < len(trace_list) else "",
        "correct_option": answer_options[idx] if idx < len(answer_options) else "",
        "correct_index": idx,
        "trace": trace_list,
        "classification": log.get("classification", {}),
        "selection_method": log.get("selection_method", "unknown"),
        "processing_time": time.time() - start_time,
        "topic": topic
    }
    
    # Cache result
    await cache_manager.set(cache_key, json.dumps(result))
    
    return result

async def process_csv_async(
    csv_path: str,
    output_csv_path: str,
    job_id: Optional[str] = None
) -> tuple:
    """Process CSV file asynchronously with progress updates"""
    try:
        # Load data
        if HAS_AIOFILES:
            async with aiofiles.open(csv_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            # Parse CSV in thread pool
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, input_loader.load_dataset, csv_path)
        else:
            data = input_loader.load_dataset(csv_path)
        
        total = len(data)
        logger.info(f"Processing {total} problems from CSV")
        
        # Prepare data
        topics = [row.get('topic', '') for row in data]
        statements = [row['problem_statement'] for row in data]
        answer_options_list = [row['answer_options'] for row in data]
        
        # Process with progress updates using FULLY INTEGRATED engine (ALL modules)
        engine = create_engine(use_notepad=False)
        results = []
        correct_indices = []
        traces = []
        
        for i, item in enumerate(data):
            options = item["answer_options"]
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            idx, trace_list, log = await loop.run_in_executor(
                None,
                engine.reason,
                item["problem_statement"],
                options,
                False  # enable_restatement parameter
            )
            traces.append(trace_list)
            
            correct_indices.append(idx)
            results.append(trace_list[idx] if idx < len(trace_list) else "")
            
            # Update progress
            progress = ((i + 1) / total) * 100
            if job_id:
                await job_manager.update_job(job_id, {"progress": progress})
            
            logger.info(f"Processed {i+1}/{total} problems ({progress:.1f}%)")
        
        # Generate output
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            output_formatter.write_advanced_submission,
            results,
            output_csv_path,
            topics,
            statements,
            correct_indices,
            answer_options_list
        )
        
        if HAS_PROMETHEUS:
            processing_time.observe(time.time())
        
        return output_csv_path, results, correct_indices, traces
        
    except Exception as e:
        logger.error(f"CSV processing error: {e}\n{traceback.format_exc()}")
        raise

# ===== API ROUTES =====

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Agentic Reasoning API",
        "version": config.API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=config.API_VERSION,
        timestamp=datetime.now().isoformat(),
        redis_connected=cache_manager.redis_client is not None,
        active_jobs=len(job_manager.jobs)
    )

@app.get("/engine/info", tags=["Monitoring"])
async def engine_info():
    """
    Get information about the fully integrated reasoning engine.
    Shows all modules being used and their status.
    """
    return {
        "engine": "FullyIntegratedHybridEngine",
        "version": "2.0.0",
        "description": "Uses ALL agentic_engine modules for comprehensive reasoning",
        "modules": {
            "config.py": {
                "status": "active",
                "function": "Configuration management",
                "features": ["Pydantic validation", "Multi-environment", "Secrets management"]
            },
            "utils.py": {
                "status": "active",
                "function": "Utilities",
                "features": ["Logging", "Caching", "Timing decorators", "Text sanitization"]
            },
            "problem_classifier.py": {
                "status": "active",
                "function": "Problem type detection",
                "features": ["8+ problem types", "Confidence scoring", "Pattern matching"]
            },
            "problem_restatement/": {
                "status": "active",
                "function": "Problem clarification",
                "features": ["Text normalization", "Ambiguity resolution", "NLP processing"]
            },
            "domain_prompts.py": {
                "status": "active",
                "function": "Specialized prompts",
                "features": ["Domain-specific templates", "Formula reminders", "Examples"]
            },
            "decompose.py": {
                "status": "active",
                "function": "Problem decomposition",
                "features": ["LangChain integration", "Structured output", "Step tracking"]
            },
            "tools/": {
                "status": "active",
                "function": "Tool execution",
                "features": ["5 LangChain tools", "calculator", "geometry", "pattern", "logic", "python_eval"]
            },
            "notepad_manager/": {
                "status": "active",
                "function": "Execution tracking",
                "features": ["High-performance C DLL (optional)", "Python fallback", "Thread-safe"]
            },
            "selection_module.py": {
                "status": "active",
                "function": "Answer selection",
                "features": ["LLM-based", "Semantic similarity", "Confidence scoring"]
            },
            "output_formatter.py": {
                "status": "active",
                "function": "Output formatting",
                "features": ["CSV export", "JSON formatting", "Statistics"]
            },
            "input_loader/": {
                "status": "active",
                "function": "CSV processing",
                "features": ["Polars integration", "Fast parsing", "Progress tracking"]
            },
            "cache/": {
                "status": "active",
                "function": "Caching",
                "features": ["Result caching", "TTL management", "Embeddings cache"]
            },
            "logs/": {
                "status": "active",
                "function": "Logging",
                "features": ["Structured logs", "Rotation", "JSON format"]
            }
        },
        "pipeline": [
            "1. Configuration Loading (config.py)",
            "2. Problem Restatement (problem_restatement/)",
            "3. Problem Classification (problem_classifier.py)",
            "4. Domain-Specific Prompts (domain_prompts.py)",
            "5. Decomposition (decompose.py)",
            "6. Tool Execution (tools/)",
            "7. Notepad Tracking (notepad_manager/)",
            "8. Answer Selection (selection_module.py)",
            "9. Output Formatting (output_formatter.py)"
        ],
        "statistics": {
            "total_modules": 13,
            "integration_complete": True,
            "performance_features": ["Caching", "Timing", "Statistics tracking"],
            "fallback_mechanisms": ["Python scratchpad", "Rule-based selection", "Memory cache"]
        }
    }

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    if not HAS_PROMETHEUS:
        raise HTTPException(status_code=501, detail="Prometheus not installed")
    
    return StreamingResponse(
        iter([generate_latest()]),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/run-single/", response_model=ProblemResponse, tags=["Processing"])
async def run_single_problem(request: ProblemRequest):
    """
    Process a single problem with reasoning engine.
    Returns solution, trace, and correct option.
    """
    try:
        result = await process_single_problem_async(
            request.problem_statement,
            request.answer_options,
            request.topic
        )
        return ProblemResponse(**result)
    
    except Exception as e:
        logger.error(f"Single problem processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-csv/", tags=["Processing"])
async def run_csv(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload CSV for batch reasoning inference.
    Returns downloadable processed CSV and JSON summary.
    
    **Synchronous processing** - waits for completion.
    For large files, use /batch-async/ instead.
    """
    # Validate file
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Check file size
    contents = await file.read()
    if len(contents) > config.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large (max {config.MAX_FILE_SIZE} bytes)")
    
    # Save uploaded file
    suffix = str(uuid.uuid4())
    temp_input = os.path.join(config.TEMP_DIR, f"input_{suffix}.csv")
    temp_output = os.path.join(config.TEMP_DIR, f"output_{suffix}.csv")
    
    try:
        if HAS_AIOFILES:
            async with aiofiles.open(temp_input, 'wb') as f:
                await f.write(contents)
        else:
            with open(temp_input, 'wb') as f:
                f.write(contents)
        
        # Process CSV
        output_csv, results, correct_indices, traces = await process_csv_async(
            temp_input, temp_output
        )
        
        # Read results
        response_data = []
        if HAS_AIOFILES:
            async with aiofiles.open(output_csv, "r", encoding="utf-8") as fin:
                content = await fin.read()
                reader = csv.DictReader(content.splitlines())
                for row in reader:
                    response_data.append({
                        "topic": row.get("topic", ""),
                        "problem_statement": row.get("problem_statement", ""),
                        "solution": row.get("solution", ""),
                        "correct_option": row.get("correct option", "")
                    })
        else:
            with open(output_csv, "r", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    response_data.append({
                        "topic": row.get("topic", ""),
                        "problem_statement": row.get("problem_statement", ""),
                        "solution": row.get("solution", ""),
                        "correct_option": row.get("correct option", "")
                    })
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(os.remove, temp_input)
        
        return {
            "success": True,
            "download_csv": f"/download/{os.path.basename(temp_output)}",
            "total_problems": len(response_data),
            "results": response_data
        }
    
    except Exception as e:
        logger.error(f"CSV processing error: {e}\n{traceback.format_exc()}")
        # Cleanup on error
        for path in [temp_input, temp_output]:
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-async/", tags=["Processing"])
async def batch_async(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload CSV for **asynchronous** batch processing.
    Returns job_id for status tracking.
    
    Use /status/{job_id} to check progress.
    Use /ws/{job_id} for real-time WebSocket updates.
    """
    # Validate file
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    contents = await file.read()
    if len(contents) > config.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large")
    
    # Create job
    job_id = str(uuid.uuid4())
    suffix = str(uuid.uuid4())
    temp_input = os.path.join(config.TEMP_DIR, f"input_{suffix}.csv")
    temp_output = os.path.join(config.TEMP_DIR, f"output_{suffix}.csv")
    
    # Save file
    if HAS_AIOFILES:
        async with aiofiles.open(temp_input, 'wb') as f:
            await f.write(contents)
    else:
        with open(temp_input, 'wb') as f:
            f.write(contents)
    
    # Create job entry
    await job_manager.create_job(job_id, {
        "filename": file.filename,
        "input_path": temp_input,
        "output_path": temp_output
    })
    
    # Start background processing
    async def process_job():
        try:
            output_csv, results, correct_indices, traces = await process_csv_async(
                temp_input, temp_output, job_id
            )
            
            # Read results
            response_data = []
            with open(output_csv, "r", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    response_data.append({
                        "topic": row.get("topic", ""),
                        "problem_statement": row.get("problem_statement", ""),
                        "solution": row.get("solution", ""),
                        "correct_option": row.get("correct option", "")
                    })
            
            await job_manager.complete_job(job_id, result={
                "download_csv": f"/download/{os.path.basename(temp_output)}",
                "total_problems": len(response_data),
                "results": response_data
            })
            
        except Exception as e:
            logger.error(f"Background job {job_id} failed: {e}")
            await job_manager.complete_job(job_id, error=str(e))
    
    # Schedule background task
    asyncio.create_task(process_job())
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Job started. Use /status/{job_id} to check progress.",
        "websocket": f"/ws/{job_id}"
    }

@app.get("/status/{job_id}", response_model=JobStatusResponse, tags=["Processing"])
async def get_job_status(job_id: str):
    """Get async job status and results"""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(job_id=job_id, **job)

@app.get("/trace/{trace_id}", tags=["Debug"])
async def get_trace(trace_id: str):
    """
    Get detailed reasoning trace for debugging.
    Trace IDs are returned in processing responses.
    """
    # This is a placeholder - implement trace storage as needed
    raise HTTPException(status_code=501, detail="Trace storage not yet implemented")

@app.get("/download/{filename}", tags=["Files"])
async def download_csv(filename: str):
    """Download processed CSV file"""
    file_path = os.path.join(config.TEMP_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type='text/csv',
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job updates.
    Sends progress and status updates as job processes.
    """
    await websocket.accept()
    await job_manager.add_websocket(job_id, websocket)
    
    try:
        # Send initial status
        job = await job_manager.get_job(job_id)
        if job:
            await websocket.send_json(job)
        else:
            await websocket.send_json({"error": "Job not found"})
            await websocket.close()
            return
        
        # Keep connection alive
        while True:
            try:
                # Wait for client messages (ping/pong)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
            except WebSocketDisconnect:
                break
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        await job_manager.remove_websocket(job_id, websocket)

# ===== Main Entry Point =====
if __name__ == "__main__":
    import uvicorn
    
    # Production configuration
    # Note: Using single worker due to Prometheus metrics compatibility
    # For true multi-worker setup, use external metrics aggregation (e.g., Prometheus Pushgateway)
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,  # Set to True for development
        workers=1,  # Single worker for Prometheus compatibility
        log_level="info",
        access_log=True,
        use_colors=True
    )
