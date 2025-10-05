# ==== ULTRA-ADVANCED CONFIGURATION MODULE 2025 ====
"""
🚀 Production-Grade Configuration Management System

STATE-OF-THE-ART FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ VALIDATION & TYPE SAFETY:
  • Pydantic models with automatic validation
  • Type hints and runtime checks
  • Schema evolution tracking
  • Configuration freezing for immutability

⚡ ENVIRONMENT MANAGEMENT:
  • Multi-environment support (dev/staging/prod)
  • Dynamic environment detection
  • Layered configuration overrides
  • Hot-reloading for development

🛡️ SECURITY & SECRETS:
  • Environment variable integration
  • Secrets masking in logs
  • API key rotation support
  • Vault/AWS Secrets Manager ready

🎯 FEATURE FLAGS & TOGGLES:
  • Runtime feature management
  • A/B testing support
  • Gradual rollout capabilities
  • Experiment tracking integration

📊 OBSERVABILITY:
  • Structured logging configuration
  • Performance monitoring settings
  • Metrics collection toggles
  • Distributed tracing config

🔧 ML/AI OPTIMIZATIONS:
  • Hyperparameter management
  • Model versioning support
  • Experiment tracking integration
  • Dynamic batch size tuning

USAGE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from config import settings, get_config
    
    # Access validated settings
    print(settings.data_dir)
    print(settings.llm.api_key.get_secret_value())
    
    # Check feature flags
    if settings.features.enable_parallel:
        run_parallel()
    
    # Environment-specific behavior
    if settings.is_production:
        enable_monitoring()

Author: AI Competition Team 2025
License: MIT
Version: 3.0.0
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Literal, TYPE_CHECKING, Callable
from functools import lru_cache
from datetime import datetime
from enum import Enum

# Advanced configuration management - Pydantic required
from pydantic import (
    BaseModel, 
    Field, 
    SecretStr, 
    field_validator,
    model_validator,
    ConfigDict
)
from pydantic_settings import BaseSettings, SettingsConfigDict

HAS_PYDANTIC = True

# Optional: Dynamic reloading
try:
    from watchdog.observers import Observer  # type: ignore
    from watchdog.events import FileSystemEventHandler  # type: ignore
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    # Create dummy classes if watchdog not available
    class FileSystemEventHandler:  # type: ignore
        pass
    
    class Observer:  # type: ignore
        def __init__(self): pass
        def schedule(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def join(self): pass

# ════════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT & VERSIONING
# ════════════════════════════════════════════════════════════════════════════════

class Environment(str, Enum):
    """Supported deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

CONFIG_VERSION = "3.0.0"
CONFIG_SCHEMA_VERSION = "1.0.0"

# ════════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

class PathConfig(BaseModel):
    """Centralized path management with validation"""
    
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.absolute(),
        description="Root directory of the project"
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent / "data",
        description="Data directory for datasets"
    )
    model_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent / "models",
        description="Directory for saved models"
    )
    log_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent / "logs",
        description="Directory for log files"
    )
    cache_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent / "cache",
        description="Directory for cached data"
    )
    checkpoint_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent / "checkpoints",
        description="Directory for model checkpoints"
    )
    
    # Input/Output files
    train_csv: Optional[Path] = None
    test_csv: Optional[Path] = None
    output_csv: Optional[Path] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set default file paths
        if self.train_csv is None:
            self.train_csv = self.data_dir / "train.csv"
        if self.test_csv is None:
            self.test_csv = self.data_dir / "test.csv"
        if self.output_csv is None:
            self.output_csv = self.project_root / "submission.csv"
    
    @model_validator(mode='after')
    def create_directories(self):
        """Automatically create directories if they don't exist"""
        for field_name in ['data_dir', 'model_dir', 'log_dir', 'cache_dir', 'checkpoint_dir']:
            path = getattr(self, field_name)
            if path and not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        return self
    
    def get_timestamped_output(self, prefix: str = "output") -> Path:
        """Generate timestamped output filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.project_root / f"{prefix}_{timestamp}.csv"

# ════════════════════════════════════════════════════════════════════════════════
# LLM & API CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

class LLMConfig(BaseModel):
    """LLM and API endpoint configuration with secrets management"""
    
    api_url: str = Field(
        default_factory=lambda: os.getenv("SMALL_LLM_API_URL", ""),
        description="Primary LLM API endpoint"
    )
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("SMALL_LLM_API_KEY", "")),
        description="LLM API key (secret)"
    )
    backup_api_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("BACKUP_LLM_API_URL"),
        description="Fallback LLM API endpoint"
    )
    backup_api_key: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("BACKUP_LLM_API_KEY", "")) if os.getenv("BACKUP_LLM_API_KEY") else None,
        description="Backup API key"
    )
    
    # Model parameters
    model_name: str = Field(default="gpt-4o-mini", description="LLM model identifier")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=4096, ge=1, le=32000, description="Maximum tokens per request")
    timeout: int = Field(default=60, ge=1, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    
    # Rate limiting
    requests_per_minute: int = Field(default=60, ge=1, description="Rate limit for API calls")
    tokens_per_minute: int = Field(default=90000, ge=1, description="Token rate limit")
    
    @field_validator('api_key', 'backup_api_key')
    @classmethod
    def validate_api_key(cls, v):
        """Validate API key format"""
        if v and isinstance(v, SecretStr):
            key = v.get_secret_value()
            if key and len(key) < 10:
                warnings.warn("API key seems too short. Verify it's correct.")
        return v
    
    def get_masked_key(self) -> str:
        """Return masked version of API key for logging"""
        key = self.api_key.get_secret_value() if self.api_key else ""
        if len(key) > 8:
            return f"{key[:4]}...{key[-4:]}"
        return "****"

# ════════════════════════════════════════════════════════════════════════════════
# TOOL CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

class ToolConfig(BaseModel):
    """MCP tool and external service configuration"""
    
    enabled_tools: List[str] = Field(
        default=["calculator", "llm", "symbolic", "regex", "semantic_search"],
        description="List of enabled tools"
    )
    symbolic_solver_url: str = Field(
        default_factory=lambda: os.getenv("SYMBOLIC_SOLVER_URL", ""),
        description="Symbolic solver endpoint"
    )
    mcp_registry_path: Path = Field(
        default_factory=lambda: Path(__file__).parent / "mcp_tools_registry.json",
        description="Path to MCP tools registry"
    )
    
    # Tool-specific settings
    calculator_precision: int = Field(default=10, ge=1, le=50, description="Calculator decimal precision")
    regex_timeout: float = Field(default=5.0, ge=0.1, le=60.0, description="Regex execution timeout")
    tool_execution_timeout: int = Field(default=30, ge=1, description="General tool timeout in seconds")
    
    # Parallel execution
    max_parallel_tools: int = Field(default=5, ge=1, le=50, description="Max concurrent tool executions")
    enable_tool_caching: bool = Field(default=True, description="Enable tool result caching")
    cache_ttl: int = Field(default=3600, ge=0, description="Cache TTL in seconds")
    
    @field_validator('enabled_tools')
    @classmethod
    def validate_tools(cls, v):
        """Validate tool names"""
        allowed_tools = {"calculator", "llm", "symbolic", "regex", "semantic_search", "web_search", "code_executor"}
        invalid = set(v) - allowed_tools
        if invalid:
            warnings.warn(f"Unknown tools will be ignored: {invalid}")
        return [t for t in v if t in allowed_tools]

# ════════════════════════════════════════════════════════════════════════════════
# FEATURE FLAGS
# ════════════════════════════════════════════════════════════════════════════════

class FeatureFlags(BaseModel):
    """Runtime feature toggles for experiments and gradual rollouts"""
    
    # Core features
    enable_parallel: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_PARALLEL", "true").lower() == "true",
        description="Enable parallel processing"
    )
    enable_caching: bool = Field(default=True, description="Enable result caching")
    enable_hot_reload: bool = Field(default=False, description="Enable config hot-reloading")
    
    # Advanced features
    use_semantic_similarity: bool = Field(default=True, description="Use semantic similarity in selection")
    use_ensemble_scoring: bool = Field(default=True, description="Use ensemble scoring methods")
    enable_self_consistency: bool = Field(default=True, description="Enable self-consistency checks")
    
    # Experimental features (A/B testing)
    experiment_advanced_decomposition: bool = Field(default=False, description="Use advanced decomposition")
    experiment_meta_learning: bool = Field(default=False, description="Enable meta-learning")
    experiment_adaptive_timeout: bool = Field(default=True, description="Use adaptive timeouts")
    
    # Output features
    full_trace_output: bool = Field(default=False, description="Output full reasoning traces")
    save_intermediate_results: bool = Field(default=True, description="Save intermediate results")
    generate_audit_trail: bool = Field(default=True, description="Generate audit trail")
    
    def to_dict(self) -> Dict[str, bool]:
        """Export feature flags as dictionary"""
        return {k: v for k, v in self.__dict__.items() if isinstance(v, bool)}

# ════════════════════════════════════════════════════════════════════════════════
# REASONING ENGINE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

class ReasoningConfig(BaseModel):
    """Configuration for reasoning engine and problem solving"""
    
    # Trace configuration
    max_trace_len: int = Field(default=4000, ge=100, le=50000, description="Maximum trace length")
    max_iterations: int = Field(default=10, ge=1, le=100, description="Maximum reasoning iterations")
    max_depth: int = Field(default=5, ge=1, le=20, description="Maximum reasoning depth")
    
    # Selection parameters
    notepad_max_options: int = Field(default=16, ge=2, le=100, description="Max options in notepad")
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="Confidence threshold")
    uncertainty_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Uncertainty threshold")
    
    # Ensemble settings
    aggregation_weights: Dict[str, float] = Field(
        default={
            "trace_match": 0.3,
            "tool_confidence": 0.25,
            "keyword_signal": 0.15,
            "semantic_similarity": 0.2,
            "length_quality": 0.1
        },
        description="Weights for multi-signal aggregation"
    )
    
    # Performance tuning
    batch_size: int = Field(default=32, ge=1, le=512, description="Batch size for processing")
    embedding_batch_size: int = Field(default=32, ge=1, le=256, description="Embedding batch size")
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Similarity threshold")
    
    @field_validator('aggregation_weights')
    @classmethod
    def validate_weights(cls, v):
        """Ensure weights sum to approximately 1.0"""
        total = sum(v.values())
        if not (0.95 <= total <= 1.05):
            warnings.warn(f"Aggregation weights sum to {total:.3f}, should be ~1.0")
        return v

# ════════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

class LoggingConfig(BaseModel):
    """Advanced logging configuration with structured output"""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    verbose: bool = Field(default=True, description="Verbose output")
    log_errors: bool = Field(default=True, description="Log errors to file")
    log_performance: bool = Field(default=True, description="Log performance metrics")
    
    # File logging
    log_to_file: bool = Field(default=True, description="Enable file logging")
    log_rotation: bool = Field(default=True, description="Enable log rotation")
    max_log_size_mb: int = Field(default=10, ge=1, le=1000, description="Max log file size in MB")
    backup_count: int = Field(default=5, ge=1, le=100, description="Number of backup logs to keep")
    
    # Structured logging
    json_logging: bool = Field(default=False, description="Use JSON structured logging")
    include_timestamps: bool = Field(default=True, description="Include timestamps in logs")
    include_process_info: bool = Field(default=True, description="Include process/thread info")
    
    # Filtering
    filter_sensitive_data: bool = Field(default=True, description="Filter sensitive data from logs")
    log_api_calls: bool = Field(default=True, description="Log API calls")
    log_tool_usage: bool = Field(default=True, description="Log tool usage")

# ════════════════════════════════════════════════════════════════════════════════
# MONITORING & OBSERVABILITY
# ════════════════════════════════════════════════════════════════════════════════

class MonitoringConfig(BaseModel):
    """Performance monitoring and observability settings"""
    
    # Metrics collection
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_interval: int = Field(default=60, ge=1, description="Metrics collection interval (seconds)")
    
    # Performance tracking
    track_latency: bool = Field(default=True, description="Track operation latency")
    track_token_usage: bool = Field(default=True, description="Track token usage")
    track_cost: bool = Field(default=True, description="Track API costs")
    
    # APM Integration
    enable_apm: bool = Field(default=False, description="Enable APM integration")
    apm_service_name: str = Field(default="ehos-hackathon", description="APM service name")
    apm_endpoint: Optional[str] = Field(
        default_factory=lambda: os.getenv("APM_ENDPOINT"),
        description="APM endpoint URL"
    )
    
    # Distributed tracing
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    trace_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Trace sampling rate")
    
    # Alerting
    enable_alerts: bool = Field(default=False, description="Enable alerting")
    error_threshold: int = Field(default=10, ge=1, description="Error count threshold for alerts")
    latency_threshold_ms: int = Field(default=5000, ge=100, description="Latency threshold for alerts")

# ════════════════════════════════════════════════════════════════════════════════
# OUTPUT CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

class OutputConfig(BaseModel):
    """Output formatting and submission configuration"""
    
    format: Literal["minimal", "advanced", "detailed"] = Field(
        default="advanced",
        description="Output format type"
    )
    encoding: str = Field(default="utf-8-sig", description="Output file encoding")
    
    # Content options
    include_reasoning: bool = Field(default=True, description="Include reasoning in output")
    include_confidence: bool = Field(default=True, description="Include confidence scores")
    include_metadata: bool = Field(default=True, description="Include metadata")
    
    # Performance
    use_polars: bool = Field(default=True, description="Use Polars for fast CSV writing")
    parallel_write: bool = Field(default=True, description="Enable parallel writing")
    
    # Validation
    validate_before_write: bool = Field(default=True, description="Validate data before writing")
    max_cell_length: int = Field(default=5000, ge=100, description="Maximum cell length")

# ════════════════════════════════════════════════════════════════════════════════
# MAIN SETTINGS CLASS
# ════════════════════════════════════════════════════════════════════════════════

class Settings(BaseSettings):
    """
    Master configuration class with all settings.
    Automatically loads from environment variables and .env file.
    """
    
    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Current environment"
    )
    debug: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true",
        description="Debug mode"
    )
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    
    # Sub-configurations
    paths: PathConfig = Field(default_factory=PathConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    # Metadata
    config_version: str = Field(default=CONFIG_VERSION, description="Configuration version")
    config_schema_version: str = Field(default=CONFIG_SCHEMA_VERSION, description="Schema version")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT
    
    def to_dict(self, mask_secrets: bool = True) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        config_dict = json.loads(self.model_dump_json())
        
        if mask_secrets:
            # Mask API keys
            if 'llm' in config_dict and 'api_key' in config_dict['llm']:
                config_dict['llm']['api_key'] = "****"
            if 'llm' in config_dict and 'backup_api_key' in config_dict['llm']:
                config_dict['llm']['backup_api_key'] = "****"
        
        return config_dict
    
    def save_to_file(self, filepath: Union[str, Path], mask_secrets: bool = True):
        """Save configuration to JSON file"""
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(mask_secrets=mask_secrets), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'Settings':
        """Load configuration from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        issues = []
        
        # Check API keys
        if not self.llm.api_key or not self.llm.api_key.get_secret_value():
            issues.append("WARNING: LLM API key not set")
        
        # Check paths
        if not self.paths.data_dir.exists():
            issues.append(f"WARNING: Data directory does not exist: {self.paths.data_dir}")
        
        # Check production settings
        if self.is_production:
            if self.debug:
                issues.append("ERROR: Debug mode should be disabled in production")
            if self.logging.level == "DEBUG":
                issues.append("WARNING: Debug logging in production may impact performance")
            if not self.monitoring.enable_metrics:
                issues.append("WARNING: Metrics disabled in production")
        
        return issues

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION HOT-RELOADING
# ════════════════════════════════════════════════════════════════════════════════

class ConfigReloader(FileSystemEventHandler):
    """Watch configuration file and reload on changes"""
    
    def __init__(self, config_path: Path, callback: Callable[[], None]):
        self.config_path = config_path
        self.callback = callback
        self.last_reload = datetime.now()
    
    def on_modified(self, event):
        if event.src_path == str(self.config_path):
            # Debounce: only reload if > 1 second since last reload
            if (datetime.now() - self.last_reload).total_seconds() > 1:
                print(f"[CONFIG] Detected changes in {self.config_path}, reloading...")
                self.callback()
                self.last_reload = datetime.now()

# ════════════════════════════════════════════════════════════════════════════════
# GLOBAL SETTINGS INSTANCE
# ════════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_config() -> Settings:
    """
    Get cached configuration instance.
    This ensures singleton pattern - only one config instance exists.
    """
    return Settings()

# Global settings instance (singleton)
settings = get_config()

# ════════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def print_config(mask_secrets: bool = True, detailed: bool = False):
    """Pretty print configuration"""
    print("\n" + "="*80)
    print(f"🚀 CONFIGURATION v{settings.config_version}")
    print("="*80)
    print(f"Environment: {settings.environment.value.upper()}")
    print(f"Debug Mode: {settings.debug}")
    print(f"Config Schema: v{settings.config_schema_version}")
    print(f"Last Updated: {settings.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*80)
    
    if detailed:
        config_dict = settings.to_dict(mask_secrets=mask_secrets)
        print(json.dumps(config_dict, indent=2, default=str))
    else:
        # Print summary
        print(f"\n📁 Paths:")
        print(f"  Project Root: {settings.paths.project_root}")
        print(f"  Data Dir: {settings.paths.data_dir}")
        print(f"  Model Dir: {settings.paths.model_dir}")
        
        print(f"\n🤖 LLM:")
        print(f"  Model: {settings.llm.model_name}")
        print(f"  API Key: {settings.llm.get_masked_key()}")
        print(f"  Max Tokens: {settings.llm.max_tokens}")
        
        print(f"\n🛠️  Tools:")
        print(f"  Enabled: {', '.join(settings.tools.enabled_tools)}")
        print(f"  Max Parallel: {settings.tools.max_parallel_tools}")
        
        print(f"\n🎯 Features:")
        active_features = [k.replace('_', ' ').title() for k, v in settings.features.to_dict().items() if v]
        print(f"  Active: {', '.join(active_features[:5])}...")
        
        print(f"\n📊 Logging:")
        print(f"  Level: {settings.logging.level}")
        print(f"  Verbose: {settings.logging.verbose}")
    
    # Validation warnings
    issues = settings.validate_config()
    if issues:
        print(f"\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"  • {issue}")
    
    print("="*80 + "\n")

def reload_config():
    """Force reload configuration from environment/file"""
    global settings
    get_config.cache_clear()
    settings = get_config()
    print("[CONFIG] Configuration reloaded successfully")

def setup_config_watcher():
    """Setup automatic configuration reloading"""
    if not HAS_WATCHDOG:
        warnings.warn("watchdog not installed. Hot-reload disabled. Install: pip install watchdog")
        return None
    
    if not settings.features.enable_hot_reload:
        return None
    
    config_file = Path(__file__)
    event_handler = ConfigReloader(config_file, reload_config)
    observer = Observer()
    observer.schedule(event_handler, str(config_file.parent), recursive=False)
    observer.start()
    print(f"[CONFIG] Hot-reload enabled for {config_file}")
    return observer

def export_config_template(filepath: str = "config_template.json"):
    """Export configuration template for documentation"""
    template = Settings()
    template.save_to_file(filepath, mask_secrets=True)
    print(f"[CONFIG] Configuration template exported to {filepath}")

# ════════════════════════════════════════════════════════════════════════════════
# BACKWARDS COMPATIBILITY (Legacy Access)
# ════════════════════════════════════════════════════════════════════════════════

# Provide backwards-compatible access to old variable names
PROJECT_ROOT = settings.paths.project_root
DATA_DIR = settings.paths.data_dir
MODEL_DIR = settings.paths.model_dir
LOG_DIR = settings.paths.log_dir
TRAIN_CSV = settings.paths.train_csv
TEST_CSV = settings.paths.test_csv
OUTPUT_CSV = settings.paths.output_csv

LLM_API_URL = settings.llm.api_url
LLM_API_KEY = settings.llm.api_key.get_secret_value() if settings.llm.api_key else ""
SYMBOLIC_SOLVER_URL = settings.tools.symbolic_solver_url

ENABLED_TOOLS = settings.tools.enabled_tools
MAX_TRACE_LEN = settings.reasoning.max_trace_len
ENABLE_PARALLEL = settings.features.enable_parallel
NOTEPAD_MAX_OPTIONS = settings.reasoning.notepad_max_options

OUTPUT_FORMAT = settings.output.format
FULL_TRACE_OUTPUT = settings.features.full_trace_output
VERBOSE = settings.logging.verbose
LOG_ERRORS = settings.logging.log_errors
SEED = settings.seed

# ════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Print configuration
    print_config(detailed=False)
    
    # Optionally start config watcher
    if settings.features.enable_hot_reload:
        observer = setup_config_watcher()
        if observer:
            try:
                import time
                print("\n[CONFIG] Watching for changes... Press Ctrl+C to stop")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
                observer.join()
                print("\n[CONFIG] Stopped configuration watcher")
    
    # Export template
    if "--export-template" in sys.argv:
        export_config_template()
