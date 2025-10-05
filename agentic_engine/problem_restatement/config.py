"""
Configuration file for the Advanced Problem Restatement System.
Adjust these settings to optimize performance for your specific use case.
"""

from typing import Optional, Dict, Any

# Core Processing Settings
PROCESSING_CONFIG = {
    # Enable advanced NLP processing with spaCy
    'enable_spacy': True,
    
    # Cache size for LRU caching (higher = more memory, better performance for repeated patterns)
    'cache_size': 1024,
    
    # Maximum text length before truncation
    'max_output_length': 350,
    
    # Minimum words required for spaCy processing (avoid overhead for short texts)
    'spacy_min_words': 5,
    
    # Enable Unicode normalization
    'enable_unicode_normalization': True,
    
    # Enable intelligent sentence boundary truncation
    'intelligent_truncation': True
}

# Parallel Processing Settings
PARALLEL_CONFIG = {
    # Maximum number of worker processes for batch processing
    'max_workers': 4,
    
    # Minimum batch size to trigger parallel processing
    'parallel_threshold': 10,
    
    # Minimum batch size to trigger pandas vectorization
    'pandas_threshold': 50,
    
    # Batch size for spaCy pipe processing
    'spacy_batch_size': 1000,
    
    # Number of processes for spaCy parallel processing
    'spacy_n_process': 2
}

# Advanced Pattern Settings
PATTERN_CONFIG = {
    # Custom filler phrases to remove (in addition to built-in patterns)
    'custom_fillers': [
        'it is worth noting that',
        'it is important to mention',
        'for the sake of argument',
        'in this particular case',
        'as we can see',
        'it goes without saying'
    ],
    
    # Character names to remove (in addition to detected proper nouns)
    'custom_character_names': [
        'Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry',
        'Ivan', 'Jack', 'Karen', 'Liam', 'Maria', 'Nancy', 'Oscar', 'Paul'
    ],
    
    # Conversational starters to remove
    'conversational_starters': [
        'what is', 'can you', 'how many', 'how much', 'where is', 'when does',
        'why would', 'which one', 'who is', 'whose'
    ],
    
    # Temporal connectors to simplify
    'temporal_connectors': [
        'first of all', 'to begin with', 'initially', 'at first',
        'after that', 'subsequently', 'then', 'next',
        'finally', 'in the end', 'lastly', 'ultimately'
    ]
}

# Performance Optimization Settings
OPTIMIZATION_CONFIG = {
    # Enable FlashText for keyword replacement (faster than regex for many keywords)
    'enable_flashtext': True,
    
    # Enable Numba JIT compilation for numerical operations
    'enable_numba': True,
    
    # Enable result caching based on text hash
    'enable_result_caching': True,
    
    # Enable pattern caching for frequently used regex patterns
    'enable_pattern_caching': True,
    
    # Maximum size of result cache (number of cached results)
    'result_cache_size': 512,
    
    # Enable performance statistics collection
    'enable_stats': True,
    
    # Enable memory-efficient processing for large texts
    'enable_memory_optimization': True
}

# spaCy Configuration
SPACY_CONFIG = {
    # spaCy model to use (install with: python -m spacy download MODEL_NAME)
    'model_name': 'en_core_web_sm',
    
    # Disable unused spaCy components for better performance
    'disable_components': ['parser', 'tagger', 'ner'],
    
    # Maximum document length for spaCy processing
    'max_length': 1000000,
    
    # Enable GPU acceleration if available
    'enable_gpu': False,
    
    # Batch size for spaCy pipe processing
    'batch_size': 1000,
    
    # Number of processes for parallel spaCy processing
    'n_process': 2
}

# Async Processing Settings
ASYNC_CONFIG = {
    # Enable asynchronous file processing
    'enable_async_files': True,
    
    # Maximum number of concurrent file operations
    'max_concurrent_files': 10,
    
    # Timeout for async operations (seconds)
    'async_timeout': 30,
    
    # Buffer size for async file reading
    'file_buffer_size': 8192
}

# Logging and Debugging
DEBUG_CONFIG = {
    # Enable detailed logging
    'enable_logging': False,
    
    # Log level (DEBUG, INFO, WARNING, ERROR)
    'log_level': 'INFO',
    
    # Enable performance profiling
    'enable_profiling': False,
    
    # Save performance reports to file
    'save_reports': True,
    
    # Directory for saving reports and logs
    'output_directory': './reports'
}

# Memory Management
MEMORY_CONFIG = {
    # Enable garbage collection optimization
    'enable_gc_optimization': True,
    
    # Frequency of manual garbage collection (0 = disabled)
    'gc_frequency': 100,
    
    # Enable memory profiling
    'enable_memory_profiling': False,
    
    # Warning threshold for memory usage (MB)
    'memory_warning_threshold': 500,
    
    # Maximum memory usage before forcing cleanup (MB)
    'memory_limit': 1000
}

# Export all configurations
CONFIG = {
    'processing': PROCESSING_CONFIG,
    'parallel': PARALLEL_CONFIG,
    'patterns': PATTERN_CONFIG,
    'optimization': OPTIMIZATION_CONFIG,
    'spacy': SPACY_CONFIG,
    'async': ASYNC_CONFIG,
    'debug': DEBUG_CONFIG,
    'memory': MEMORY_CONFIG
}

# Configuration validation
def validate_config():
    """Validate configuration settings and warn about potential issues."""
    warnings = []
    
    # Check cache sizes
    if CONFIG['processing']['cache_size'] > 2048:
        warnings.append("Large cache size may consume significant memory")
    
    # Check parallel settings
    if CONFIG['parallel']['max_workers'] > 8:
        warnings.append("High worker count may cause resource contention")
    
    # Check memory limits
    if CONFIG['memory']['memory_limit'] < 100:
        warnings.append("Memory limit too low, may cause frequent cleanup")
    
    return warnings

# Helper function to get configuration
def get_config(section: Optional[str] = None):
    """Get configuration section or entire config."""
    if section:
        return CONFIG.get(section, {})
    return CONFIG

# Helper function to update configuration
def update_config(section: str, key: str, value):
    """Update a specific configuration value."""
    if section in CONFIG:
        CONFIG[section][key] = value
    else:
        raise ValueError(f"Unknown configuration section: {section}")

# Profile-based configurations for different use cases
PROFILES = {
    'development': {
        'processing': {'cache_size': 256, 'enable_spacy': True},
        'parallel': {'max_workers': 2, 'parallel_threshold': 5},
        'debug': {'enable_logging': True, 'log_level': 'DEBUG'}
    },
    
    'production': {
        'processing': {'cache_size': 2048, 'enable_spacy': True},
        'parallel': {'max_workers': 8, 'parallel_threshold': 20},
        'optimization': {'enable_result_caching': True, 'enable_stats': True},
        'debug': {'enable_logging': True, 'log_level': 'WARNING'}
    },
    
    'high_performance': {
        'processing': {'cache_size': 4096, 'enable_spacy': True},
        'parallel': {'max_workers': 16, 'parallel_threshold': 50},
        'optimization': {'enable_flashtext': True, 'enable_numba': True},
        'spacy': {'n_process': 4, 'batch_size': 2000}
    },
    
    'memory_efficient': {
        'processing': {'cache_size': 128, 'enable_spacy': False},
        'parallel': {'max_workers': 2, 'pandas_threshold': 100},
        'memory': {'enable_gc_optimization': True, 'gc_frequency': 50},
        'optimization': {'enable_memory_optimization': True}
    }
}

def apply_profile(profile_name: str):
    """Apply a predefined configuration profile."""
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(PROFILES.keys())}")
    
    profile = PROFILES[profile_name]
    for section, settings in profile.items():
        if section in CONFIG:
            CONFIG[section].update(settings)
        else:
            CONFIG[section] = settings

# Default to development profile for safety
if __name__ == "__main__":
    # Validate default configuration
    warnings = validate_config()
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("Available configuration profiles:")
    for profile in PROFILES.keys():
        print(f"  - {profile}")
    
    print("\nTo use a profile:")
    print("from config import apply_profile")
    print("apply_profile('production')")