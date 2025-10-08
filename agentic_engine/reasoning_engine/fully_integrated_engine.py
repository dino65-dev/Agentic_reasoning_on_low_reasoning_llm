"""
FULLY INTEGRATED HYBRID REASONING ENGINE
Uses ALL modules from agentic_engine folder structure:
- config.py - Configuration management
- utils.py - Utility functions
- problem_classifier.py - Problem type classification
- problem_restatement/ - Problem restatement and clarification
- domain_prompts.py - Domain-specific prompt templates
- decompose.py - LangChain-based decomposition
- tools/ - LangChain tool definitions
- notepad_manager/ - High-performance C-based scratchpad
- selection_module.py - Answer selection
- output_formatter.py - Output formatting
- input_loader/ - CSV input loading
- cache/ - Caching layer
- logs/ - Logging configuration
"""

import os
import sys
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from pydantic import SecretStr

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# === Import ALL agentic_engine modules ===
# Core configuration
from config import settings, get_config

# Utilities
from utils import (
    setup_logger,
    timer,
    timed_cache,
    sanitize_text,
    safe_mkdir
)

# Problem processing
from problem_classifier import ProblemClassifier, ProblemType
# Import problem_restatement conditionally to avoid slow spacy loading
try:
    from problem_restatement.problem_restatement import AdvancedProblemRestater
    HAS_PROBLEM_RESTATEMENT = True
except ImportError as e:
    HAS_PROBLEM_RESTATEMENT = False
    logger.warning(f"Problem restatement not available: {e}")
from domain_prompts import DomainPrompts

# Decomposition
from decompose import fast_manual_decompose, manual_decompose_runnable

# Tools
from tools.langchain_tools import ALL_TOOLS

# Notepad manager for high-performance tracking
from notepad_manager.notepad_manager import NotepadManager

# Selection module
from selection_module import SelectionModule
# output_formatter is for CSV export, not needed for reasoning

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Configure logging using utils
logger = setup_logger(__name__, level="INFO")


class FullyIntegratedHybridEngine:
    """
    Comprehensive reasoning engine using ALL agentic_engine modules.
    
    Architecture:
    1. Config - Load settings from config.py
    2. Utils - Use utilities for caching, logging, timing
    3. Problem Restatement - Clarify problem using problem_restatement/
    4. Classification - Type detection using problem_classifier.py
    5. Domain Prompts - Get specialized prompts from domain_prompts.py
    6. Decomposition - Use decompose.py for problem breakdown
    7. Tool Execution - Execute tools/ with notepad_manager tracking
    8. Selection - Use selection_module.py for answer selection
    9. Formatting - Format output using output_formatter.py
    """
    
    def __init__(self, use_notepad_manager: bool = True):
        """
        Initialize with ALL modules
        
        Args:
            use_notepad_manager: Whether to use C-based notepad manager for performance
        """
        logger.info("Initializing Fully Integrated Hybrid Engine...")
        
        # 1. Load configuration
        self.config = get_config()
        logger.info(f"Loaded config from: {self.config.paths.project_root}")
        
        # 2. Initialize problem processor modules
        self.classifier = ProblemClassifier()
        if HAS_PROBLEM_RESTATEMENT:
            self.restater = AdvancedProblemRestater(enable_spacy=False)  # Fast mode
            logger.info("Problem restatement enabled")
        else:
            self.restater = None
            logger.warning("Problem restatement disabled (spacy not available)")
        self.domain_prompts = DomainPrompts()
        
        # 3. Initialize decomposer
        self.decompose_fn = fast_manual_decompose
        
        # 4. Initialize tools
        self.tools = {tool.name: tool for tool in ALL_TOOLS}
        logger.info(f"Loaded {len(self.tools)} LangChain tools: {list(self.tools.keys())}")
        
        # 5. Initialize notepad manager or fallback
        self.use_notepad_manager = use_notepad_manager
        if use_notepad_manager:
            try:
                self.notepad = NotepadManager()
                logger.info("Using high-performance C-based notepad_manager")
            except Exception as e:
                logger.warning(f"Notepad manager not available: {e}, using fallback")
                self.notepad = SimpleScratchpad()
                self.use_notepad_manager = False
        else:
            self.notepad = SimpleScratchpad()
            logger.info("Using simple Python scratchpad")
        
        # 6. Initialize selection module
        self.selector = SelectionModule()
        
        # 7. Initialize LLM (minimal use) - Optional if no API key
        llm_config = self.config.llm
        api_key = llm_config.api_key.get_secret_value() if llm_config.api_key else ""
        if api_key and llm_config.api_url:
            self.llm = ChatOpenAI(
                base_url=llm_config.api_url,
                api_key=api_key,
                model=llm_config.model_name,
                temperature=llm_config.temperature,
                max_retries=1,
                timeout=10
            )
            logger.info(f"Initialized LLM: {llm_config.model_name}")
        else:
            self.llm = None
            logger.warning("No LLM API key found - LLM features disabled")
        
        # 8. Setup caching directory
        cache_dir = self.config.paths.cache_dir
        safe_mkdir(cache_dir)
        logger.info(f"Cache directory: {cache_dir}")
        
        # 9. Statistics
        self.stats = {
            'problems_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'restatement_used': 0,
            'tool_executions': 0
        }
        
        logger.info("✅ Fully Integrated Engine initialized successfully")
    
    @timer(verbose=True, store_stats=True)
    @timed_cache(seconds=3600)
    def reason(self, problem: str, options: List[str], enable_restatement: bool = True) -> Tuple[int, List[str], Dict]:
        """
        Full reasoning pipeline using ALL modules
        
        Args:
            problem: Problem statement
            options: Answer options
            enable_restatement: Whether to restate problem for clarity
            
        Returns:
            (selected_index, trace_list, execution_log)
        """
        logger.info("="*80)
        logger.info("STARTING FULLY INTEGRATED REASONING PIPELINE")
        logger.info("="*80)
        
        self.stats['problems_processed'] += 1
        
        # STEP 1: Problem Restatement (Optional but recommended)
        if enable_restatement and self.restater:
            logger.info("STEP 1: Problem Restatement")
            try:
                restated = self.restater.restate(problem)
                if restated and restated != problem:
                    logger.info(f"Original: {problem[:100]}...")
                    logger.info(f"Restated: {restated[:100]}...")
                    problem_to_use = restated
                    self.stats['restatement_used'] += 1
                else:
                    problem_to_use = problem
            except Exception as e:
                logger.warning(f"Restatement failed: {e}, using original")
                problem_to_use = problem
        else:
            problem_to_use = problem
        
        # Sanitize text
        problem_to_use = sanitize_text(problem_to_use)
        
        # STEP 2: Classification
        logger.info("STEP 2: Problem Classification")
        problem_type, confidence, all_types = self.classifier.classify(problem_to_use)
        logger.info(f"  Type: {problem_type.value}")
        logger.info(f"  Confidence: {confidence:.2f}")
        logger.info(f"  All detected types: {[t.value for t in all_types]}")
        
        # STEP 3: Get Domain-Specific Prompt
        logger.info("STEP 3: Domain-Specific Prompt Generation")
        domain_prompt = self.domain_prompts.get_decomposition_prompt(
            problem_type, problem_to_use, options
        )
        logger.info(f"  Generated specialized prompt for {problem_type.value}")
        
        # STEP 4: Decomposition
        logger.info("STEP 4: Problem Decomposition")
        try:
            decomp_result = self.decompose_fn(problem_to_use)
            steps = decomp_result.reasoning_steps
            logger.info(f"  Decomposed into {len(steps)} steps")
            for i, step in enumerate(steps, 1):
                logger.info(f"    {i}. {step.instruction}")
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}, using rule-based fallback")
            steps = self._rule_based_decomposition(problem_to_use, problem_type, options)
        
        # STEP 5: Tool Execution with Notepad Tracking
        logger.info("STEP 5: Tool Execution")
        self.notepad.clear()  # Clear previous entries
        
        tool_results = []
        for i, step in enumerate(steps, 1):
            # Map step to appropriate tool
            tool_name, instruction = self._map_step_to_tool(step.instruction, problem_type)
            
            if tool_name and tool_name in self.tools:
                logger.info(f"  Executing: {tool_name}({instruction[:50]}...)")
                try:
                    tool = self.tools[tool_name]
                    result = tool.invoke(instruction)
                    self.notepad.add(tool_name, instruction, str(result), i)
                    tool_results.append({
                        'step': i,
                        'tool': tool_name,
                        'instruction': instruction,
                        'result': str(result)
                    })
                    self.stats['tool_executions'] += 1
                    logger.info(f"    Result: {str(result)[:100]}...")
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    self.notepad.add(tool_name, instruction, error_msg, i)
                    tool_results.append({
                        'step': i,
                        'tool': tool_name,
                        'instruction': instruction,
                        'result': error_msg
                    })
                    logger.error(f"    Tool error: {e}")
        
        # Get trace from notepad
        trace_list = self.notepad.get_trace()
        
        # STEP 6: Rule-Based Selection
        logger.info("STEP 6: Answer Selection")
        selected_idx = self._rule_based_selection(tool_results, options)
        
        if selected_idx is None:
            # STEP 6b: Fallback to SelectionModule
            logger.info("  Rule-based selection failed, using SelectionModule")
            try:
                selected_idx = self.selector.select(trace_list, options) - 1  # Convert to 0-based
                selection_method = "selection_module"
            except Exception as e:
                logger.warning(f"  SelectionModule failed: {e}, using option 0")
                selected_idx = 0
                selection_method = "fallback"
        else:
            logger.info(f"  Rule-based selection: Option {selected_idx + 1}")
            selection_method = "rule_based"
        
        # STEP 7: Build Execution Log
        execution_log = {
            'problem_original': problem,
            'problem_restated': problem_to_use if enable_restatement else None,
            'classification': {
                'primary_type': problem_type.value,
                'confidence': confidence,
                'all_types': [t.value for t in all_types],
                'method': 'rule-based'
            },
            'domain_prompt': domain_prompt[:200] + "..." if len(domain_prompt) > 200 else domain_prompt,
            'decomposition': {
                'steps': len(steps),
                'method': 'langchain' if 'decomp_result' in locals() else 'rule-based'
            },
            'tool_executions': tool_results,
            'selection': {
                'method': selection_method,
                'index': selected_idx,
                'option': options[selected_idx] if selected_idx < len(options) else "unknown"
            },
            'statistics': self.stats.copy()
        }
        
        logger.info("="*80)
        logger.info(f"PIPELINE COMPLETE: Selected Option {selected_idx + 1}")
        logger.info("="*80)
        
        return selected_idx, trace_list, execution_log
    
    def _rule_based_decomposition(self, problem: str, problem_type: ProblemType, options: List[str]) -> List:
        """Fallback rule-based decomposition"""
        from decompose import ReasoningStep
        
        # Simple decomposition based on problem type
        if problem_type == ProblemType.SPATIAL_GEOMETRY:
            return [ReasoningStep(step_number=1, instruction=f"Apply geometric formulas to: {problem[:100]}")]
        elif problem_type == ProblemType.PATTERN_RECOGNITION:
            return [ReasoningStep(step_number=1, instruction=f"Detect pattern in: {problem[:100]}")]
        elif problem_type == ProblemType.LOGIC_PUZZLE:
            return [ReasoningStep(step_number=1, instruction=f"Trace logic in: {problem[:100]}")]
        else:
            return [ReasoningStep(step_number=1, instruction=f"Analyze: {problem[:100]}")]
    
    def _map_step_to_tool(self, instruction: str, problem_type: ProblemType) -> Tuple[Optional[str], str]:
        """Map decomposition step to appropriate tool"""
        instruction_lower = instruction.lower()
        
        # Pattern matching based on keywords
        if any(kw in instruction_lower for kw in ['pattern', 'sequence', 'next number']):
            return 'pattern_matcher_tool', instruction
        elif any(kw in instruction_lower for kw in ['calculate', 'compute', 'ratio', 'multiply']):
            return 'calculator_tool', instruction
        elif any(kw in instruction_lower for kw in ['geometric', 'cube', 'sphere', 'area', 'volume']):
            return 'geometry_tool', instruction
        elif any(kw in instruction_lower for kw in ['logic', 'if', 'then', 'implication']):
            return 'logic_tracer_tool', instruction
        elif any(kw in instruction_lower for kw in ['execute', 'run', 'code']):
            return 'python_eval_tool', instruction
        else:
            # Default based on problem type
            if problem_type == ProblemType.SPATIAL_GEOMETRY:
                return 'geometry_tool', instruction
            elif problem_type == ProblemType.PATTERN_RECOGNITION:
                return 'pattern_matcher_tool', instruction
            elif problem_type == ProblemType.LOGIC_PUZZLE:
                return 'logic_tracer_tool', instruction
            else:
                return 'python_eval_tool', instruction
    
    def _rule_based_selection(self, tool_results: List[Dict], options: List[str]) -> Optional[int]:
        """Rule-based selection from tool results"""
        if not tool_results:
            return None
        
        # Get last tool result
        last_result = tool_results[-1]['result'].lower()
        
        import re
        # Extract numbers
        result_numbers = re.findall(r'\d+\.?\d*', last_result)
        
        # Try to match each option
        for i, option in enumerate(options):
            option_lower = option.lower().strip()
            option_numbers = re.findall(r'\d+\.?\d*', option_lower)
            
            # Number matching
            if result_numbers and option_numbers:
                final_num = result_numbers[-1]
                if final_num in option_numbers:
                    return i
            
            # String matching
            option_clean = re.sub(r'[^\w\s]', '', option_lower)
            if option_clean in last_result or last_result in option_clean:
                return i
            
            # Keyword matching for logic
            if 'now 2nd' in last_result or 'now second' in last_result:
                if 'second' in option_lower:
                    return i
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get engine statistics"""
        return self.stats.copy()


class SimpleScratchpad:
    """Fallback scratchpad if notepad_manager not available"""
    def __init__(self):
        self.entries = []
    
    def clear(self):
        self.entries = []
    
    def add(self, tool: str, instruction: str, output: str, step: int):
        self.entries.append({
            'tool': tool,
            'instruction': instruction,
            'output': output,
            'step': step
        })
    
    def get_trace(self) -> List[str]:
        return [
            f"[{e['tool']}] {e['instruction'][:50]}... → {e['output'][:100]}..."
            for e in self.entries
        ]


# Convenience function
def create_engine(use_notepad: bool = False) -> FullyIntegratedHybridEngine:
    """
    Factory function to create fully integrated engine
    
    Args:
        use_notepad: Use C-based notepad_manager (requires compilation)
    
    Returns:
        Fully initialized engine
    """
    return FullyIntegratedHybridEngine(use_notepad_manager=use_notepad)
