"""
LangChain Hybrid Reasoning System for Qwen2 1.5B
=================================================

Architecture:
- Rule-based problem decomposition (deterministic)
- LangChain custom tools for math, logic, and computation
- Qwen2 1.5B as NLP helper only (not planner)
- Transparent scratchpad for audit trail
- Deterministic selection when LLM fails

Principles:
1. Tools do the work, not the LLM
2. Keep context minimal per step
3. Rule-based routing > LLM routing
4. Explicit fallbacks for every decision
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# LangChain imports
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Tool imports
import sys
from pathlib import Path
tools_path = str(Path(__file__).parent.parent / "tools")
if tools_path not in sys.path:
    sys.path.insert(0, tools_path)

# Try to import tools, but don't fail if unavailable
try:
    from calculator import AdvancedCalculator
    HAS_CALCULATOR = True
except ImportError:
    HAS_CALCULATOR = False
    logger.warning("AdvancedCalculator not available, using basic eval")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PROBLEM TYPE DETECTION (Rule-Based)
# ============================================================================

class ProblemType(Enum):
    """Deterministic problem classification"""
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    LOGIC = "logic"
    PATTERN = "pattern"
    COMPARISON = "comparison"
    SCHEDULING = "scheduling"
    UNKNOWN = "unknown"


class RuleBasedClassifier:
    """
    Pure rule-based classifier - NO LLM calls.
    Fast, deterministic, transparent.
    """
    
    # Pattern rules (regex + keywords)
    RULES = {
        ProblemType.ARITHMETIC: [
            r'\d+\s*[\+\-\*\/\^]\s*\d+',  # Math expressions
            r'calculate|compute|sum|product|multiply|divide',
            r'how much|how many.*total',
        ],
        ProblemType.ALGEBRA: [
            r'solve.*equation|find.*x|variable',
            r'[a-z]\s*=|equation|expression',
        ],
        ProblemType.GEOMETRY: [
            r'\d+x\d+x\d+|cube|sphere|cylinder|pyramid',
            r'painted.*sides?|faces?|edges?|corners?',
            r'area|volume|perimeter|surface',
            r'geometric|shape',
        ],
        ProblemType.LOGIC: [
            r'if.*then|overtake|race|position',
            r'all.*except|true.*false|statement',
            r'riddle|puzzle',
        ],
        ProblemType.PATTERN: [
            r'sequence|series|pattern',
            r'\d+,\s*\d+,\s*\d+.*\?',
            r'next (number|term)',
        ],
        ProblemType.COMPARISON: [
            r'which.*(more|less|greater|smaller|larger)',
            r'compare|contrast|difference between',
        ],
        ProblemType.SCHEDULING: [
            r'machine[s]?.*process|order|sequence',
            r'maximize.*items|minimize.*time',
            r'schedule|arrange',
        ],
    }
    
    @staticmethod
    def classify(problem: str) -> Tuple[ProblemType, float]:
        """
        Classify problem using pattern matching.
        Returns (type, confidence).
        """
        problem_lower = problem.lower()
        
        scores = {}
        for ptype, patterns in RuleBasedClassifier.RULES.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, problem_lower, re.IGNORECASE):
                    score += 1
            if score > 0:
                scores[ptype] = score
        
        if not scores:
            return (ProblemType.UNKNOWN, 0.0)
        
        best_type = max(scores.items(), key=lambda x: x[1])
        confidence = min(best_type[1] / 3.0, 1.0)  # Normalize
        
        return (best_type[0], confidence)


# ============================================================================
# LANGCHAIN CUSTOM TOOLS (Deterministic)
# ============================================================================

@tool
def calculator_tool(expression: str) -> str:
    """
    Evaluate mathematical expressions. 
    Examples: '2+3', '12*(10-2)', 'sqrt(16)', 'sin(pi/2)'
    """
    try:
        if HAS_CALCULATOR:
            calc = AdvancedCalculator()
            result = calc.calculate(expression)
            return str(result)
        else:
            # Fallback to basic eval
            import math
            safe_dict = {
                '__builtins__': {},
                'abs': abs, 'round': round,
                'pow': pow, 'sqrt': math.sqrt,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'pi': math.pi, 'e': math.e,
            }
            result = eval(expression, safe_dict)
            return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def python_eval_tool(code: str) -> str:
    """
    Execute Python code safely. 
    Example: 'sum([1,2,3,4,5])'
    """
    try:
        # Safe eval with limited namespace
        allowed_names = {
            'sum': sum, 'len': len, 'max': max, 'min': min,
            'abs': abs, 'round': round, 'sorted': sorted,
            'range': range, 'list': list, 'dict': dict,
        }
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def geometry_tool(instruction: str) -> str:
    """
    Apply geometric formulas for cubes.
    Format: 'cube_edge_n=10' for edge cubes with 2 painted faces
    Formula: 12 * (n-2) for nxnxn cube
    """
    try:
        # Parse instruction like "cube_edge_n=10"
        if 'cube_edge' in instruction or '2' in instruction.lower() and 'face' in instruction.lower():
            # Extract n
            n_match = re.search(r'n=(\d+)', instruction)
            if not n_match:
                n_match = re.search(r'(\d+)x\d+x\d+', instruction)
            
            if n_match:
                n = int(n_match.group(1))
                result = 12 * (n - 2)
                return f"12 × ({n}-2) = 12 × {n-2} = {result}"
            else:
                return "Could not extract cube dimension"
        
        elif 'corner' in instruction.lower():
            return "8 (cubes always have 8 corners)"
        
        else:
            return f"Unknown geometry instruction: {instruction}"
    
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def pattern_matcher_tool(numbers: str) -> str:
    """
    Find patterns in number sequences.
    Input: comma-separated numbers like '2,4,8,16'
    """
    try:
        nums = [int(x.strip()) for x in numbers.split(',') if x.strip()]
        
        if len(nums) < 2:
            return "Need at least 2 numbers"
        
        # Check arithmetic sequence
        diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
        if len(set(diffs)) == 1:
            next_num = nums[-1] + diffs[0]
            return f"Arithmetic sequence (diff={diffs[0]}), next={next_num}"
        
        # Check geometric sequence
        if all(n != 0 for n in nums[:-1]):
            ratios = [nums[i+1] / nums[i] for i in range(len(nums)-1)]
            if all(abs(r - ratios[0]) < 0.001 for r in ratios):
                next_num = nums[-1] * ratios[0]
                return f"Geometric sequence (ratio={ratios[0]}), next={int(next_num)}"
        
        # Check Fibonacci-like
        if len(nums) >= 3:
            is_fib = all(nums[i] == nums[i-1] + nums[i-2] for i in range(2, len(nums)))
            if is_fib:
                next_num = nums[-1] + nums[-2]
                return f"Fibonacci-like sequence, next={next_num}"
        
        return "No clear pattern detected"
    
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def logic_tracer_tool(statement: str) -> str:
    """
    Trace logical implications step by step.
    Example: 'You overtake 2nd place' -> 'You are now in 2nd place'
    """
    # Simple logic rules
    if "overtake" in statement.lower() and "second" in statement.lower():
        return "If you overtake 2nd place, you take their position → You are now 2nd"
    
    if "overtake" in statement.lower() and "first" in statement.lower():
        return "If you overtake 1st place, you take their position → You are now 1st"
    
    return "Logic trace: " + statement


# ============================================================================
# SCRATCHPAD MANAGER (Minimal Context)
# ============================================================================

class Scratchpad:
    """
    Minimal, append-only scratchpad for transparent audit trail.
    """
    
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
    
    def add(self, tool: str, input_data: str, output: str, step: int):
        """Add a tool execution record"""
        self.entries.append({
            'step': step,
            'tool': tool,
            'input': input_data[:200],  # Truncate
            'output': output[:200],     # Truncate
        })
    
    def get_summary(self) -> str:
        """Get concise summary for LLM if needed"""
        if not self.entries:
            return "No steps yet"
        
        summary = []
        for entry in self.entries:
            summary.append(f"Step {entry['step']}: {entry['tool']}('{entry['input']}') → {entry['output']}")
        
        return "\n".join(summary)
    
    def get_full_trace(self) -> List[str]:
        """Get detailed trace for output"""
        return [
            f"[{e['tool']}] {e['input']} → {e['output']}"
            for e in self.entries
        ]


# ============================================================================
# RULE-BASED DECOMPOSER (No LLM)
# ============================================================================

class DeterministicDecomposer:
    """
    Decomposes problems into tool-executable steps WITHOUT using LLM.
    Pure rule-based logic.
    """
    
    @staticmethod
    def decompose(problem: str, problem_type: ProblemType) -> List[Dict[str, str]]:
        """
        Generate step-by-step plan based on problem type.
        Returns list of {tool, instruction}
        """
        
        if problem_type == ProblemType.GEOMETRY:
            # Extract dimensions
            cube_match = re.search(r'(\d+)x\d+x\d+', problem)
            if cube_match:
                n = cube_match.group(1)
                
                if "two sides painted" in problem.lower() or "two faces" in problem.lower():
                    return [
                        {
                            'tool': 'geometry_tool',
                            'instruction': f'cube_edge_n={n}',
                            'description': f'Apply edge cube formula: 12×({n}-2) for 2 painted faces'
                        }
                    ]
        
        if problem_type == ProblemType.ARITHMETIC:
            # Extract mathematical expressions
            expr_match = re.search(r'\d+\s*[\+\-\*\/\^]\s*\d+', problem)
            if expr_match:
                expr = expr_match.group(0)
                return [
                    {
                        'tool': 'calculator_tool',
                        'instruction': expr,
                        'description': f'Calculate {expr}'
                    }
                ]
        
        if problem_type == ProblemType.PATTERN:
            # Extract number sequence
            numbers = re.findall(r'\d+', problem)
            if len(numbers) >= 3:
                seq = ','.join(numbers[:5])
                return [
                    {
                        'tool': 'pattern_matcher_tool',
                        'instruction': seq,
                        'description': 'Find pattern in sequence'
                    }
                ]
        
        if problem_type == ProblemType.LOGIC:
            return [
                {
                    'tool': 'logic_tracer_tool',
                    'instruction': problem[:200],
                    'description': 'Trace logical implications'
                }
            ]
        
        # Default: simple analysis
        return [
            {
                'tool': 'python_eval_tool',
                'instruction': 'None',
                'description': 'No clear decomposition available'
            }
        ]


# ============================================================================
# TOOL EXECUTOR
# ============================================================================

class ToolExecutor:
    """Execute tools with scratchpad tracking"""
    
    TOOLS = {
        'calculator_tool': calculator_tool,
        'python_eval_tool': python_eval_tool,
        'geometry_tool': geometry_tool,
        'pattern_matcher_tool': pattern_matcher_tool,
        'logic_tracer_tool': logic_tracer_tool,
    }
    
    @staticmethod
    def execute(tool_name: str, instruction: str, scratchpad: Scratchpad, step: int) -> str:
        """Execute tool and record in scratchpad"""
        try:
            if tool_name not in ToolExecutor.TOOLS:
                result = f"Unknown tool: {tool_name}"
            else:
                tool_func = ToolExecutor.TOOLS[tool_name]
                result = tool_func.invoke(instruction)
            
            scratchpad.add(tool_name, instruction, result, step)
            return result
        
        except Exception as e:
            error = f"Tool error: {str(e)}"
            scratchpad.add(tool_name, instruction, error, step)
            return error


# ============================================================================
# NLP HELPER (Qwen2 - Minimal Use)
# ============================================================================

class NLPHelper:
    """
    Qwen2 1.5B used ONLY for NLP tasks, not planning.
    Kept minimal and isolated.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=os.getenv("SMALL_LLM_API_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("SMALL_LLM_API_KEY", "lm-studio"),
            model=os.getenv("LLM__MODEL_NAME", "qwen2-1.5b-instruct"),
            temperature=0.1,
        )
    
    def select_option_nlp(self, problem: str, options: List[str], 
                         tool_results: str) -> int:
        """
        Use LLM ONLY for final option selection when results are unclear.
        Keep prompt minimal.
        """
        try:
            prompt = f"""Problem: {problem[:200]}

Tool Results: {tool_results[:300]}

Options:
{chr(10).join([f'{i+1}. {opt}' for i, opt in enumerate(options)])}

Select best option number (1-{len(options)}). Respond with ONLY the number."""

            messages = [
                SystemMessage(content="You select the best answer option. Respond with only a number."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Extract number
            match = re.search(r'\d+', content)
            if match:
                num = int(match.group())
                if 1 <= num <= len(options):
                    return num - 1  # 0-indexed
            
            # Fallback
            return 0
        
        except Exception as e:
            logger.error(f"NLP helper error: {e}")
            return 0


# ============================================================================
# HYBRID REASONING ENGINE
# ============================================================================

class HybridReasoningEngine:
    """
    Main engine: Rule-based + LangChain tools + minimal LLM.
    """
    
    def __init__(self):
        self.classifier = RuleBasedClassifier()
        self.decomposer = DeterministicDecomposer()
        self.executor = ToolExecutor()
        self.nlp_helper = NLPHelper()
    
    def reason(self, problem: str, options: List[str]) -> Tuple[int, List[str], Dict]:
        """
        Main reasoning pipeline.
        
        Returns:
            (selected_index, trace_list, execution_log)
        """
        log = {
            'classification': {},
            'steps': [],
            'selection': {},
        }
        
        scratchpad = Scratchpad()
        
        # Step 1: Rule-based classification
        problem_type, confidence = self.classifier.classify(problem)
        log['classification'] = {
            'type': problem_type.value,
            'confidence': confidence,
            'method': 'rule-based'
        }
        
        logger.info(f"Classified as {problem_type.value} (confidence: {confidence:.2f})")
        
        # Step 2: Deterministic decomposition
        steps = self.decomposer.decompose(problem, problem_type)
        log['steps'] = steps
        
        logger.info(f"Decomposed into {len(steps)} steps")
        
        # Step 3: Execute tools
        for i, step in enumerate(steps, 1):
            tool_name = step['tool']
            instruction = step['instruction']
            
            logger.info(f"Step {i}: {tool_name}({instruction[:50]}...)")
            
            result = self.executor.execute(tool_name, instruction, scratchpad, i)
            
            logger.info(f"Result: {result[:100]}...")
        
        # Step 4: Deterministic selection (rule-based first)
        selected_idx = self._rule_based_selection(scratchpad, options)
        
        if selected_idx is None:
            # Fallback to NLP helper
            logger.info("Using NLP helper for selection")
            tool_summary = scratchpad.get_summary()
            selected_idx = self.nlp_helper.select_option_nlp(problem, options, tool_summary)
            log['selection']['method'] = 'nlp_helper'
        else:
            log['selection']['method'] = 'rule_based'
        
        log['selection']['index'] = selected_idx
        log['selection']['option'] = options[selected_idx] if selected_idx < len(options) else 'Error'
        
        # Build trace
        trace_list = scratchpad.get_full_trace()
        
        return (selected_idx, trace_list, log)
    
    def _rule_based_selection(self, scratchpad: Scratchpad, options: List[str]) -> Optional[int]:
        """
        Try to select option using deterministic rules from tool outputs.
        Returns None if unclear.
        """
        if not scratchpad.entries:
            return None
        
        # Get all numbers from tool outputs (rightmost = final result)
        output_numbers = []
        for entry in scratchpad.entries:
            numbers = re.findall(r'\d+', entry['output'])
            if numbers:
                output_numbers.extend(numbers)
        
        # Check for exact matches with options
        for entry in scratchpad.entries:
            output = entry['output'].lower()
            
            for i, option in enumerate(options):
                option_clean = option.strip().lower()
                
                # Direct string match
                if option_clean in output or output in option_clean:
                    logger.info(f"Rule-based string match: option {i+1}")
                    return i
        
        # Check if final computed number matches an option exactly
        if output_numbers:
            final_num = output_numbers[-1]  # Rightmost number is usually the answer
            for i, option in enumerate(options):
                option_stripped = option.strip()
                if option_stripped == final_num or option_stripped == str(final_num):
                    logger.info(f"Rule-based number match: option {i+1} = {final_num}")
                    return i
        
        # Check for keyword matches (logic problems)
        for entry in scratchpad.entries:
            output_lower = entry['output'].lower()
            for i, option in enumerate(options):
                option_lower = option.lower()
                
                # For logic: "you are now 2nd" should match "Second"
                if ('now 2nd' in output_lower or 'now second' in output_lower) and 'second' in option_lower:
                    logger.info(f"Logic keyword match 'second': option {i+1}")
                    return i
                
                if ('now 1st' in output_lower or 'now first' in output_lower) and 'first' in option_lower:
                    logger.info(f"Logic keyword match 'first': option {i+1}")
                    return i
                
                if ('now 3rd' in output_lower or 'now third' in output_lower) and 'third' in option_lower:
                    logger.info(f"Logic keyword match 'third': option {i+1}")
                    return i
        
        return None


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    engine = HybridReasoningEngine()
    
    # Test 1: Geometry
    problem1 = "A 10x10x10 cube is painted red, then cut into 1x1x1 cubes. How many have exactly two sides painted?"
    options1 = ["88", "96", "104", "112", "Another"]
    
    print("=" * 80)
    print("TEST 1: Geometry")
    print(f"Problem: {problem1}")
    print()
    
    idx, trace, log = engine.reason(problem1, options1)
    
    print(f"Classification: {log['classification']}")
    print(f"Selected: Option {idx+1} - {options1[idx]}")
    print(f"Trace: {trace}")
    print()
    
    # Test 2: Logic
    problem2 = "If you overtake the person in second place, what position are you in?"
    options2 = ["First", "Second", "Third", "Depends", "Another"]
    
    print("=" * 80)
    print("TEST 2: Logic")
    print(f"Problem: {problem2}")
    print()
    
    idx, trace, log = engine.reason(problem2, options2)
    
    print(f"Classification: {log['classification']}")
    print(f"Selected: Option {idx+1} - {options2[idx]}")
    print(f"Trace: {trace}")
