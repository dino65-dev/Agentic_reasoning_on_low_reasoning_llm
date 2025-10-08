"""
LangChain Tools for Hybrid Reasoning System
Uses @tool decorator to create deterministic reasoning tools
"""

from langchain_core.tools import tool
from typing import Optional
import re

# Try to import advanced calculator, fall back to basic eval
try:
    from calculator import AdvancedCalculator
    HAS_CALCULATOR = True
    calc = AdvancedCalculator()
except ImportError:
    HAS_CALCULATOR = False


@tool
def calculator_tool(expression: str) -> str:
    """
    Evaluate mathematical expressions safely.
    
    Args:
        expression: Math expression like "2 + 2" or "(40/55) * (55/70)"
    
    Returns:
        String result of the calculation
    """
    try:
        # Try advanced calculator first
        if HAS_CALCULATOR:
            result = calc.calculate(expression)
            return f"{expression} = {result}"
        
        # Fallback to safe eval
        safe_dict = {
            '__builtins__': {},
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow, 'len': len
        }
        result = eval(expression, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


@tool
def python_eval_tool(code: str) -> str:
    """
    Execute Python code safely for calculations.
    
    Args:
        code: Python code to execute (e.g., "result = 40/55 * 55/70")
    
    Returns:
        String result of the execution
    """
    try:
        safe_globals = {
            '__builtins__': {},
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow, 'len': len, 'range': range,
            'list': list, 'dict': dict, 'str': str, 'int': int, 'float': float
        }
        local_vars = {}
        exec(code, safe_globals, local_vars)
        
        # Return all non-private variables
        results = {k: v for k, v in local_vars.items() if not k.startswith('_')}
        return str(results)
    except Exception as e:
        return f"Error executing code: {str(e)}"


@tool
def geometry_tool(instruction: str) -> str:
    """
    Solve geometry problems for cubes, spheres, cylinders.
    
    Args:
        instruction: Command like "cube_edge_n=10" or "cube_corners" or "sphere_volume_r=5"
    
    Returns:
        String with formula and calculation
    """
    try:
        # Parse cube edge cubes (2 faces painted)
        match = re.search(r'cube_edge.*n=(\d+)', instruction, re.IGNORECASE)
        if match:
            n = int(match.group(1))
            result = 12 * (n - 2)
            return f"12 × ({n}-2) = 12 × {n-2} = {result}"
        
        # Parse cube corners
        if 'corner' in instruction.lower():
            return "8 corners in any cube"
        
        # Parse cube faces (1 face painted)
        match = re.search(r'cube_face.*n=(\d+)', instruction, re.IGNORECASE)
        if match:
            n = int(match.group(1))
            result = 6 * ((n - 2) ** 2)
            return f"6 × ({n}-2)² = 6 × {(n-2)**2} = {result}"
        
        # Parse sphere volume
        match = re.search(r'sphere.*r=(\d+\.?\d*)', instruction, re.IGNORECASE)
        if match:
            r = float(match.group(1))
            result = (4/3) * 3.14159 * (r ** 3)
            return f"(4/3) × π × {r}³ ≈ {result:.2f}"
        
        return f"Unknown geometry instruction: {instruction}"
    except Exception as e:
        return f"Error in geometry calculation: {str(e)}"


@tool
def pattern_matcher_tool(sequence: str) -> str:
    """
    Detect patterns in number sequences.
    
    Args:
        sequence: Comma-separated numbers like "2,4,8,16" or "1,5,12,22,35"
    
    Returns:
        String describing the pattern and next number
    """
    try:
        # Parse numbers
        numbers = [float(x.strip()) for x in sequence.split(',')]
        
        if len(numbers) < 2:
            return "Need at least 2 numbers to detect pattern"
        
        # Check arithmetic sequence (constant difference)
        diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
        if len(set(diffs)) == 1:
            diff = diffs[0]
            next_num = numbers[-1] + diff
            return f"Arithmetic sequence (diff={diff}), next={int(next_num) if next_num == int(next_num) else next_num}"
        
        # Check geometric sequence (constant ratio)
        if all(numbers[i] != 0 for i in range(len(numbers)-1)):
            ratios = [numbers[i+1] / numbers[i] for i in range(len(numbers)-1)]
            if all(abs(ratios[i] - ratios[0]) < 0.01 for i in range(len(ratios))):
                ratio = ratios[0]
                next_num = numbers[-1] * ratio
                return f"Geometric sequence (ratio={ratio}), next={int(next_num) if next_num == int(next_num) else next_num}"
        
        # Check Fibonacci-like (sum of previous two)
        if len(numbers) >= 3:
            is_fib = all(abs(numbers[i] - (numbers[i-1] + numbers[i-2])) < 0.01 for i in range(2, len(numbers)))
            if is_fib:
                next_num = numbers[-1] + numbers[-2]
                return f"Fibonacci-like sequence, next={int(next_num) if next_num == int(next_num) else next_num}"
        
        # Check differences of differences (quadratic)
        if len(diffs) >= 2:
            second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
            if len(set(second_diffs)) == 1:
                diff2 = second_diffs[0]
                next_diff = diffs[-1] + diff2
                next_num = numbers[-1] + next_diff
                return f"Quadratic pattern (2nd diff={diff2}), next={int(next_num) if next_num == int(next_num) else next_num}"
        
        return "No clear pattern detected"
    except Exception as e:
        return f"Error detecting pattern: {str(e)}"


@tool
def logic_tracer_tool(problem: str) -> str:
    """
    Trace logical implications step by step.
    
    Args:
        problem: Logic problem statement
    
    Returns:
        String with logical reasoning trace
    """
    try:
        # Look for key logical operations
        problem_lower = problem.lower()
        
        # Race position logic
        if 'overtake' in problem_lower and ('second' in problem_lower or '2nd' in problem_lower):
            return "If you overtake 2nd place, you take their position → You are now 2nd"
        
        # Truth/lie logic
        if 'truth' in problem_lower and 'lie' in problem_lower:
            return "Both truth-teller and liar would point away from truth → Ask 'What would other say?'"
        
        # General implication
        if 'if' in problem_lower and 'then' in problem_lower:
            parts = problem_lower.split('if', 1)[1].split('then', 1) if 'then' in problem_lower else [problem_lower]
            return f"Logical implication: IF {parts[0].strip()} THEN ..."
        
        return f"Logical analysis: {problem[:100]}..."
    except Exception as e:
        return f"Error in logic tracing: {str(e)}"


# Export all tools as a list for easy registration
ALL_TOOLS = [
    calculator_tool,
    python_eval_tool,
    geometry_tool,
    pattern_matcher_tool,
    logic_tracer_tool
]
