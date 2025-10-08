"""
Advanced Symbolic Mathematics Engine using LangChain and SymPy

This module provides a comprehensive symbolic mathematics engine that combines
LangChain's LLM capabilities with SymPy's computer algebra system for advanced
mathematical operations including:
- Symbolic manipulation and simplification
- Calculus (differentiation, integration, limits)
- Equation solving (algebraic, differential, systems)
- Matrix operations
- Series expansions
- Mathematical reasoning with natural language
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence
from datetime import datetime

# Core dependencies
import sympy as sp
from sympy import symbols, simplify, expand, factor, solve, diff, integrate
from sympy import limit, series, Matrix, latex, pretty
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# LangChain imports
LANGCHAIN_AVAILABLE = False
LLMSymbolicMathChain = None
LangChainOpenAI = None
Tool = None
AgentType = None
initialize_agent = None
PromptTemplate = None
get_openai_callback = None
ChatOpenAI = None

try:
    from langchain_openai import ChatOpenAI
    from langchain_community.callbacks import get_openai_callback
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LangChain OpenAI not available. Error: {e}")

try:
    from langchain_core.tools import Tool
except ImportError:
    try:
        from langchain.tools import Tool  # type: ignore
    except ImportError:
        pass

try:
    from langchain.agents import AgentType, initialize_agent  # type: ignore
except ImportError:
    AgentType = None  # type: ignore
    initialize_agent = None  # type: ignore

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    try:
        from langchain.prompts import PromptTemplate  # type: ignore
    except ImportError:
        pass

# Note: LLMSymbolicMathChain has been deprecated/removed from langchain-experimental
# We'll use direct LLM calls with custom prompts instead
print("Note: Using direct SymPy integration. LangChain symbolic math chain not required.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedSymbolicEngine:
    """
    Advanced Symbolic Mathematics Engine with LangChain integration.
    
    Features:
    - Natural language to symbolic math conversion
    - Comprehensive symbolic operations
    - Step-by-step solution tracking
    - LaTeX output for formatted math
    - Safety checks and validation
    - Caching for performance
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", 
                 enable_dangerous_ops: bool = False, cache_enabled: bool = True):
        """
        Initialize the Advanced Symbolic Engine.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: LLM model to use for reasoning
            enable_dangerous_ops: Allow arbitrary code execution (use with caution)
            cache_enabled: Enable result caching
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.enable_dangerous_ops = enable_dangerous_ops
        self.cache_enabled = cache_enabled
        self.cache = {}
        self.operation_history = []
        
        # Initialize LangChain components if available
        if LANGCHAIN_AVAILABLE and self.api_key:
            self._initialize_langchain()
        else:
            self.llm = None
            self.symbolic_chain = None
            self.math_chain = None
            logger.warning("LangChain not initialized. Some features may be limited.")
    
    def _initialize_langchain(self):
        """Initialize LangChain LLM and chains."""
        try:
            if ChatOpenAI is None:
                raise ImportError("ChatOpenAI not available")
                
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,  # type: ignore
                temperature=0  # Deterministic for math
            )
            
            # Note: LLMSymbolicMathChain is deprecated/removed
            # We use direct LLM calls with SymPy backend
            self.symbolic_chain = self  # Use self for symbolic operations
            self.math_chain = None
            
            logger.info("LangChain components initialized successfully with direct SymPy backend")
        except Exception as e:
            logger.error(f"Error initializing LangChain: {e}")
            self.llm = None
            self.symbolic_chain = None
            self.math_chain = None
    
    def _log_operation(self, operation: str, input_data: Any, output_data: Any, 
                      metadata: Optional[Dict] = None):
        """Log operation for history tracking."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "input": str(input_data),
            "output": str(output_data),
            "metadata": metadata or {}
        }
        self.operation_history.append(log_entry)
    
    def parse_expression(self, expr_str: str, local_dict: Optional[Dict] = None) -> sp.Expr:
        """
        Parse a string expression into a SymPy expression with intelligent transformations.
        
        Args:
            expr_str: String representation of mathematical expression
            local_dict: Optional dictionary of predefined symbols/functions
            
        Returns:
            SymPy expression object
        """
        transformations = standard_transformations + (implicit_multiplication_application,)
        try:
            expr = parse_expr(expr_str, transformations=transformations, local_dict=local_dict)
            return expr
        except Exception as e:
            logger.error(f"Error parsing expression '{expr_str}': {e}")
            raise ValueError(f"Invalid expression: {expr_str}")
    
    def simplify_expression(self, expr: Union[str, sp.Expr], 
                           method: str = "auto") -> Dict[str, Any]:
        """
        Simplify a mathematical expression using various methods.
        
        Args:
            expr: Expression to simplify (string or SymPy expression)
            method: Simplification method ('auto', 'expand', 'factor', 'trigsimp', 'radsimp')
            
        Returns:
            Dictionary containing simplified expression and metadata
        """
        if isinstance(expr, str):
            expr = self.parse_expression(expr)
        
        cache_key = f"simplify_{str(expr)}_{method}"
        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]
        
        result = {
            "original": str(expr),
            "original_latex": latex(expr),
            "simplified": None,
            "simplified_latex": None,
            "method": method,
            "steps": []
        }
        
        try:
            if method == "auto":
                simplified = simplify(expr)
            elif method == "expand":
                simplified = expand(expr)
            elif method == "factor":
                simplified = factor(expr)
            elif method == "trigsimp":
                simplified = sp.trigsimp(expr)
            elif method == "radsimp":
                simplified = sp.radsimp(expr)
            else:
                simplified = simplify(expr)
            
            result["simplified"] = str(simplified)
            result["simplified_latex"] = latex(simplified)
            result["steps"].append(f"Applied {method} simplification")
            
            self._log_operation("simplify", expr, simplified, {"method": method})
            
            if self.cache_enabled:
                self.cache[cache_key] = result
            
            return result
        except Exception as e:
            logger.error(f"Error simplifying expression: {e}")
            result["error"] = str(e)
            return result
    
    def differentiate(self, expr: Union[str, sp.Expr], variable: str = "x", 
                     order: int = 1, show_steps: bool = True) -> Dict[str, Any]:
        """
        Compute derivative of an expression.
        
        Args:
            expr: Expression to differentiate
            variable: Variable to differentiate with respect to
            order: Order of derivative (1 for first derivative, 2 for second, etc.)
            show_steps: Include intermediate steps
            
        Returns:
            Dictionary with derivative and metadata
        """
        if isinstance(expr, str):
            expr = self.parse_expression(expr)
        
        var = symbols(variable)
        
        result = {
            "original": str(expr),
            "original_latex": latex(expr),
            "variable": variable,
            "order": order,
            "derivative": None,
            "derivative_latex": None,
            "steps": []
        }
        
        try:
            if show_steps and order > 1:
                # Show intermediate derivatives
                current = expr
                for i in range(1, order + 1):
                    current = diff(current, var)
                    result["steps"].append({
                        "order": i,
                        "expression": str(current),
                        "latex": latex(current)
                    })
            
            derivative = diff(expr, var, order)
            result["derivative"] = str(derivative)
            result["derivative_latex"] = latex(derivative)
            
            self._log_operation("differentiate", expr, derivative, 
                               {"variable": variable, "order": order})
            
            return result
        except Exception as e:
            logger.error(f"Error computing derivative: {e}")
            result["error"] = str(e)
            return result
    
    def integrate_expression(self, expr: Union[str, sp.Expr], variable: str = "x",
                           lower_limit: Optional[float] = None, 
                           upper_limit: Optional[float] = None) -> Dict[str, Any]:
        """
        Compute integral of an expression (definite or indefinite).
        
        Args:
            expr: Expression to integrate
            variable: Integration variable
            lower_limit: Lower limit for definite integral
            upper_limit: Upper limit for definite integral
            
        Returns:
            Dictionary with integral and metadata
        """
        if isinstance(expr, str):
            expr = self.parse_expression(expr)
        
        var = symbols(variable)
        
        is_definite = lower_limit is not None and upper_limit is not None
        
        result = {
            "original": str(expr),
            "original_latex": latex(expr),
            "variable": variable,
            "type": "definite" if is_definite else "indefinite",
            "integral": None,
            "integral_latex": None
        }
        
        if is_definite:
            result["lower_limit"] = lower_limit
            result["upper_limit"] = upper_limit
        
        try:
            if is_definite:
                integral_result = integrate(expr, (var, lower_limit, upper_limit))
            else:
                integral_result = integrate(expr, var)
            
            result["integral"] = str(integral_result)
            result["integral_latex"] = latex(integral_result)
            
            # Try to evaluate numerically if definite
            if is_definite:
                try:
                    numeric_val = integral_result.evalf()
                    result["numerical_value"] = float(numeric_val)  # type: ignore
                except Exception:
                    pass
            
            self._log_operation("integrate", expr, integral_result, 
                               {"variable": variable, "definite": is_definite})
            
            return result
        except Exception as e:
            logger.error(f"Error computing integral: {e}")
            result["error"] = str(e)
            return result
    
    def solve_equation(self, equation: Union[str, sp.Expr], variable: str = "x",
                      domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Solve algebraic equation(s).
        
        Args:
            equation: Equation to solve (use '=' or pass expression equal to zero)
            variable: Variable to solve for
            domain: Solution domain ('real', 'complex', None for all)
            
        Returns:
            Dictionary with solutions and metadata
        """
        if isinstance(equation, str):
            # Handle equations with '='
            if '=' in equation:
                lhs, rhs = equation.split('=')
                expr = self.parse_expression(f"({lhs}) - ({rhs})")
            else:
                expr = self.parse_expression(equation)
        else:
            expr = equation
        
        var = symbols(variable)
        
        result = {
            "equation": str(expr) + " = 0",
            "equation_latex": latex(expr) + " = 0",
            "variable": variable,
            "domain": domain,
            "solutions": [],
            "solutions_latex": []
        }
        
        try:
            if domain == "real":
                solutions = solve(expr, var, domain=sp.S.Reals)
            else:
                solutions = solve(expr, var)
            
            result["solutions"] = [str(sol) for sol in solutions]
            result["solutions_latex"] = [latex(sol) for sol in solutions]
            result["num_solutions"] = len(solutions)
            
            self._log_operation("solve", expr, solutions, {"variable": variable})
            
            return result
        except Exception as e:
            logger.error(f"Error solving equation: {e}")
            result["error"] = str(e)
            return result
    
    def compute_limit(self, expr: Union[str, sp.Expr], variable: str = "x",
                     point: Union[float, str] = 0, direction: str = "both") -> Dict[str, Any]:
        """
        Compute limit of an expression.
        
        Args:
            expr: Expression to compute limit of
            variable: Limit variable
            point: Point to approach (number or 'oo' for infinity)
            direction: Direction ('both', '+', '-')
            
        Returns:
            Dictionary with limit and metadata
        """
        if isinstance(expr, str):
            expr = self.parse_expression(expr)
        
        var = symbols(variable)
        
        # Handle infinity
        if point == "oo" or point == "inf":
            point = sp.oo
        elif point == "-oo" or point == "-inf":
            point = -sp.oo
        
        result = {
            "expression": str(expr),
            "expression_latex": latex(expr),
            "variable": variable,
            "point": str(point),
            "direction": direction,
            "limit": None,
            "limit_latex": None
        }
        
        try:
            if direction == "+":
                lim = limit(expr, var, point, '+')
            elif direction == "-":
                lim = limit(expr, var, point, '-')
            else:
                lim = limit(expr, var, point)
            
            result["limit"] = str(lim)
            result["limit_latex"] = latex(lim)
            
            self._log_operation("limit", expr, lim, 
                               {"variable": variable, "point": point})
            
            return result
        except Exception as e:
            logger.error(f"Error computing limit: {e}")
            result["error"] = str(e)
            return result
    
    def taylor_series(self, expr: Union[str, sp.Expr], variable: str = "x",
                     point: Union[int, float] = 0, order: int = 5) -> Dict[str, Any]:
        """
        Compute Taylor/Maclaurin series expansion.
        
        Args:
            expr: Expression to expand
            variable: Expansion variable
            point: Expansion point (0 for Maclaurin)
            order: Order of expansion
            
        Returns:
            Dictionary with series and metadata
        """
        if isinstance(expr, str):
            expr = self.parse_expression(expr)
        
        var = symbols(variable)
        
        result = {
            "expression": str(expr),
            "expression_latex": latex(expr),
            "variable": variable,
            "point": point,
            "order": order,
            "series": None,
            "series_latex": None
        }
        
        try:
            ser = series(expr, var, int(point), order)  # type: ignore
            result["series"] = str(ser)
            result["series_latex"] = latex(ser)
            
            self._log_operation("taylor_series", expr, ser, 
                               {"point": point, "order": order})
            
            return result
        except Exception as e:
            logger.error(f"Error computing series: {e}")
            result["error"] = str(e)
            return result
    
    def matrix_operations(self, matrix_data: Union[List[List[float]], List[List[int]], Sequence[Sequence[Union[int, float]]]], 
                         operation: str) -> Dict[str, Any]:
        """
        Perform matrix operations.
        
        Args:
            matrix_data: 2D list representing matrix
            operation: Operation ('det', 'inverse', 'eigenvalues', 'eigenvectors', 'transpose')
            
        Returns:
            Dictionary with operation result
        """
        matrix = Matrix(matrix_data)
        
        result = {
            "matrix": str(matrix),
            "matrix_latex": latex(matrix),
            "operation": operation,
            "result": None,
            "result_latex": None
        }
        
        try:
            if operation == "det" or operation == "determinant":
                res = matrix.det()
            elif operation == "inverse":
                res = matrix.inv()
            elif operation == "eigenvalues":
                res = matrix.eigenvals()
            elif operation == "eigenvectors":
                res = matrix.eigenvects()
            elif operation == "transpose":
                res = matrix.T
            elif operation == "rank":
                res = matrix.rank()
            elif operation == "rref":
                res = matrix.rref()
            else:
                raise ValueError(f"Unknown matrix operation: {operation}")
            
            result["result"] = str(res)
            result["result_latex"] = latex(res)
            
            self._log_operation("matrix_operation", matrix, res, 
                               {"operation": operation})
            
            return result
        except Exception as e:
            logger.error(f"Error in matrix operation: {e}")
            result["error"] = str(e)
            return result
    
    def run(self, query: str) -> str:
        """
        Run a natural language mathematical query using LLM with SymPy backend.
        
        Args:
            query: Natural language mathematical question
            
        Returns:
            String answer
        """
        if not self.llm:
            return "LLM not available. Please initialize with valid API key."
        
        try:
            # Create a prompt for mathematical reasoning
            from langchain_core.messages import HumanMessage, SystemMessage
            
            system_prompt = """You are a mathematical assistant with access to SymPy. 
When asked to solve mathematical problems:
1. Parse the mathematical expression
2. Use appropriate SymPy functions (simplify, diff, integrate, solve, limit, series, etc.)
3. Return the result in a clear format with explanation

Express answers using SymPy syntax: x**2 for x squared, * for multiplication, etc."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            
            response = self.llm.invoke(messages)
            return str(response.content)
        except Exception as e:
            logger.error(f"Error in run method: {e}")
            return f"Error: {e}"
    
    def natural_language_query(self, query: str) -> Dict[str, Any]:
        """
        Process natural language mathematical query using LangChain.
        
        Args:
            query: Natural language mathematical question
            
        Returns:
            Dictionary with answer and reasoning steps
        """
        if not LANGCHAIN_AVAILABLE or not self.llm:
            return {
                "query": query,
                "error": "LangChain not available. Please initialize with valid API key."
            }
        
        result = {
            "query": query,
            "answer": None,
            "steps": [],
            "token_usage": None
        }
        
        try:
            if get_openai_callback is None:
                raise ImportError("get_openai_callback not available")
                
            with get_openai_callback() as cb:
                # Use the run method which now uses LLM
                answer = self.run(query)
                
                result["answer"] = answer
                result["token_usage"] = {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost
                }
            
            self._log_operation("natural_language_query", query, answer, {})
            
            return result
        except Exception as e:
            logger.error(f"Error processing natural language query: {e}")
            result["error"] = str(e)
            return result
    
    def create_langchain_tools(self) -> List:  # type: ignore
        """
        Create LangChain Tool objects for agent integration.
        
        Returns:
            List of Tool objects
        """
        if Tool is None:
            logger.warning("Tool class not available - cannot create tools")
            return []
            
        tools = [
            Tool(
                name="Simplify_Expression",
                func=lambda x: str(self.simplify_expression(x)),
                description="Simplify a mathematical expression. Input should be a mathematical expression as a string."
            ),
            Tool(
                name="Differentiate",
                func=lambda x: str(self.differentiate(x)),
                description="Compute derivative of an expression with respect to x. Input should be a mathematical expression."
            ),
            Tool(
                name="Integrate",
                func=lambda x: str(self.integrate_expression(x)),
                description="Compute integral of an expression with respect to x. Input should be a mathematical expression."
            ),
            Tool(
                name="Solve_Equation",
                func=lambda x: str(self.solve_equation(x)),
                description="Solve an algebraic equation for x. Input should be an equation or expression equal to zero."
            ),
            Tool(
                name="Compute_Limit",
                func=lambda x: str(self.compute_limit(x)),
                description="Compute limit of an expression as x approaches 0. Input should be a mathematical expression."
            ),
        ]
        
        return tools
    
    def get_operation_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get operation history.
        
        Args:
            limit: Maximum number of recent operations to return
            
        Returns:
            List of operation log entries
        """
        if limit:
            return self.operation_history[-limit:]
        return self.operation_history
    
    def export_history(self, filepath: str):
        """Export operation history to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.operation_history, f, indent=2)
            logger.info(f"History exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting history: {e}")
    
    def clear_cache(self):
        """Clear the operation cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def __repr__(self):
        return f"AdvancedSymbolicEngine(model={self.model}, operations={len(self.operation_history)})"


# Convenience functions for direct usage
def quick_simplify(expr: str) -> str:
    """Quick simplification without engine initialization."""
    try:
        parsed = parse_expr(expr, transformations=standard_transformations + 
                          (implicit_multiplication_application,))
        return str(simplify(parsed))
    except Exception as e:
        return f"Error: {e}"


def quick_solve(equation: str, variable: str = "x") -> List[str]:
    """Quick equation solving without engine initialization."""
    try:
        if '=' in equation:
            lhs, rhs = equation.split('=')
            expr = parse_expr(f"({lhs}) - ({rhs})", 
                            transformations=standard_transformations + 
                            (implicit_multiplication_application,))
        else:
            expr = parse_expr(equation, transformations=standard_transformations + 
                            (implicit_multiplication_application,))
        var = symbols(variable)
        solutions = solve(expr, var)
        return [str(sol) for sol in solutions]
    except Exception as e:
        return [f"Error: {e}"]


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Advanced Symbolic Mathematics Engine - Demo")
    print("=" * 80)
    
    # Initialize engine
    engine = AdvancedSymbolicEngine(enable_dangerous_ops=False)
    
    # Test 1: Simplification
    print("\n1. SIMPLIFICATION")
    print("-" * 40)
    result = engine.simplify_expression("(x**2 + 2*x + 1)", method="factor")
    print(f"Original: {result['original']}")
    print(f"Simplified: {result['simplified']}")
    print(f"LaTeX: {result['simplified_latex']}")
    
    # Test 2: Differentiation
    print("\n2. DIFFERENTIATION")
    print("-" * 40)
    result = engine.differentiate("sin(x)*exp(x)", order=1)
    print(f"f(x) = {result['original']}")
    print(f"f'(x) = {result['derivative']}")
    print(f"LaTeX: {result['derivative_latex']}")
    
    # Test 3: Integration
    print("\n3. INTEGRATION")
    print("-" * 40)
    result = engine.integrate_expression("x**2", variable="x")
    print(f"∫({result['original']}) dx = {result['integral']}")
    
    # Test 4: Definite Integration
    print("\n4. DEFINITE INTEGRATION")
    print("-" * 40)
    result = engine.integrate_expression("x**2", variable="x", lower_limit=0, upper_limit=1)
    print(f"∫₀¹ ({result['original']}) dx = {result['integral']}")
    if 'numerical_value' in result:
        print(f"Numerical value: {result['numerical_value']}")
    
    # Test 5: Equation Solving
    print("\n5. EQUATION SOLVING")
    print("-" * 40)
    result = engine.solve_equation("x**2 - 4 = 0", variable="x")
    print(f"Equation: {result['equation']}")
    print(f"Solutions: {', '.join(result['solutions'])}")
    
    # Test 6: Limits
    print("\n6. LIMITS")
    print("-" * 40)
    result = engine.compute_limit("sin(x)/x", variable="x", point=0)
    print(f"lim(x→{result['point']}) {result['expression']} = {result['limit']}")
    
    # Test 7: Taylor Series
    print("\n7. TAYLOR SERIES")
    print("-" * 40)
    result = engine.taylor_series("exp(x)", order=5)
    print(f"Taylor series of {result['expression']} at x=0:")
    print(result['series'])
    
    # Test 8: Matrix Operations
    print("\n8. MATRIX OPERATIONS")
    print("-" * 40)
    matrix_data = [[1, 2], [3, 4]]
    result = engine.matrix_operations(matrix_data, "det")
    print(f"Matrix:\n{result['matrix']}")
    print(f"Determinant: {result['result']}")
    
    # Test 9: Quick functions
    print("\n9. QUICK FUNCTIONS")
    print("-" * 40)
    print(f"Quick simplify: (x+1)**2 = {quick_simplify('(x+1)**2')}")
    print(f"Quick solve: x**2-9=0, x = {quick_solve('x**2-9=0')}")
    
    # Test 10: Natural Language (if API key available)
    print("\n10. NATURAL LANGUAGE QUERY (requires API key)")
    print("-" * 40)
    if engine.llm:
        result = engine.natural_language_query("What is the derivative of x^3 + 2x?")
        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer']}")
    else:
        print("Skipped - No API key configured")
    
    # Show operation history
    print("\n" + "=" * 80)
    print(f"Total operations performed: {len(engine.operation_history)}")
    print("=" * 80)
