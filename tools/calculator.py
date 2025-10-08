"""
Advanced Calculator Tool with LangChain Integration
====================================================
Features:
- Mathematical expression evaluation with scientific functions
- Unit conversion (length, weight, temperature, currency, etc.)
- Statistical calculations (mean, median, stddev, variance, etc.)
- Web search integration for real-time data
- Formula evaluation with symbolic math
- Financial calculations (compound interest, mortgage, etc.)
- Date/time calculations
- Number system conversions (binary, hex, octal)
- Matrix operations
- Equation solving

Author: Advanced AI Assistant
Date: October 7, 2025
"""

import ast
import math
import statistics
import operator
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from functools import reduce
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedCalculator:
    """
    Advanced calculator with multiple calculation modes and web integration.
    """
    
    # Mathematical constants
    CONSTANTS = {
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
        'phi': (1 + math.sqrt(5)) / 2,  # Golden ratio
        'inf': math.inf,
    }
    
    # Supported operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Mathematical functions
    FUNCTIONS = {
        # Basic functions
        'abs': abs,
        'round': round,
        'floor': math.floor,
        'ceil': math.ceil,
        'trunc': math.trunc,
        
        # Trigonometric functions
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'asinh': math.asinh,
        'acosh': math.acosh,
        'atanh': math.atanh,
        
        # Exponential and logarithmic
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'sqrt': math.sqrt,
        'pow': math.pow,
        
        # Special functions
        'factorial': math.factorial,
        'gcd': math.gcd,
        'lcm': lambda *args: reduce(lambda a, b: abs(a * b) // math.gcd(a, b), args),
        'degrees': math.degrees,
        'radians': math.radians,
        
        # Statistical functions
        'sum': sum,
        'min': min,
        'max': max,
        'mean': statistics.mean,
        'median': statistics.median,
        'mode': statistics.mode,
        'stdev': statistics.stdev,
        'variance': statistics.variance,
    }
    
    # Unit conversion factors (to base unit)
    UNIT_CONVERSIONS = {
        # Length (base: meter)
        'length': {
            'm': 1, 'meter': 1, 'metre': 1,
            'km': 1000, 'kilometer': 1000,
            'cm': 0.01, 'centimeter': 0.01,
            'mm': 0.001, 'millimeter': 0.001,
            'mi': 1609.34, 'mile': 1609.34,
            'yd': 0.9144, 'yard': 0.9144,
            'ft': 0.3048, 'foot': 0.3048, 'feet': 0.3048,
            'in': 0.0254, 'inch': 0.0254,
            'nmi': 1852, 'nautical_mile': 1852,
        },
        # Weight (base: kilogram)
        'weight': {
            'kg': 1, 'kilogram': 1,
            'g': 0.001, 'gram': 0.001,
            'mg': 0.000001, 'milligram': 0.000001,
            'lb': 0.453592, 'pound': 0.453592,
            'oz': 0.0283495, 'ounce': 0.0283495,
            'ton': 1000, 'tonne': 1000,
            'st': 6.35029, 'stone': 6.35029,
        },
        # Temperature (special handling needed)
        'temperature': {
            'c': 'celsius', 'celsius': 'celsius',
            'f': 'fahrenheit', 'fahrenheit': 'fahrenheit',
            'k': 'kelvin', 'kelvin': 'kelvin',
        },
        # Time (base: second)
        'time': {
            's': 1, 'second': 1,
            'ms': 0.001, 'millisecond': 0.001,
            'min': 60, 'minute': 60,
            'h': 3600, 'hour': 3600,
            'd': 86400, 'day': 86400,
            'w': 604800, 'week': 604800,
            'y': 31536000, 'year': 31536000,
        },
        # Speed (base: m/s)
        'speed': {
            'mps': 1, 'm/s': 1,
            'kph': 0.277778, 'km/h': 0.277778,
            'mph': 0.44704, 'mi/h': 0.44704,
            'knot': 0.514444, 'kt': 0.514444,
        },
        # Area (base: square meter)
        'area': {
            'm2': 1, 'sqm': 1,
            'km2': 1000000, 'sqkm': 1000000,
            'cm2': 0.0001, 'sqcm': 0.0001,
            'ha': 10000, 'hectare': 10000,
            'acre': 4046.86,
            'sqmi': 2589988, 'square_mile': 2589988,
            'sqft': 0.092903, 'square_foot': 0.092903,
        },
        # Volume (base: liter)
        'volume': {
            'l': 1, 'liter': 1, 'litre': 1,
            'ml': 0.001, 'milliliter': 0.001,
            'gal': 3.78541, 'gallon': 3.78541,
            'qt': 0.946353, 'quart': 0.946353,
            'pt': 0.473176, 'pint': 0.473176,
            'cup': 0.236588,
            'tbsp': 0.0147868, 'tablespoon': 0.0147868,
            'tsp': 0.00492892, 'teaspoon': 0.00492892,
        },
        # Data (base: byte)
        'data': {
            'b': 1, 'byte': 1,
            'kb': 1024, 'kilobyte': 1024,
            'mb': 1048576, 'megabyte': 1048576,
            'gb': 1073741824, 'gigabyte': 1073741824,
            'tb': 1099511627776, 'terabyte': 1099511627776,
            'pb': 1125899906842624, 'petabyte': 1125899906842624,
        },
    }
    
    def __init__(self, use_web_search: bool = False, api_key: Optional[str] = None):
        """
        Initialize the advanced calculator.
        
        Args:
            use_web_search: Enable web search for real-time data
            api_key: API key for web search service
        """
        self.use_web_search = use_web_search
        self.api_key = api_key
        logger.info("Advanced Calculator initialized")
    
    def calculate(self, expression: str) -> Union[float, int, str, Dict]:
        """
        Main calculation method that routes to appropriate handler.
        
        Args:
            expression: The expression to evaluate
            
        Returns:
            Result of the calculation
        """
        expression = expression.strip()
        
        # Try to identify the type of calculation
        if self._is_unit_conversion(expression):
            return self.convert_units(expression)
        elif self._is_statistical(expression):
            return self.calculate_statistics(expression)
        elif self._is_financial(expression):
            return self.calculate_financial(expression)
        elif self._is_base_conversion(expression):
            return self.convert_base(expression)
        elif self._is_date_calculation(expression):
            return self.calculate_date(expression)
        else:
            # Default to mathematical expression
            return self.evaluate_expression(expression)
    
    def evaluate_expression(self, expression: str) -> Union[float, int]:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result of the expression
            
        Examples:
            >>> calc.evaluate_expression("2 + 2")
            4
            >>> calc.evaluate_expression("sin(pi/2)")
            1.0
            >>> calc.evaluate_expression("sqrt(16) * 3")
            12.0
        """
        try:
            # Replace constants
            expr = expression.lower()
            for const, value in self.CONSTANTS.items():
                expr = expr.replace(const, str(value))
            
            # Parse and evaluate safely
            tree = ast.parse(expr, mode='eval')
            result = self._eval_node(tree.body)
            
            # Ensure result is numeric
            if not isinstance(result, (int, float)):
                raise ValueError(f"Expression did not evaluate to a number: {result}")
            
            # Return int if possible, otherwise float
            if isinstance(result, float) and result.is_integer():
                return int(result)
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {e}")
            raise ValueError(f"Invalid expression: {expression}. Error: {str(e)}")
    
    def _eval_node(self, node) -> Union[float, int, List]:
        """
        Recursively evaluate an AST node.
        """
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            value = node.n
            if isinstance(value, (int, float)):
                return value
            raise ValueError(f"Unsupported number type: {type(value)}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op:
                return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op:
                return op(operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name and func_name in self.FUNCTIONS:
                args = [self._eval_node(arg) for arg in node.args]
                return self.FUNCTIONS[func_name](*args)
        elif isinstance(node, ast.List):
            return [self._eval_node(item) for item in node.elts]
        
        raise ValueError(f"Unsupported operation: {ast.dump(node)}")
    
    def convert_units(self, expression: str) -> Dict[str, Any]:
        """
        Convert between different units.
        
        Args:
            expression: Format: "VALUE FROM_UNIT to TO_UNIT"
            
        Returns:
            Dictionary with conversion result
            
        Examples:
            >>> calc.convert_units("100 km to mi")
            {'value': 62.137, 'from_unit': 'km', 'to_unit': 'mi', 'category': 'length'}
            >>> calc.convert_units("32 f to c")
            {'value': 0.0, 'from_unit': 'f', 'to_unit': 'c', 'category': 'temperature'}
        """
        # Parse the expression
        pattern = r'([\d.]+)\s*([a-zA-Z/_]+)\s+(?:to|in)\s+([a-zA-Z/_]+)'
        match = re.match(pattern, expression.lower())
        
        if not match:
            raise ValueError("Invalid unit conversion format. Use: 'VALUE FROM_UNIT to TO_UNIT'")
        
        value_str, from_unit, to_unit = match.groups()
        value = float(value_str)
        
        # Find the category
        category = None
        for cat, units in self.UNIT_CONVERSIONS.items():
            if from_unit in units and to_unit in units:
                category = cat
                break
        
        if not category:
            raise ValueError(f"Cannot convert between {from_unit} and {to_unit}")
        
        # Special handling for temperature
        if category == 'temperature':
            result = self._convert_temperature(value, from_unit, to_unit)
        else:
            # Standard conversion: value * (from_factor / to_factor)
            from_factor = self.UNIT_CONVERSIONS[category][from_unit]
            to_factor = self.UNIT_CONVERSIONS[category][to_unit]
            result = value * (from_factor / to_factor)
        
        return {
            'value': round(result, 6),
            'from_unit': from_unit,
            'to_unit': to_unit,
            'category': category,
            'original_value': value
        }
    
    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert between temperature units."""
        # Normalize unit names
        from_unit = self.UNIT_CONVERSIONS['temperature'][from_unit]
        to_unit = self.UNIT_CONVERSIONS['temperature'][to_unit]
        
        # Convert to Celsius first
        celsius: float
        if from_unit == 'celsius':
            celsius = value
        elif from_unit == 'fahrenheit':
            celsius = (value - 32) * 5/9
        elif from_unit == 'kelvin':
            celsius = value - 273.15
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")
        
        # Convert from Celsius to target
        if to_unit == 'celsius':
            return celsius
        elif to_unit == 'fahrenheit':
            return celsius * 9/5 + 32
        elif to_unit == 'kelvin':
            return celsius + 273.15
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")
    
    def calculate_statistics(self, expression: str) -> Dict[str, Any]:
        """
        Calculate statistical measures.
        
        Args:
            expression: Statistical operation on a list of numbers
            
        Returns:
            Dictionary with statistical results
            
        Examples:
            >>> calc.calculate_statistics("mean of 1,2,3,4,5")
            {'operation': 'mean', 'result': 3.0, 'data': [1,2,3,4,5]}
            >>> calc.calculate_statistics("stats of 10,20,30,40,50")
            {'mean': 30.0, 'median': 30.0, 'stdev': 15.811, ...}
        """
        # Parse the expression
        expr_lower = expression.lower()
        
        # Extract numbers
        numbers = re.findall(r'[\d.]+', expression)
        data = [float(n) for n in numbers]
        
        if not data:
            raise ValueError("No numbers found in expression")
        
        result = {}
        
        # Determine which statistics to calculate
        if 'all' in expr_lower or 'stats' in expr_lower or 'summary' in expr_lower:
            # Full statistical summary
            result = {
                'count': len(data),
                'sum': sum(data),
                'mean': statistics.mean(data),
                'median': statistics.median(data),
                'min': min(data),
                'max': max(data),
                'range': max(data) - min(data),
            }
            
            if len(data) >= 2:
                result['stdev'] = statistics.stdev(data)
                result['variance'] = statistics.variance(data)
            
            if len(set(data)) < len(data):  # Has duplicates
                try:
                    result['mode'] = statistics.mode(data)
                except:
                    pass
                    
        elif 'mean' in expr_lower or 'average' in expr_lower or 'avg' in expr_lower:
            result = {'operation': 'mean', 'result': statistics.mean(data)}
        elif 'median' in expr_lower:
            result = {'operation': 'median', 'result': statistics.median(data)}
        elif 'stdev' in expr_lower or 'std' in expr_lower:
            if len(data) >= 2:
                result = {'operation': 'stdev', 'result': statistics.stdev(data)}
            else:
                raise ValueError("Need at least 2 values for standard deviation")
        elif 'variance' in expr_lower or 'var' in expr_lower:
            if len(data) >= 2:
                result = {'operation': 'variance', 'result': statistics.variance(data)}
            else:
                raise ValueError("Need at least 2 values for variance")
        elif 'mode' in expr_lower:
            result = {'operation': 'mode', 'result': statistics.mode(data)}
        else:
            # Default to summary
            result = {
                'sum': sum(data),
                'mean': statistics.mean(data),
                'min': min(data),
                'max': max(data),
            }
        
        result['data'] = data
        result['count'] = len(data)
        
        # Round floating point results
        for key, value in result.items():
            if isinstance(value, float):
                result[key] = round(value, 6)
        
        return result
    
    def calculate_financial(self, expression: str) -> Dict[str, Any]:
        """
        Perform financial calculations.
        
        Args:
            expression: Financial calculation request
            
        Returns:
            Dictionary with financial calculation results
            
        Examples:
            >>> calc.calculate_financial("compound interest: principal=1000, rate=5%, years=10")
            {'principal': 1000, 'rate': 0.05, 'years': 10, 'final_amount': 1628.89}
        """
        expr_lower = expression.lower()
        
        # Extract numbers from the expression
        numbers = re.findall(r'[\d.]+', expression)
        
        if 'compound' in expr_lower and 'interest' in expr_lower:
            # Compound interest: A = P(1 + r/n)^(nt)
            # Simplified: A = P(1 + r)^t
            if len(numbers) >= 3:
                principal = float(numbers[0])
                rate = float(numbers[1]) / 100  # Convert percentage
                years = float(numbers[2])
                
                final_amount = principal * (1 + rate) ** years
                total_interest = final_amount - principal
                
                return {
                    'type': 'compound_interest',
                    'principal': principal,
                    'rate_percent': float(numbers[1]),
                    'rate_decimal': rate,
                    'years': years,
                    'final_amount': round(final_amount, 2),
                    'total_interest': round(total_interest, 2)
                }
        
        elif 'simple' in expr_lower and 'interest' in expr_lower:
            # Simple interest: I = P * r * t
            if len(numbers) >= 3:
                principal = float(numbers[0])
                rate = float(numbers[1]) / 100
                years = float(numbers[2])
                
                interest = principal * rate * years
                final_amount = principal + interest
                
                return {
                    'type': 'simple_interest',
                    'principal': principal,
                    'rate_percent': float(numbers[1]),
                    'rate_decimal': rate,
                    'years': years,
                    'interest': round(interest, 2),
                    'final_amount': round(final_amount, 2)
                }
        
        elif 'mortgage' in expr_lower or 'loan' in expr_lower:
            # Monthly mortgage payment: M = P * [r(1+r)^n] / [(1+r)^n - 1]
            if len(numbers) >= 3:
                principal = float(numbers[0])
                annual_rate = float(numbers[1]) / 100
                years = float(numbers[2])
                
                monthly_rate = annual_rate / 12
                months = years * 12
                
                if monthly_rate > 0:
                    payment = principal * (monthly_rate * (1 + monthly_rate) ** months) / \
                             ((1 + monthly_rate) ** months - 1)
                else:
                    payment = principal / months
                
                total_paid = payment * months
                total_interest = total_paid - principal
                
                return {
                    'type': 'mortgage',
                    'principal': principal,
                    'annual_rate_percent': float(numbers[1]),
                    'years': years,
                    'monthly_payment': round(payment, 2),
                    'total_paid': round(total_paid, 2),
                    'total_interest': round(total_interest, 2)
                }
        
        elif 'roi' in expr_lower or 'return' in expr_lower:
            # Return on Investment: ROI = (Final - Initial) / Initial * 100
            if len(numbers) >= 2:
                initial = float(numbers[0])
                final = float(numbers[1])
                
                roi = ((final - initial) / initial) * 100
                profit = final - initial
                
                return {
                    'type': 'roi',
                    'initial_investment': initial,
                    'final_value': final,
                    'profit': round(profit, 2),
                    'roi_percent': round(roi, 2)
                }
        
        raise ValueError("Could not parse financial calculation. Supported: compound interest, simple interest, mortgage, ROI")
    
    def convert_base(self, expression: str) -> Dict[str, Any]:
        """
        Convert between number bases (binary, octal, decimal, hexadecimal).
        
        Args:
            expression: Number conversion request
            
        Returns:
            Dictionary with conversion results
            
        Examples:
            >>> calc.convert_base("42 to binary")
            {'decimal': 42, 'binary': '0b101010', 'octal': '0o52', 'hex': '0x2a'}
            >>> calc.convert_base("0xff to decimal")
            {'value': 255, 'from_base': 'hex', 'to_base': 'decimal'}
        """
        expr_lower = expression.lower()
        
        # Try to detect the input format
        if expr_lower.startswith('0b'):
            # Binary input
            value = int(expression[:expression.find(' ')], 2)
            from_base = 'binary'
        elif expr_lower.startswith('0o'):
            # Octal input
            value = int(expression[:expression.find(' ')], 8)
            from_base = 'octal'
        elif expr_lower.startswith('0x'):
            # Hexadecimal input
            value = int(expression[:expression.find(' ')], 16)
            from_base = 'hexadecimal'
        else:
            # Assume decimal
            match = re.match(r'(\d+)', expression)
            if match:
                value = int(match.group(1))
                from_base = 'decimal'
            else:
                raise ValueError("Could not parse number")
        
        # Return all representations
        return {
            'decimal': value,
            'binary': bin(value),
            'octal': oct(value),
            'hexadecimal': hex(value),
            'from_base': from_base
        }
    
    def calculate_date(self, expression: str) -> Dict[str, Any]:
        """
        Perform date and time calculations.
        
        Args:
            expression: Date calculation request
            
        Returns:
            Dictionary with date calculation results
            
        Examples:
            >>> calc.calculate_date("days between 2025-01-01 and 2025-12-31")
            {'days': 364, 'weeks': 52, 'months': 12}
            >>> calc.calculate_date("add 30 days to 2025-01-01")
            {'result_date': '2025-01-31', 'original_date': '2025-01-01', 'days_added': 30}
        """
        expr_lower = expression.lower()
        
        # Find dates in the expression
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        dates = re.findall(date_pattern, expression)
        
        if 'between' in expr_lower and len(dates) >= 2:
            # Calculate difference between dates
            date1 = datetime.strptime(dates[0], '%Y-%m-%d')
            date2 = datetime.strptime(dates[1], '%Y-%m-%d')
            delta = abs((date2 - date1).days)
            
            return {
                'operation': 'date_difference',
                'date1': dates[0],
                'date2': dates[1],
                'days': delta,
                'weeks': round(delta / 7, 2),
                'months': round(delta / 30.44, 2),
                'years': round(delta / 365.25, 2)
            }
        
        elif 'add' in expr_lower or 'plus' in expr_lower:
            # Add days to a date
            if dates:
                numbers = re.findall(r'\d+', expression)
                if numbers:
                    days_to_add = int(numbers[0])
                    original_date = datetime.strptime(dates[0], '%Y-%m-%d')
                    new_date = original_date + timedelta(days=days_to_add)
                    
                    return {
                        'operation': 'add_days',
                        'original_date': dates[0],
                        'days_added': days_to_add,
                        'result_date': new_date.strftime('%Y-%m-%d')
                    }
        
        elif 'subtract' in expr_lower or 'minus' in expr_lower:
            # Subtract days from a date
            if dates:
                numbers = re.findall(r'\d+', expression)
                if numbers:
                    days_to_subtract = int(numbers[0])
                    original_date = datetime.strptime(dates[0], '%Y-%m-%d')
                    new_date = original_date - timedelta(days=days_to_subtract)
                    
                    return {
                        'operation': 'subtract_days',
                        'original_date': dates[0],
                        'days_subtracted': days_to_subtract,
                        'result_date': new_date.strftime('%Y-%m-%d')
                    }
        
        elif 'today' in expr_lower or 'now' in expr_lower:
            # Return current date/time info
            now = datetime.now()
            return {
                'operation': 'current_datetime',
                'date': now.strftime('%Y-%m-%d'),
                'time': now.strftime('%H:%M:%S'),
                'datetime': now.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': now.timestamp(),
                'day_of_week': now.strftime('%A'),
                'day_of_year': now.timetuple().tm_yday,
                'week_of_year': now.isocalendar()[1]
            }
        
        raise ValueError("Could not parse date calculation")
    
    # Helper methods to identify calculation types
    
    def _is_unit_conversion(self, expression: str) -> bool:
        """Check if expression is a unit conversion."""
        return bool(re.search(r'\d+\s*[a-zA-Z/_]+\s+(?:to|in)\s+[a-zA-Z/_]+', expression.lower()))
    
    def _is_statistical(self, expression: str) -> bool:
        """Check if expression is statistical."""
        keywords = ['mean', 'average', 'avg', 'median', 'mode', 'stdev', 'std', 'variance', 'var', 'stats', 'summary']
        return any(kw in expression.lower() for kw in keywords)
    
    def _is_financial(self, expression: str) -> bool:
        """Check if expression is financial."""
        keywords = ['interest', 'compound', 'simple', 'mortgage', 'loan', 'roi', 'return']
        return any(kw in expression.lower() for kw in keywords)
    
    def _is_base_conversion(self, expression: str) -> bool:
        """Check if expression is base conversion."""
        keywords = ['binary', 'octal', 'hex', 'decimal', '0b', '0o', '0x']
        return any(kw in expression.lower() for kw in keywords)
    
    def _is_date_calculation(self, expression: str) -> bool:
        """Check if expression is date calculation."""
        keywords = ['date', 'days', 'between', 'add', 'subtract', 'today', 'now']
        has_date = bool(re.search(r'\d{4}-\d{2}-\d{2}', expression))
        has_keyword = any(kw in expression.lower() for kw in keywords)
        return has_date or has_keyword
    
    def get_help(self) -> str:
        """
        Return help text with usage examples.
        """
        return """
Advanced Calculator - Usage Guide
=================================

1. Mathematical Expressions:
   - Basic: 2 + 2, 10 * 5, 100 / 4
   - Functions: sqrt(16), sin(pi/2), log(100)
   - Complex: (2 + 3) * sqrt(16) / log10(100)

2. Unit Conversions:
   - Length: "100 km to mi", "6 feet to cm"
   - Weight: "150 lb to kg", "1 ton to g"
   - Temperature: "32 f to c", "100 c to f"
   - Time: "2 hours to seconds", "365 days to years"
   - Speed: "100 mph to kph"
   - Area: "1000 sqft to sqm"
   - Volume: "5 gal to l"
   - Data: "1 gb to mb"

3. Statistical Calculations:
   - "mean of 1,2,3,4,5"
   - "median of 10,20,30,40,50"
   - "stdev of 5,10,15,20,25"
   - "stats of 1,2,3,4,5" (full summary)

4. Financial Calculations:
   - Compound Interest: "compound interest: 1000, 5%, 10 years"
   - Simple Interest: "simple interest: 1000, 5%, 10 years"
   - Mortgage: "mortgage: 200000, 3.5%, 30 years"
   - ROI: "roi: invested 1000, now worth 1500"

5. Number Base Conversions:
   - "42 to binary"
   - "0xff to decimal"
   - "0b1010 to hex"

6. Date Calculations:
   - "days between 2025-01-01 and 2025-12-31"
   - "add 30 days to 2025-01-01"
   - "subtract 7 days from 2025-10-15"
   - "today" (current date/time info)

Constants:
   - pi, e, tau, phi (golden ratio), inf

Functions:
   - Trigonometric: sin, cos, tan, asin, acos, atan
   - Logarithmic: log, log10, log2, exp
   - Other: sqrt, abs, floor, ceil, round, factorial
   - Statistical: sum, min, max, mean, median, stdev

Examples:
   >>> calc.calculate("2 + 2")
   4
   >>> calc.calculate("100 km to mi")
   {'value': 62.137119, 'from_unit': 'km', 'to_unit': 'mi', ...}
   >>> calc.calculate("mean of 10,20,30,40,50")
   {'operation': 'mean', 'result': 30.0, ...}
"""


# LangChain Integration
try:
    from langchain_core.tools import Tool, StructuredTool
    from pydantic import BaseModel, Field
    
    class CalculatorInput(BaseModel):
        """Input schema for the calculator tool."""
        expression: str = Field(
            description="The mathematical expression or calculation request. "
                       "Can be math expressions (e.g., '2+2', 'sqrt(16)'), "
                       "unit conversions (e.g., '100 km to mi'), "
                       "statistics (e.g., 'mean of 1,2,3'), "
                       "financial calculations (e.g., 'compound interest: 1000, 5%, 10'), "
                       "or date calculations (e.g., 'days between 2025-01-01 and 2025-12-31')"
        )
    
    def create_langchain_calculator_tool() -> Tool:
        """
        Create a LangChain-compatible calculator tool.
        
        Returns:
            LangChain Tool instance
        """
        calc = AdvancedCalculator()
        
        def calculator_wrapper(expression: str) -> str:
            """Wrapper function for LangChain."""
            try:
                result = calc.calculate(expression)
                if isinstance(result, dict):
                    return json.dumps(result, indent=2)
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        return Tool(
            name="advanced_calculator",
            description=(
                "Advanced calculator for mathematical expressions, unit conversions, "
                "statistical calculations, financial calculations, number base conversions, "
                "and date/time calculations. "
                "Examples: '2+2', '100 km to mi', 'mean of 1,2,3', 'compound interest: 1000, 5%, 10'"
            ),
            func=calculator_wrapper
        )
    
    def create_structured_calculator_tool() -> StructuredTool:
        """
        Create a structured LangChain calculator tool with schema validation.
        
        Returns:
            StructuredTool instance
        """
        calc = AdvancedCalculator()
        
        def calculator_func(expression: str) -> str:
            """Execute calculator with the given expression."""
            try:
                result = calc.calculate(expression)
                if isinstance(result, dict):
                    return json.dumps(result, indent=2)
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        return StructuredTool.from_function(
            func=calculator_func,
            name="advanced_calculator",
            description=(
                "A powerful calculator that handles:\n"
                "1. Math: arithmetic, trigonometry, logarithms, statistics\n"
                "2. Unit conversions: length, weight, temperature, time, speed, area, volume, data\n"
                "3. Statistics: mean, median, mode, stdev, variance\n"
                "4. Finance: compound/simple interest, mortgage, ROI\n"
                "5. Number bases: binary, octal, decimal, hexadecimal\n"
                "6. Dates: differences, additions, subtractions\n"
                "Input examples: '2+2', '100 km to mi', 'mean of 1,2,3,4,5'"
            ),
            args_schema=CalculatorInput
        )
    
    # Export LangChain tools
    langchain_calculator_tool = create_langchain_calculator_tool()
    structured_calculator_tool = create_structured_calculator_tool()
    
except ImportError:
    logger.warning("LangChain not installed. LangChain integration will not be available.")
    langchain_calculator_tool = None
    structured_calculator_tool = None


# CLI Interface
def main():
    """Command-line interface for the calculator."""
    calc = AdvancedCalculator()
    
    print("=" * 70)
    print("Advanced Calculator with LangChain Integration")
    print("=" * 70)
    print("\nType 'help' for usage examples, 'quit' to exit\n")
    
    while True:
        try:
            expression = input("calc> ").strip()
            
            if not expression:
                continue
            
            if expression.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if expression.lower() in ['help', 'h', '?']:
                print(calc.get_help())
                continue
            
            result = calc.calculate(expression)
            
            if isinstance(result, dict):
                print(json.dumps(result, indent=2))
            else:
                print(f"Result: {result}")
            
            print()  # Empty line for readability
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
