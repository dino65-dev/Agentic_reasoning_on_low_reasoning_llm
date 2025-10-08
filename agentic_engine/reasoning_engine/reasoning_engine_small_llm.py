"""
═══════════════════════════════════════════════════════════════════════════════
MCP-ENHANCED SMALL LLM REASONING ENGINE
═══════════════════════════════════════════════════════════════════════════════

Purpose: Enable Qwen2 1.5B (non-reasoning LLM) to perform complex reasoning tasks
         by leveraging Model Context Protocol (MCP) tools as external cognitive aids.

Key Strategy:
- Small LLM = Task decomposer + Coordinator
- MCP Tools = Actual reasoning engines (calculator, sequential thinking, memory, etc.)
- Pattern: Break down complex problems → Use MCP tools → Aggregate results

This approach transforms a weak LLM into a powerful reasoning system through tool use.
"""

import os
import sys
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
except ImportError:
    logger.warning("langchain not installed. Install with: pip install langchain-openai")
    ChatOpenAI = None

# Import MCP tool infrastructure
try:
    from tool_invoker import (
        load_mcp_tools_from_registry, 
        create_mock_tools,
        LLM_API_URL, 
        LLM_API_KEY, 
        LLM_MODEL_NAME
    )
except ImportError:
    logger.warning("Could not import tool_invoker. Using fallback.")
    LLM_API_URL = os.getenv("SMALL_LLM_API_URL", "http://localhost:1234/v1")
    LLM_API_KEY = os.getenv("SMALL_LLM_API_KEY", "lm-studio")
    LLM_MODEL_NAME = os.getenv("LLM__MODEL_NAME", "qwen2-1.5b-instruct")

# ═══════════════════════════════════════════════════════════════════════════════
# CORE PROMPTS FOR SMALL LLM ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

DECOMPOSITION_PROMPT = """You are a problem decomposition assistant. Your job is to break down complex problems into simple, tool-solvable steps.

Available tools:
- calculator: For math calculations
- sequential_thinking: For multi-step logical reasoning
- memory: For storing and retrieving information
- search: For finding patterns or information

Problem: {problem}
Answer Options: {options}

Break this into 3-5 simple steps that can be solved using the available tools.
Format your response as a JSON list:
[
  {{"step": 1, "description": "...", "tool": "calculator|sequential_thinking|memory|search"}},
  {{"step": 2, "description": "...", "tool": "..."}}
]

Output ONLY the JSON array, nothing else."""

TOOL_INSTRUCTION_PROMPT = """Generate a specific instruction for the {tool} tool to solve this step:

Step: {step_description}
Context: {context}

Provide a clear, concise instruction (1-2 sentences) for the tool."""

AGGREGATION_PROMPT = """Based on the following reasoning steps and their results, determine which answer option is correct.

Problem: {problem}
Options: {options}

Steps and Results:
{steps_and_results}

Analyze the results and select the correct option number (1-{num_options}).
Respond with ONLY the number, nothing else."""

VERIFICATION_PROMPT = """Verify if this answer is logically consistent with the reasoning:

Problem: {problem}
Selected Answer: {selected_answer}
Reasoning Steps: {reasoning}

Is this answer correct? Respond with YES or NO, followed by a brief explanation."""

# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOL SIMULATORS (Fallback when MCP servers unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

class MCPToolSimulator:
    """Simulates MCP tools for reasoning augmentation"""
    
    @staticmethod
    def calculator(instruction: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates calculator tool"""
        try:
            # Extract mathematical expressions
            import re
            numbers = re.findall(r'\d+\.?\d*', instruction)
            
            # Simple operations
            if 'add' in instruction.lower() or '+' in instruction:
                result = sum(float(n) for n in numbers)
                return {"result": result, "success": True}
            elif 'multiply' in instruction.lower() or '*' in instruction or '×' in instruction:
                result = 1
                for n in numbers:
                    result *= float(n)
                return {"result": result, "success": True}
            elif 'subtract' in instruction.lower() or '-' in instruction:
                if len(numbers) >= 2:
                    result = float(numbers[0]) - float(numbers[1])
                    return {"result": result, "success": True}
            elif 'divide' in instruction.lower() or '/' in instruction or '÷' in instruction:
                if len(numbers) >= 2 and float(numbers[1]) != 0:
                    result = float(numbers[0]) / float(numbers[1])
                    return {"result": result, "success": True}
            
            # Try eval as fallback
            expr = re.sub(r'[^\d+\-*/().\s]', '', instruction)
            if expr:
                result = eval(expr, {"__builtins__": {}}, {})
                return {"result": result, "success": True}
            
            return {"result": "Cannot parse calculation", "success": False}
        except Exception as e:
            return {"result": f"Error: {str(e)}", "success": False}
    
    @staticmethod
    def sequential_thinking(instruction: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Uses MCP sequential thinking for step-by-step reasoning"""
        try:
            # This would call the actual MCP sequential thinking tool
            # For now, we simulate a structured thinking process
            steps = []
            
            # Break down the instruction into logical steps
            if 'step' in instruction.lower():
                steps.append(f"1. Understand: {instruction[:100]}")
                steps.append(f"2. Analyze: Consider the implications")
                steps.append(f"3. Conclude: {context.get('expected_outcome', 'Determine result')}")
            else:
                steps.append(f"Reasoning: {instruction}")
            
            return {
                "result": " → ".join(steps),
                "success": True,
                "steps": steps
            }
        except Exception as e:
            return {"result": f"Error: {str(e)}", "success": False}
    
    @staticmethod
    def memory(instruction: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates memory storage/retrieval"""
        memory_store = context.get('memory_store', {})
        
        if 'store' in instruction.lower() or 'remember' in instruction.lower():
            # Extract key-value to store
            parts = instruction.split(':')
            if len(parts) >= 2:
                key = parts[0].strip()
                value = ':'.join(parts[1:]).strip()
                memory_store[key] = value
                context['memory_store'] = memory_store
                return {"result": f"Stored: {key}", "success": True}
        elif 'recall' in instruction.lower() or 'retrieve' in instruction.lower():
            # Try to find relevant stored information
            for key, value in memory_store.items():
                if key.lower() in instruction.lower():
                    return {"result": value, "success": True}
            return {"result": "No matching memory found", "success": False}
        
        return {"result": "Memory operation unclear", "success": False}
    
    @staticmethod
    def search(instruction: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates search/pattern finding"""
        # Look for patterns in the problem or options
        problem = context.get('problem', '')
        options = context.get('options', [])
        
        search_term = instruction.lower()
        matches = []
        
        for i, option in enumerate(options, 1):
            if any(term in option.lower() for term in search_term.split()):
                matches.append(f"Option {i}: {option}")
        
        if matches:
            return {"result": "\n".join(matches), "success": True, "matches": matches}
        else:
            return {"result": "No matches found", "success": False}


# ═══════════════════════════════════════════════════════════════════════════════
# MCP-ENHANCED SMALL LLM WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class MCPEnhancedSmallLLM:
    """
    Wrapper that enhances a small LLM (Qwen2 1.5B) with MCP tool capabilities.
    
    The small LLM acts as an orchestrator that:
    1. Decomposes problems into tool-solvable steps
    2. Calls appropriate MCP tools for each step
    3. Aggregates tool results to produce final answer
    """
    
    def __init__(self, 
                 api_url: str = None,
                 api_key: str = None,
                 model_name: str = None,
                 mcp_config_path: str = None,
                 use_real_mcp: bool = False):
        """
        Initialize MCP-enhanced small LLM.
        
        Args:
            api_url: LLM API endpoint (default: LM Studio local)
            api_key: API key for LLM
            model_name: Model name (default: qwen2-1.5b-instruct)
            mcp_config_path: Path to MCP tools configuration
            use_real_mcp: Whether to use real MCP servers (requires setup)
        """
        self.api_url = api_url or LLM_API_URL
        self.api_key = api_key or LLM_API_KEY
        self.model_name = model_name or LLM_MODEL_NAME
        self.use_real_mcp = use_real_mcp
        
        # Initialize LLM
        if ChatOpenAI:
            self.llm = ChatOpenAI(
                base_url=self.api_url,
                api_key=self.api_key,
                model=self.model_name,
                temperature=0.3,  # Lower temperature for more focused reasoning
                max_tokens=2048,
                timeout=30
            )
            logger.info(f"Initialized ChatOpenAI with {self.model_name} at {self.api_url}")
        else:
            self.llm = None
            logger.warning("ChatOpenAI not available. Using fallback mode.")
        
        # Initialize MCP tools
        self.tool_simulator = MCPToolSimulator()
        self.mcp_tools = []
        
        if use_real_mcp and mcp_config_path and os.path.exists(mcp_config_path):
            try:
                self.mcp_tools = load_mcp_tools_from_registry(mcp_config_path)
                logger.info(f"Loaded {len(self.mcp_tools)} real MCP tools")
            except Exception as e:
                logger.warning(f"Failed to load MCP tools: {e}. Using simulator.")
        else:
            logger.info("Using MCP tool simulator")
    
    def _call_llm(self, prompt: str, system_msg: str = None) -> str:
        """Call the small LLM with a prompt"""
        try:
            if self.llm:
                messages = []
                if system_msg:
                    messages.append(SystemMessage(content=system_msg))
                messages.append(HumanMessage(content=prompt))
                
                response = self.llm.invoke(messages)
                return response.content
            else:
                # Fallback: return empty response
                logger.warning("LLM not available, returning fallback")
                return "{}"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "{}"
    
    def decompose_problem(self, problem: str, options: List[str]) -> List[Dict[str, Any]]:
        """
        Decompose problem into tool-executable steps.
        
        This is where the small LLM shines - not in reasoning,
        but in breaking down problems into tool-solvable chunks.
        """
        prompt = DECOMPOSITION_PROMPT.format(
            problem=problem,
            options="\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
        )
        
        response = self._call_llm(prompt, "You are a problem decomposition expert.")
        
        try:
            # Parse JSON response
            steps = json.loads(response)
            if isinstance(steps, list):
                return steps
        except json.JSONDecodeError:
            logger.warning("Failed to parse decomposition JSON, using heuristic")
        
        # Fallback: Generate heuristic steps based on problem type
        return self._heuristic_decomposition(problem, options)
    
    def _heuristic_decomposition(self, problem: str, options: List[str]) -> List[Dict[str, Any]]:
        """Fallback decomposition using heuristics"""
        steps = []
        problem_lower = problem.lower()
        
        # Detect problem type and suggest appropriate tools
        if any(word in problem_lower for word in ['calculate', 'add', 'multiply', 'divide', 'sum', 'total']):
            steps.append({
                "step": 1,
                "description": f"Extract numbers from: {problem[:100]}",
                "tool": "calculator"
            })
            steps.append({
                "step": 2,
                "description": "Perform calculation with extracted numbers",
                "tool": "calculator"
            })
        
        if any(word in problem_lower for word in ['sequence', 'pattern', 'next', 'series']):
            steps.append({
                "step": len(steps) + 1,
                "description": "Identify pattern in the sequence",
                "tool": "sequential_thinking"
            })
        
        if any(word in problem_lower for word in ['reason', 'logical', 'if', 'then', 'because']):
            steps.append({
                "step": len(steps) + 1,
                "description": "Apply logical reasoning to the problem",
                "tool": "sequential_thinking"
            })
        
        # Always add a final aggregation step
        steps.append({
            "step": len(steps) + 1,
            "description": f"Compare results with options and select best match",
            "tool": "search"
        })
        
        return steps if steps else [{"step": 1, "description": problem, "tool": "sequential_thinking"}]
    
    def execute_tool(self, tool_name: str, instruction: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool (real MCP or simulated)"""
        try:
            if self.use_real_mcp and self.mcp_tools:
                # Use real MCP tools
                for tool in self.mcp_tools:
                    if tool.name == tool_name:
                        result = tool.func(instruction)
                        return {"result": result, "success": True}
            
            # Use simulator
            if hasattr(self.tool_simulator, tool_name):
                return getattr(self.tool_simulator, tool_name)(instruction, context)
            else:
                return {"result": f"Tool {tool_name} not available", "success": False}
        
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"result": f"Error: {str(e)}", "success": False}
    
    def aggregate_results(self, problem: str, options: List[str], 
                         steps_and_results: List[Dict[str, Any]]) -> Tuple[int, str]:
        """
        Aggregate tool results to select final answer.
        
        Returns:
            (selected_index, reasoning)
        """
        # Format steps and results for LLM
        formatted_results = []
        for item in steps_and_results:
            formatted_results.append(
                f"Step {item['step']}: {item['description']}\n"
                f"Tool: {item['tool']}\n"
                f"Result: {item['result']}\n"
            )
        
        prompt = AGGREGATION_PROMPT.format(
            problem=problem,
            options="\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options)),
            steps_and_results="\n".join(formatted_results),
            num_options=len(options)
        )
        
        response = self._call_llm(prompt, "You are an answer selection expert.")
        
        # Extract number from response
        try:
            import re
            numbers = re.findall(r'\b([1-9]|[1-9][0-9])\b', response)
            if numbers:
                selected = int(numbers[0])
                if 1 <= selected <= len(options):
                    reasoning = "\n".join(formatted_results)
                    return selected, reasoning
        except Exception as e:
            logger.warning(f"Failed to parse aggregation response: {e}")
        
        # Fallback: Use heuristic selection
        return self._heuristic_selection(options, steps_and_results)
    
    def _heuristic_selection(self, options: List[str], 
                            steps_and_results: List[Dict[str, Any]]) -> Tuple[int, str]:
        """Fallback selection using heuristics"""
        # Look for successful tool results that match options
        for result_dict in steps_and_results:
            result_str = str(result_dict.get('result', '')).lower()
            
            for i, option in enumerate(options, 1):
                option_lower = option.lower()
                # Check for exact or partial matches
                if option_lower in result_str or result_str in option_lower:
                    reasoning = f"Matched result '{result_str}' with option {i}"
                    return i, reasoning
        
        # Default to first option if no match
        reasoning = "No clear match found, selected first option as default"
        return 1, reasoning
    
    def reason(self, problem: str, options: List[str]) -> Tuple[int, str, List[Dict[str, Any]]]:
        """
        Main reasoning method that orchestrates MCP tools.
        
        Returns:
            (selected_index, reasoning_trace, tool_execution_log)
        """
        logger.info(f"Reasoning over problem with {len(options)} options")
        
        # Step 1: Decompose problem
        steps = self.decompose_problem(problem, options)
        logger.info(f"Decomposed into {len(steps)} steps")
        
        # Step 2: Execute each step using appropriate tools
        context = {
            'problem': problem,
            'options': options,
            'memory_store': {}
        }
        
        steps_and_results = []
        for step in steps:
            tool_name = step.get('tool', 'sequential_thinking')
            instruction = step.get('description', '')
            
            logger.info(f"Executing step {step.get('step')}: {tool_name}")
            result = self.execute_tool(tool_name, instruction, context)
            
            steps_and_results.append({
                'step': step.get('step'),
                'description': instruction,
                'tool': tool_name,
                'result': result.get('result'),
                'success': result.get('success', False)
            })
        
        # Step 3: Aggregate results and select answer
        selected_index, reasoning = self.aggregate_results(problem, options, steps_and_results)
        
        # Build comprehensive reasoning trace
        trace = f"Problem Analysis:\n{problem}\n\n"
        trace += "Reasoning Steps:\n"
        for item in steps_and_results:
            trace += f"\nStep {item['step']} ({item['tool']}):\n"
            trace += f"Task: {item['description']}\n"
            trace += f"Result: {item['result']}\n"
        trace += f"\nConclusion: Selected option {selected_index}\n"
        trace += f"Rationale: {reasoning}"
        
        logger.info(f"Selected option {selected_index}")
        
        return selected_index, trace, steps_and_results


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION FOR INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_mcp_enhanced_engine(api_url: str = None, 
                               api_key: str = None,
                               model_name: str = None,
                               use_real_mcp: bool = False) -> MCPEnhancedSmallLLM:
    """
    Factory function to create MCP-enhanced reasoning engine.
    
    Usage:
        engine = create_mcp_enhanced_engine()
        index, trace, log = engine.reason(problem, options)
    """
    return MCPEnhancedSmallLLM(
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        use_real_mcp=use_real_mcp
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("MCP-ENHANCED SMALL LLM REASONING ENGINE TEST")
    print("=" * 80)
    
    # Create engine
    engine = create_mcp_enhanced_engine()
    
    # Test problem
    problem = """You are in a race and you overtake the second person. 
    What position are you in now?"""
    
    options = ["First", "Second", "Third", "Fourth", "Another answer"]
    
    print(f"\nProblem: {problem}")
    print(f"\nOptions:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    # Run reasoning
    print("\n" + "=" * 80)
    print("REASONING PROCESS")
    print("=" * 80)
    
    selected_index, trace, log = engine.reason(problem, options)
    
    print(f"\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"\nSelected: Option {selected_index} - {options[selected_index-1]}")
    print(f"\nReasoning Trace:\n{trace}")
    print(f"\nExecution Log:")
    for item in log:
        print(f"  - Step {item['step']}: {item['tool']} -> {item['success']}")
