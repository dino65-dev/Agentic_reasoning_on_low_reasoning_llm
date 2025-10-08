"""
═══════════════════════════════════════════════════════════════════════════════
LANGCHAIN-ENHANCED SMALL LLM REASONING ENGINE
═══════════════════════════════════════════════════════════════════════════════

Purpose: Enable Qwen2 1.5B (non-reasoning LLM) to perform complex reasoning tasks
         by leveraging LangChain tools, agents, and chains as external cognitive aids.

Key Strategy:
- Small LLM = Task decomposer + Coordinator
- LangChain Tools = Actual reasoning engines (calculator, sequential thinking, memory, etc.)
- LangChain Agents = Tool orchestration with ReAct pattern
- Pattern: Break down complex problems → Use LangChain tools → Aggregate results

This approach transforms a weak LLM into a powerful reasoning system through LangChain's
tool use and agent framework.

Key LangChain Components:
- Tools: Calculator, Reasoning, Search, Memory
- Agents: ReAct agent for tool orchestration
- Chains: For decomposition, aggregation, verification
- PromptTemplates: Structured prompts for each step
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import Tool
    from langchain.chains import LLMChain
    logger.info("Successfully imported LangChain components")
except ImportError as e:
    logger.warning(f"LangChain not fully installed: {e}")
    ChatOpenAI = None
    Tool = None

# Configuration
LLM_API_URL = os.getenv("SMALL_LLM_API_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("SMALL_LLM_API_KEY", "lm-studio")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2-1.5b-instruct")

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
# LANGCHAIN TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class LangChainToolImplementations:
    """LangChain Tool implementations for reasoning augmentation"""
    
    @staticmethod
    def create_calculator_tool():
        """Creates a calculator tool using LangChain Tool wrapper"""
        
        def calculator_func(instruction: str) -> str:
            """Performs mathematical calculations"""
            try:
                import re
                numbers = re.findall(r'\d+\.?\d*', instruction)
                
                # Simple operations
                if 'add' in instruction.lower() or '+' in instruction:
                    result = sum(float(n) for n in numbers)
                    return f"Sum: {result}"
                elif 'multiply' in instruction.lower() or '*' in instruction or '×' in instruction:
                    result = 1
                    for n in numbers:
                        result *= float(n)
                    return f"Product: {result}"
                elif 'subtract' in instruction.lower() or '-' in instruction:
                    if len(numbers) >= 2:
                        result = float(numbers[0]) - float(numbers[1])
                        return f"Difference: {result}"
                elif 'divide' in instruction.lower() or '/' in instruction or '÷' in instruction:
                    if len(numbers) >= 2 and float(numbers[1]) != 0:
                        result = float(numbers[0]) / float(numbers[1])
                        return f"Quotient: {result}"
                
                # Try eval as fallback
                expr = re.sub(r'[^\d+\-*/().\s]', '', instruction)
                if expr.strip():
                    result = eval(expr, {"__builtins__": {}}, {})
                    return f"Result: {result}"
                
                return f"Numbers found: {numbers}. Cannot determine operation."
            except Exception as e:
                return f"Calculation error: {str(e)}"
        
        if Tool:
            return Tool(
                name="calculator",
                func=calculator_func,
                description="Useful for mathematical calculations. Input should be a math problem or expression."
            )
        return None
    
    @staticmethod
    def create_sequential_thinking_tool(llm):
        """Creates a sequential reasoning tool using LangChain"""
        
        def thinking_func(instruction: str) -> str:
            """Uses LangChain for step-by-step reasoning"""
            try:
                if not llm:
                    # Fallback: structured thinking process
                    steps = [
                        f"1. Understand: {instruction[:100]}",
                        "2. Analyze: Consider the implications",
                        "3. Conclude: Determine result"
                    ]
                    return " → ".join(steps)
                
                # Use LangChain chain for reasoning
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a logical reasoning assistant. Think step by step."),
                    ("human", "Reason through this step by step:\n{instruction}")
                ])
                
                chain = prompt | llm | StrOutputParser()
                result = chain.invoke({"instruction": instruction})
                return f"Reasoning: {result}"
                
            except Exception as e:
                return f"Reasoning: {instruction} (processed with basic logic)"
        
        if Tool:
            return Tool(
                name="sequential_thinking",
                func=thinking_func,
                description="Useful for step-by-step logical reasoning and analysis."
            )
        return None
    
    @staticmethod
    def create_memory_tool():
        """Creates memory storage/retrieval tool"""
        memory_store = {}
        
        def memory_func(instruction: str) -> str:
            """Simulates memory storage/retrieval"""
            if 'store' in instruction.lower() or 'remember' in instruction.lower():
                parts = instruction.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    memory_store[key] = value
                    return f"Stored: {key}"
            elif 'recall' in instruction.lower() or 'retrieve' in instruction.lower():
                for key, value in memory_store.items():
                    if key.lower() in instruction.lower():
                        return f"Retrieved: {value}"
                return "No matching memory found"
            return "Memory operation unclear"
        
        if Tool:
            return Tool(
                name="memory",
                func=memory_func,
                description="Useful for storing and retrieving information during reasoning."
            )
        return None
    
    @staticmethod
    def create_search_tool():
        """Creates search/pattern finding tool"""
        
        def search_func(instruction: str) -> str:
            """Simulates search/pattern finding"""
            # This would interface with actual search in production
            return f"Searching for patterns in: {instruction[:100]}. Pattern analysis complete."
        
        if Tool:
            return Tool(
                name="search",
                func=search_func,
                description="Useful for finding patterns, matching information, or searching through data."
            )
        return None
    
    @classmethod
    def create_all_tools(cls, llm=None):
        """Creates all available LangChain tools"""
        tools = []
        
        calc_tool = cls.create_calculator_tool()
        if calc_tool:
            tools.append(calc_tool)
        
        thinking_tool = cls.create_sequential_thinking_tool(llm)
        if thinking_tool:
            tools.append(thinking_tool)
        
        memory_tool = cls.create_memory_tool()
        if memory_tool:
            tools.append(memory_tool)
        
        search_tool = cls.create_search_tool()
        if search_tool:
            tools.append(search_tool)
        
        return tools


# ═══════════════════════════════════════════════════════════════════════════════
# LANGCHAIN-ENHANCED SMALL LLM WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class LangChainEnhancedSmallLLM:
    """
    Wrapper that enhances a small LLM (Qwen2 1.5B) with LangChain tool capabilities.
    
    The small LLM acts as an orchestrator that:
    1. Decomposes problems into tool-solvable steps using LangChain chains
    2. Calls appropriate LangChain tools for each step
    3. Aggregates tool results using LangChain chains to produce final answer
    """
    
    def __init__(self, 
                 api_url: str = None,
                 api_key: str = None,
                 model_name: str = None,
                 use_react_agent: bool = False):
        """
        Initialize LangChain-enhanced small LLM.
        
        Args:
            api_url: LLM API endpoint (default: LM Studio local)
            api_key: API key for LLM
            model_name: Model name (default: qwen2-1.5b-instruct)
            use_react_agent: Whether to use ReAct agent for tool orchestration
        """
        self.api_url = api_url or LLM_API_URL
        self.api_key = api_key or LLM_API_KEY
        self.model_name = model_name or LLM_MODEL_NAME
        self.use_react_agent = use_react_agent
        
        # Initialize LLM
        if ChatOpenAI:
            self.llm = ChatOpenAI(
                base_url=self.api_url,
                api_key=self.api_key,
                model=self.model_name,
                temperature=0.3,
                max_tokens=2048,
                timeout=30
            )
            logger.info(f"Initialized ChatOpenAI with {self.model_name} at {self.api_url}")
        else:
            self.llm = None
            logger.warning("ChatOpenAI not available. Using fallback mode.")
        
        # Initialize LangChain tools
        self.tools = LangChainToolImplementations.create_all_tools(self.llm)
        logger.info(f"Created {len(self.tools)} LangChain tools")
        
        # Optionally create ReAct agent
        self.agent_executor = None
        if self.use_react_agent and self.llm and self.tools:
            self.agent_executor = self._create_react_agent()
    
    def _create_react_agent(self):
        """Creates a LangChain ReAct agent for tool orchestration"""
        try:
            # Create ReAct prompt
            react_prompt = PromptTemplate.from_template(
                """Answer the following question by reasoning and using tools.
                
You have access to the following tools:
{tools}

Use this format:
Thought: Consider what to do
Action: tool name
Action Input: input to the tool
Observation: tool result
... (repeat as needed)
Thought: I now know the final answer
Final Answer: the final answer

Question: {input}

{agent_scratchpad}"""
            )
            
            # Create agent and executor
            agent = create_react_agent(self.llm, self.tools, react_prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=10,
                handle_parsing_errors=True
            )
            
            logger.info("Created ReAct agent executor")
            return agent_executor
            
        except Exception as e:
            logger.warning(f"Could not create ReAct agent: {e}")
            return None
    
    def _call_llm(self, prompt: str, system_msg: str = None) -> str:
        """Call the small LLM with a prompt using LangChain"""
        try:
            if self.llm:
                messages = []
                if system_msg:
                    messages.append(SystemMessage(content=system_msg))
                messages.append(HumanMessage(content=prompt))
                
                response = self.llm.invoke(messages)
                return response.content
            else:
                logger.warning("LLM not available, returning fallback")
                return "{}"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "{}"
    
    def decompose_problem(self, problem: str, options: List[str]) -> List[Dict[str, Any]]:
        """
        Decompose problem into tool-executable steps using LangChain chains.
        
        This is where the small LLM shines - not in reasoning,
        but in breaking down problems into tool-solvable chunks.
        """
        # Create decomposition chain
        prompt = ChatPromptTemplate.from_template(DECOMPOSITION_PROMPT)
        
        if self.llm:
            try:
                chain = prompt | self.llm | JsonOutputParser()
                steps = chain.invoke({
                    "problem": problem,
                    "options": "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
                })
                
                if isinstance(steps, list) and len(steps) > 0:
                    return steps
            except:
                # Fallback: try without JSON parser
                try:
                    chain = prompt | self.llm | StrOutputParser()
                    response = chain.invoke({
                        "problem": problem,
                        "options": "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
                    })
                    
                    # Try to extract JSON
                    import re
                    json_match = re.search(r'\[.*\]', response, re.DOTALL)
                    if json_match:
                        steps = json.loads(json_match.group())
                        if isinstance(steps, list):
                            return steps
                except Exception as e:
                    logger.warning(f"Failed to parse decomposition: {e}")
        
        # Fallback: heuristic decomposition
        logger.warning("Using heuristic decomposition")
        return self._heuristic_decomposition(problem, options)
    
    def _heuristic_decomposition(self, problem: str, options: List[str]) -> List[Dict[str, Any]]:
        """Fallback decomposition using heuristics"""
        steps = []
        problem_lower = problem.lower()
        
        # Detect problem type and suggest appropriate tools
        if any(word in problem_lower for word in ['calculate', 'add', 'multiply', 'divide', 'sum', 'total']):
            steps.append({
                "step": 1,
                "description": f"Extract numbers and perform calculation from: {problem[:100]}",
                "tool": "calculator"
            })
        
        if any(word in problem_lower for word in ['reason', 'logical', 'if', 'then', 'because', 'think']):
            steps.append({
                "step": len(steps) + 1,
                "description": "Apply logical reasoning to analyze the problem",
                "tool": "sequential_thinking"
            })
        
        # Always add a final step
        steps.append({
            "step": len(steps) + 1,
            "description": "Compare results with options and select best match",
            "tool": "search"
        })
        
        return steps if steps else [{"step": 1, "description": problem, "tool": "sequential_thinking"}]
    
    def execute_tool(self, tool_name: str, instruction: str) -> Dict[str, Any]:
        """Execute a LangChain tool"""
        try:
            # Find the tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            
            if tool:
                result = tool.func(instruction)
                return {"result": result, "success": True}
            else:
                return {
                    "result": f"Tool {tool_name} not found. Available: {[t.name for t in self.tools]}", 
                    "success": False
                }
        
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"result": f"Error: {str(e)}", "success": False}
    
    def aggregate_results(self, problem: str, options: List[str], 
                         steps_and_results: List[Dict[str, Any]]) -> Tuple[int, str]:
        """
        Aggregate tool results using LangChain chain to select final answer.
        
        Returns:
            (selected_index, reasoning)
        """
        # Format steps and results
        formatted_results = []
        for item in steps_and_results:
            formatted_results.append(
                f"Step {item['step']}: {item['description']}\n"
                f"Tool: {item['tool']}\n"
                f"Result: {item['result']}\n"
            )
        
        # Create aggregation chain
        prompt = ChatPromptTemplate.from_template(AGGREGATION_PROMPT)
        
        if self.llm:
            try:
                chain = prompt | self.llm | StrOutputParser()
                response = chain.invoke({
                    "problem": problem,
                    "options": "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options)),
                    "steps_and_results": "\n".join(formatted_results),
                    "num_options": len(options)
                })
                
                # Extract number from response
                import re
                numbers = re.findall(r'\b([1-9]|[1-9][0-9])\b', response)
                if numbers:
                    selected = int(numbers[0])
                    if 1 <= selected <= len(options):
                        reasoning = "\n".join(formatted_results)
                        return selected, reasoning
            except Exception as e:
                logger.warning(f"Aggregation chain failed: {e}")
        
        # Fallback: heuristic selection
        return self._heuristic_selection(options, steps_and_results)
    
    def _heuristic_selection(self, options: List[str], 
                            steps_and_results: List[Dict[str, Any]]) -> Tuple[int, str]:
        """Fallback selection using heuristics"""
        for result_dict in steps_and_results:
            result_str = str(result_dict.get('result', '')).lower()
            
            for i, option in enumerate(options, 1):
                option_lower = option.lower()
                if option_lower in result_str or result_str in option_lower:
                    reasoning = f"Matched result '{result_str}' with option {i}"
                    return i, reasoning
        
        reasoning = "No clear match found, selected first option as default"
        return 1, reasoning
    
    def reason(self, problem: str, options: List[str]) -> Tuple[int, str, List[Dict[str, Any]]]:
        """
        Main reasoning method using LangChain tools and chains.
        
        Returns:
            (selected_index, reasoning_trace, tool_execution_log)
        """
        logger.info(f"Reasoning over problem with {len(options)} options")
        
        # If using ReAct agent, delegate to it
        if self.agent_executor:
            try:
                result = self.agent_executor.invoke({
                    "input": f"Problem: {problem}\nOptions: {', '.join(options)}\nSelect the correct option."
                })
                # Parse agent output
                output = result.get('output', '')
                # Try to extract option number
                import re
                numbers = re.findall(r'\b([1-9])\b', output)
                if numbers:
                    selected_index = int(numbers[0])
                    return selected_index, output, []
            except Exception as e:
                logger.warning(f"ReAct agent failed: {e}, using manual orchestration")
        
        # Manual orchestration with LangChain tools
        # Step 1: Decompose problem using LangChain chain
        steps = self.decompose_problem(problem, options)
        logger.info(f"Decomposed into {len(steps)} steps")
        
        # Step 2: Execute each step using LangChain tools
        steps_and_results = []
        for step in steps:
            tool_name = step.get('tool', 'sequential_thinking')
            instruction = step.get('description', '')
            
            logger.info(f"Executing step {step.get('step')}: {tool_name}")
            result = self.execute_tool(tool_name, instruction)
            
            steps_and_results.append({
                'step': step.get('step'),
                'description': instruction,
                'tool': tool_name,
                'result': result.get('result'),
                'success': result.get('success', False)
            })
        
        # Step 3: Aggregate results using LangChain chain
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

def create_langchain_enhanced_engine(api_url: str = None, 
                                     api_key: str = None,
                                     model_name: str = None,
                                     use_react_agent: bool = False) -> LangChainEnhancedSmallLLM:
    """
    Factory function to create LangChain-enhanced reasoning engine.
    
    Usage:
        engine = create_langchain_enhanced_engine()
        index, trace, log = engine.reason(problem, options)
    """
    return LangChainEnhancedSmallLLM(
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        use_react_agent=use_react_agent
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("LANGCHAIN-ENHANCED SMALL LLM REASONING ENGINE TEST")
    print("=" * 80)
    
    # Create engine
    engine = create_langchain_enhanced_engine(use_react_agent=False)
    
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
    
    print("\n" + "=" * 80)
    print("LANGCHAIN TOOLS USED:")
    print("=" * 80)
    for tool in engine.tools:
        print(f"  - {tool.name}: {tool.description}")
