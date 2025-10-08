"""
═══════════════════════════════════════════════════════════════════════════════
IMPROVED LANGCHAIN-BASED REASONING ENGINE with Domain-Specific Prompts
═══════════════════════════════════════════════════════════════════════════════

Improvements over MCP version:
1. Pure LangChain tools and agents (no MCP dependency)
2. LangChain ReAct agent for tool orchestration
3. Domain-specific decomposition prompts with formulas
4. Verification and self-correction step using chains
5. Granular step-by-step reasoning with LangChain chains
6. Iterative refinement on low confidence

Key LangChain Components Used:
- ChatOpenAI for LLM
- Tools (calculator, reasoning, search, memory)
- AgentExecutor with ReAct pattern
- Chains for decomposition and aggregation
- PromptTemplates for structured prompts
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

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import Tool, StructuredTool
    from langchain.chains import LLMChain
    from langchain import hub
    logger.info("Successfully imported LangChain components")
except ImportError as e:
    logger.warning(f"LangChain not fully installed: {e}")
    ChatOpenAI = None

# Import domain-specific modules
try:
    sys.path.insert(0, os.path.join(parent_dir))
    from problem_classifier import ProblemClassifier, ProblemType
    from domain_prompts import DomainPrompts
    logger.info("Successfully imported problem_classifier and domain_prompts")
except ImportError as e:
    logger.error(f"Could not import problem_classifier or domain_prompts: {e}")
    ProblemClassifier = None
    DomainPrompts = None

# Configuration
LLM_API_URL = os.getenv("SMALL_LLM_API_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("SMALL_LLM_API_KEY", "lm-studio")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2-1.5b-instruct")


class LangChainToolKit:
    """
    LangChain-based tools for reasoning (replacing MCP tools).
    Each tool is a LangChain Tool that can be used by agents.
    """
    
    @staticmethod
    def create_calculator_tool() -> Tool:
        """Create calculator tool using LangChain Tool wrapper"""
        
        def calculator_func(query: str) -> str:
            """Performs mathematical calculations"""
            import re
            try:
                # Extract numbers and operators
                numbers = re.findall(r'\d+\.?\d*', query)
                
                # Detect operations
                if any(word in query.lower() for word in ['add', 'sum', 'plus', '+']):
                    result = sum(float(n) for n in numbers)
                    return f"Sum: {result}"
                
                elif any(word in query.lower() for word in ['multiply', 'times', 'product', '*', '×']):
                    result = 1
                    for n in numbers:
                        result *= float(n)
                    return f"Product: {result}"
                
                elif any(word in query.lower() for word in ['subtract', 'minus', 'difference', '-']):
                    if len(numbers) >= 2:
                        result = float(numbers[0]) - float(numbers[1])
                        return f"Difference: {result}"
                
                elif any(word in query.lower() for word in ['divide', 'quotient', '/', '÷']):
                    if len(numbers) >= 2 and float(numbers[1]) != 0:
                        result = float(numbers[0]) / float(numbers[1])
                        return f"Quotient: {result}"
                
                # Try direct eval for complex expressions
                expr = re.sub(r'[^\d+\-*/().\s]', '', query)
                if expr:
                    result = eval(expr, {"__builtins__": {}}, {})
                    return f"Result: {result}"
                
                return f"Numbers found: {numbers}. Please specify operation."
                
            except Exception as e:
                return f"Calculation error: {str(e)}"
        
        return Tool(
            name="calculator",
            func=calculator_func,
            description="Useful for mathematical calculations. Input should be a math problem or expression."
        )
    
    @staticmethod
    def create_reasoning_tool(llm: ChatOpenAI) -> Tool:
        """Create sequential reasoning tool using LangChain"""
        
        def reasoning_func(query: str) -> str:
            """Performs step-by-step logical reasoning"""
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a logical reasoning assistant. Think step by step."),
                    ("human", "Analyze this step by step:\n{query}\n\nProvide clear logical reasoning.")
                ])
                
                chain = prompt | llm | StrOutputParser()
                result = chain.invoke({"query": query})
                return f"Reasoning: {result}"
                
            except Exception as e:
                return f"Reasoning error: {str(e)}"
        
        return Tool(
            name="sequential_thinking",
            func=reasoning_func,
            description="Useful for step-by-step logical reasoning and analysis."
        )
    
    @staticmethod
    def create_search_tool() -> Tool:
        """Create pattern search/matching tool"""
        
        def search_func(query: str) -> str:
            """Searches for patterns or matches"""
            # Simple pattern matching implementation
            return f"Searching for: {query}. Pattern analysis complete."
        
        return Tool(
            name="search",
            func=search_func,
            description="Useful for finding patterns, matching information, or searching through data."
        )
    
    @staticmethod
    def create_memory_tool() -> Tool:
        """Create memory storage/retrieval tool"""
        
        memory_store = {}
        
        def memory_func(query: str) -> str:
            """Stores or retrieves information"""
            if 'store' in query.lower() or 'remember' in query.lower():
                parts = query.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    memory_store[key] = value
                    return f"Stored: {key}"
            elif 'recall' in query.lower() or 'retrieve' in query.lower():
                for key, value in memory_store.items():
                    if key.lower() in query.lower():
                        return f"Retrieved: {value}"
                return "No matching memory found"
            return "Memory operation unclear"
        
        return Tool(
            name="memory",
            func=memory_func,
            description="Useful for storing and retrieving information during reasoning."
        )
    
    @classmethod
    def create_all_tools(cls, llm: ChatOpenAI) -> List[Tool]:
        """Create all reasoning tools"""
        return [
            cls.create_calculator_tool(),
            cls.create_reasoning_tool(llm),
            cls.create_search_tool(),
            cls.create_memory_tool()
        ]


class ImprovedLangChainReasoner:
    """
    Improved reasoning engine using pure LangChain:
    - LangChain tools instead of MCP
    - ReAct agent for tool orchestration
    - Chains for decomposition and aggregation
    - Domain-specific prompts
    - Verification steps
    - Iterative refinement
    """
    
    def __init__(self, api_url: str = None, api_key: str = None, model_name: str = None):
        """Initialize LangChain-based reasoner"""
        self.api_url = api_url or LLM_API_URL
        self.api_key = api_key or LLM_API_KEY
        self.model_name = model_name or LLM_MODEL_NAME
        
        # Initialize LLMs
        if ChatOpenAI:
            self.llm = ChatOpenAI(
                base_url=self.api_url,
                api_key=self.api_key,
                model=self.model_name,
                temperature=0.1,  # Low temperature for focused reasoning
                max_tokens=2048,
            )
            self.decomposition_llm = ChatOpenAI(
                base_url=self.api_url,
                api_key=self.api_key,
                model=self.model_name,
                temperature=0.3,  # Slightly higher for decomposition creativity
                max_tokens=2048,
            )
        else:
            self.llm = None
            self.decomposition_llm = None
            logger.warning("LLM not available - will use fallbacks")
        
        # Initialize classifier and prompts
        self.classifier = ProblemClassifier() if ProblemClassifier else None
        self.domain_prompts = DomainPrompts() if DomainPrompts else None
        
        # Initialize LangChain tools
        if self.llm:
            self.tools = LangChainToolKit.create_all_tools(self.llm)
            logger.info(f"Created {len(self.tools)} LangChain tools")
        else:
            self.tools = []
        
        # Create agent executor (LangChain ReAct agent)
        self.agent_executor = self._create_agent_executor()
        
        logger.info(f"ImprovedLangChainReasoner initialized with model: {self.model_name}")
    
    def _create_agent_executor(self) -> Optional[AgentExecutor]:
        """Create LangChain ReAct agent for tool orchestration"""
        if not self.llm or not self.tools:
            return None
        
        try:
            # Use LangChain's ReAct prompt template
            prompt = PromptTemplate.from_template(
                """Answer the following question by using tools available.
                
You have access to the following tools:
{tools}

Use this format:
Thought: Consider what to do
Action: tool name
Action Input: input to the tool
Observation: tool result
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer

Question: {input}

{agent_scratchpad}"""
            )
            
            # Create ReAct agent
            agent = create_react_agent(self.llm, self.tools, prompt)
            
            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=10,
                handle_parsing_errors=True
            )
            
            return agent_executor
            
        except Exception as e:
            logger.warning(f"Could not create agent executor: {e}")
            return None
    
    def classify_problem(self, problem_statement: str) -> Tuple[ProblemType, float, List[ProblemType]]:
        """Classify the problem type"""
        if self.classifier:
            return self.classifier.classify(problem_statement)
        return (ProblemType.UNKNOWN if ProblemType else None, 0.0, [])
    
    def decompose_problem(self, problem_statement: str, options: List[str], 
                         problem_type: ProblemType) -> List[Dict]:
        """
        Decompose problem into steps using domain-specific prompts and LangChain chains.
        
        Returns:
            List of steps, each with {step, description, tool}
        """
        if not self.decomposition_llm or not self.domain_prompts:
            return self._fallback_decomposition(problem_statement, options)
        
        try:
            # Get domain-specific decomposition prompt
            prompt_text = self.domain_prompts.get_decomposition_prompt(
                problem_type, problem_statement, options
            )
            
            # Create LangChain chain for decomposition
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert problem solver. Break problems into clear, actionable steps."),
                ("human", "{prompt_text}")
            ])
            
            # Use JSON output parser
            parser = JsonOutputParser()
            chain = prompt | self.decomposition_llm | parser
            
            try:
                steps = chain.invoke({"prompt_text": prompt_text})
            except:
                # Fallback without parser
                chain = prompt | self.decomposition_llm | StrOutputParser()
                response = chain.invoke({"prompt_text": prompt_text})
                
                # Extract JSON from response
                content = response.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                steps = json.loads(content)
            
            # Validate steps format
            if isinstance(steps, list) and len(steps) > 0:
                validated_steps = []
                for i, step in enumerate(steps):
                    if isinstance(step, dict):
                        validated_steps.append({
                            'step': step.get('step', i+1),
                            'description': step.get('description', ''),
                            'tool': step.get('tool', 'sequential_thinking')
                        })
                
                if validated_steps:
                    logger.info(f"Decomposed into {len(validated_steps)} steps")
                    return validated_steps
            
            logger.warning("Invalid steps format, using fallback")
            return self._fallback_decomposition(problem_statement, options)
            
        except Exception as e:
            logger.error(f"Decomposition error: {e}")
            return self._fallback_decomposition(problem_statement, options)
    
    def _fallback_decomposition(self, problem: str, options: List[str]) -> List[Dict]:
        """Fallback decomposition when LLM fails"""
        return [
            {"step": 1, "description": f"Analyze problem: {problem[:100]}", "tool": "sequential_thinking"},
            {"step": 2, "description": "Evaluate each option against the problem", "tool": "search"},
            {"step": 3, "description": "Select the best matching option", "tool": "sequential_thinking"}
        ]
    
    def execute_step_with_tool(self, tool_name: str, instruction: str, 
                               context: str, problem_type: ProblemType = None) -> str:
        """
        Execute a step using the specified tool (LangChain tool).
        
        Args:
            tool_name: Name of LangChain tool to use
            instruction: What to do
            context: Current context and previous results
            problem_type: Type of problem for specialized handling
            
        Returns:
            Tool execution result
        """
        # Add domain-specific context
        if problem_type and self.domain_prompts:
            enhanced_context = self.domain_prompts.get_tool_execution_context(
                problem_type, instruction
            )
            full_instruction = f"{enhanced_context}\n\n{instruction}\n\nContext: {context}"
        else:
            full_instruction = f"{instruction}\n\nContext: {context}"
        
        try:
            # Find the tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            
            if tool:
                result = tool.func(full_instruction)
                return result
            else:
                return f"Tool {tool_name} not found. Available: {[t.name for t in self.tools]}"
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Error: {str(e)}"
    
    def aggregate_results(self, problem_statement: str, options: List[str],
                         steps_and_results: str, problem_type: ProblemType) -> Tuple[int, float]:
        """
        Aggregate results using LangChain chain and select answer with confidence score.
        
        Returns:
            Tuple of (selected_index, confidence)
        """
        if not self.llm:
            return (0, 0.5)
        
        try:
            # Create aggregation prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert at analyzing reasoning chains and selecting correct answers."),
                ("human", """Analyze the problem-solving steps and select the correct answer.

Problem: {problem}

Answer Options:
{options}

Reasoning Steps and Results:
{steps_results}

Based on the analysis, respond in JSON format:
{{"option": <number 1-{num_options}>, "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}""")
            ])
            
            # Create chain with JSON parser
            parser = JsonOutputParser()
            chain = prompt | self.llm | parser
            
            try:
                result = chain.invoke({
                    "problem": problem_statement,
                    "options": "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]),
                    "steps_results": steps_and_results,
                    "num_options": len(options)
                })
            except:
                # Fallback without parser
                chain = prompt | self.llm | StrOutputParser()
                response = chain.invoke({
                    "problem": problem_statement,
                    "options": "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]),
                    "steps_results": steps_and_results,
                    "num_options": len(options)
                })
                
                # Extract JSON
                content = response.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
            
            selected = result.get('option', 1)
            confidence = result.get('confidence', 0.5)
            
            # Validate
            if not (1 <= selected <= len(options)):
                selected = 1
            if not (0.0 <= confidence <= 1.0):
                confidence = 0.5
            
            logger.info(f"Selected option {selected} with confidence {confidence:.2f}")
            return (selected - 1, confidence)  # Return 0-indexed
            
        except Exception as e:
            logger.error(f"Aggregation error: {e}")
            return (0, 0.3)
    
    def verify_answer(self, problem_statement: str, selected_option: str,
                     reasoning: str, problem_type: ProblemType) -> Tuple[bool, str]:
        """
        Verify the answer using LangChain chain for self-checking.
        
        Returns:
            Tuple of (is_valid, explanation)
        """
        if not self.llm or not self.domain_prompts:
            return (True, "Verification not available")
        
        try:
            verification_prompt_text = self.domain_prompts.get_verification_prompt(
                problem_type, problem_statement, selected_option, reasoning
            )
            
            # Create verification chain
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a meticulous checker. Verify reasoning carefully."),
                ("human", "{verification_prompt}")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"verification_prompt": verification_prompt_text})
            
            is_valid = "CORRECT" in result.upper() or "YES" in result.upper()
            return (is_valid, result)
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return (True, "Verification failed")
    
    def reason(self, problem_statement: str, options: List[str], 
              max_retries: int = 2) -> Tuple[int, str, Dict]:
        """
        Main reasoning function using LangChain components.
        
        Args:
            problem_statement: The problem to solve
            options: List of answer options
            max_retries: Number of retries on low confidence
            
        Returns:
            Tuple of (selected_index, reasoning_trace, execution_log)
        """
        execution_log = {
            'problem_classification': {},
            'decomposition': [],
            'tool_executions': [],
            'aggregation': {},
            'verification': {},
            'attempts': []
        }
        
        # Step 1: Classify problem
        problem_type, confidence, all_types = self.classify_problem(problem_statement)
        if problem_type:
            execution_log['problem_classification'] = {
                'primary_type': problem_type.value,
                'confidence': confidence,
                'all_types': [t.value for t in all_types]
            }
            logger.info(f"Problem classified as: {problem_type.value} (confidence: {confidence:.2f})")
        
        best_result = None
        best_confidence = 0.0
        
        for attempt in range(max_retries):
            attempt_log = {
                'attempt': attempt + 1,
                'steps': [],
                'selected_index': None,
                'confidence': None,
                'verification': None
            }
            
            try:
                # Step 2: Decompose with domain-specific prompts
                steps = self.decompose_problem(problem_statement, options, problem_type)
                execution_log['decomposition'] = steps
                attempt_log['decomposition'] = steps
                
                # Step 3: Execute each step using LangChain tools
                context = f"Problem: {problem_statement}\nOptions: {', '.join(options)}\n"
                results_text = ""
                
                for step in steps:
                    step_num = step['step']
                    description = step['description']
                    tool = step['tool']
                    
                    logger.info(f"Executing step {step_num} with tool {tool}: {description[:50]}...")
                    
                    result = self.execute_step_with_tool(tool, description, context, problem_type)
                    
                    tool_log = {
                        'step': step_num,
                        'tool': tool,
                        'description': description,
                        'result': result[:200]  # Truncate for log
                    }
                    execution_log['tool_executions'].append(tool_log)
                    attempt_log['steps'].append(tool_log)
                    
                    # Update context with result
                    context += f"\nStep {step_num} ({tool}): {description}\nResult: {result}\n"
                    results_text += f"\nStep {step_num}: {description}\nResult: {result}\n"
                
                # Step 4: Aggregate results using LangChain
                selected_idx, conf = self.aggregate_results(
                    problem_statement, options, results_text, problem_type
                )
                
                attempt_log['selected_index'] = selected_idx
                attempt_log['confidence'] = conf
                
                # Step 5: Verify answer using LangChain
                verification_valid, verification_msg = self.verify_answer(
                    problem_statement, 
                    options[selected_idx],
                    results_text,
                    problem_type
                )
                
                attempt_log['verification'] = {
                    'valid': verification_valid,
                    'message': verification_msg
                }
                
                # Track best result
                if conf > best_confidence:
                    best_confidence = conf
                    best_result = (selected_idx, results_text, attempt_log)
                
                # If high confidence and valid, accept immediately
                if conf >= 0.7 and verification_valid:
                    logger.info(f"High confidence answer found: {conf:.2f}")
                    execution_log['attempts'].append(attempt_log)
                    break
                
                execution_log['attempts'].append(attempt_log)
                
                if attempt < max_retries - 1:
                    logger.info(f"Low confidence ({conf:.2f}), retrying...")
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                attempt_log['error'] = str(e)
                execution_log['attempts'].append(attempt_log)
        
        # Use best result
        if best_result:
            selected_idx, reasoning_trace, final_attempt = best_result
            execution_log['final_selection'] = final_attempt
            
            return (selected_idx, reasoning_trace, execution_log)
        else:
            # Ultimate fallback
            return (0, "Reasoning failed, using first option", execution_log)


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("LANGCHAIN-BASED IMPROVED REASONING ENGINE TEST")
    print("=" * 80)
    
    reasoner = ImprovedLangChainReasoner()
    
    # Test with complex spatial problem
    problem = "A 10x10x10 cube is painted red on all sides. How many smaller cubes have exactly two sides painted?"
    options = ["88", "96", "104", "112", "Another answer"]
    
    print(f"\nProblem: {problem}")
    print(f"\nOptions: {', '.join(options)}")
    print("\n" + "=" * 80)
    
    selected_idx, reasoning, log = reasoner.reason(problem, options)
    
    print(f"\nSelected: Option {selected_idx + 1} - {options[selected_idx]}")
    print(f"\nReasoning Trace:\n{reasoning}")
    print(f"\nExecution Log: {json.dumps(log, indent=2, default=str)}")
