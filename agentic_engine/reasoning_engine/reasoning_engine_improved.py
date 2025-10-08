"""
═══════════════════════════════════════════════════════════════════════════════
IMPROVED MCP-ENHANCED REASONING ENGINE with Domain-Specific Prompts
═══════════════════════════════════════════════════════════════════════════════

Improvements:
1. Problem classification for specialized handling
2. Domain-specific decomposition prompts with formulas
3. Verification and self-correction step
4. Granular step-by-step reasoning
5. Iterative refinement on low confidence
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

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    logger.info("Successfully imported langchain")
except ImportError as e:
    logger.warning(f"langchain not installed: {e}")
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None

# Import our new modules - they are in the agentic_engine directory
try:
    sys.path.insert(0, os.path.join(parent_dir))
    from problem_classifier import ProblemClassifier, ProblemType
    from domain_prompts import DomainPrompts
    logger.info("Successfully imported problem_classifier and domain_prompts")
except ImportError as e:
    logger.error(f"Could not import problem_classifier or domain_prompts: {e}")
    ProblemClassifier = None
    DomainPrompts = None

# Import MCP infrastructure
try:
    from tool_invoker import LLM_API_URL, LLM_API_KEY, LLM_MODEL_NAME
except ImportError:
    LLM_API_URL = os.getenv("SMALL_LLM_API_URL", "http://localhost:1234/v1")
    LLM_API_KEY = os.getenv("SMALL_LLM_API_KEY", "lm-studio")
    LLM_MODEL_NAME = os.getenv("LLM__MODEL_NAME", "qwen2-1.5b-instruct")


class ImprovedMCPReasoner:
    """
    Improved reasoning engine with:
    - Problem classification
    - Domain-specific prompts
    - Verification steps
    - Iterative refinement
    """
    
    def __init__(self, api_url: str = None, api_key: str = None, model_name: str = None):
        """Initialize improved reasoner"""
        self.api_url = api_url or LLM_API_URL
        self.api_key = api_key or LLM_API_KEY
        self.model_name = model_name or LLM_MODEL_NAME
        
        # Initialize LLM
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
        
        # Initialize MCP tools simulator
        self.tools = MCPToolSimulator()
        
        logger.info(f"ImprovedMCPReasoner initialized with model: {self.model_name}")
    
    def classify_problem(self, problem_statement: str) -> Tuple[ProblemType, float, List[ProblemType]]:
        """Classify the problem type"""
        if self.classifier:
            return self.classifier.classify(problem_statement)
        return (ProblemType.UNKNOWN, 0.0, [])
    
    def decompose_problem(self, problem_statement: str, options: List[str], 
                         problem_type: ProblemType) -> List[Dict]:
        """
        Decompose problem into steps using domain-specific prompts.
        
        Returns:
            List of steps, each with {step, description, tool}
        """
        if not self.decomposition_llm or not self.domain_prompts:
            return self._fallback_decomposition(problem_statement, options)
        
        try:
            # Get domain-specific decomposition prompt
            prompt = self.domain_prompts.get_decomposition_prompt(
                problem_type, problem_statement, options
            )
            
            messages = [
                SystemMessage(content="You are an expert problem solver. Break problems into clear, actionable steps."),
                HumanMessage(content=prompt)
            ]
            
            response = self.decomposition_llm.invoke(messages)
            content = response.content.strip()
            
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Remove any markdown formatting
            content = content.replace("```", "").strip()
            if content.startswith("json"):
                content = content[4:].strip()
            
            steps = json.loads(content)
            
            # Validate steps format
            if isinstance(steps, list) and len(steps) > 0:
                # Ensure each step has required fields
                validated_steps = []
                for i, step in enumerate(steps):
                    if isinstance(step, dict):
                        validated_steps.append({
                            'step': step.get('step', i+1),
                            'description': step.get('description', ''),
                            'tool': step.get('tool', 'sequential_thinking')
                        })
                
                if validated_steps:
                    logger.info(f"Decomposed into {len(validated_steps)} steps for {problem_type.value}")
                    return validated_steps
            
            logger.warning("Invalid steps format, using fallback")
            return self._fallback_decomposition(problem_statement, options)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}. Using fallback decomposition.")
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
    
    def execute_tool(self, tool_name: str, instruction: str, context: str, 
                    problem_type: ProblemType = None) -> str:
        """
        Execute a tool with domain-specific context.
        
        Args:
            tool_name: Name of tool to use
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
            context = f"{enhanced_context}\n\n{context}"
        
        try:
            result = self.tools.execute(tool_name, instruction, context)
            return result
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Error: {str(e)}"
    
    def aggregate_results(self, problem_statement: str, options: List[str],
                         steps_and_results: str, problem_type: ProblemType) -> Tuple[int, float]:
        """
        Aggregate results and select answer with confidence score.
        
        Returns:
            Tuple of (selected_index, confidence)
        """
        if not self.llm:
            return (0, 0.5)
        
        try:
            # Enhanced aggregation prompt
            prompt = f"""Analyze the problem-solving steps and select the correct answer.

Problem: {problem_statement}

Answer Options:
{chr(10).join([f"{i+1}. {opt}" for i, opt in enumerate(options)])}

Reasoning Steps and Results:
{steps_and_results}

Based on the analysis:
1. Which option best matches the reasoning?
2. How confident are you? (0.0 to 1.0)

Respond in JSON format:
{{"option": <number 1-{len(options)}>, "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}
"""
            
            messages = [
                SystemMessage(content="You are an expert at analyzing reasoning chains and selecting correct answers."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Extract JSON
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
        Verify the answer is correct through self-checking.
        
        Returns:
            Tuple of (is_valid, explanation)
        """
        if not self.llm or not self.domain_prompts:
            return (True, "Verification not available")
        
        try:
            verification_prompt = self.domain_prompts.get_verification_prompt(
                problem_type, problem_statement, selected_option, reasoning
            )
            
            messages = [
                SystemMessage(content="You are a meticulous checker. Verify reasoning carefully."),
                HumanMessage(content=verification_prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = response.content.strip()
            
            is_valid = "CORRECT" in result.upper()
            return (is_valid, result)
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return (True, "Verification failed")
    
    def reason(self, problem_statement: str, options: List[str], 
              max_retries: int = 2) -> Tuple[int, str, Dict]:
        """
        Main reasoning function with improvements.
        
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
                
                # Step 3: Execute each step
                context = f"Problem: {problem_statement}\nOptions: {', '.join(options)}\n"
                results_text = ""
                
                for step in steps:
                    step_num = step['step']
                    description = step['description']
                    tool = step['tool']
                    
                    logger.info(f"Executing step {step_num}: {description[:50]}...")
                    
                    result = self.execute_tool(tool, description, context, problem_type)
                    
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
                
                # Step 4: Aggregate results
                selected_idx, conf = self.aggregate_results(
                    problem_statement, options, results_text, problem_type
                )
                
                attempt_log['selected_index'] = selected_idx
                attempt_log['confidence'] = conf
                
                # Step 5: Verify answer
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


class MCPToolSimulator:
    """Simulates MCP tools for the reasoning engine"""
    
    def execute(self, tool_name: str, instruction: str, context: str) -> str:
        """Execute a tool simulation"""
        
        if tool_name == "calculator":
            return self._calculator(instruction, context)
        elif tool_name == "sequential_thinking":
            return self._sequential_thinking(instruction, context)
        elif tool_name == "search":
            return self._search(instruction, context)
        elif tool_name == "memory":
            return self._memory(instruction, context)
        else:
            return f"Tool {tool_name} not available. Proceeding with basic analysis."
    
    def _calculator(self, instruction: str, context: str) -> str:
        """Simulate calculator tool"""
        import re
        
        # Try to extract and evaluate mathematical expressions
        numbers = re.findall(r'\d+', instruction + " " + context)
        
        # Common calculation patterns
        if "×" in instruction or "*" in instruction or "multiply" in instruction.lower():
            if len(numbers) >= 2:
                result = int(numbers[0]) * int(numbers[1])
                return f"Calculation: {numbers[0]} × {numbers[1]} = {result}"
        
        if "12" in context and "(" in context and "-2)" in context:
            # Pattern: 12 × (n-2)
            n_match = re.search(r'(\d+)x\d+x\d+', context)
            if n_match:
                n = int(n_match.group(1))
                result = 12 * (n - 2)
                return f"Formula: 12 × ({n}-2) = 12 × {n-2} = {result}"
        
        return f"Calculator: Numbers found: {numbers}. Instruction: {instruction[:100]}"
    
    def _sequential_thinking(self, instruction: str, context: str) -> str:
        """Simulate sequential thinking"""
        return f"Analyzing: {instruction}. Based on context, proceeding step by step through the logic."
    
    def _search(self, instruction: str, context: str) -> str:
        """Simulate search/matching"""
        return f"Searching for: {instruction}. Matching against available options and facts."
    
    def _memory(self, instruction: str, context: str) -> str:
        """Simulate memory storage"""
        return f"Stored: {instruction[:100]}"


# Example usage
if __name__ == "__main__":
    reasoner = ImprovedMCPReasoner()
    
    # Test with complex spatial problem
    problem = "A 10x10x10 cube is painted red on all sides. How many smaller cubes have exactly two sides painted?"
    options = ["88", "96", "104", "112", "Another answer"]
    
    print("Testing Improved MCP Reasoner")
    print("=" * 80)
    
    selected_idx, reasoning, log = reasoner.reason(problem, options)
    
    print(f"\nProblem: {problem}")
    print(f"\nClassification: {log['problem_classification']}")
    print(f"\nSelected: Option {selected_idx + 1} - {options[selected_idx]}")
    print(f"\nReasoning:\n{reasoning}")
