"""
Domain-Specific Prompt Templates for Enhanced Reasoning
Provides specialized prompts based on problem type to improve
decomposition and reasoning quality for small LLMs.
"""

from typing import Dict
from agentic_engine.problem_classifier import ProblemType

class DomainPrompts:
    """
    Provides domain-specific prompts and guidance for different problem types.
    Uses explicit formulas, strategies, and step-by-step instructions.
    """
    
    @staticmethod
    def get_decomposition_prompt(problem_type: ProblemType, problem_statement: str, options: list) -> str:
        """
        Get specialized decomposition prompt based on problem type.
        
        Args:
            problem_type: The classified problem type
            problem_statement: The actual problem
            options: Available answer options
            
        Returns:
            Specialized decomposition prompt
        """
        
        base_template = f"""Problem: {problem_statement}

Answer options:
{chr(10).join([f"{i+1}. {opt}" for i, opt in enumerate(options)])}

"""
        
        if problem_type == ProblemType.SPATIAL_GEOMETRY:
            return base_template + """This is a SPATIAL GEOMETRY problem. Follow these steps:

IMPORTANT FORMULAS TO REMEMBER:
- For a cube painted on all sides, smaller cubes fall into categories:
  * Corner cubes: Always 8 corners (3 painted faces each)
  * Edge cubes (not corners): 12 edges × (n-2) cubes per edge (2 painted faces each)
  * Face cubes (not edges): 6 faces × (n-2)² cubes per face (1 painted face each)
  * Interior cubes: (n-2)³ cubes (0 painted faces)
  Where n = dimension of the cube

STEP-BY-STEP APPROACH:
1. Extract the cube dimensions (e.g., "10x10x10" means n=10)
2. Identify what type of smaller cubes we're counting (how many painted faces?)
3. Apply the correct formula based on face count
4. For 2 painted faces: Use edge formula = 12 × (n-2)
5. Calculate the result using the calculator tool
6. Verify the calculation matches one of the given options

Generate your decomposition as a JSON array with these exact steps."""

        elif problem_type == ProblemType.OPTIMIZATION or problem_type == ProblemType.SCHEDULING:
            return base_template + """This is an OPTIMIZATION/SCHEDULING problem. Follow these steps:

IMPORTANT CONCEPTS:
- Total time = Sum of processing times for all operations
- For sequential machines, consider queueing effects
- Faster machines first can reduce total queue time
- Calculate actual throughput for each sequence option

STEP-BY-STEP APPROACH:
1. List all machines/processes with their individual times
2. Understand the goal: minimize time, maximize output, or optimize sequence?
3. For each answer option (sequence), calculate:
   - Total cycle time
   - Number of complete cycles in available time
   - Total items produced
4. Use calculator to compute each option's result
5. Compare all options to find the optimum
6. Account for queueing: items wait for slower machines

FORMULA FOR THROUGHPUT:
- If machines are sequential: Bottleneck = slowest machine determines rate
- Items per cycle = 1
- Cycles in time T = floor(T / cycle_time)
- But watch for parallel vs sequential processing!

Generate your decomposition as a JSON array following these steps."""

        elif problem_type == ProblemType.LOGIC_PUZZLE:
            return base_template + """This is a LOGIC PUZZLE. Follow these steps:

APPROACH FOR LOGIC PROBLEMS:
1. Identify the initial condition (e.g., "you are in a race...")
2. Identify the action/event (e.g., "you overtake the second person")
3. Apply the logical consequence step by step:
   - What was true before the action?
   - What changes with the action?
   - What is true after the action?
4. Verify your reasoning is consistent
5. Match your conclusion to one of the answer options

COMMON LOGIC PATTERNS:
- Position changes: If you overtake position N, you take their place
- "All except": Count total, subtract exceptions
- True/false statements: Check each against facts
- If-then chains: Follow implications step by step

Generate your decomposition as a JSON array with clear logical steps."""

        elif problem_type == ProblemType.PATTERN_RECOGNITION:
            return base_template + """This is a PATTERN RECOGNITION problem. Follow these steps:

STEP-BY-STEP APPROACH:
1. Write out the given sequence clearly
2. Look for common patterns:
   - Arithmetic: constant difference (e.g., 2,4,6,8... diff=2)
   - Geometric: constant ratio (e.g., 2,4,8,16... ratio=2)
   - Polynomial: differences of differences
   - Fibonacci-like: sum of previous terms
   - Alternating patterns: two interleaved sequences
3. Calculate differences between consecutive terms
4. If differences aren't constant, calculate second differences
5. Apply the identified pattern to generate next term
6. Verify the pattern holds for all given terms

Generate your decomposition as a JSON array with these analysis steps."""

        elif problem_type == ProblemType.MATHEMATICAL:
            return base_template + """This is a MATHEMATICAL problem. Follow these steps:

STEP-BY-STEP APPROACH:
1. Identify all given numbers and values
2. Identify what needs to be calculated
3. Determine which mathematical operations to apply
4. Break complex calculations into simple steps
5. Use calculator tool for each arithmetic step
6. Verify the calculation is correct
7. Check if result needs rounding or special formatting

OPERATION ORDER (PEMDAS/BODMAS):
1. Parentheses/Brackets first
2. Exponents/Orders
3. Multiplication and Division (left to right)
4. Addition and Subtraction (left to right)

Generate your decomposition as a JSON array with calculation steps."""

        elif problem_type == ProblemType.PROBABILITY:
            return base_template + """This is a PROBABILITY problem. Follow these steps:

PROBABILITY FORMULAS:
- Basic probability = (Favorable outcomes) / (Total outcomes)
- Probability always between 0 and 1
- Multiple events: P(A and B) = P(A) × P(B) if independent
- Either event: P(A or B) = P(A) + P(B) - P(A and B)

COUNTING PRINCIPLES:
- Permutations: Order matters, n!/(n-r)!
- Combinations: Order doesn't matter, n!/(r!(n-r)!)
- Fundamental counting: Multiply choices

STEP-BY-STEP APPROACH:
1. Identify the sample space (all possible outcomes)
2. Count total number of outcomes
3. Identify favorable outcomes for the event
4. Count favorable outcomes
5. Calculate probability = favorable/total
6. Verify answer is between 0 and 1

Generate your decomposition as a JSON array with counting and calculation steps."""

        elif problem_type == ProblemType.COMPARISON:
            return base_template + """This is a COMPARISON problem. Follow these steps:

STEP-BY-STEP APPROACH:
1. List all items/options to compare
2. Identify comparison criteria from the problem
3. For each option, evaluate against each criterion
4. Create a comparison table mentally
5. Determine which option best meets the criteria
6. Verify the winning option is clearly superior

COMPARISON STRATEGIES:
- Quantitative: Calculate numerical values, compare directly
- Qualitative: List pros/cons for each option
- Elimination: Remove options that fail criteria
- Ranking: Score each option, highest wins

Generate your decomposition as a JSON array with systematic comparison steps."""

        else:  # UNKNOWN or general
            return base_template + """Follow these GENERAL reasoning steps:

UNIVERSAL PROBLEM-SOLVING APPROACH:
1. Read the problem carefully and identify:
   - What information is given?
   - What is being asked?
   - What are the constraints?
2. Break the problem into smaller sub-problems
3. For each sub-problem:
   - Determine what calculation or reasoning is needed
   - Choose appropriate tool (calculator, sequential_thinking, search)
   - Execute the step
4. Combine results from all sub-problems
5. Select the answer option that matches your conclusion
6. Double-check your reasoning

Generate your decomposition as a JSON array. Each step should:
- Be specific and actionable
- Specify which tool to use (calculator, sequential_thinking, search, memory)
- Include what to calculate or reason about"""

    @staticmethod
    def get_verification_prompt(problem_type: ProblemType, problem_statement: str, 
                               selected_option: str, reasoning: str) -> str:
        """
        Get verification prompt to check answer validity.
        
        Args:
            problem_type: The problem type
            problem_statement: Original problem
            selected_option: The option we selected
            reasoning: Our reasoning trace
            
        Returns:
            Verification prompt
        """
        
        base_prompt = f"""VERIFICATION TASK

Original Problem: {problem_statement}

Our Selected Answer: {selected_option}

Our Reasoning:
{reasoning}

"""
        
        if problem_type == ProblemType.SPATIAL_GEOMETRY:
            return base_prompt + """Verify this answer by checking:
1. Did we identify the cube dimensions correctly?
2. Did we apply the correct formula for the type of cubes asked?
3. Did we calculate correctly? Re-do the calculation:
   - For 2 painted faces: 12 × (n-2) where n is cube dimension
   - Double-check the arithmetic
4. Does our answer match one of the given options?
5. Is our answer reasonable for the problem?

Answer with "CORRECT" if verified, or "ERROR: [specific issue]" if wrong."""

        elif problem_type == ProblemType.OPTIMIZATION or problem_type == ProblemType.SCHEDULING:
            return base_prompt + """Verify this answer by checking:
1. Did we calculate throughput for ALL sequence options?
2. For scheduling, did we account for the bottleneck (slowest machine)?
3. Did we verify which sequence produces the MOST items in the time limit?
4. Double-check calculations: items = floor(time_limit / cycle_time)
5. Is the selected sequence clearly superior to others?

Answer with "CORRECT" if verified, or "ERROR: [specific issue]" if wrong."""

        elif problem_type == ProblemType.LOGIC_PUZZLE:
            return base_prompt + """Verify this answer by checking:
1. Did we correctly identify the initial state?
2. Did we correctly apply the logical transformation?
3. Is our conclusion consistent with the problem statement?
4. Did we avoid common logical errors?
5. Does our answer make intuitive sense?

Answer with "CORRECT" if verified, or "ERROR: [specific issue]" if wrong."""

        else:
            return base_prompt + """Verify this answer by checking:
1. Did we understand the question correctly?
2. Did we use all relevant information from the problem?
3. Are our calculations correct?
4. Is our reasoning logically sound?
5. Does our answer actually address what was asked?
6. Is our answer one of the given options?

Answer with "CORRECT" if verified, or "ERROR: [specific issue]" if wrong."""

    @staticmethod
    def get_tool_execution_context(problem_type: ProblemType, step_description: str) -> str:
        """
        Provide context for tool execution based on problem type.
        
        Args:
            problem_type: The problem type
            step_description: What the step is trying to do
            
        Returns:
            Enhanced context for tool execution
        """
        
        contexts = {
            ProblemType.SPATIAL_GEOMETRY: "Use geometric formulas. For cubes: corners=8, edges=12, faces=6.",
            ProblemType.OPTIMIZATION: "Compare all options systematically. Calculate exact values.",
            ProblemType.SCHEDULING: "Consider bottlenecks. Slowest machine limits throughput.",
            ProblemType.LOGIC_PUZZLE: "Trace logical steps carefully. Verify consistency.",
            ProblemType.PATTERN_RECOGNITION: "Look for arithmetic, geometric, or polynomial patterns.",
            ProblemType.MATHEMATICAL: "Follow order of operations. Check arithmetic carefully.",
            ProblemType.PROBABILITY: "Count outcomes systematically. Verify 0 ≤ probability ≤ 1.",
            ProblemType.COMPARISON: "List criteria. Evaluate each option against all criteria.",
        }
        
        context = contexts.get(problem_type, "Reason step by step.")
        return f"{context} Current step: {step_description}"


# Testing
if __name__ == "__main__":
    prompts = DomainPrompts()
    
    # Test spatial geometry prompt
    problem = "A 10x10x10 cube is painted red. How many cubes have exactly 2 faces painted?"
    options = ["88", "96", "104", "112", "Another"]
    
    print("SPATIAL GEOMETRY PROMPT:")
    print("=" * 80)
    print(prompts.get_decomposition_prompt(ProblemType.SPATIAL_GEOMETRY, problem, options))
    print("\n" + "=" * 80)
    
    # Test verification prompt
    print("\nVERIFICATION PROMPT:")
    print("=" * 80)
    reasoning = "Applied formula 12 × (10-2) = 12 × 8 = 96"
    print(prompts.get_verification_prompt(ProblemType.SPATIAL_GEOMETRY, problem, "96", reasoning))
