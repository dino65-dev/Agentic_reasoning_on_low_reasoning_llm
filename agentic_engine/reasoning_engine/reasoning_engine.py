"""
This file is maintained for backward compatibility.
For advanced features, use reasoning_engine_v2.py

UPGRADED VERSION (reasoning_engine_v2.py) includes:
- Quantum-enhanced parallel evaluation
- Bayesian scoring with confidence updates
- Self-healing error recovery
- Adaptive performance optimization
- JIT compilation and vectorization
- Zero-copy C integration
"""

# Import corrected modules
from problem_restatement.problem_restatement import AdvancedProblemRestater
from decompose import run_decomposition_chain
from tool_invoker import run_advanced_mcp_pipeline, API_KEY
from notepad_manager.notepad_manager import NotepadManager

# Import advanced engine
try:
    from reasoning_engine_v2 import QuantumEnhancedReasoningEngine
    ADVANCED_ENGINE_AVAILABLE = True
except ImportError:
    ADVANCED_ENGINE_AVAILABLE = False
    QuantumEnhancedReasoningEngine = None

class ReasoningEngine:
    """
    Enhanced Reasoning Engine with fixed imports and improved functionality.
    
    For advanced features (Bayesian scoring, async evaluation, etc.), 
    use QuantumEnhancedReasoningEngine from reasoning_engine_v2.py
    """
    
    def __init__(self, tools=None, llm_api_url=None, llm_api_key=None):
        self.tools = tools or ['calculator', 'symbolic', 'llm']
        self.llm_api_url = llm_api_url
        self.llm_api_key = llm_api_key or API_KEY
        
        # Initialize helper objects
        self.restater = AdvancedProblemRestater(enable_spacy=False)
        
        # Use advanced engine if available
        if ADVANCED_ENGINE_AVAILABLE and QuantumEnhancedReasoningEngine is not None:
            self.advanced_engine = QuantumEnhancedReasoningEngine(
                tools=self.tools,
                llm_api_url=llm_api_url,
                llm_api_key=llm_api_key
            )
        else:
            self.advanced_engine = None

    def reason_over_options(self, problem_statement, answer_options):
        """
        Main orchestration logic:
        - Restates problem
        - Decomposes steps
        - Creates parallel notepads
        - Runs tool-based reasoning per notepad
        Returns: List of (reasoning trace, final verdict) per option
        """
        # Use advanced engine if available
        if self.advanced_engine:
            traces, verdicts = self.advanced_engine.reason_over_options(
                problem_statement, answer_options
            )
            # Convert verdict format for backward compatibility
            simple_verdicts = [v.get('verdict', False) for v in verdicts]
            return traces, simple_verdicts
        
        # ---- RESTATEMENT ----
        restated = self.restater.manual_restate_problem(problem_statement)

        # ---- DECOMPOSITION ----
        decomp_result = run_decomposition_chain(restated)
        steps = [step.instruction for step in decomp_result.reasoning_steps]

        # ---- NOTEPAD CREATION ----
        num_options = len(answer_options)
        notepad_manager = NotepadManager(min(num_options, 16))
        notepad_manager.run_all()  # Run fast, parallelized C-side stub for scratchpad setup
        
        # Get scratchpads safely
        try:
            scratchpads = notepad_manager.get_scratchpads() if hasattr(notepad_manager, 'get_scratchpads') else [''] * num_options
        except:
            scratchpads = [''] * num_options

        # ---- Option-wise Reasoning (Python -> C bridge) ----
        traces, verdicts = [], []
        for idx, option in enumerate(answer_options):
            # You may fetch or store prior C-side scratchpad text if needed:
            c_scratch = scratchpads[idx] if idx < len(scratchpads) else ''
            context = {
                "problem_statement": restated,
                "answer_option": option,
                "scratchpad": c_scratch,
                "stepwise": []
            }
            # For each decomposition step:
            for step_txt in steps:
                # Select tool - simple heuristic-based selection
                tool_name = self._select_tool_simple(step_txt)
                
                # Invoke tool - simplified version
                result = self._invoke_tool_simple(tool_name, step_txt, context)
                
                context["stepwise"].append({
                    "step": step_txt,
                    "tool": tool_name,
                    "result": result
                })

            # ---- Final verdict: Use last step’s result + trace
            trace = "\n".join([f"[{s['tool']}] {s['step']} --> {s['result']}" for s in context["stepwise"]])
            verdict = self.assess_option(trace, option)
            traces.append(trace)
            verdicts.append(verdict)

        return traces, verdicts

    def _select_tool_simple(self, step_text):
        """Simple tool selection based on keywords"""
        step_lower = step_text.lower()
        
        if any(word in step_lower for word in ['calculate', 'compute', 'add', 'multiply', 'sum']):
            return 'calculator'
        elif any(word in step_lower for word in ['prove', 'derive', 'logical']):
            return 'symbolic'
        else:
            return 'llm'
    
    def _invoke_tool_simple(self, tool_name, step_text, context):
        """Simple synchronous tool invocation"""
        try:
            if tool_name == 'calculator':
                # Try to evaluate mathematical expressions
                import re
                expr = re.search(r'[\d+\-*/().\s]+', step_text)
                if expr:
                    result = eval(expr.group(), {"__builtins__": {}}, {})
                    return f"Result: {result}"
            elif tool_name == 'symbolic':
                return f"Symbolic analysis of: {step_text[:50]}"
            else:  # llm
                # Use MCP pipeline
                try:
                    result = run_advanced_mcp_pipeline(
                        step_text,
                        "mcp_tools_registry.yaml",
                        self.llm_api_key
                    )
                    return result.get('final_answer', 'No answer')
                except Exception as e:
                    return f"LLM error: {str(e)}"
        except Exception as e:
            return f"Tool error: {str(e)}"
        
        return "No result"

    def assess_option(self, reasoning_trace, answer_option):
        """
        Assess whether this option is correct/logically valid using custom rules, scoring, or final tool call.
        """
        # Simple example: If any step contains 'CORRECT', treat as valid. Replace with advanced scoring!
        if "CORRECT" in reasoning_trace.upper():
            return True
        
        # Check for successful calculations
        if "Result:" in reasoning_trace and "error" not in reasoning_trace.lower():
            return True
        
        # Advanced: Add scoring, error detection, or aggregation modules here.
        return False

# === Example usage ===
if __name__ == "__main__":
    engine = ReasoningEngine(tools=['calculator', 'symbolic', 'llm'])
    problem = "Tom has three boxes. First, he adds 2 to each box. Then, he removes 1 from the first box."
    answer_options = ["Box 1 has 3", "Box 2 has 2", "Box 3 has 2"]
    traces, verdicts = engine.reason_over_options(problem, answer_options)
    for i, (trace, verdict) in enumerate(zip(traces, verdicts)):
        print(f"Option {i+1}: {'CORRECT' if verdict else 'INCORRECT'}\n{trace}\n---\n")

