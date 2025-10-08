from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser

# 1. Define a structured schema for reasoning steps
class ReasoningStep(BaseModel):
    step_number: int = Field(..., description="Sequence number of the step")
    instruction: str = Field(..., description="Instruction or subproblem to solve")

class DecompositionResult(BaseModel):
    problem_statement: str = Field(..., description="Original logic problem statement")
    reasoning_steps: List[ReasoningStep] = Field(..., description="Decomposed ordered steps")

# 2. Manual decomposition function (efficient & granular)
def fast_manual_decompose(problem_statement: str) -> DecompositionResult:
    import re
    # Use connectives and sentence tokenization for steps
    connectives = r"(first|second|third|next|then|after that|finally|step \d+|task \d+|before that|now|also|meanwhile|subsequently)[\s,:-]*"
    segments = re.split(connectives, problem_statement, flags=re.I)
    sentences = []
    connective_words = {'first', 'second', 'third', 'next', 'then', 'after that', 'finally', 'before that', 'now', 'also', 'meanwhile', 'subsequently'}
    for seg in segments:
        seg = seg.strip()
        # Skip empty segments, pure connective words, or very short segments
        if not seg or len(seg.split()) < 2 or seg.lower() in connective_words:
            continue
        # If segments still too large, split by sentence
        if len(seg.split()) > 16:
            sentences.extend([s.strip() for s in re.split(r"[.?!]", seg) if len(s.strip()) > 5])
        elif len(seg) > 5:
            sentences.append(seg)
    steps = list(dict.fromkeys(sentences))
    # Build ReasoningStep objects
    step_objs = [ReasoningStep(step_number=i+1, instruction=txt) for i, txt in enumerate(steps)]
    return DecompositionResult(problem_statement=problem_statement, reasoning_steps=step_objs)

# 3. LangChain v3 structured extraction using manual runner
manual_decompose_runnable = RunnableLambda(fast_manual_decompose)
parser = PydanticOutputParser(pydantic_object=DecompositionResult)

# 4. Example prompt structure (for LLM fallback if needed)
structured_prompt = ChatPromptTemplate.from_template(
    """
You are a logic reasoning assistant.
Given the following problem statement, decompose it into an ordered list of atomic steps.
Format your output as a list of ReasoningStep objects with fields: step_number, instruction.

Problem: {problem_statement}
Output (structured list of steps):
"""
)

# 5. Chaining together (LangChain v3 compatible)
def run_decomposition_chain(problem_statement: str):
    # Manual chain - this already returns a DecompositionResult object
    result = manual_decompose_runnable.invoke(problem_statement)
    # The manual function already returns the structured object, no need to parse
    return result

# === Example usage ===
if __name__ == "__main__":
    problem = ("First, Alice sets up the experiment. Next, she records the results. Finally, "
               "she analyzes the data to produce the final report. After that, the team presents their findings.")
    structured_result = run_decomposition_chain(problem)
    print(f"Problem: {structured_result.problem_statement}")
    for step in structured_result.reasoning_steps:
        print(f"Step {step.step_number}: {step.instruction}")

# Output:
# Problem: First, Alice sets up the experiment. Next, she records the results. Finally, she analyzes the data to produce the final report. After that, the team presents their findings.
# Step 1: Alice sets up the experiment
# Step 2: she records the results
# Step 3: she analyzes the data to produce the final report
# Step 4: the team presents their findings
