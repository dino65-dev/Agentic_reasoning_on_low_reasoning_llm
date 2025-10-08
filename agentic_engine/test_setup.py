"""
Simple test runner to verify MCP-enhanced small LLM setup
"""

import sys
import os

print("=" * 80)
print("MCP-ENHANCED SMALL LLM - VERIFICATION TEST")
print("=" * 80)
print()

# Test 1: Check imports
print("Test 1: Checking imports...")
try:
    from langchain_openai import ChatOpenAI
    print("  ✓ langchain_openai available")
except ImportError:
    print("  ❌ langchain_openai not found")
    print("     Install with: pip install langchain-openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print("  ✓ python-dotenv available")
except ImportError:
    print("  ❌ python-dotenv not found")
    print("     Install with: pip install python-dotenv")
    sys.exit(1)

# Test 2: Check environment
print("\nTest 2: Checking environment variables...")
load_dotenv()

llm_url = os.getenv('SMALL_LLM_API_URL', 'http://192.168.0.100:1234')
llm_key = os.getenv('SMALL_LLM_API_KEY', 'lm-studio')
llm_model = os.getenv('LLM__MODEL_NAME', 'qwen2-1.5b-instruct')

print(f"  LLM URL: {llm_url}")
print(f"  LLM Key: {llm_key}")
print(f"  LLM Model: {llm_model}")

# Test 3: Check LLM connection
print("\nTest 3: Testing LLM connection...")
try:
    llm = ChatOpenAI(
        base_url=llm_url,
        api_key=llm_key,
        model=llm_model,
        temperature=0.3,
        timeout=10
    )
    
    response = llm.invoke("Say 'Hello'")
    print(f"  ✓ LLM responded: {response.content[:50]}")
except Exception as e:
    print(f"  ❌ LLM connection failed: {str(e)}")
    print("\n  Make sure LM Studio is running with qwen2-1.5b-instruct")
    print("  Server should be at: http://localhost:1234")
    sys.exit(1)

# Test 4: Check MCP engine
print("\nTest 4: Testing MCP-enhanced engine...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'reasoning_engine'))
    from reasoning_engine_small_llm import create_mcp_enhanced_engine
    
    engine = create_mcp_enhanced_engine()
    print("  ✓ MCP engine created successfully")
    
    # Simple test
    problem = "What is 2 + 2?"
    options = ["3", "4", "5", "6"]
    
    print(f"\n  Test problem: {problem}")
    print(f"  Options: {options}")
    print(f"\n  Running reasoning...")
    
    selected_index, trace, log = engine.reason(problem, options)
    
    print(f"\n  ✓ Selected: Option {selected_index} - {options[selected_index-1]}")
    print(f"  ✓ Reasoning steps: {len(log)}")
    
except Exception as e:
    print(f"  ❌ MCP engine test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check CSV processing
print("\nTest 5: Checking CSV processing setup...")
try:
    csv_processor_path = os.path.join(os.path.dirname(__file__), 'process_csv_with_mcp.py')
    if os.path.exists(csv_processor_path):
        print(f"  ✓ CSV processor found: {csv_processor_path}")
    else:
        print(f"  ❌ CSV processor not found")
        sys.exit(1)
    
    test_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test.csv')
    if os.path.exists(test_csv):
        print(f"  ✓ test.csv found: {test_csv}")
    else:
        print(f"  ⚠ test.csv not found at: {test_csv}")
        
except Exception as e:
    print(f"  ❌ CSV check failed: {str(e)}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print("\n✓ All tests passed!")
print("\nYou can now process test.csv:")
print("  python process_csv_with_mcp.py --test")
print("\nOr process all problems:")
print("  python process_csv_with_mcp.py --input ../test.csv --output ../output.csv")
print()
