"""
Use API Server to Process test.csv and Generate output.csv
Uses the fully integrated engine with ALL agentic_engine modules
"""

import requests
import csv
import time
import json
from typing import List, Dict, Any
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8080"
INPUT_CSV = "test.csv"
OUTPUT_CSV = "api_output.csv"
BATCH_SIZE = 10  # Process in batches to avoid timeouts
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds

def read_test_csv(filepath: str) -> List[Dict[str, Any]]:
    """Read test.csv and parse into structured format"""
    problems = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract answer options (handling 5 options)
            answer_options = []
            for i in range(1, 6):
                key = f'answer_option_{i}'
                if key in row and row[key]:
                    answer_options.append(row[key])
            
            problems.append({
                'topic': row['topic'],
                'problem_statement': row['problem_statement'],
                'answer_options': answer_options
            })
    
    return problems

def call_api_single(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Call /run-single/ endpoint for a single problem"""
    url = f"{API_BASE_URL}/run-single/"
    
    payload = {
        "problem_statement": problem['problem_statement'],
        "answer_options": problem['answer_options'],
        "topic": problem['topic']
    }
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"⚠️  Timeout on attempt {attempt + 1}/{RETRY_ATTEMPTS}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise
    
    raise Exception("Max retry attempts reached")

def process_problems(problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process all problems using API server"""
    results = []
    total = len(problems)
    
    print(f"\n🚀 Processing {total} problems using API server...")
    print(f"📡 API URL: {API_BASE_URL}")
    print(f"🔧 Using: FullyIntegratedHybridEngine (ALL 13 modules)")
    print("="*80)
    
    for i, problem in enumerate(problems, 1):
        print(f"\n[{i}/{total}] Processing: {problem['topic']}")
        print(f"   Problem: {problem['problem_statement'][:80]}...")
        
        try:
            result = call_api_single(problem)
            
            # Extract solution
            solution = result.get('solution', '')
            correct_option = result.get('correct_option', '')
            correct_index = result.get('correct_index', -1)
            processing_time = result.get('processing_time', 0)
            classification = result.get('classification', {})
            
            # Display results
            print(f"   ✓ Classification: {classification.get('type', 'unknown')}")
            print(f"   ✓ Selected: Option {correct_index + 1} - {correct_option}")
            print(f"   ✓ Processing time: {processing_time:.2f}s")
            
            results.append({
                'topic': problem['topic'],
                'problem_statement': problem['problem_statement'],
                'solution': solution,
                'correct_option': correct_option,
                'correct_index': correct_index,
                'processing_time': processing_time,
                'classification': classification
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            # Add error placeholder
            results.append({
                'topic': problem['topic'],
                'problem_statement': problem['problem_statement'],
                'solution': f"ERROR: {str(e)}",
                'correct_option': "ERROR",
                'correct_index': -1,
                'processing_time': 0,
                'classification': {}
            })
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    return results

def write_output_csv(results: List[Dict[str, Any]], filepath: str):
    """Write results to CSV in output.csv format"""
    print(f"\n📝 Writing results to {filepath}...")
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        # Match output.csv format exactly
        fieldnames = ['topic', 'problem_statement', 'solution', 'correct option']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'topic': result['topic'],
                'problem_statement': result['problem_statement'],
                'solution': result['solution'],
                'correct option': result['correct_option']
            })
    
    print(f"✅ Output written successfully!")

def write_detailed_json(results: List[Dict[str, Any]], filepath: str):
    """Write detailed results including classification and timing to JSON"""
    print(f"\n📊 Writing detailed results to {filepath}...")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Detailed JSON written successfully!")

def print_statistics(results: List[Dict[str, Any]]):
    """Print processing statistics"""
    print("\n" + "="*80)
    print("📊 PROCESSING STATISTICS")
    print("="*80)
    
    total = len(results)
    errors = sum(1 for r in results if r['correct_index'] == -1)
    successful = total - errors
    
    total_time = sum(r['processing_time'] for r in results)
    avg_time = total_time / total if total > 0 else 0
    
    print(f"Total Problems: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Errors: {errors} ({errors/total*100:.1f}%)")
    print(f"Total Processing Time: {total_time:.2f}s")
    print(f"Average Time per Problem: {avg_time:.2f}s")
    
    # Classification breakdown
    classifications = {}
    for r in results:
        if r['classification']:
            cls_type = r['classification'].get('type', 'unknown')
            classifications[cls_type] = classifications.get(cls_type, 0) + 1
    
    if classifications:
        print("\n📋 Classification Breakdown:")
        for cls_type, count in sorted(classifications.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cls_type}: {count} problems ({count/total*100:.1f}%)")
    
    print("="*80)

def check_api_health():
    """Check if API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        health = response.json()
        print(f"✅ API Server is healthy")
        print(f"   Version: {health.get('version', 'unknown')}")
        print(f"   Status: {health.get('status', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ API Server not responding: {e}")
        print(f"   Make sure the server is running on {API_BASE_URL}")
        return False

def check_engine_info():
    """Display engine information"""
    try:
        response = requests.get(f"{API_BASE_URL}/engine/info", timeout=5)
        response.raise_for_status()
        info = response.json()
        
        print("\n" + "="*80)
        print(f"🔧 ENGINE: {info.get('engine', 'unknown')}")
        print("="*80)
        print(f"Description: {info.get('description', '')}")
        print(f"Total Modules: {info.get('statistics', {}).get('total_modules', 0)}")
        print(f"Integration Complete: {info.get('statistics', {}).get('integration_complete', False)}")
        
        print("\n📦 Active Modules:")
        modules = info.get('modules', {})
        for module_name, module_info in modules.items():
            print(f"   ✓ {module_name} - {module_info.get('function', '')}")
        
        print("\n🔄 Processing Pipeline:")
        pipeline = info.get('pipeline', [])
        for step in pipeline:
            print(f"   {step}")
        
        print("="*80)
        
    except Exception as e:
        print(f"⚠️  Could not fetch engine info: {e}")

def main():
    """Main execution function"""
    print("="*80)
    print("🚀 API-BASED TEST.CSV PROCESSING")
    print("="*80)
    print(f"Using Fully Integrated Engine with ALL 13 agentic_engine modules")
    print()
    
    # Check API health
    if not check_api_health():
        print("\n⚠️  Please start the API server first:")
        print("   python api_server.py")
        return
    
    # Show engine info
    check_engine_info()
    
    # Check input file
    if not Path(INPUT_CSV).exists():
        print(f"\n❌ Input file not found: {INPUT_CSV}")
        return
    
    try:
        # Read problems
        print(f"\n📖 Reading input from: {INPUT_CSV}")
        problems = read_test_csv(INPUT_CSV)
        print(f"✅ Loaded {len(problems)} problems")
        
        # Process using API
        results = process_problems(problems)
        
        # Write outputs
        write_output_csv(results, OUTPUT_CSV)
        write_detailed_json(results, OUTPUT_CSV.replace('.csv', '_detailed.json'))
        
        # Print statistics
        print_statistics(results)
        
        print("\n" + "="*80)
        print("✨ PROCESSING COMPLETE!")
        print("="*80)
        print(f"📄 Output CSV: {OUTPUT_CSV}")
        print(f"📊 Detailed JSON: {OUTPUT_CSV.replace('.csv', '_detailed.json')}")
        print(f"🎯 Ready for submission!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
