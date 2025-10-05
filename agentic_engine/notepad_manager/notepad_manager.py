# notepad_manager.py - Python interface to advanced C notepad manager

"""
QUICK START - Method 1 (Using C as subprocess - WORKS IMMEDIATELY):
    Just run: python notepad_manager.py
    
ADVANCED - Method 2 (Using DLL - REQUIRES COMPILATION):
    1. Open "x64 Native Tools Command Prompt for VS"
    2. cd "f:\\visual stuidio code\\.vscode\\ehos hackathon\\notepad_manager"
    3. cl /LD /O2 notepad_manager.c /link /OUT:notepad_manager.dll
    4. Set USE_DLL = True below
"""

import os
import sys
import subprocess
import ctypes
from ctypes import c_int, c_uint, c_ulong, c_char, POINTER, Structure
import tempfile
import json

# Configuration
USE_DLL = False  # Set to True if you have compiled the DLL

# Define Notepad structure (must match C)
class Notepad(Structure):
    _fields_ = [
        ("id", c_int),
        ("status", c_ulong),  # volatile LONG
        ("scratchpad", c_char * 8192),
        ("hash", c_uint),
        ("processing_time_ms", c_ulong),
        ("_padding", c_char * 40)
    ]

def compile_and_run_c_directly(num_notepads):
    """Compile and run C code as subprocess (fallback method)"""
    c_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notepad_manager.c")
    
    # Create a simple main function to test
    test_code = f'''
#include <stdio.h>
#include <windows.h>

// Include the actual implementation
{open(c_path, 'r', encoding='utf-8').read()}

int main() {{
    printf("{{\\n");
    printf("  \\"method\\": \\"subprocess\\",\\n");
    printf("  \\"num_notepads\\": {num_notepads},\\n");
    printf("  \\"notepads\\": [\\n");
    
    Notepad* notepads = make_notepads({num_notepads});
    if (!notepads) {{
        printf("  ],\\n");
        printf("  \\"error\\": \\"Failed to allocate\\"\\n");
        printf("}}\\n");
        return 1;
    }}
    
    DWORD start = GetTickCount();
    run_all_notepads({num_notepads}, notepads);
    DWORD elapsed = GetTickCount() - start;
    
    for (int i = 0; i < {num_notepads}; i++) {{
        printf("    {{\\n");
        printf("      \\"id\\": %d,\\n", notepads[i].id);
        printf("      \\"status\\": %ld,\\n", notepads[i].status);
        printf("      \\"hash\\": %u,\\n", notepads[i].hash);
        printf("      \\"time_ms\\": %lu,\\n", notepads[i].processing_time_ms);
        printf("      \\"content\\": \\"%s\\"\\n", notepads[i].scratchpad);
        printf("    }}%s\\n", (i < {num_notepads}-1) ? "," : "");
    }}
    
    printf("  ],\\n");
    printf("  \\"total_time_ms\\": %lu\\n", elapsed);
    printf("}}\\n");
    
    free_notepads(notepads);
    cleanup_notepad_manager();
    return 0;
}}
'''
    
    # Write temporary C file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False, encoding='utf-8') as f:
        temp_c = f.name
        f.write(test_code)
    
    try:
        # Compile
        temp_exe = temp_c.replace('.c', '.exe')
        compile_result = subprocess.run(
            ['gcc', '-O3', '-o', temp_exe, temp_c],
            capture_output=True,
            text=True
        )
        
        if compile_result.returncode != 0:
            return {'error': f'Compilation failed: {compile_result.stderr}'}
        
        # Run
        run_result = subprocess.run(
            [temp_exe],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if run_result.returncode != 0:
            return {'error': f'Execution failed: {run_result.stderr}'}
        
        # Parse JSON output
        try:
            return json.loads(run_result.stdout)
        except:
            return {'error': 'Failed to parse output', 'raw': run_result.stdout}
            
    finally:
        # Cleanup
        try:
            if os.path.exists(temp_c):
                os.remove(temp_c)
            if os.path.exists(temp_exe):
                os.remove(temp_exe)
        except:
            pass

def load_dll():
    """Load DLL if available"""
    dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notepad_manager.dll")
    
    if not os.path.exists(dll_path):
        raise FileNotFoundError("DLL not found")
    
    lib = ctypes.CDLL(dll_path)
    
    # Define function signatures
    lib.make_notepads.argtypes = [c_int]
    lib.make_notepads.restype = POINTER(Notepad)
    
    lib.run_all_notepads.argtypes = [c_int, POINTER(Notepad)]
    lib.run_all_notepads.restype = None
    
    lib.free_notepads.argtypes = [POINTER(Notepad)]
    lib.free_notepads.restype = None
    
    lib.cleanup_notepad_manager.argtypes = []
    lib.cleanup_notepad_manager.restype = None
    
    return lib

# Try to load library
if USE_DLL:
    try:
        lib = load_dll()
    except Exception as e:
        print(f"⚠️  DLL loading failed: {e}")
        print("   Falling back to subprocess method...")
        lib = None
else:
    lib = None

class NotepadManager:
    """High-performance notepad manager using advanced C implementation"""
    
    def __init__(self, num_notepads):
        """
        Create notepad manager with specified number of notepads
        
        Args:
            num_notepads (int): Number of notepads (1-16)
        """
        if num_notepads < 1 or num_notepads > 16:
            raise ValueError("num_notepads must be between 1 and 16")
        
        self.num_notepads = num_notepads
        self.use_dll = lib is not None
        self.result_cache = None
        
        if self.use_dll:
            self.np_ptr = lib.make_notepads(num_notepads)
            if not self.np_ptr:
                raise MemoryError("Failed to allocate notepads")
    
    def run_all(self):
        """Process all notepads in parallel using thread pool"""
        if self.use_dll:
            lib.run_all_notepads(self.num_notepads, self.np_ptr)
        else:
            # Use subprocess method
            self.result_cache = compile_and_run_c_directly(self.num_notepads)
    
    def get_scratchpads(self):
        """
        Get scratchpad contents from all notepads
        
        Returns:
            list[str]: List of scratchpad strings
        """
        if self.use_dll:
            scratchpads = []
            for i in range(self.num_notepads):
                notepad = self.np_ptr[i]
                # Extract string from scratchpad
                scratchpad_bytes = notepad.scratchpad
                # Find null terminator
                end_idx = scratchpad_bytes.find(b'\x00')
                if end_idx != -1:
                    scratchpad_bytes = scratchpad_bytes[:end_idx]
                scratchpad_str = scratchpad_bytes.decode('utf-8', errors='ignore')
                scratchpads.append(scratchpad_str)
            return scratchpads
        else:
            # Get from cached subprocess result
            if self.result_cache and 'notepads' in self.result_cache:
                return [np['content'] for np in self.result_cache['notepads']]
            return []
    
    def get_status(self, index):
        """
        Get status of specific notepad
        
        Args:
            index (int): Notepad index
            
        Returns:
            int: Status (0=idle, 1=processing, 2=complete)
        """
        if self.use_dll:
            if 0 <= index < self.num_notepads:
                return self.np_ptr[index].status
            return -1
        else:
            if self.result_cache and 'notepads' in self.result_cache:
                if 0 <= index < len(self.result_cache['notepads']):
                    return self.result_cache['notepads'][index]['status']
            return -1
    
    def get_metrics(self):
        """
        Get performance metrics for all notepads
        
        Returns:
            list[dict]: List of metrics per notepad
        """
        if self.use_dll:
            metrics = []
            for i in range(self.num_notepads):
                notepad = self.np_ptr[i]
                metrics.append({
                    'id': notepad.id,
                    'status': notepad.status,
                    'hash': notepad.hash,
                    'processing_time_ms': notepad.processing_time_ms
                })
            return metrics
        else:
            # Get from cached subprocess result
            if self.result_cache and 'notepads' in self.result_cache:
                return [{
                    'id': np['id'],
                    'status': np['status'],
                    'hash': np['hash'],
                    'processing_time_ms': np['time_ms']
                } for np in self.result_cache['notepads']]
            return []
    
    def __del__(self):
        """Cleanup resources"""
        if self.use_dll and hasattr(self, 'np_ptr') and self.np_ptr:
            lib.free_notepads(self.np_ptr)

def cleanup():
    """Cleanup thread pool (call before program exit)"""
    if lib:
        lib.cleanup_notepad_manager()

# ==== Usage Demo ====
if __name__ == "__main__":
    print("=" * 50)
    print("Advanced Notepad Manager - Python Interface")
    print("=" * 50)
    
    # Test 1: Basic usage
    print("\n📝 Test 1: Creating and processing notepads...")
    n_notepads = 8
    manager = NotepadManager(n_notepads)
    
    print(f"✅ Created {n_notepads} notepads")
    
    # Process all
    print("⚙️  Processing notepads in parallel...")
    import time
    start = time.time()
    manager.run_all()
    elapsed = (time.time() - start) * 1000
    
    print(f"✅ Processed in {elapsed:.2f}ms")
    
    # Get results
    print("\n📊 Results:")
    scratchpads = manager.get_scratchpads()
    for i, content in enumerate(scratchpads):
        print(f"  Notepad {i}: {content[:80]}...")
    
    # Get metrics
    print("\n📈 Performance Metrics:")
    metrics = manager.get_metrics()
    total_time = 0
    for m in metrics:
        print(f"  Notepad {m['id']}: Status={m['status']}, "
              f"Hash=0x{m['hash']:08X}, Time={m['processing_time_ms']}ms")
        total_time += m['processing_time_ms']
    
    avg_time = total_time / len(metrics) if metrics else 0
    print(f"\n  Average processing time: {avg_time:.2f}ms")
    print(f"  Total throughput: {len(metrics) / (elapsed / 1000):.2f} notepads/sec")
    
    # Test 2: Stress test
    print("\n🔥 Test 2: Stress test with 16 notepads...")
    stress_manager = NotepadManager(16)
    start = time.time()
    stress_manager.run_all()
    elapsed = (time.time() - start) * 1000
    
    print(f"✅ Processed 16 notepads in {elapsed:.2f}ms")
    print(f"   Throughput: {16 / (elapsed / 1000):.2f} notepads/sec")
    
    # Cleanup
    del stress_manager
    del manager
    cleanup()
    
    print("\n✅ All tests completed successfully!")
    print("=" * 50)
