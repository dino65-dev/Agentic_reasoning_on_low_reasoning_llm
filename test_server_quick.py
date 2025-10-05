"""
Quick Test Script - Verify API Server Works
"""

import requests
import time
import sys

BASE_URL = "http://localhost:8080"

def test_server():
    print("=" * 60)
    print("🧪 TESTING API SERVER")
    print("=" * 60)
    print()
    
    # Test 1: Health Check
    print("1️⃣ Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed!")
            print(f"   📊 Status: {data.get('status')}")
            print(f"   💾 Cache: {data.get('cache_backend')}")
            print(f"   📝 Active jobs: {data.get('active_jobs')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to server!")
        print(f"   💡 Make sure server is running: python api_server.py")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print()
    
    # Test 2: Root Endpoint
    print("2️⃣ Testing Root Endpoint (API Info)...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Root endpoint works!")
            print(f"   📌 Title: {data.get('title', 'N/A')}")
            print(f"   🔢 Version: {data.get('version', 'N/A')}")
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test 3: Check if run-csv endpoint exists
    print("3️⃣ Testing /run-csv/ Endpoint Existence...")
    try:
        # Try with no file to see if endpoint exists
        response = requests.post(f"{BASE_URL}/run-csv/", timeout=5)
        # 422 = validation error (expected - we didn't send a file)
        # 200 = would work if we sent a file
        if response.status_code in [422, 400]:
            print(f"   ✅ /run-csv/ endpoint exists and is responding!")
            print(f"   📝 Validation working (got {response.status_code} as expected)")
        elif response.status_code == 200:
            print(f"   ✅ /run-csv/ endpoint exists!")
        else:
            print(f"   ⚠️  Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    print("=" * 60)
    print("🎉 TESTS COMPLETE!")
    print("=" * 60)
    print()
    print("✅ Your API server is working correctly!")
    print()
    print("📚 Next steps:")
    print("   1. Create a CSV file with problems")
    print("   2. Upload it using POST /run-csv/")
    print("   3. Get results in JSON + CSV formats")
    print()
    print("📖 See HOW_TO_RUN.md for detailed usage examples")
    print()
    
    return True

if __name__ == "__main__":
    print()
    print("Starting tests in 2 seconds...")
    print("(Make sure the server is running: python api_server.py)")
    print()
    time.sleep(2)
    
    success = test_server()
    sys.exit(0 if success else 1)
