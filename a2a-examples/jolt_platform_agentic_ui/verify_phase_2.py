import requests
import time
import subprocess
import sys
import os

def run_verification():
    print("Starting Orchestrator for verification...")
    # Start the orchestrator in the background
    process = subprocess.Popen(
        [sys.executable, "-m", "orchestrator.main"],
        cwd="/home/amit.wankhede@GSLAB.COM/.gemini/antigravity/playground/azure-nebula",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # Wait for server to start
        time.sleep(5)
        
        # Test 1: Health Check
        print("Testing Health Check...")
        try:
            resp = requests.get("http://localhost:8000/health")
            if resp.status_code == 200:
                print("✅ Health Check Passed")
            else:
                print(f"❌ Health Check Failed: {resp.status_code}")
        except Exception as e:
            print(f"❌ Health Check Error: {e}")

        # Test 2: MCP Read File (Valid)
        print("Testing MCP Read File (Valid)...")
        try:
            resp = requests.get("http://localhost:8000/test-mcp?path=input.json&token=valid_token")
            data = resp.json()
            if "content" in data and "rating" in data["content"]:
                print("✅ MCP Read File Passed")
            else:
                print(f"❌ MCP Read File Failed: {data}")
        except Exception as e:
            print(f"❌ MCP Read File Error: {e}")

        # Test 3: MCP Read File (Invalid Token)
        print("Testing MCP Read File (Invalid Token)...")
        try:
            resp = requests.get("http://localhost:8000/test-mcp?path=input.json&token=invalid")
            data = resp.json()
            if "Error: Invalid Authentication Token" in data["content"]:
                print("✅ MCP Auth Check Passed")
            else:
                print(f"❌ MCP Auth Check Failed: {data}")
        except Exception as e:
            print(f"❌ MCP Auth Check Error: {e}")

    finally:
        print("Stopping Orchestrator...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    run_verification()
