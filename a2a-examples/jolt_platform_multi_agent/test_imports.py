#!/usr/bin/env python3
"""
Simple Import Test - Avoids circular import issues
Tests each module independently
"""

def test_module(module_name, import_statement):
    """Test a single module import"""
    import subprocess
    import sys
    
    test_code = f"""
import sys
try:
    {import_statement}
    print("OK")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""
    
    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()


print("="*60)
print("Testing Multi-Agent Platform Dependencies")
print("="*60)

tests = [
    ("CrewAI", "from crewai import Agent, Task, Crew"),
    ("LangChain Core", "from langchain_core.prompts import ChatPromptTemplate"),
    ("LangChain OpenAI", "from langchain_openai import ChatOpenAI"),
    ("FastAPI", "from fastapi import FastAPI"),
    ("Python-dotenv", "from dotenv import load_dotenv"),
    ("Pydantic", "from pydantic import BaseModel"),
]

results = []
for name, import_stmt in tests:
    print(f"\n{len(results)+1}. Testing {name}...")
    success, stdout, stderr = test_module(name, import_stmt)
    
    if success:
        print(f"   ✅ {name} works!")
        results.append(True)
    else:
        print(f"   ❌ {name} failed")
        if stderr:
            # Print first line of error for debugging
            error_line = stderr.split('\n')[-1] if stderr else "Unknown error"
            print(f"      Error: {error_line[:100]}")
        results.append(False)

print("\n" + "="*60)
if all(results):
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\n🎉 You can now run:")
    print("  • python3 quickstart.py")
    print("  • python3 example_workflow.py")
    print("  • python3 platform/api_server.py")
    print()
else:
    failed = sum(1 for r in results if not r)
    print(f"❌ {failed} of {len(tests)} tests failed")
    print("="*60)
    print("\n📝 To install missing dependencies:")
    print("  pip3 install --user crewai langchain langchain-core langchain-openai python-dotenv pydantic fastapi uvicorn")
    print()
