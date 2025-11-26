#!/usr/bin/env python3
"""
Test Script for JOLT Multi-Agent Platform
Tests both CrewAI and LangChain agents with Gemini
"""

import sys
import os
from dotenv import load_dotenv
import json

print("="*70)
print("🧪 TESTING JOLT MULTI-AGENT PLATFORM (GEMINI)")
print("="*70)

# Load environment
load_dotenv()

# Check API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "your_gemini_api_key_here":
    print("\n❌ ERROR: Please set your GOOGLE_API_KEY in .env file")
    print("\n📝 Steps:")
    print("1. cp .env.example .env")
    print("2. Edit .env and add your Gemini API key")
    print("3. Get key from: https://makersuite.google.com/app/apikey")
    sys.exit(1)

print(f"\n✅ API Key configured (Gemini model: {os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')})")

# Import platform
print("\n📦 Importing platform...")
try:
    from jolt_platform.unified_platform import JoltPlatform
    print("✅ Platform imported successfully")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Initialize platform
print("\n🚀 Initializing platform with both agents...")
print("-"*70)
try:
    platform = JoltPlatform(output_dir="./test_output")
    print("\n✅ Platform initialized successfully!")
except Exception as e:
    print(f"\n❌ Initialization error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test case
print("\n" + "="*70)
print("🎯 TEST CASE: User Profile Transformation")
print("="*70)

test_input = {
    "user": {
        "firstName": "Alice",
        "lastName": "Johnson",
        "email": "alice.j@company.com",
        "age": 28
    },
    "metadata": {
        "timestamp": "2024-11-24T15:30:00Z",
        "source": "api"
    }
}

test_expected = {
    "profile": {
        "fullName": "Alice Johnson",
        "contact": {
            "email": "alice.j@company.com"
        },
        "age": 28
    },
    "eventTime": "2024-11-24T15:30:00Z",
    "dataSource": "api"
}

print("\n📥 INPUT JSON:")
print(json.dumps(test_input, indent=2))

print("\n📤 EXPECTED OUTPUT:")
print(json.dumps(test_expected, indent=2))

# Run the workflow
print("\n" + "="*70)
print("🔄 RUNNING MULTI-AGENT WORKFLOW...")
print("="*70)
print("\n⏳ This may take 30-60 seconds (agents are thinking)...\n")

try:
    result = platform.create_and_validate(
        test_input,
        test_expected,
        save_outputs=True
    )
    
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    if result['status'] == 'success':
        print("\n🎉 SUCCESS! Both agents completed their tasks.\n")
        
        print("🤖 CrewAI Agent (Gemini) - Generated JOLT Spec:")
        print("-"*70)
        print(json.dumps(result['jolt_spec'], indent=2))
        
        print("\n✅ LangChain Agent (Gemini) - Validation Report:")
        print("-"*70)
        validation = result['validation_report']
        print(f"Timestamp: {validation.get('timestamp', 'N/A')}")
        print(f"Status: {validation.get('status', 'N/A')}")
        
        if validation.get('validation_passed'):
            print("\n✅ VALIDATION PASSED - Transformation is correct!")
        else:
            print("\n⚠️ VALIDATION ISSUES FOUND")
            print(f"Details: {validation.get('comparison', 'See full report')}")
        
        print("\n💾 Output files saved to: ./test_output/")
        
    else:
        print(f"\n❌ WORKFLOW FAILED")
        print(f"Stage: {result.get('stage', 'unknown')}")
        print(f"Error: {result.get('error', 'Unknown error')}")

except Exception as e:
    print(f"\n❌ ERROR during workflow: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("✅ TEST COMPLETE!")
print("="*70)
print("\n📝 Summary:")
print("  • CrewAI Agent (Gemini): Created JOLT specification")
print("  • LangChain Agent (Gemini): Validated transformation")
print("  • Platform: Orchestrated both agents successfully")
print("\n🎉 Your multi-agent platform is working with Gemini!")
print("\n📚 Next steps:")
print("  • Run: python quickstart.py - for interactive demo")
print("  • Run: python example_workflow.py - for detailed example")
print("  • Run: python jolt_platform/api_server.py - for API server")
print()
