#!/bin/bash

# Complete Installation Script for JOLT Multi-Agent Platform
# This fixes ALL import errors

echo "🚀 Installing JOLT Multi-Agent Platform Dependencies"
echo "=" * 60

echo ""
echo "📦 Installing Core Dependencies..."
pip3 install \
    crewai \
    langchain \
    langchain-core \
    langchain-openai \
    langchain-community \
    python-dotenv \
    pydantic \
    fastapi \
    uvicorn \
    requests

echo ""
echo "✅ Installation Complete!"
echo ""
echo "🧪 Testing imports..."
python3 << 'EOF'
import sys
errors = []

print("\n1. Testing CrewAI...")
try:
    import crewai
    print("   ✅ CrewAI installed")
except ImportError as e:
    print(f"   ❌ CrewAI: {e}")
    errors.append("crewai")

print("\n2. Testing LangChain Core...")
try:
    from langchain_core.prompts import ChatPromptTemplate
    print("   ✅ LangChain Core installed")
except ImportError as e:
    print(f"   ❌ LangChain Core: {e}")
    errors.append("langchain-core")

print("\n3. Testing LangChain OpenAI...")
try:
    from langchain_openai import ChatOpenAI
    print("   ✅ LangChain OpenAI installed")
except ImportError as e:
    print(f"   ❌ LangChain OpenAI: {e}")
    errors.append("langchain-openai")

print("\n4. Testing FastAPI...")
try:
    from fastapi import FastAPI
    print("   ✅ FastAPI installed")
except ImportError as e:
    print(f"   ❌ FastAPI: {e}")
    errors.append("fastapi")

print("\n5. Testing Python-dotenv...")
try:
    from dotenv import load_dotenv
    print("   ✅ Python-dotenv installed")
except ImportError as e:
    print(f"   ❌ Python-dotenv: {e}")
    errors.append("python-dotenv")

if errors:
    print(f"\n❌ Missing packages: {', '.join(errors)}")
    print("Run: pip3 install --user " + " ".join(errors))
    sys.exit(1)
else:
    print("\n" + "="*60)
    print("✅ ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
    print("="*60)
    print("\nYou can now run:")
    print("  • python3 quickstart.py")
    print("  • python3 example_workflow.py")
    print("  • python3 platform/api_server.py")
    sys.exit(0)
EOF

echo ""
echo "Done!"
