#!/usr/bin/env python
"""
Quick Start Script for JOLT Multi-Agent Platform

This script demonstrates the basic usage of both agents:
1. CrewAI agent for JOLT spec creation
2. LangChain agent for validation
"""

import sys
from dotenv import load_dotenv
from jolt_platform.unified_platform import JoltPlatform
import json

def print_banner():
    """Print welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║         JOLT Multi-Agent Platform - Quick Start         ║
    ║                                                          ║
    ║   🤖 CrewAI Agent    → JOLT Spec Creation               ║
    ║   ✅ LangChain Agent → Validation & Reports             ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_environment():
    """Check if environment is properly configured."""
    import os
    
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found!")
        print("📝 Please create a .env file with your GOOGLE_API_KEY")
        print("\nExample:")
        print("  cp .env.example .env")
        print("  # Then edit .env and add your API key\n")
        
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        print("\n⚠️  Warning: GOOGLE_API_KEY not properly configured!")
        print(f"   Current value: {api_key}")
        print("   Please check your .env file.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print(f"\n✅  Environment configured (Model: {model})")

def run_demo():
    """Run demonstration workflow."""
    print_banner()
    
    # Load environment
    load_dotenv()
    check_environment()
    
    
    print("\n🔄 Workflow Mode Selection")
    print("=" * 60)
    print("1. Traditional Mode (Platform orchestrates agents)")
    print("2. A2A Mode (Event-driven, agents communicate via messages)")
    print("=" * 60)
    
    mode_choice = input("\nSelect mode (1 or 2, default=1): ").strip()
    use_a2a = mode_choice == "2"
    
    try:
        platform = JoltPlatform(output_dir="./quickstart_output")
        platform.load_default_agents()
        
        if use_a2a:
            platform.enable_a2a_mode()
            print("\n📡 A2A Mode Enabled - Agents will communicate via Message Bus")
        else:
            print("\n🔧 Traditional Mode - Platform will orchestrate agents")
    except Exception as e:
        print(f"\n❌ Error initializing platform: {e}")
        print("\nPlease ensure:")
        print("  1. You have created a .env file with OPENAI_API_KEY")
        print("  2. All dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    
    # Example transformation
    print("\n" + "=" * 60)
    print("📝 Example: User Profile Transformation")
    print("=" * 60)
    
    input_json = {
        "user": {
            "firstName": "Jane",
            "lastName": "Smith",
            "email": "jane.smith@example.com",
            "age": 30
        },
        "metadata": {
            "timestamp": "2024-01-15T10:30:00Z",
            "source": "web"
        }
    }
    
    expected_output = {
        "profile": {
            "fullName": "Jane Smith",
            "contact": {
                "email": "jane.smith@example.com"
            },
            "age": 30
        },
        "eventTime": "2024-01-15T10:30:00Z",
        "dataSource": "web"
    }
    
    print("\n📥 INPUT JSON:")
    print(json.dumps(input_json, indent=2))
    
    print("\n📤 EXPECTED OUTPUT JSON:")
    print(json.dumps(expected_output, indent=2))
    
    print("\n" + "=" * 60)
    print("🔄 Running Multi-Agent Workflow...")
    print("=" * 60)
    
    try:
        if use_a2a:
            # A2A Mode - Event-driven workflow
            result = platform.run_a2a_workflow(
                input_json,
                expected_output
            )
        else:
            # Traditional Mode - Procedural workflow
            result = platform.create_and_validate(
                input_json,
                expected_output,
                save_outputs=True
            )
        
        if result.get('status') == 'success' or 'jolt_spec' in result:
            print("\n" + "=" * 60)
            print("✅ SUCCESS!")
            print("=" * 60)
            
            print("\n📋 GENERATED JOLT SPECIFICATION (by CrewAI):")
            print(json.dumps(result['jolt_spec'], indent=2))
            
            print("\n📊 VALIDATION REPORT (by LangChain):")
            validation_output = result['validation_report'].get('validation_result', 'See full report in output files')
            print(validation_output)
            
            print("\n" + "=" * 60)
            print("💾 Output files saved to: ./quickstart_output/")
            print("=" * 60)
        else:
            print(f"\n❌ Workflow failed at stage: {result.get('stage', 'unknown')}")
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"\n❌ Error during workflow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Next steps
    print("\n" + "=" * 60)
    print("🎉 Quick Start Complete!")
    print("=" * 60)
    print("\n📚 Next Steps:")
    print("  1. Check ./quickstart_output/ for generated files")
    print("  2. Try the API server: python platform/api_server.py")
    print("  3. Read README.md for more examples")
    print("  4. Explore individual agents in agents/ directory")
    print("\n✨ Happy transforming!\n")

if __name__ == "__main__":
    run_demo()
