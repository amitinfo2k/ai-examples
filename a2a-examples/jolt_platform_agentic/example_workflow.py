"""
Example Workflow: How CrewAI and LangChain Agents Work Together

This script demonstrates the complete workflow showing how both agents
collaborate on the same platform.
"""

from jolt_platform.unified_platform import JoltPlatform
from dotenv import load_dotenv
import json

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def example_workflow():
    """Demonstrate the complete workflow."""
    load_dotenv()
    
    print_section("🎯 MULTI-AGENT JOLT WORKFLOW EXAMPLE")
    
    print("\nThis example shows how CrewAI and LangChain agents work together:")
    print("  1️⃣  CrewAI Agent creates the JOLT specification")
    print("  2️⃣  LangChain Agent validates the specification")
    print("  3️⃣  Platform orchestrates both on a single system")
    
    # Choose workflow mode
    print("\n🔄 Workflow Mode Selection")
    print("=" * 60)
    print("1. Traditional Mode (Platform orchestrates agents)")
    print("2. A2A Mode (Event-driven, agents communicate via messages)")
    print("=" * 60)
    
    mode_choice = input("\nSelect mode (1 or 2, default=1): ").strip()
    use_a2a = mode_choice == "2"
    
    # Initialize platform
    print("\n🚀 Initializing JOLT Platform...")
    platform = JoltPlatform(output_dir="./workflow_output")
    platform.load_default_agents()
    
    if use_a2a:
        platform.enable_a2a_mode()
        print("\n📡 A2A Mode Enabled - Agents will communicate via Message Bus")
    else:
        print("\n🔧 Traditional Mode - Platform will orchestrate agents")
    
    print("✅ Platform initialized with both agents!")
    
    # Define transformation
    print_section("📝 Step 2: Define Your Transformation")
    
    input_json = {
        "employee": {
            "id": "EMP001",
            "personalInfo": {
                "firstName": "Sarah",
                "lastName": "Johnson",
                "email": "sarah.j@company.com"
            },
            "department": "Engineering",
            "salary": 95000
        },
        "hireDate": "2024-01-15"
    }
    
    expected_output = {
        "employeeId": "EMP001",
        "fullName": "Sarah Johnson",
        "contact": {
            "email": "sarah.j@company.com"
        },
        "dept": "Engineering",
        "compensation": 95000,
        "startDate": "2024-01-15"
    }
    
    print("\n📥 INPUT JSON:")
    print(json.dumps(input_json, indent=2))
    
    print("\n📤 EXPECTED OUTPUT JSON:")
    print(json.dumps(expected_output, indent=2))
    
    # Run the workflow
    print_section("🔄 Step 3: Run Multi-Agent Workflow")
    
    
    if use_a2a:
        print("\n📡 A2A MODE: Triggering workflow via message bus...")
        print("   Platform will publish START_WORKFLOW message")
        print("   Agents will react asynchronously")
        result = platform.run_a2a_workflow(
            input_json,
            expected_output
        )
    else:
        print("\n🤖 TRADITIONAL MODE: Platform orchestrating agents...")
        print("   PHASE 1: CrewAI Agent is creating JOLT specification...")
        print("   (This agent analyzes the JSON structures and generates the spec)")
        result = platform.create_and_validate(
            input_json,
            expected_output,
            save_outputs=True
        )
    
    # Display results
    if result.get('status') == 'success' or 'jolt_spec' in result:
        print_section("✅ Step 4: View Results")
        
        print("\n🤖 CREWAI AGENT OUTPUT:")
        print("   Generated JOLT Specification:")
        print(json.dumps(result['jolt_spec'], indent=2))
        
        print("\n✅ LANGCHAIN AGENT OUTPUT:")
        print("   Validation Report:")
        validation_result = result['validation_report'].get('validation_result', 'See report file')
        print(f"   {validation_result}")
        
        print_section("💾 Step 5: Files Saved")
        print("\n✅ Output files saved to: ./example_output/")
        print("   • JOLT Specification (from CrewAI)")
        print("   • Validation Report (from LangChain)")
        
        print_section("🎉 WORKFLOW COMPLETE!")
        print("\n📊 Summary:")
        print(f"   ✅ CrewAI Agent: Successfully created JOLT spec")
        print(f"   ✅ LangChain Agent: Successfully validated spec")
        print(f"   ✅ Platform: Orchestrated both agents on single system")
        print(f"   ✅ Status: {result['status']}")
        
    else:
        print(f"\n❌ Workflow failed: {result.get('error')}")
    
    print_section("🎓 What Just Happened?")
    print("""
    1. The UNIFIED PLATFORM orchestrated both agents
    
    2. The CREWAI AGENT:
       • Received input and expected output JSON
       • Analyzed the structure and field mappings
       • Generated a JOLT specification
       • Used GPT-4 to understand the transformation
    
    3. The LANGCHAIN AGENT:
       • Received the JOLT spec from CrewAI
       • Applied the transformation to the input
       • Compared result with expected output
       • Generated a detailed validation report
    
    4. Both agents ran on the SAME PLATFORM, working together!
    """)
    
    print_section("🚀 Next Steps")
    print("""
    Try these:
    
    1. Create your own transformation:
       from jolt_platform.unified_platform import JoltPlatform
       platform = JoltPlatform()
       platform.load_default_agents()
       result = platform.create_and_validate(your_input, your_output)
    
    2. Use individual agents:
       # Only CrewAI
       spec = platform.create_spec_only(input_json, output_json)
       
       # Only LangChain
       report = platform.validate_spec_only(spec, input_json, expected)
    
    3. Use the REST API:
       python platform/api_server.py
       # Then visit http://localhost:8000/docs
    
    4. Check the documentation:
       • README.md - Features overview
       • GETTING_STARTED.md - Step-by-step guide
       • ARCHITECTURE.md - System design
    """)
    
    print("\n" + "=" * 70)
    print("  ✨ Multi-Agent Platform Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    example_workflow()
