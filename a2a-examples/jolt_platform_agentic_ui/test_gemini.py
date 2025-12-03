import os
import sys
from crewai import LLM, Agent, Task, Crew

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

def test_gemini():
    print("Testing Gemini Connection...")
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables!")
        return
    
    model = os.environ.get("GEMINI_MODEL", "gemini/gemini-1.5-flash")
    print(f"Using Model: {model}")
    
    try:
        llm = LLM(
            model=model,
            api_key=api_key
        )
        
        agent = Agent(
            role="Test Agent",
            goal="Say hello",
            backstory="You are a friendly test agent.",
            llm=llm,
            verbose=True
        )
        
        task = Task(
            description="Say 'Hello, World!' and nothing else.",
            expected_output="Hello, World!",
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()
        
        print(f"\n✅ Success! Result: {result}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gemini()
