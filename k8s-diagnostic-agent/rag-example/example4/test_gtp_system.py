#!/usr/bin/env python3
"""
Test Script for GTP Analysis System

This script demonstrates the complete workflow of the GTP packet analysis system.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'scapy',
        'langchain_google_genai',
        'langchain_community',
        'langchain_text_splitters',
        'langchain',
        'chromadb',
        'pysqlite3_binary'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✓ All dependencies are available")
    return True

def check_api_key():
    """Check if Google API key is set"""
    print("\nChecking Google API key...")
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        print("✓ GOOGLE_API_KEY is set")
        return True
    else:
        print("✗ GOOGLE_API_KEY is not set")
        print("Please set it using:")
        print("export GOOGLE_API_KEY='your_api_key_here'")
        return False

def check_pcap_file():
    """Check if a PCAP file is available for testing"""
    print("\nChecking for PCAP file...")
    
    # Look for any .pcap files in current directory
    pcap_files = list(Path(".").glob("*.pcap"))
    
    if pcap_files:
        print(f"✓ Found PCAP files: {[f.name for f in pcap_files]}")
        return True
    else:
        print("⚠ No PCAP files found in current directory")
        print("You can:")
        print("1. Copy your GTP PCAP file to this directory")
        print("2. Use the --pcap argument to specify a file path")
        print("3. Run: python rag-gtp-analysis.py --pcap /path/to/your/file.pcap")
        return False

def test_gtp_analysis(pcap_file):
    """Test the GTP analysis system"""
    print(f"\nTesting GTP analysis system with {pcap_file}...")
    
    try:
        # Run the GTP analyzer with the PCAP file
        result = subprocess.run([
            sys.executable, "rag-gtp-analysis.py", "--pcap", pcap_file
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ GTP analysis completed successfully")
            print("\nOutput:")
            print(result.stdout)
            return True
        else:
            print(f"✗ Error in GTP analysis:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error running GTP analyzer: {e}")
        return False

def run_interactive_demo():
    """Run interactive demo if user wants"""
    print("\n" + "="*60)
    print("GTP ANALYSIS SYSTEM DEMO")
    print("="*60)
    
    response = input("\nWould you like to run an interactive demo? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        print("\nRunning interactive demo...")
        print("You can ask questions about the GTP packets.")
        print("Type 'quit' to exit.")
        
        try:
            # Import and run the interactive system
            
            
            # Check API key
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                print("Please set GOOGLE_API_KEY first")
                return
            
            # Initialize system
            print("Interactive demo requires running the main GTP analyzer directly.")
            print("Please run: python rag-gtp-analysis.py --pcap your_file.pcap --interactive")
            return
            
        except Exception as e:
            print(f"Error in interactive demo: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main test function"""
    print("GTP Packet Analysis System - Test Suite")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing packages.")
        return
    
    # Check API key
    if not check_api_key():
        print("\n❌ API key check failed. Please set GOOGLE_API_KEY.")
        return
    
    # Check for PCAP files
    if not check_pcap_file():
        print("\n⚠ No PCAP files available for testing.")
        print("Please provide a PCAP file to test the system.")
        return
    
    # Get the first available PCAP file for testing
    pcap_files = list(Path(".").glob("*.pcap"))
    if pcap_files:
        test_pcap = pcap_files[0].name
        print(f"Using {test_pcap} for testing...")
        
        # Test analysis system
        if not test_gtp_analysis(test_pcap):
            print("\n❌ GTP analysis test failed.")
            return
    else:
        print("\n❌ No PCAP files available for testing.")
        return
    
    print("\n✅ All tests passed!")
    
    # Offer interactive demo
    run_interactive_demo()
    
    print("\n🎉 Test suite completed successfully!")
    print("\nNext steps:")
    print("1. Use your own PCAP files with GTP packets")
            print("2. Modify the analysis parameters in rag-gtp-analysis.py")
    print("3. Extend the system with additional GTP protocol features")
    print("4. Integrate with your existing network monitoring tools")

if __name__ == "__main__":
    main()
