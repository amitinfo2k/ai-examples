#!/usr/bin/env python3
"""
Test script for workflow progress tracking

This script tests the new workflow progress tracking endpoints.
Run this after starting the orchestrator to verify everything works.
"""

import requests
import time
import json
import sys


def test_workflow_progress():
    """Test the workflow progress tracking system"""
    
    base_url = "http://localhost:8088"
    
    print("=" * 60)
    print("Workflow Progress Tracking Test")
    print("=" * 60)
    print()
    
    # Step 1: Health check
    print("1. Testing orchestrator health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Orchestrator is healthy")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot connect to orchestrator: {e}")
        print(f"   Make sure orchestrator is running on {base_url}")
        return False
    
    print()
    
    # Step 2: Start workflow
    print("2. Starting workflow...")
    request_data = {
        "task_type": "generate_spec",
        "input_file_path": "input.json",
        "output_file_path": "output.json",
        "auth_mapping": {
            "user_id": "test_user",
            "gdrive_token": "valid_token",
            "target_directory_id": "mock_storage"
        },
        "description": "Test workflow for progress tracking"
    }
    
    try:
        response = requests.post(
            f"{base_url}/workflow/generate-and-validate",
            json=request_data,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"   ❌ Failed to start workflow: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        result = response.json()
        task_id = result.get("task_id")
        
        if not task_id:
            print(f"   ❌ No task_id in response: {result}")
            return False
        
        print(f"   ✅ Workflow started with task_id: {task_id}")
        
    except Exception as e:
        print(f"   ❌ Error starting workflow: {e}")
        return False
    
    print()
    
    # Step 3: Poll for progress
    print("3. Polling for progress (max 60 seconds)...")
    print()
    
    max_polls = 60
    poll_count = 0
    last_progress = -1
    
    while poll_count < max_polls:
        time.sleep(1)
        poll_count += 1
        
        try:
            response = requests.get(
                f"{base_url}/workflow/status/{task_id}",
                timeout=5
            )
            
            if response.status_code != 200:
                print(f"   ❌ Failed to get status: {response.status_code}")
                break
            
            status_data = response.json()
            
            # Extract info
            status = status_data.get("status", "unknown")
            current_step = status_data.get("current_step", "N/A")
            progress = status_data.get("progress_percentage", 0)
            logs = status_data.get("logs", [])
            
            # Only print when progress changes
            if progress != last_progress:
                print(f"   [{poll_count}s] {progress:3d}% | {status:12s} | {current_step}")
                
                # Show latest log
                if logs:
                    latest_log = logs[-1]
                    print(f"        └─ {latest_log}")
                
                last_progress = progress
            
            # Check if completed
            if status in ["completed", "failed", "needs_review"]:
                print()
                print("=" * 60)
                print(f"   🏁 Workflow finished with status: {status.upper()}")
                print("=" * 60)
                print()
                
                # Show final logs
                print("Final Logs (last 5):")
                for log in logs[-5:]:
                    print(f"   - {log}")
                print()
                
                # Show result summary
                result = status_data.get("result")
                if result:
                    print("Result Summary:")
                    print(f"   - Input file: {result.get('input_file_path', 'N/A')}")
                    print(f"   - Output file: {result.get('output_file_path', 'N/A')}")
                    
                    validation = result.get("validation", {})
                    is_valid = validation.get("is_valid", False)
                    print(f"   - Validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
                    
                    if not is_valid:
                        errors = validation.get("errors", [])
                        print(f"   - Errors found: {len(errors)}")
                    
                    jolt_spec = result.get("jolt_spec", [])
                    print(f"   - JOLT operations: {len(jolt_spec) if isinstance(jolt_spec, list) else 1}")
                
                print()
                
                if status == "completed":
                    print("✅ Test PASSED - Workflow completed successfully!")
                    return True
                elif status == "needs_review":
                    print("⚠️  Test PASSED - Workflow needs review (expected for some inputs)")
                    return True
                else:
                    print("❌ Test FAILED - Workflow failed")
                    error = status_data.get("error")
                    if error:
                        print(f"   Error: {error}")
                    return False
        
        except requests.exceptions.Timeout:
            # Continue polling
            continue
        except Exception as e:
            print(f"   ❌ Error polling status: {e}")
            return False
    
    print()
    print(f"⏱️  Test TIMEOUT - No completion after {max_polls} seconds")
    print("   (Workflow may still be running)")
    return False


if __name__ == "__main__":
    print()
    success = test_workflow_progress()
    print()
    
    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed. Check logs above.")
        sys.exit(1)
