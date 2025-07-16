#!/usr/bin/env python3
"""
Test script for the Kubernetes CrashLoopBackOff diagnosis system.

This script helps deploy a test pod in CrashLoopBackOff state and run the diagnosis system.
"""

import argparse
import os
import subprocess
import sys
import time

# Add parent directory to path to import from main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_user_agent


def deploy_crashloop_pod(namespace="default"):
    """Deploy a pod that will enter CrashLoopBackOff state."""
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_crashloopbackoff_pod.yaml")
    
    # Create namespace if it doesn't exist
    subprocess.run(["kubectl", "create", "namespace", namespace], 
                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Apply the pod YAML
    result = subprocess.run(
        ["kubectl", "apply", "-f", yaml_path, "-n", namespace],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error deploying pod: {result.stderr}")
        return False
    
    print(f"Pod deployed in namespace {namespace}")
    return True


def wait_for_crashloopbackoff(namespace="default", pod_name="crashloop-pod", timeout=60):
    """Wait for the pod to enter CrashLoopBackOff state."""
    print(f"Waiting for pod {pod_name} to enter CrashLoopBackOff state...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        result = subprocess.run(
            ["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "jsonpath={.status.phase},{.status.containerStatuses[0].state.waiting.reason}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print("Error checking pod status")
            time.sleep(2)
            continue
        
        status = result.stdout.split(",")
        if len(status) > 1 and status[1] == "CrashLoopBackOff":
            print(f"Pod {pod_name} is now in CrashLoopBackOff state")
            return True
        
        print(f"Current pod status: {result.stdout}")
        time.sleep(5)
    
    print(f"Timeout waiting for pod {pod_name} to enter CrashLoopBackOff state")
    return False


def delete_crashloop_pod(namespace="default", pod_name="crashloop-pod"):
    """Delete the test pod."""
    subprocess.run(
        ["kubectl", "delete", "pod", pod_name, "-n", namespace],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print(f"Pod {pod_name} deleted from namespace {namespace}")


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(
        description="Test the Kubernetes CrashLoopBackOff diagnosis system"
    )
    
    parser.add_argument("--namespace", "-n", default="default", help="Namespace to deploy the test pod in")
    parser.add_argument("--pod-name", "-p", default="crashloop-pod", help="Name of the test pod")
    parser.add_argument("--deploy-only", action="store_true", help="Only deploy the test pod, don't run the diagnosis")
    parser.add_argument("--delete-only", action="store_true", help="Only delete the test pod")
    parser.add_argument("--diagnose-only", action="store_true", help="Only run the diagnosis on an existing pod")
    
    args = parser.parse_args()
    
    if args.delete_only:
        delete_crashloop_pod(args.namespace, args.pod_name)
        return
    
    if not args.diagnose_only:
        # Deploy the test pod
        if not deploy_crashloop_pod(args.namespace):
            return
        
        # Wait for the pod to enter CrashLoopBackOff state
        if not wait_for_crashloopbackoff(args.namespace, args.pod_name):
            return
    
    if args.deploy_only:
        print("Pod deployed successfully. Run with --diagnose-only to diagnose it.")
        return
    
    # Run the diagnosis
    print("\nStarting diagnosis system...")
    
    # Import here to avoid circular imports
    import asyncio
    
    try:
        # Run the user agent to diagnose the pod
        asyncio.run(run_user_agent(args.namespace, args.pod_name))
    except KeyboardInterrupt:
        print("\nDiagnosis interrupted")


if __name__ == "__main__":
    main()
