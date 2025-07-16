#!/usr/bin/env python3
"""
Full test script for the Kubernetes CrashLoopBackOff diagnosis system.

This script deploys a test pod, starts all services, and runs the diagnosis.
"""

import argparse
import asyncio
import os
import subprocess
import sys
import threading
import time

# Add parent directory to path to import from main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_mcp_server, run_k8s_agent, run_log_agent, run_orchestrator_agent, run_user_agent


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


def run_mcp_server_thread():
    """Run the MCP server in a separate thread."""
    run_mcp_server()


async def run_all_agents():
    """Run all agents concurrently."""
    # Create tasks for each agent
    k8s_agent_task = asyncio.create_task(run_k8s_agent())
    log_agent_task = asyncio.create_task(run_log_agent())
    orchestrator_task = asyncio.create_task(run_orchestrator_agent())
    
    # Wait for all tasks to complete (they won't unless cancelled)
    try:
        await asyncio.gather(k8s_agent_task, log_agent_task, orchestrator_task)
    except asyncio.CancelledError:
        print("Agents stopped")


async def run_full_test(namespace="default", pod_name="crashloop-pod"):
    """Run a full test of the system."""
    # Start the MCP server in a separate thread
    mcp_thread = threading.Thread(target=run_mcp_server_thread)
    mcp_thread.daemon = True
    mcp_thread.start()
    print("MCP server started")
    
    # Wait for MCP server to initialize
    time.sleep(2)
    
    # Start all agents
    agents_task = asyncio.create_task(run_all_agents())
    print("All agents started")
    
    # Wait for agents to initialize
    time.sleep(5)
    
    # Run the user agent to diagnose the pod
    print(f"\nStarting diagnosis for pod {pod_name} in namespace {namespace}...")
    await run_user_agent(namespace, pod_name)
    
    # Cancel the agents task
    agents_task.cancel()
    try:
        await agents_task
    except asyncio.CancelledError:
        pass


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(
        description="Run a full test of the Kubernetes CrashLoopBackOff diagnosis system"
    )
    
    parser.add_argument("--namespace", "-n", default="default", help="Namespace to deploy the test pod in")
    parser.add_argument("--pod-name", "-p", default="crashloop-pod", help="Name of the test pod")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip deploying the test pod")
    parser.add_argument("--cleanup", action="store_true", help="Delete the test pod after diagnosis")
    
    args = parser.parse_args()
    
    if not args.skip_deploy:
        # Deploy the test pod
        if not deploy_crashloop_pod(args.namespace):
            return
        
        # Wait for the pod to enter CrashLoopBackOff state
        if not wait_for_crashloopbackoff(args.namespace, args.pod_name):
            return
    
    # Run the full test
    try:
        asyncio.run(run_full_test(args.namespace, args.pod_name))
    except KeyboardInterrupt:
        print("\nTest interrupted")
    
    if args.cleanup:
        delete_crashloop_pod(args.namespace, args.pod_name)


if __name__ == "__main__":
    main()
