#!/usr/bin/env python3
"""
Main script for the Kubernetes CrashLoopBackOff diagnosis system.

This script provides commands to run the MCP server and agents.
"""

import argparse
import asyncio
import logging
import subprocess
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_mcp_server():
    """Run the Kubernetes MCP server."""
    from kubernetes_mcp_server.server import mcp
    import uvicorn
    
    logger.info("Starting Kubernetes MCP server...")
    # With the latest FastMCP version, we can directly pass the mcp object to uvicorn
    uvicorn.run(mcp, host="0.0.0.0", port=8000)


async def run_k8s_agent():
    """Run the Kubernetes Info Agent."""
    from agents.kubernetes_info_agent import main as k8s_agent_main
    
    logger.info("Starting Kubernetes Info Agent...")
    await k8s_agent_main()


async def run_log_agent():
    """Run the Log Analysis Agent."""
    from agents.log_analysis_agent import main as log_agent_main
    
    logger.info("Starting Log Analysis Agent...")
    await log_agent_main()


async def run_orchestrator_agent():
    """Run the Orchestrator Agent."""
    from agents.orchestrator_agent import main as orchestrator_agent_main
    
    logger.info("Starting Orchestrator Agent...")
    await orchestrator_agent_main()


async def run_user_agent():
    """Run the User Agent."""
    from agents.user_agent import main as user_agent_main
    
    logger.info("Starting User Agent...")
    await user_agent_main()


def run_all_services():
    """Run all services in separate processes."""
    # Start MCP server
    mcp_server_process = subprocess.Popen(
        [sys.executable, "-c", "from main import run_mcp_server; run_mcp_server()"]
    )
    logger.info("MCP server started")
    
    # Wait for MCP server to start
    time.sleep(2)
    
    # Start Kubernetes Info Agent
    k8s_agent_process = subprocess.Popen(
        [sys.executable, "-c", "import asyncio; from main import run_k8s_agent; asyncio.run(run_k8s_agent())"]
    )
    logger.info("Kubernetes Info Agent started")
    
    # Start Log Analysis Agent
    log_agent_process = subprocess.Popen(
        [sys.executable, "-c", "import asyncio; from main import run_log_agent; asyncio.run(run_log_agent())"]
    )
    logger.info("Log Analysis Agent started")
    
    # Start Orchestrator Agent
    orchestrator_process = subprocess.Popen(
        [sys.executable, "-c", "import asyncio; from main import run_orchestrator_agent; asyncio.run(run_orchestrator_agent())"]
    )
    logger.info("Orchestrator Agent started")
    
    # Start User Agent
    user_agent_process = subprocess.Popen(
        [sys.executable, "-c", "import asyncio; from main import run_user_agent; asyncio.run(run_user_agent())"]
    )
    logger.info("User Agent started")
    
    # Wait for all processes to finish (which they won't unless terminated)
    try:
        mcp_server_process.wait()
    except KeyboardInterrupt:
        logger.info("Stopping all services...")
        mcp_server_process.terminate()
        k8s_agent_process.terminate()
        log_agent_process.terminate()
        orchestrator_process.terminate()
        user_agent_process.terminate()


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Kubernetes CrashLoopBackOff Diagnosis System"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # MCP server command
    subparsers.add_parser("mcp-server", help="Run the Kubernetes MCP server")
    
    # K8s agent command
    subparsers.add_parser("k8s-agent", help="Run the Kubernetes Info Agent")
    
    # Log agent command
    subparsers.add_parser("log-agent", help="Run the Log Analysis Agent")
    
    # Orchestrator agent command
    subparsers.add_parser("orchestrator", help="Run the Orchestrator Agent")
    
    # User agent command
    subparsers.add_parser("user-agent", help="Run the User Agent")
    
    # Run all services command
    subparsers.add_parser("run-all", help="Run all services")
    
    args = parser.parse_args()
    
    if args.command == "mcp-server":
        run_mcp_server()
    elif args.command == "k8s-agent":
        asyncio.run(run_k8s_agent())
    elif args.command == "log-agent":
        asyncio.run(run_log_agent())
    elif args.command == "orchestrator":
        asyncio.run(run_orchestrator_agent())
    elif args.command == "user-agent":
        asyncio.run(run_user_agent())
    elif args.command == "run-all":
        run_all_services()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
