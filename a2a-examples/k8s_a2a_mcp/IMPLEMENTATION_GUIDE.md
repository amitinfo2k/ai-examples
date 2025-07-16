### Option 2: Run Services Individually

You can also run each service individually in separate terminals:

1. Start the Kubernetes MCP Server:
   ```bash   
   uv run k8s-a2a-mcp --run mcp-server --transport sse
   ```

2. Start the Kubernetes Info Agent:
   ```bash
   run src/k8s_debug/agents/ --agent-card agent_cards/k8s_info_agent.json --port 10103
   ```

3. Start the Log Analysis Agent:
   ```bash
   run src/k8s_debug/agents/ --agent-card agent_cards/log_analysis_agent.json --port 10104
   ```

4. Start the Orchestrator Agent:
   ```bash
   run src/k8s_debug/agents/ --agent-card agent_cards/orchestrator_agent.json --port 10101
   ```
5. Start the Langgraph planner Agent:
   ```bash
   run src/k8s_debug/agents/ --agent-card agent_cards/planner_agent.json --port 10102
   ```