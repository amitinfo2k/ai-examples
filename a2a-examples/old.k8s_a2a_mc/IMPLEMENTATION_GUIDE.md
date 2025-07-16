# Implementation Guide: Kubernetes CrashLoopBackOff Diagnosis System

This guide provides detailed instructions on how to set up and use the Kubernetes CrashLoopBackOff Diagnosis System.

## System Architecture

The system consists of the following components:

1. **Kubernetes MCP Server**: Exposes tools for kubectl commands to interact with a Kubernetes cluster.
2. **Kubernetes Info Agent**: Acts as an MCP client to interact with the Kubernetes MCP server and as an A2A server to expose skills to the Orchestrator Agent.
3. **Log Analysis Agent**: Analyzes logs from Kubernetes pods to identify issues that might be causing CrashLoopBackOff states.
4. **Orchestrator Agent**: Acts as the central coordinator, delegating tasks to specialized agents and aggregating their findings.
5. **User Agent**: Serves as the interface for users to interact with the system and initiate the pod troubleshooting process.

## Prerequisites

- Python 3.10.12 or higher
- Kubernetes v1.30.4 or higher
- Access to a Kubernetes cluster with kubectl configured
- Gemini API key (for log analysis and orchestration)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/k8s-a2a-mc.git
   cd k8s-a2a-mc
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

3. Create a `.env` file from the template:
   ```bash
   cp .env.example .env
   ```

4. Edit the `.env` file and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Running the System

### Option 1: Run All Services at Once

To start all services (MCP server and agents) in one command:

```bash
python main.py run-all
```

This will start:
- Kubernetes MCP Server on port 8000
- Kubernetes Info Agent on port 8001
- Log Analysis Agent on port 8002
- Orchestrator Agent on port 8003

### Option 2: Run Services Individually

You can also run each service individually in separate terminals:

1. Start the Kubernetes MCP Server:
   ```bash
   python main.py mcp-server
   ```

2. Start the Kubernetes Info Agent:
   ```bash
   python main.py k8s-agent
   ```

3. Start the Log Analysis Agent:
   ```bash
   python main.py log-agent
   ```

4. Start the Orchestrator Agent:
   ```bash
   python main.py orchestrator
   ```

## Troubleshooting a Pod

Once all services are running, you can troubleshoot a pod in CrashLoopBackOff state:

```bash
python main.py troubleshoot --namespace <namespace> --pod <pod-name>
```

For example:
```bash
python main.py troubleshoot --namespace default --pod my-app-pod
```

This will:
1. Retrieve information about the pod from the Kubernetes cluster
2. Analyze the pod logs to identify issues
3. Generate a comprehensive troubleshooting report with recommendations

## Development

### Project Structure

```
k8s-a2a-mc/
├── agents/                     # Agent implementations
│   ├── __init__.py
│   ├── kubernetes_info_agent.py
│   ├── log_analysis_agent.py
│   ├── orchestrator_agent.py
│   └── user_agent.py
├── kubernetes_mcp_server/      # MCP server implementation
│   ├── __init__.py
│   └── server.py
├── utils/                      # Utility modules
│   ├── __init__.py
│   └── config.py
├── .env.example                # Environment variables template
├── main.py                     # Main entry point
├── pyproject.toml              # Project configuration
└── README.md                   # Project documentation
```

### Adding New Features

#### Adding a New Tool to the MCP Server

To add a new tool to the Kubernetes MCP Server:

1. Open `kubernetes_mcp_server/server.py`
2. Add a new function with the `@mcp.tool()` decorator:
   ```python
   @mcp.tool()
   def my_new_tool(param1: str, param2: int) -> Dict[str, Any]:
       """
       Description of the tool.
       
       Args:
           param1: Description of param1
           param2: Description of param2
           
       Returns:
           Tool result
       """
       # Tool implementation
       return {"result": "some result"}
   ```

#### Adding a New Skill to an Agent

To add a new skill to an agent:

1. Open the agent file (e.g., `agents/kubernetes_info_agent.py`)
2. Define a new request model if needed:
   ```python
   class MyNewSkillRequest(BaseModel):
       """Request model for the new skill."""
       param1: str
       param2: int
   ```
3. Add a new method to the agent class:
   ```python
   async def my_new_skill(self, request: TaskRequest) -> TaskResult:
       """
       Description of the skill.
       
       Args:
           request: Task request
           
       Returns:
           Task result
       """
       # Skill implementation
       return TaskResult(artifacts={"result": "some result"})
   ```
4. Register the skill in the agent's `__init__` method:
   ```python
   self.register_skill(
       Skill(
           name="my_new_skill",
           description="Description of the skill",
           handler=self.my_new_skill,
           input_schema=MyNewSkillRequest,
       )
   )
   ```

## Troubleshooting

### Common Issues

1. **MCP Server Connection Error**:
   - Ensure the MCP server is running on the correct port
   - Check if the Kubernetes cluster is accessible
   - Verify that kubectl is properly configured

2. **Agent Communication Error**:
   - Ensure all agents are running on their respective ports
   - Check if the agent URLs in the `.env` file are correct

3. **Log Analysis Error**:
   - Verify that the Gemini API key is correctly set in the `.env` file
   - Check if the pod has any logs available

4. **Pod Not Found Error**:
   - Verify that the pod exists in the specified namespace
   - Check if you have permission to access the pod

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
