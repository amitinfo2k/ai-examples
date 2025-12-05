# Project Brief: Multi-Agent Jolt Specification Generation & Validation System

## 1. Executive Summary
This project aims to demonstrate a sophisticated multi-agent system designed to automate the generation and validation of Jolt (JSON to JSON transformation) specifications. The system leverages distinct agentic frameworks (CrewAI and LangChain/LangGraph) to showcase interoperability and specialized agent roles. A key component is the integration of a Model Context Protocol (MCP) server to securely access Google Drive resources, managed by an Agentic Orchestrator with robust Authentication and Authorization mechanisms.

## 2. Key Objectives
1.  **Heterogeneous Multi-Agent Architecture**: Implement agents using different technologies to demonstrate flexibility.
    *   **Generation Agent**: Built with **CrewAI**.
    *   **Validation Agent**: Built with **LangChain/LangGraph**.
2.  **Secure Resource Access via MCP**: Utilize an MCP server to provide agents with controlled access to files (Google Drive).
3.  **Security & Orchestration**: Implement an Agentic Orchestrator handling Authentication (AuthN) and Authorization (AuthZ), ensuring users explicitly map their credentials for resource access.
4.  **Closed-Loop Workflow**: Automate the cycle of generation, validation, and potential regeneration or human-in-the-loop intervention.
5.  **Agent-to-Agent (A2A) Collaboration**: Implement a "Collaborative Debugging" protocol where the Validator and Generator agents directly exchange structured messages (Error Reports, Diagnostic Queries, Patch Proposals) to iteratively fix issues without Orchestrator intervention.

## 3. System Architecture & Components

### 3.1. User Interface (Streamlit)
*   **Role**: The front-end dashboard for user interaction and visualization.
*   **Features**:
    *   **Auth UI**: User-friendly login and Google Drive OAuth mapping.
    *   **Task Management**: File selector to trigger generation tasks.
    *   **Live Visualization**: Real-time chat interface showing the "A2A Collaborative Debugging" session between agents.
    *   **Report Viewer**: Side-by-side JSON diff viewer for validation reports.

### 3.2. Agentic Orchestrator (FastAPI Backend)
*   **Role**: The central REST API managing security, state, and agent coordination.
*   **Role**: The central hub managing user interactions, security context, and task delegation.
*   **Features**:
    *   **Authentication**: Verifies user identity.
    *   **Authorization**: Manages permissions for agents to access specific tools and resources.
    *   **Auth Mapping**: Provides a mechanism for users to securely map their Google Drive authentication tokens/credentials to the MCP server context.

### 3.3. MCP Server (Google Drive Integration)
*   **Function**: Acts as the bridge between the agents and the external file system (Google Drive).
*   **Capabilities**:
    *   Exposes a `read_file` tool.
    *   Restricted scope: Read-only access to a specific, user-defined directory.
    *   Requires valid auth tokens passed from the Orchestrator.

### 3.4. Agent 1: The Generator (CrewAI)
*   **Tech Stack**: CrewAI.
*   **Responsibility**:
    *   Reads `input.json` (source) and `output.json` (target) from the MCP-connected drive.
    *   Analyzes the structure and data transformation requirements.
    *   Generates a `jolt_spec.json` file designed to transform the input to the output.

### 3.5. Agent 2: The Validator (LangChain/LangGraph)
*   **Tech Stack**: LangChain / LangGraph.
*   **Responsibility**:
    *   Reads the original `input.json` and the generated `jolt_spec.json`.
    *   Executes the Jolt transformation.
    *   Compares the result against the expected `output.json`.
    *   Generates a **Validation Report**.
*   **Decision Logic**:
    *   **Pass**: Marks the task as complete.
    *   **Fail**:
        *   Analyzes the failure severity.
        *   **Regenerate**: If the error is programmatic or logic-based, sends feedback back to Agent 1 to regenerate the spec.
        *   **Human-in-the-loop**: If the error is ambiguous or requires domain expertise, flags the task for human review.

## 4. Operational Workflow

1.  **Initialization & Auth**:
    *   User logs in to the Orchestrator.
    *   User provides Google Drive credentials/mapping for the specific working directory.
    *   Orchestrator initializes the MCP server connection with these credentials.

2.  **Task Trigger**:
    *   User initiates a "Generate Jolt Spec" task, pointing to specific `input.json` and `output.json` files.

3.  **Generation Phase (Agent 1 - CrewAI)**:
    *   Agent 1 receives the task.
    *   Uses MCP tool to read `input.json` and `output.json`.
    *   Synthesizes the transformation logic.
    *   Writes/Returns the `jolt_spec.json`.

4.  **Validation & Collaborative Debugging (A2A Phase)**:
    *   Agent 2 picks up the `jolt_spec.json` and `input.json`.
    *   Performs the transformation locally or via a utility.
    *   **Outcome A (Success)**: Returns a success report to the user.
    *   **Outcome B (Failure - A2A Trigger)**:
        *   Validator initiates a **Direct A2A Session** with Agent 1.
        *   **Protocol Loop**:
            *   Validator sends `ERROR_REPORT` (e.g., "Mismatch at path $.foo").
            *   Generator analyzes and sends `PATCH_PROPOSAL` (updated spec snippet).
            *   Validator tests patch and sends `VERIFICATION_RESULT`.
        *   Loop continues until success or retry limit reached.
    *   **Final Result**: The agreed-upon spec is returned to the Orchestrator.

## 5. Technical Requirements
*   **Language**: Python.
*   **Backend**: FastAPI (Orchestrator API).
*   **Frontend**: Streamlit (UI & Visualization).
*   **Frameworks**: CrewAI, LangChain, LangGraph.
*   **Protocol**: Model Context Protocol (MCP) for tool/resource abstraction.
*   **External Service**: Google Drive API.
