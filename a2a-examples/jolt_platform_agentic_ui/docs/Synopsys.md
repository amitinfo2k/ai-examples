# Technical Synopsis: Heterogeneous Multi-Agent Jolt System

## 1. Executive Summary

This project demonstrates a sophisticated **heterogeneous multi-agent architecture** designed to automate the generation and validation of Jolt (JSON-to-JSON) transformation specifications. The system addresses the challenge of interoperability by integrating two distinct agentic frameworks—**CrewAI** (Generation) and **LangGraph** (Validation)—into a cohesive, secure workflow.

Critically, the entire system is deployed as **distributed microservices within a Kubernetes (K8s) environment**. This ensures that each agent operates in its own isolated container with independent dependencies, proving that agents built on entirely different stacks can collaborate effectively over a network.

A defining feature is the **Agent-to-Agent (A2A) Collaborative Debugging Protocol**. Instead of relying on a central orchestrator to manage every retrial step, the Validator agent autonomously initiates a direct feedback loop with the Generator agent to resolve errors iteratively, mimicking a human developer-tester workflow.

## 2. Distributed Microservices Architecture

The system is architected not as a monolithic application, but as a cluster of independent services running in a `jolt-platform` Kubernetes namespace. This decoupling allows for true heterogeneous environments where Python versions, dependencies, and resources are isolated per agent.

### 2.1 Kubernetes Service Topology

* **Orchestrator Service (`jolt-orchestrator-service`):**
  * **Role:** The central API gateway (FastAPI) managing Authentication (AuthN) and Authorization (AuthZ).
  * **Function:** Handles user requests and delegates initial tasks. It exposes the system to the frontend but steps back during high-frequency debugging loops.

* **Generator Service (`jolt-generator-service`):**
  * **Role:** A specialized **CrewAI** container.
  * **Function:** Analyzes input/output JSON and synthesizes Jolt specifications. By running as an independent service, it isolates the CrewAI heavy dependencies from the rest of the system.

* **Validator Service (`jolt-validator-service`):**
  * **Role:** A specialized **LangGraph** container.
  * **Function:** Executes validation logic. It acts as an internal client within the cluster, sending HTTP requests directly to the Generator service during debugging sessions.

* **Google Drive MCP Service (`drive-mcp-service`):**
  * **Role:** Secure file access interface.
  * **Function:** Exposes file reading capabilities via MCP to the Generator agent, ensuring access is scoped to authorized directories.

* **Jolt MCP Service (`jolt-mcp-service`):**
  * **Role:** A dedicated Model Context Protocol server (written in Go or Python).
  * **Function:** Provides the raw transformation engine. This service exposes `transform` tools via the MCP protocol, accessible only to authorized agents (specifically the Validator).

### 2.2 Heterogeneous Communication Flow

Communication occurs over the Kubernetes internal network using HTTP/REST and WebSockets. This enforces a strict contract between agents, allowing them to be written in different languages or frameworks without integration issues.

**The Workflow:**

1. **Delegation:** The Orchestrator calls the **Generator Service**.
2. **Hand-off:** The Orchestrator passes the result to the **Validator Service**.
3. **A2A Loop:** If validation fails, the **Validator** initiates a direct service-to-service call to the **Generator's** `/refine` endpoint. This loop happens entirely within the cluster network, bypassing the Orchestrator for efficiency.

## 3. Architecture & Flow Diagrams

### 3.1 Detailed Technical Flow (Kubernetes/Service Level)

The following diagram illustrates the interaction between these distributed services. Note that "Gen", "Val", and "Orch" represent distinct Kubernetes Pods communicating over the cluster network.

```mermaid
sequenceDiagram
    participant User as Streamlit Frontend
    participant Orch as K8s: Orchestrator Service
    participant Gen as K8s: Generator Service (CrewAI)
    participant Drive as K8s: Google Drive MCP Service
    participant Val as K8s: Validator Service (LangGraph)
    participant Jolt as K8s: Jolt MCP Service

    Note over User, Orch: Phase 1: Task Initiation
    User->>Orch: POST /workflow/generate-and-validate
    
    Note over Orch, Gen: Phase 2: Generation
    Orch->>Gen: POST /generate (Input/Output JSON)
    Gen->>Drive: Read Files (input.json/output.json)
    Drive-->>Gen: File Content
    Gen-->>Orch: Returns initial jolt_spec.json

    Note over Orch, Val: Phase 3: Validation
    Orch->>Val: POST /validate-with-a2a (jolt_spec + input)
    Val->>Drive: Read Files (input.json/output.json)
    Drive-->>Val: Returns Source & Expected Data
    
    rect rgb(240, 240, 255)
        note right of Val: Phase 4: A2A Collaborative Debugging (Service-to-Service)
        Val->>Jolt: Transform (via Jolt MCP)
        Jolt-->>Val: Transformation Result
        
        alt Validation Fails
            Val->>Gen: POST /refine (ERROR_REPORT)
            Note right of Gen: Generator Pod analyzes error<br/>and updates spec
            Gen-->>Val: PATCH_PROPOSAL (New Spec)
            Val->>Val: Re-test with New Spec
        else Validation Succeeds
            Val->>Val: Mark Complete
        end
    end
    
    Val-->>Orch: Final Validated Spec & Report
    Orch-->>User: Display Results
````

### 3.2 Simplified Logical Flow

This simplified diagram abstracts the infrastructure details to highlight the logical interaction and the autonomous nature of the agent collaboration.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Orch as Orchestrator
    participant Gen as Generator
    participant Val as Validator

    User->>Orch: Start Workflow
    Orch->>Gen: "Create Jolt Spec"
    Gen-->>Orch: Draft Spec
    Orch->>Val: "Validate this Spec"
    
    rect rgb(200, 255, 200)
        Note over Gen, Val: Autonomous "A2A" Debugging Loop
        loop Until Valid or Max Retries
            Val->>Gen: "Error Report: Fix required"
            Gen->>Val: "Patch Proposal: Try this"
        end
    end
    
    Val-->>Orch: Verified Spec
    Orch-->>User: Final Result
```

## 4\. Key Technical Achievements

  * **Distributed Isolation:** Each agent runs in its own Kubernetes Pod. This proves that an agent built on `CrewAI` (potentially requiring specific Python libraries) can collaborate seamlessly with an agent built on `LangGraph`, as they only share an API contract, not a runtime environment.

  * **Decoupled Refinement:** The Orchestrator does not manage the retry logic; the Validator "owns" the quality assurance process, directly pinging the Generator service to fix issues.

  * **Scalability:** Because they are microservices, the Generator and Validator can be scaled independently (e.g., spinning up more Generator pods for heavy workloads) without affecting the rest of the system.

  * **Secure Tooling:** The Jolt MCP runs as a standalone service (`jolt-mcp-service`), adhering to the "least privilege" principle. Agents must authenticate via the Orchestrator's token system to access these tools.
