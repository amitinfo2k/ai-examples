# Technical Synopsis: Heterogeneous Multi-Agent Jolt System

## 1. Executive Summary

This project demonstrates a sophisticated **heterogeneous multi-agent architecture** designed to automate the generation and validation of Jolt (JSON-to-JSON) transformation specifications. The system addresses the challenge of interoperability by integrating two distinct agentic frameworks—**CrewAI** (Generation) and **LangGraph** (Validation)—into a cohesive, secure workflow.

Critically, the entire system is deployed as **distributed microservices within a Kubernetes (K8s) environment**. This ensures that each agent operates in its own isolated container with independent dependencies, proving that agents built on entirely different stacks can collaborate effectively over a network.

### Key Features

| Feature | Description |
|---------|-------------|
| **A2A Protocol** | Agent-to-Agent Collaborative Debugging between Validator and Generator |
| **HITL Support** | Human-in-the-Loop fallback when automatic refinement exhausts retries |
| **Observability** | End-to-end tracing via LangSmith for both CrewAI and LangGraph agents |
| **MCP Integration** | Model Context Protocol for secure file access and JOLT transformations |

## 2. Distributed Microservices Architecture

The system is architected not as a monolithic application, but as a cluster of independent services running in a `jolt-platform` Kubernetes namespace. This decoupling allows for true heterogeneous environments where Python versions, dependencies, and resources are isolated per agent.

### 2.1 Kubernetes Service Topology

* **Orchestrator Service (`jolt-orchestrator-service`):**
  * **Role:** The central API gateway (FastAPI) managing Authentication (AuthN) and Authorization (AuthZ).
  * **Function:** Handles user requests and delegates initial tasks. It exposes the system to the frontend but steps back during high-frequency debugging loops.

* **Generator Service (`jolt-generator-service`):**
  * **Role:** A specialized **CrewAI** container with OpenInference instrumentation.
  * **Function:** Analyzes input/output JSON and synthesizes Jolt specifications. Supports both automatic refinement (A2A) and human-guided refinement (HITL).

* **Validator Service (`jolt-validator-service`):**
  * **Role:** A specialized **LangGraph** container with native LangChain tracing.
  * **Function:** Executes validation logic. It acts as an internal client within the cluster, sending HTTP requests directly to the Generator service during debugging sessions.

* **Google Drive MCP Service (`drive-mcp-service`):**
  * **Role:** Secure file access interface.
  * **Function:** Exposes file reading capabilities via MCP to the Generator agent, ensuring access is scoped to authorized directories.

* **Jolt MCP Service (`jolt-mcp-service`):**
  * **Role:** A dedicated Model Context Protocol server (written in Go).
  * **Function:** Provides the raw transformation engine. This service exposes `transform` tools via the MCP protocol, accessible only to authorized agents (specifically the Validator).

* **LangSmith (External):**
  * **Role:** Observability and tracing platform.
  * **Function:** Receives telemetry from all agents for monitoring, debugging, and performance analysis.

### 2.2 Heterogeneous Communication Flow

Communication occurs over the Kubernetes internal network using HTTP/REST and WebSockets. This enforces a strict contract between agents, allowing them to be written in different languages or frameworks without integration issues.

**The Workflow:**

1. **Delegation:** The Orchestrator calls the **Generator Service**.
2. **Hand-off:** The Orchestrator passes the result to the **Validator Service**.
3. **A2A Loop:** If validation fails, the **Validator** initiates a direct service-to-service call to the **Generator's** `/refine` endpoint.
4. **HITL Fallback:** If A2A exhausts retries, the user can provide natural language feedback or manually edit the spec.

## 3. Architecture & Flow Diagrams

### 3.0 High-Level System Architecture

This block diagram depicts the structural composition of the system, highlighting the relationships between internal microservices and external cloud dependencies.

```mermaid
graph TB
    %% Nodes
    User((User))
    
    subgraph "External Cloud Services"
        Gemini[Google Gemini API]
        LangSmith[LangSmith Observability]
    end

    subgraph "Kubernetes Cluster (Namespace: jolt-platform)"
        Frontend[Streamlit Frontend]
        Orchestrator[Orchestrator Service]
        
        subgraph "Agent Services"
            Generator["Generator Agent<br/>(CrewAI)"]
            Validator["Validator Agent<br/>(LangGraph)"]
        end
        
        subgraph "MCP Services"
            DriveMCP[Google Drive MCP]
            JoltMCP[Jolt Transformation MCP]
        end
    end

    %% Edges
    User <-->|HTTPS| Frontend
    Frontend <-->|REST| Orchestrator
    
    Orchestrator -->|REST| Generator
    Orchestrator -->|REST| Validator
    
    Validator <-->|A2A Protocol / REST| Generator
    
    Generator -->|MCP Protocol| DriveMCP
    Validator -->|MCP Protocol| DriveMCP
    Validator -->|MCP Protocol| JoltMCP
    
    Generator <-->|HTTPS| Gemini
    
    Generator -.->|HTTPS / Tracing| LangSmith
    Validator -.->|HTTPS / Tracing| LangSmith
    
    %% Styling
    classDef k8s fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef ext fill:#fff3e0,stroke:#ff6f00,stroke-width:2px;
    classDef user fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    
    class Frontend,Orchestrator,Generator,Validator,DriveMCP,JoltMCP k8s;
    class Gemini,LangSmith ext;
    class User user;
```

### 3.1 Detailed Technical Flow (with HITL & Observability)

The following diagram illustrates the complete interaction including **Human-in-the-Loop (HITL)** fallback when automatic A2A debugging fails, and **LangSmith observability** for tracing all agent activities.

```mermaid
sequenceDiagram
    participant User as Streamlit Frontend
    participant Orch as K8s: Orchestrator Service
    participant Gen as K8s: Generator (CrewAI)
    participant Drive as K8s: GDrive MCP
    participant Val as K8s: Validator (LangGraph)
    participant Jolt as K8s: Jolt MCP
    participant LS as LangSmith (Tracing)

    Note over User, LS: Phase 1: Task Initiation
    User->>Orch: POST /workflow/generate-and-validate
    
    Note over Orch, Gen: Phase 2: Generation (Traced)
    Orch->>Gen: POST /generate (Input/Output paths)
    Gen->>LS: 📊 Trace: crewai_generate_jolt_spec
    Gen->>Drive: Read Files (input.json/output.json)
    Drive-->>Gen: File Content
    Gen-->>Orch: Returns initial jolt_spec.json
    Gen->>LS: 📊 Trace: Generation Complete

    Note over Orch, Val: Phase 3: Validation (Traced)
    Orch->>Val: POST /validate-with-a2a (jolt_spec)
    Val->>LS: 📊 Trace: langgraph_validate_spec
    Val->>Drive: Read Files (input.json/output.json)
    Drive-->>Val: Returns Source & Expected Data
    
    rect rgb(240, 240, 255)
        Note right of Val: Phase 4: A2A Collaborative Debugging
        Val->>Jolt: Transform (via MCP)
        Jolt-->>Val: Transformation Result
        
        loop Until Valid or Max Retries (3)
            alt Validation Fails
                Val->>Gen: POST /refine (ERROR_REPORT)
                Gen->>LS: 📊 Trace: crewai_refine_jolt_spec
                Gen-->>Val: PATCH_PROPOSAL (New Spec)
                Val->>Jolt: Re-transform with New Spec
                Jolt-->>Val: New Result
            else Validation Succeeds
                Val->>Val: Mark Complete ✅
            end
        end
    end

    alt A2A Succeeded
        Val-->>Orch: Final Validated Spec & Report
        Orch-->>User: Display Results ✅
    else A2A Failed (Max Retries)
        Val-->>Orch: Partial Result + Errors
        Orch-->>User: Display HITL Interface ⚠️
        
        rect rgb(255, 250, 220)
            Note over User, Gen: Phase 5: Human-in-the-Loop (HITL)
            alt AI-Assisted Refinement
                User->>Orch: Natural Language Feedback
                Orch->>Gen: POST /refine-with-prompt
                Gen->>LS: 📊 Trace: crewai_refine_with_prompt
                Gen-->>Orch: Refined Spec
                Orch->>Val: POST /validate (new spec)
                Val-->>Orch: Validation Result
                Orch-->>User: Updated Results
            else Manual Edit
                User->>User: Edit Jolt Spec JSON
                User->>Orch: POST /validate (edited spec)
                Orch->>Val: Validate edited spec
                Val-->>Orch: Validation Result
                Orch-->>User: Updated Results
            end
        end
    end
```

### 3.2 Simplified Logical Flows

#### 3.2.1 Autonomous A2A Flow
This diagram shows the standard success path where the agents collaborate to solve the problem without human intervention.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Orch as Orchestrator
    participant Gen as Generator
    participant Val as Validator
    participant LLM as 🧠 Gemini
    participant LS as 📊 LangSmith

    User->>Orch: Start Workflow
    Orch->>Gen: "Create Jolt Spec"
    Gen->>LS: Trace: Generation Started
    Gen->>LLM: Generate Spec
    LLM-->>Gen: Spec Content
    Gen-->>Orch: Draft Spec
    Gen->>LS: Trace: Generation Complete
    
    Orch->>Val: "Validate this Spec"
    Val->>LS: Trace: Validation Started
    
    rect rgb(200, 255, 200)
        Note over Gen, Val: Autonomous "A2A" Debugging Loop
        loop Until Valid or Max Retries (3)
            Val->>LLM: Analyze Errors
            LLM-->>Val: Fix Suggestions
            Val->>Gen: "Error Report + AI Analysis"
            Gen->>LS: Trace: Refinement Attempt
            Gen->>LLM: Refine Spec (Error Report)
            LLM-->>Gen: Refined Spec
            Gen->>Val: "Patch Proposal: Try this"
        end
    end
    
    Val->>LS: Trace: Validation Success
    Val-->>Orch: Verified Spec
    Orch-->>User: Final Result
```

#### 3.2.2 Human-in-the-Loop (HITL) Flow
This diagram illustrates the fallback process when the autonomous agents cannot resolve all validation errors, requiring human guidance.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Orch as Orchestrator
    participant Gen as Generator
    participant Val as Validator
    participant LLM as 🧠 Gemini
    participant LS as 📊 LangSmith

    Note over Val, Orch: Scenario: A2A Failed (Max Retries)
    Val->>LS: Trace: Validation Failed - HITL Required
    Val-->>Orch: Partial Result + Errors
    Orch-->>User: Show HITL Interface
    
    rect rgb(255, 245, 200)
        Note over User, Gen: Human-in-the-Loop Intervention
        alt 💬 AI-Assisted
            User->>Gen: Natural Language Feedback
            Gen->>LS: Trace: HITL Prompt Refinement
            Gen->>LLM: Refine with Prompt
            LLM-->>Gen: Refined Spec
            Gen-->>Val: Refined Spec
        else ✏️ Manual Edit
            User->>Val: Edited Spec JSON
        end
        Val->>Val: Re-validate
        Val-->>Orch: Updated Result
        Orch-->>User: Display Result
    end
```

## 4. Human-in-the-Loop (HITL) Support

When the automatic A2A collaborative debugging exhausts its retry limit (default: 3 attempts), the system provides **Human-in-the-Loop** capabilities for manual intervention.

### 4.1 HITL Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **💬 AI-Assisted** | User provides natural language feedback; Generator refines spec accordingly | "Map the product.name field to product_name" |
| **✏️ Manual Edit** | User directly edits the Jolt specification JSON | Precise control over transformation rules |
| **📝 Expected Output Edit** | User modifies the expected output for re-validation | Adjusting test expectations |

### 4.2 HITL Workflow

1. **A2A Exhaustion**: Validator returns after max retries with partial result and error report
2. **User Choice**: Frontend presents HITL interface with validation errors
3. **Intervention**: User provides feedback (AI-assisted) or edits spec (manual)
4. **Re-validation**: System validates the modified spec
5. **Iteration**: Process repeats until success or user accepts result

### 4.3 API Endpoints for HITL

* `POST /refine-with-prompt` - AI-assisted refinement with natural language
* `POST /validate` - Direct validation of manually edited specs

## 5. Observability & Tracing

The system implements comprehensive observability through **LangSmith** integration, enabling end-to-end visibility into agent operations.

### 5.1 Tracing Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       LangSmith Platform                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Agent Traces   │  │   LLM Calls     │  │  Token Usage    │  │
│  └────────▲────────┘  └────────▲────────┘  └────────▲────────┘  │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
    ┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐
    │  OpenInference │     │  LangChain    │     │   LiteLLM     │
    │  Instrumentation│     │  Native       │     │  Instrumentation│
    └───────▲───────┘     └───────▲───────┘     └───────▲───────┘
            │                     │                     │
    ┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐
    │   Generator   │     │   Validator   │     │  Gemini LLM   │
    │   (CrewAI)    │     │  (LangGraph)  │     │    Calls      │
    └───────────────┘     └───────────────┘     └───────────────┘
```

### 5.2 Traced Operations

| Agent | Framework | Instrumentation | Traced Events |
|-------|-----------|-----------------|---------------|
| Generator | CrewAI | OpenInference | Task execution, tool calls, LLM requests |
| Validator | LangGraph | Native LangChain | State transitions, node execution, A2A messages |

### 5.3 Trace Data Captured

**CrewAI Generator Traces:**
- `crewai_generate_jolt_spec` - Initial spec generation
- `crewai_refine_jolt_spec` - A2A-triggered refinement
- `crewai_refine_jolt_spec_with_prompt` - HITL-triggered refinement
- MCP tool calls (file reads)
- LLM request/response pairs

**LangGraph Validator Traces:**
- State graph traversal
- Node execution timing
- A2A protocol messages (ERROR_REPORT, PATCH_PROPOSAL)
- Validation results and diffs

### 5.4 Enabling Tracing

```yaml
# Environment Variables (ConfigMap)
LANGCHAIN_TRACING_V2: "true"
LANGCHAIN_PROJECT: "jolt-platform"
LANGCHAIN_ENDPOINT: "https://api.smith.langchain.com"

# Secret
LANGCHAIN_API_KEY: <your-api-key>
```

## 6. Key Technical Achievements

* **Distributed Isolation:** Each agent runs in its own Kubernetes Pod. This proves that an agent built on `CrewAI` can collaborate seamlessly with an agent built on `LangGraph`, as they only share an API contract, not a runtime environment.

* **Decoupled Refinement:** The Orchestrator does not manage the retry logic; the Validator "owns" the quality assurance process, directly pinging the Generator service to fix issues.

* **Scalability:** Because they are microservices, the Generator and Validator can be scaled independently (e.g., spinning up more Generator pods for heavy workloads) without affecting the rest of the system.

* **Secure Tooling:** The Jolt MCP runs as a standalone service (`jolt-mcp-service`), adhering to the "least privilege" principle. Agents must authenticate via the Orchestrator's token system to access these tools.

* **Human-in-the-Loop:** When automatic debugging fails, users can intervene with natural language guidance or direct edits, bridging the gap between automation and human expertise.

* **End-to-End Observability:** All agent operations are traced to LangSmith, enabling debugging, performance monitoring, and audit trails across the heterogeneous system.

---

**Last Updated**: 2025-12-05  
**Version**: 2.0 (Added HITL & Observability)
