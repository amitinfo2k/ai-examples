# MCP JOLT Transformation Server (Go Implementation)

This server provides a robust and scalable solution for transforming JSON data based on JOLT specification files. It acts as a Microservice Communication Protocol (MCP) endpoint, taking raw JSON input and a JOLT transformation rule set, and returning the structured, transformed JSON output.

## Features

Declarative Transformation: Uses a Go implementation of the JOLT library for declarative JSON-to-JSON transformations.

Decoupled Logic: Separation of data (input.json), business logic (jolt_spec.json), and execution logic (the server).

Single Binary: Built with Go for easy, self-contained deployment without external runtime dependencies (beyond the OS).

Simple Input/Output: Designed to accept the input JSON and the JOLT specification as primary inputs.

## Getting Started

To get the server up and running, follow these steps:

Prerequisites

Ensure you have the following installed on your system:

Go (Golang) 1.22 or higher

Go CLI (for building and running the executable)

Docker (for containerized deployment)

1. Project Structure

Your server implementation should handle the inputs based on a request payload or file system structure. A typical setup involves four core files:

/mcp-jolt-server/
├── input.json             # The raw JSON data to be transformed.
├── jolt_spec.json         # The JOLT specification defining the transformation rules.
├── Dockerfile             # Defines the steps to build the container image.
└── mcp-jolt-server        # The compiled Go executable binary (pre-built or inside the container).


2. Input Files

input.json (Example)

This is the source data that needs to be restructured.

{
  "customerId": "CUST-4001",
  "data": {
    "firstName": "Alex",
    "lastName": "Johnson",
    "orderHistory": [
      { "id": "ORD100", "date": "2023-11-20", "total": 45.99 },
      { "id": "ORD101", "date": "2023-11-25", "total": 12.50 }
    ]
  }
}


jolt_spec.json (Example)

This JOLT specification defines the transformation logic. This example flattens the customer details and extracts the order IDs.

[
  {
    "operation": "shift",
    "spec": {
      "customerId": "account.id",
      "data": {
        "firstName": "account.contact.first",
        "lastName": "account.contact.last",
        "orderHistory": {
          "*": {
            "id": "orders[&1]"
          }
        }
      }
    }
  }
]


3. Execution

The server should expose an endpoint (e.g., via REST API) or a command-line interface (CLI) to trigger the transformation.

A. CLI Execution (Direct Execution)

Once compiled, you can run the binary directly, passing the input and spec files as arguments:

# Build the executable (run once)
go build -o mcp-jolt-server main.go 

# Run the transformation
./mcp-jolt-server transform \
  --input-file resources/input.json \
  --spec-file resources/jolt_spec.json \
  --output-file transformed_output.json


B. API Endpoint (If deployed as a RESTful service)

If the server is running on http://localhost:8081, you would typically send a POST request with a payload containing both the input data and the specification.

Endpoint: POST /api/v1/transform/jolt

Request Body (JSON):

{
  "input_json": {
    "customerId": "CUST-4001",
    "...": "..."
  },
  "jolt_spec": [
    {
      "operation": "shift",
      "...": "..."
    }
  ]
}


C. Docker Container Execution

Follow these steps to build and run the application as a Docker container.

1. Build the Docker Image
Run this command from the root directory containing your Dockerfile and Go source code:

docker build -t mcp-jolt-server:latest .


2. Run the Container (RESTful API Mode)
If your application runs as an HTTP server by default (using CMD ["server"] in the Dockerfile):

docker run -d -p 8081:8081 --name jolt-api mcp-jolt-server:latest
# The API is now available at http://localhost:8081/api/v1/transform/jolt


3. Run the Container (CLI Transformation Mode)
To run a one-off transformation using the container's CLI mode, you need to mount the input files as volumes and override the default CMD:

# Example: Running the 'transform' command within the container
docker run --rm \
  -v "$(pwd)/input.json:/app/input.json" \
  -v "$(pwd)/jolt_spec.json:/app/jolt_spec.json" \
  -v "$(pwd):/app/output" \
  mcp-jolt-server:latest transform \
  --input-file input.json \
  --spec-file jolt_spec.json \
  --output-file /app/output/transformed_output.json


Note: This command mounts the current directory ($(pwd)) to allow the container to read input files and write the transformed_output.json back to your host machine.

## 🧪 Testing the MCP Server with curl

Once the server is running (either locally or via Docker), you can test the transformation endpoint using curl.

### Basic curl Test

**Using the example files from the `resources/` directory:**

```bash
curl -X POST http://localhost:8081/api/v1/transform/jolt \
  -H "Content-Type: application/json" \
  -d '{
    "input_json": {
      "category_uid": 2,
      "category_name": "Findings",
      "class_uid": 2004,
      "class_name": "Detection Finding",
      "metadata": {
        "version": "1.6.0",
        "product": {
          "name": "Acme Security Defender",
          "vendor_name": "Acme Security",
          "version": "5.2.1"
        },
        "uid": "12345678-abcd-1234-abcd-1234567890ab",
        "event_dt": "2025-11-15T15:00:00.123Z"
      }
    },
    "jolt_spec": [
      {
        "operation": "shift",
        "spec": {
          "category_uid": "events[0].category_id",
          "class_uid": "events[0].baseeventid",
          "class_name": "events[0].class_name",
          "metadata": {
            "product": {
              "name": "events[0].product_name"
            }
          }
        }
      }
    ]
  }'
```

### Using Files with curl

If you prefer to keep your test data in files:

```bash
# Create a test request payload
cat > test_request.json <<'EOF'
{
  "input_json": {
    "category_uid": 2,
    "class_uid": 2004,
    "class_name": "Detection Finding",
    "metadata": {
      "product": {
        "name": "Acme Security Defender"
      }
    }
  },
  "jolt_spec": [
    {
      "operation": "shift",
      "spec": {
        "category_uid": "events[0].category_id",
        "class_uid": "events[0].baseeventid",
        "class_name": "events[0].class_name",
        "metadata": {
          "product": {
            "name": "events[0].product_name"
          }
        }
      }
    }
  ]
}
EOF

# Send the request
curl -X POST http://localhost:8081/api/v1/transform/jolt \
  -H "Content-Type: application/json" \
  -d @test_request.json | jq '.'
```

### Expected Response

```json
{
  "success": true,
  "result": {
    "events": [
      {
        "baseeventid": 2004,
        "category_id": 2,
        "class_name": "Detection Finding",
        "product_name": "Acme Security Defender"
      }
    ]
  },
  "error": ""
}
```

### Health Check (Optional)

To verify the server is running:

```bash
# Check if the server is accepting connections
curl -X POST http://localhost:8081/api/v1/transform/jolt \
  -H "Content-Type: application/json" \
  -d '{"input_json": {}, "jolt_spec": [{"operation": "shift", "spec": {}}]}' \
  && echo -e "\n✅ Server is running!" \
  || echo "❌ Server is not responding"
```

💡 Expected Output

The server will return the transformed JSON based on the rules in jolt_spec.json.

Transformed JSON (transformed_output.json):

{
  "account": {
    "id": "CUST-4001",
    "contact": {
      "first": "Alex",
      "last": "Johnson"
    }
  },
  "orders": [
    "ORD100",
    "ORD101"
  ]
}


🧪 Testing and Validation

Validate JOLT Spec: Always validate your jolt_spec.json using an online JOLT Validator before deploying complex changes.

Unit Tests: Ensure your core Go transformation function is covered by unit tests. Go's built-in testing features makes this easy.

Error Handling: The server must handle errors gracefully, such as:

Invalid JSON format in input.json or jolt_spec.json.

JOLT transformation errors (e.g., invalid spec syntax).

File not found errors.

🤝 Contribution

If this is an open-source project, add details on how others can contribute here (e.g., fork the repository, submit pull requests, etc.).