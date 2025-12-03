package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os/exec"
)

// JSON-RPC 2.0 Types

type JSONRPCRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      int         `json:"id"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

type JSONRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *JSONRPCError   `json:"error,omitempty"`
}

type JSONRPCError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

type InitializeParams struct {
	ProtocolVersion string          `json:"protocolVersion"`
	Capabilities    map[string]bool `json:"capabilities"`
	ClientInfo      ClientInfo      `json:"clientInfo"`
}

type ClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type CallToolParams struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

func main() {
	// Build the server first to ensure we have the latest binary
	// Or we can just run "go run cmd/server/main.go mcp"
	cmd := exec.Command("go", "run", "cmd/server/main.go", "mcp")

	stdin, err := cmd.StdinPipe()
	if err != nil {
		log.Fatalf("Failed to create stdin pipe: %v", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatalf("Failed to create stdout pipe: %v", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Fatalf("Failed to create stderr pipe: %v", err)
	}

	// Start the server
	if err := cmd.Start(); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
	defer cmd.Process.Kill()

	// Goroutine to print stderr
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			log.Printf("[Server Log] %s", scanner.Text())
		}
	}()

	reader := bufio.NewReader(stdout)

	// Helper to send request
	sendRequest := func(req JSONRPCRequest) {
		bytes, err := json.Marshal(req)
		if err != nil {
			log.Fatalf("Failed to marshal request: %v", err)
		}
		fmt.Fprintf(stdin, "%s\n", string(bytes))
		log.Printf("Sent: %s", string(bytes))
	}

	// Helper to read response
	readResponse := func() JSONRPCResponse {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				log.Fatal("Server closed connection")
			}
			log.Fatalf("Failed to read response: %v", err)
		}
		log.Printf("Received: %s", line)

		var resp JSONRPCResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			log.Fatalf("Failed to unmarshal response: %v", err)
		}
		return resp
	}

	// 1. Initialize
	initReq := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "initialize",
		Params: InitializeParams{
			ProtocolVersion: "2024-11-05",
			Capabilities:    map[string]bool{},
			ClientInfo: ClientInfo{
				Name:    "test-client",
				Version: "1.0.0",
			},
		},
	}
	sendRequest(initReq)
	initResp := readResponse()
	if initResp.Error != nil {
		log.Fatalf("Initialize failed: %s", initResp.Error.Message)
	}
	fmt.Println("Initialization successful!")

	// 2. List Tools
	listToolsReq := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/list",
		Params:  map[string]interface{}{},
	}
	sendRequest(listToolsReq)
	listToolsResp := readResponse()
	if listToolsResp.Error != nil {
		log.Fatalf("List tools failed: %s", listToolsResp.Error.Message)
	}
	fmt.Printf("Tools: %s\n", string(listToolsResp.Result))

	// 3. Call transform
	inputJSON := `{"category_uid":2,"class_uid":2004,"class_name":"Detection Finding","metadata":{"product":{"name":"Acme Security Defender"}}}`
	joltSpec := `[{"operation":"shift","spec":{"category_uid":"events[0].category_id","class_uid":"events[0].baseeventid","class_name":"events[0].class_name","metadata":{"product":{"name":"events[0].product_name"}}}}]`

	callToolReq := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      3,
		Method:  "tools/call",
		Params: CallToolParams{
			Name: "transform",
			Arguments: map[string]interface{}{
				"input_json": inputJSON,
				"jolt_spec":  joltSpec,
			},
		},
	}
	sendRequest(callToolReq)
	callToolResp := readResponse()
	if callToolResp.Error != nil {
		log.Fatalf("Call tool failed: %s", callToolResp.Error.Message)
	}

	// Parse the result content
	var toolResult struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(callToolResp.Result, &toolResult); err != nil {
		log.Fatalf("Failed to parse tool result: %v", err)
	}

	fmt.Println("\nTransformation Result:")
	for _, content := range toolResult.Content {
		if content.Type == "text" {
			fmt.Println(content.Text)
		}
	}
}
