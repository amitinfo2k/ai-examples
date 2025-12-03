package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log" // This import is necessary for log.Printf and log.Fatal
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/buger/jsonparser"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/tidwall/sjson"
)

type TransformRequest struct {
	InputJSON interface{} `json:"input_json"`
	JOLTSpec  interface{} `json:"jolt_spec"`
}

type TransformResponse struct {
	Success bool        `json:"success"` // true if transformation was successful
	Result  interface{} `json:"result"`  // transformed result
	Error   string      `json:"error"`   // error message if transformation failed
}

func main() {
	mcpCmd := flag.NewFlagSet("mcp", flag.ExitOnError)

	mcpSSECmd := flag.NewFlagSet("mcp-sse", flag.ExitOnError)
	mcpSSEPort := mcpSSECmd.String("port", "8081", "Port to run the MCP SSE server on")
	mcpSSEBaseURL := mcpSSECmd.String("base-url", "", "Base URL for SSE server (e.g., http://localhost:8081)")

	serverCmd := flag.NewFlagSet("server", flag.ExitOnError)
	serverPort := serverCmd.String("port", "8081", "Port to run the server on")

	transformCmd := flag.NewFlagSet("transform", flag.ExitOnError)
	inputFile := transformCmd.String("input-file", "", "Path to input JSON file")
	specFile := transformCmd.String("spec-file", "", "Path to JOLT spec file")
	outputFile := transformCmd.String("output-file", "", "Path to output file (optional, prints to stdout if not provided)")

	if len(os.Args) < 2 {
		fmt.Println("expected 'mcp', 'mcp-sse', 'server', or 'transform' subcommands")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "mcp":
		log.Println("Starting in MCP Server mode")
		mcpCmd.Parse(os.Args[2:])
		runMCPServer()
	case "mcp-sse":
		log.Println("Starting in MCP SSE Server mode")
		mcpSSECmd.Parse(os.Args[2:])
		runMCPSSEServer(*mcpSSEPort, *mcpSSEBaseURL)
	case "server":
		log.Println("Starting in HTTP Server mode")
		serverCmd.Parse(os.Args[2:])
		runServer(*serverPort)
	case "transform":
		transformCmd.Parse(os.Args[2:])
		if *inputFile == "" || *specFile == "" {
			transformCmd.Usage()
			os.Exit(1)
		}
		transformFromFiles(*inputFile, *specFile, *outputFile)
	default:
		fmt.Println("expected 'mcp', 'mcp-sse', 'server', or 'transform' subcommands")
		os.Exit(1)
	}
}

func runMCPServer() {
	// Create MCP server
	s := server.NewMCPServer(
		"jolt-transformer",
		"1.0.0",
	)

	// Register the transform_json tool
	tool := mcp.NewTool("transform",
		mcp.WithDescription("Transform JSON data using JOLT (JSON to JSON transformation) specification"),
		mcp.WithString("input_json",
			mcp.Required(),
			mcp.Description("The input JSON data to be transformed (as a JSON string)"),
		),
		mcp.WithString("jolt_spec",
			mcp.Required(),
			mcp.Description("The JOLT specification defining the transformation rules (as a JSON array string)"),
		),
	)

	// Set the tool handler
	s.AddTool(tool, handleTransformTool)

	// Start the server with stdio transport
	log.Println("Starting MCP server with stdio transport...")
	if err := server.ServeStdio(s); err != nil {
		log.Fatal(err)
	}
}

func runMCPSSEServer(port, baseURL string) {
	// Create MCP server
	s := server.NewMCPServer(
		"jolt-transformer",
		"1.0.0",
	)

	// Register the transform tool
	tool := mcp.NewTool("transform",
		mcp.WithDescription("Transform JSON data using JOLT (JSON to JSON transformation) specification"),
		mcp.WithString("input_json",
			mcp.Required(),
			mcp.Description("The input JSON data to be transformed (as a JSON string)"),
		),
		mcp.WithString("jolt_spec",
			mcp.Required(),
			mcp.Description("The JOLT specification defining the transformation rules (as a JSON array string)"),
		),
	)

	// Set the tool handler
	s.AddTool(tool, handleTransformTool)

	// Determine base URL
	if baseURL == "" {
		// Use SERVICE_NAME environment variable if set, otherwise use the service name that's used in k8s
		host := os.Getenv("SERVICE_NAME")
		if host == "" {
			host = "jolt-mcp-service" // Default service name in k8s
		}
		baseURL = fmt.Sprintf("http://%s:%s", host, port)
		log.Printf("Using service URL: %s\n", baseURL)
	}

	// Create SSE server
	sseServer := server.NewSSEServer(s, baseURL)

	// Start the server
	log.Printf("Starting MCP server with SSE transport on :%s...\n", port)
	log.Printf("SSE endpoint: %s/sse\n", baseURL)
	log.Printf("Message endpoint: %s/message\n", baseURL)
	log.Println("Note: Use TCP socket probes for health checks in Kubernetes")

	// Bind to all network interfaces
	if err := sseServer.Start("0.0.0.0:" + port); err != nil {
		log.Fatal(err)
	}
}

func handleTransformTool(arguments map[string]interface{}) (*mcp.CallToolResult, error) {
	log.Printf("Received transform tool request with arguments: %+v\n", arguments)
	// Extract arguments
	inputJSONStr, ok := arguments["input_json"].(string)
	if !ok {
		return mcp.NewToolResultError("input_json must be a JSON string"), nil
	}

	joltSpecStr, ok := arguments["jolt_spec"].(string)
	if !ok {
		return mcp.NewToolResultError("jolt_spec must be a JSON array string"), nil
	}

	// Parse the JSON strings
	var inputData interface{}
	if err := json.Unmarshal([]byte(inputJSONStr), &inputData); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid input_json: %v", err)), nil
	}

	var specData interface{}
	if err := json.Unmarshal([]byte(joltSpecStr), &specData); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid jolt_spec: %v", err)), nil
	}

	// Perform the transformation
	result, err := transform(inputData, specData)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("transformation error: %v", err)), nil
	}

	// Convert result to JSON string
	resultJSON, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("error formatting result: %v", err)), nil
	}

	// Return the result
	return mcp.NewToolResultText(string(resultJSON)), nil
}

func runServer(port string) {
	http.HandleFunc("/transform", handleTransformRequest)
	http.HandleFunc("/health", handleHealthCheck)
	log.Printf("HTTP Server starting on port %s...\n", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func handleHealthCheck(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func handleTransformRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Read body to print it
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		resp := TransformResponse{
			Success: false,
			Error:   "Error reading request body",
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(resp)
		return
	}
	// Restore body for decoder
	r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

	log.Printf("Received request: %s\n", string(bodyBytes))

	var req TransformRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		resp := TransformResponse{
			Success: false,
			Error:   "Invalid request body",
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(resp)
		return
	}

	result, err := transform(req.InputJSON, req.JOLTSpec)
	if err != nil {
		log.Printf("Transformation error: %v\n", err)
		resp := TransformResponse{
			Success: false,
			Error:   err.Error(),
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(resp)
		return
	}

	// Create response object
	resp := TransformResponse{
		Success: true,
		Result:  result,
	}

	// Print result
	resultBytes, _ := json.MarshalIndent(resp, "", "  ")
	log.Printf("Sending response: %s\n", string(resultBytes))

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func transformFromFiles(inputFile, specFile, outputFile string) {
	log.Printf("Starting transformation from files: input=%s, spec=%s, output=%s\n", inputFile, specFile, outputFile)
	inputJSON, err := os.ReadFile(inputFile)
	if err != nil {
		log.Fatalf("Error reading input file: %v", err)
	}

	specJSON, err := os.ReadFile(specFile)
	if err != nil {
		log.Fatalf("Error reading spec file: %v", err)
	}

	var inputData, specData interface{}
	if err := json.Unmarshal(inputJSON, &inputData); err != nil {
		log.Fatalf("Error parsing input JSON: %v", err)
	}
	if err := json.Unmarshal(specJSON, &specData); err != nil {
		log.Fatalf("Error parsing JOLT spec: %v", err)
	}

	result, err := transform(inputData, specData)
	if err != nil {
		log.Fatalf("Error transforming data: %v", err)
	}

	output, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		log.Fatalf("Error formatting output: %v", err)
	}

	if outputFile != "" {
		if err := os.WriteFile(outputFile, output, 0644); err != nil {
			log.Fatalf("Error writing output file: %v", err)
		}
		fmt.Printf("Transformation successful. Output written to %s\n", outputFile)
	} else {
		fmt.Println(string(output))
	}
}

func transform(input, spec interface{}) (interface{}, error) {
	log.Println("Executing core transformation logic")
	// Convert input to JSON bytes
	inputBytes, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("error marshaling input: %v", err)
	}

	// Convert spec to JSON bytes
	specBytes, err := json.Marshal(spec)
	if err != nil {
		return nil, fmt.Errorf("error marshaling spec: %v", err)
	}

	// Parse the JOLT spec
	var specArray []map[string]interface{}
	if err := json.Unmarshal(specBytes, &specArray); err != nil {
		return nil, fmt.Errorf("invalid JOLT spec format: %v", err)
	}

	// Apply each operation in the spec
	result := inputBytes
	for _, op := range specArray {
		operation, ok := op["operation"].(string)
		if !ok {
			return nil, fmt.Errorf("missing or invalid 'operation' in spec")
		}

		switch operation {
		case "shift":
			spec, ok := op["spec"].(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("missing or invalid 'spec' in shift operation")
			}
			result, err = applyShift(result, spec)
			if err != nil {
				return nil, fmt.Errorf("error in shift operation: %v", err)
			}
		case "default":
			spec, ok := op["spec"].(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("missing or invalid 'spec' in default operation")
			}
			result, err = applyDefault(result, spec)
			if err != nil {
				return nil, fmt.Errorf("error in default operation: %v", err)
			}
		// Add support for more JOLT operations here
		default:
			return nil, fmt.Errorf("unsupported operation: %s", operation)
		}
	}

	var transformed interface{}
	if err := json.Unmarshal(result, &transformed); err != nil {
		return nil, fmt.Errorf("error unmarshaling result: %v", err)
	}

	return transformed, nil
}

func applyShift(input []byte, spec map[string]interface{}) ([]byte, error) {
	var outputJSON = []byte("{}")
	var err error

	// Recursive function to traverse the spec
	var traverse func(currentSpec map[string]interface{}, path []string)
	traverse = func(currentSpec map[string]interface{}, path []string) {
		for k, v := range currentSpec {
			// Create a new slice for the path to avoid sharing underlying arrays in recursion
			currentPath := make([]string, len(path))
			copy(currentPath, path)
			currentPath = append(currentPath, k)

			switch target := v.(type) {
			case string:
				// Leaf node: extract from input
				val, dataType, _, errGet := jsonparser.Get(input, currentPath...)
				if errGet != nil {
					continue // Skip if path not found in input
				}

				// Normalize path for sjson (convert events[0].id -> events.0.id)
				targetPath := strings.ReplaceAll(target, "[", ".")
				targetPath = strings.ReplaceAll(targetPath, "]", "")

				// Handle different data types
				if dataType == jsonparser.Object || dataType == jsonparser.Array {
					// For complex types, set raw bytes to preserve structure
					outputJSON, err = sjson.SetRawBytes(outputJSON, targetPath, val)
				} else {
					// For primitive types, parse and set
					var realVal interface{}
					switch dataType {
					case jsonparser.String:
						realVal, _ = jsonparser.ParseString(val)
					case jsonparser.Number:
						realVal, _ = strconv.ParseFloat(string(val), 64)
					case jsonparser.Boolean:
						realVal, _ = strconv.ParseBool(string(val))
					default:
						realVal = string(val)
					}

					// sjson.Set takes string, so we need to convert
					var strJSON string
					strJSON, err = sjson.Set(string(outputJSON), targetPath, realVal)
					outputJSON = []byte(strJSON)
				}

				if err != nil {
					log.Printf("Error setting value for path %s: %v", targetPath, err)
				}

			case map[string]interface{}:
				// Nested spec: recurse
				traverse(target, currentPath)
			}
		}
	}

	traverse(spec, []string{})
	return outputJSON, nil
}

func applyDefault(input []byte, spec map[string]interface{}) ([]byte, error) {
	// For default operation, we merge the spec into the input
	// Only fields that don't exist in the input will be added
	var inputMap map[string]interface{}
	if err := json.Unmarshal(input, &inputMap); err != nil {
		return nil, fmt.Errorf("error unmarshaling input: %v", err)
	}

	// Recursive function to apply defaults
	var applyDefaults func(target map[string]interface{}, defaults map[string]interface{}, path []string)
	applyDefaults = func(target map[string]interface{}, defaults map[string]interface{}, path []string) {
		for key, defaultVal := range defaults {
			currentPath := append(path, key)

			if existingVal, exists := target[key]; exists {
				// If the value exists and both are maps, recurse
				if existingMap, isMap := existingVal.(map[string]interface{}); isMap {
					if defaultMap, isDefaultMap := defaultVal.(map[string]interface{}); isDefaultMap {
						applyDefaults(existingMap, defaultMap, currentPath)
					}
				}
				// If value exists and is not a map, keep existing value (don't override)
			} else {
				// Value doesn't exist, set the default
				target[key] = defaultVal
			}
		}
	}

	applyDefaults(inputMap, spec, []string{})

	// Convert back to JSON
	result, err := json.Marshal(inputMap)
	if err != nil {
		return nil, fmt.Errorf("error marshaling result: %v", err)
	}

	return result, nil
}
