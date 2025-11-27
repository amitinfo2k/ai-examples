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
	serverCmd := flag.NewFlagSet("server", flag.ExitOnError)
	serverPort := serverCmd.String("port", "8081", "Port to run the server on")

	transformCmd := flag.NewFlagSet("transform", flag.ExitOnError)
	inputFile := transformCmd.String("input-file", "", "Path to input JSON file")
	specFile := transformCmd.String("spec-file", "", "Path to JOLT spec file")
	outputFile := transformCmd.String("output-file", "", "Path to output file (optional, prints to stdout if not provided)")

	if len(os.Args) < 2 {
		fmt.Println("expected 'server' or 'transform' subcommands")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "server":
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
		fmt.Println("expected 'server' or 'transform' subcommands")
		os.Exit(1)
	}
}

func runServer(port string) {
	http.HandleFunc("/transform", handleTransformRequest)
	http.HandleFunc("/health", handleHealthCheck)
	log.Printf("Server starting on port %s...\n", port)
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
