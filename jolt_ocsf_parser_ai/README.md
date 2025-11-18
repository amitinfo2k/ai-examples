# OCSF to JOLT Specification Generator

This tool converts OCSF (Open Cybersecurity Schema Framework) event logs into JOLT (JSON to JSON Transformation Language) specifications using a local Ollama model. The generated JOLT specifications can transform OCSF-formatted logs into your desired JSON output format.

## Features

- Converts OCSF event logs to JOLT specifications
- Uses local Ollama models for processing (default: llama3)
- Supports custom field mappings
- Handles nested JSON structures
- Generates human-readable JOLT specifications

## Prerequisites

- Python >=3.10 <3.14
- For Ollama (default):
  - [Ollama](https://ollama.ai/) installed and running locally
  - Ollama model (default: llama3)
- For Google's Gemini:
  - Google API key with access to Gemini Pro
  - `google-generativeai` package installed

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/jolt_ocsf_parser_ai.git
   cd jolt_ocsf_parser_ai
   ```

2. **Install the package**:
   ```bash
   make install
   ```

3. **For development**:
   ```bash
   make dev-install
   ```

4. **For Gemini support**:
   ```bash
   pip install google-generativeai
   ```

## Configuration

### Agent Configuration (`config/agents.yaml`)

Customize the AI agent that generates JOLT specifications:

```yaml
ocsf_jolt_parser:
  role: Senior OCSF to JOLT Specification Expert
  goal: Convert OCSF event logs to JOLT specifications using a local Ollama model
  backstory: |
    You are an expert in OCSF event logs and JOLT specifications.
    You specialize in analyzing OCSF event logs and generating accurate JOLT
    transformation specifications that convert these logs into desired JSON output formats.
  
  # Model configuration
  # For Ollama:
  # llm:
  #   model: ollama/llama3  # or any other suitable model you have available locally
  #   base_url: http://localhost:11434
  #   temperature: 0.1
  #   max_tokens: 4000
  
  # For Gemini:
  llm:
    model: gemini-2.5-pro  # or any other supported Gemini model
    provider: google
    api_key: ${GOOGLE_API_KEY}  # Set this environment variable
    temperature: 0.1
    max_tokens: 4000
```

### Task Configuration (`config/tasks.yaml`)

Configure the JOLT specification generation task:

```yaml
generate_jolt_spec:
  description: |
    Analyze the provided OCSF event log file and the expected output JSON template.
    Generate a JOLT specification that can transform the OCSF log into the desired JSON format.
  expected_output: |
    A complete JOLT specification that can transform the input OCSF log to the desired output format.
    The output will be a valid JSON object containing the JOLT spec.
  agent: ocsf_jolt_parser
  output_file: jolt_spec.json
```

## Usage

### Basic Usage

**Using Ollama (default):**
```bash
python -m jolt_ocsf_parser_ai input_ocsf.json output_template.json
```

**Using Google's Gemini:**
```bash
GOOGLE_API_KEY='your-api-key' python -m jolt_ocsf_parser_ai --model gemini input_ocsf.json output_template.json
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m`, `--mappings` | Path to JSON file with field mappings | None |
| `-o`, `--output` | Output file path for JOLT spec | `jolt_spec.json` |
| `--model` | LLM to use (`ollama` or `gemini`) | `ollama` |
| `--debug` | Enable debug logging | False |

### Examples

**Basic conversion**:
```bash
python -m jolt_ocsf_parser_ai logs/security_events.json templates/standard_output.json
```

**With custom mappings and output file**:
```bash
python -m jolt_ocsf_parser_ai \
  --mappings config/field_mappings.json \
  --output output/my_jolt_spec.json \
  logs/events.json \
  templates/custom_output.json
```

## Testing

To test the converter with sample data:

```bash
python -m jolt_ocsf_parser_ai test
```

This will create sample input files, run the conversion, and save the JOLT specification to `sample_jolt_spec.json`.

## Input Format

### OCSF Input File

The OCSF input file should be a JSON file containing OCSF-formatted event logs. Example:

```json
{
  "activity_id": 1,
  "activity_name": "Process Start",
  "category_uid": 1,
  "class_uid": 1,
  "cloud": {
    "provider": "AWS",
    "region": "us-west-2"
  },
  "metadata": {
    "product": {
      "name": "Test Product"
    },
    "version": "1.0.0"
  },
  "severity": "Low",
  "severity_id": 1,
  "time": "2023-01-01T00:00:00Z",
  "type_uid": 1
}
```

### Output Template

The output template defines the desired structure of the transformed JSON. Example:

```json
{
  "event": {
    "id": "",
    "name": "",
    "severity": "",
    "timestamp": "",
    "cloud": {
      "provider": "",
      "region": ""
    }
  },
  "metadata": {
    "product": "",
    "version": ""
  }
}
```

### Field Mappings (Optional)

You can provide a JSON file with field mappings to guide the conversion:

```json
{
  "activity_id": "event.id",
  "activity_name": "event.name",
  "severity": "event.severity",
  "time": "event.timestamp",
  "cloud.provider": "event.cloud.provider",
  "cloud.region": "event.cloud.region",
  "metadata.product.name": "metadata.product",
  "metadata.version": "metadata.version"
}
```

## Output

The tool generates a JOLT specification that can be used with the JOLT CLI or library to transform OCSF logs into the desired format.

## Customization

### Using a Different Ollama Model

To use a different Ollama model, modify the `model` parameter in `config/agents.yaml`:

```yaml
llm:
  model: ollama/your-model-name  # e.g., mistral, codellama, etc.
  base_url: http://localhost:11434
  temperature: 0.1
  max_tokens: 4000
```

### Adjusting Task Parameters

You can modify the task parameters in `config/tasks.yaml` to adjust how the JOLT specification is generated.

## Troubleshooting

- **Ollama not running**: Ensure the Ollama server is running with `ollama serve`
- **Model not found**: Make sure you've pulled the model with `ollama pull <model-name>`
- **JSON parsing errors**: Verify that your input files contain valid JSON

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
