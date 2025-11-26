"""
LangChain Agent for JOLT Specification Validation
This agent is responsible for validating JOLT transformations and generating
detailed validation reports.
Uses Google Gemini for LLM capabilities.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from typing import Dict, Any, List
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime


class JoltValidator:
    """LangChain-based agent for validating JOLT specifications using Google Gemini."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the JOLT Validator agent with Google Gemini.
        
        Args:
            model_name: Gemini model to use (defaults to gemini-1.5-pro)
        """
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        self.llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0)
    
    def _execute_jolt_tool(self, input_data: str) -> str:
        """
        Tool to execute JOLT transformation using MCP server.
        
        Args:
            input_data: JSON string containing 'jolt_spec' and 'input_json'
            
        Returns:
            Transformed JSON as string
        """
        try:
            data = json.loads(input_data)
            jolt_spec = data['jolt_spec']
            input_json = data['input_json']
            
            # Call MCP server for transformation
            result = self._call_mcp_jolt_transform(jolt_spec, input_json)
            
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error executing JOLT transformation: {str(e)}"
    
    def _call_mcp_jolt_transform(self, jolt_spec: List[Dict], input_json: Dict) -> Dict:
        """
        Call the JOLT MCP server via HTTP to perform transformation.
        
        Args:
            jolt_spec: JOLT specification
            input_json: Input JSON to transform
            
        Returns:
            Transformed JSON
        """
        import requests
        import os
        import sys
        import json
        
        # Get MCP server URL from environment
        mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8080")
        
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"🌐 Validator: Calling MCP Server at {mcp_server_url}", file=sys.stderr)
        print(f"📋 Validator: JOLT Spec ({len(jolt_spec)} operations):", file=sys.stderr)
        print(json.dumps(jolt_spec, indent=2), file=sys.stderr)
        print(f"📥 Validator: Input JSON:", file=sys.stderr)
        print(json.dumps(input_json, indent=2), file=sys.stderr)
        
        try:
            # Call MCP server via HTTP
            print(f"⏳ Validator: Making POST request to {mcp_server_url}/transform...", file=sys.stderr)
            response = requests.post(
                f"{mcp_server_url}/transform",
                json={
                    "jolt_spec": jolt_spec,
                    "input_json": input_json
                },
                timeout=10
            )
            
            print(f"📡 Validator: Received response with status code {response.status_code}", file=sys.stderr)
            
            if response.status_code != 200:
                error_msg = f"MCP Server returned status {response.status_code}: {response.text}"
                print(f"❌ Validator: {error_msg}", file=sys.stderr)
                raise Exception(error_msg)
            
            result = response.json()
            print(f"📥 Validator: Response JSON: {result}", file=sys.stderr)
            
            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                print(f"❌ Validator: MCP transformation failed: {error_msg}", file=sys.stderr)
                raise Exception(f"MCP transformation failed: {error_msg}")
            
            transformed_result = result.get("result", {})
            print(f"✅ Validator: MCP Server returned transformed result: {transformed_result}", file=sys.stderr)
            print(f"{'=' * 60}\n", file=sys.stderr)
            
            return transformed_result
            
        except requests.exceptions.Timeout:
            error_message = "MCP Server request timed out"
            print(f"⚠️ Validator: {error_message}, using fallback implementation", file=sys.stderr)
            print(f"{'=' * 60}\n", file=sys.stderr)
            return self._apply_jolt_transformation_fallback(jolt_spec, input_json)
        except requests.exceptions.ConnectionError as e:
            error_message = f"Cannot connect to MCP Server at {mcp_server_url}: {str(e)}"
            print(f"⚠️ Validator: {error_message}, using fallback implementation", file=sys.stderr)
            print(f"{'=' * 60}\n", file=sys.stderr)
            return self._apply_jolt_transformation_fallback(jolt_spec, input_json)
        except Exception as e:
            # Fallback to internal implementation if MCP server fails
            print(f"⚠️ Validator: MCP server failed ({e}), using fallback implementation", file=sys.stderr)
            print(f"{'=' * 60}\n", file=sys.stderr)
            return self._apply_jolt_transformation_fallback(jolt_spec, input_json)
    
    def _apply_jolt_transformation_fallback(self, jolt_spec: List[Dict], input_json: Dict) -> Dict:
        """
        Apply JOLT transformation to input JSON.
        This is a simplified implementation. In production, use the actual JOLT library.
        
        Args:
            jolt_spec: JOLT specification
            input_json: Input JSON to transform
            
        Returns:
            Transformed JSON
        """
        # This is a placeholder - in production, use actual JOLT library
        # For now, we'll return a basic transformation
        result = {}
        
        for operation in jolt_spec:
            op_type = operation.get('operation', '')
            spec = operation.get('spec', {})
            
            if op_type == 'shift':
                result = self._apply_shift(spec, input_json)
            elif op_type == 'default':
                result = self._apply_default(spec, result)
        
        return result
    
    def _apply_shift(self, spec: Dict, input_json: Dict, current_path: str = "") -> Dict:
        """Apply shift operation."""
        result = {}
        
        for key, value in spec.items():
            if isinstance(value, dict):
                # Nested transformation
                if key in input_json and isinstance(input_json[key], dict):
                    nested_result = self._apply_shift(value, input_json[key], f"{current_path}.{key}")
                    result.update(nested_result)
            elif isinstance(value, str):
                # Simple mapping
                if key in input_json:
                    # Parse the target path
                    target_parts = value.split('.')
                    self._set_nested_value(result, target_parts, input_json[key])
        
        return result
    
    def _apply_default(self, spec: Dict, data: Dict) -> Dict:
        """Apply default operation."""
        result = data.copy()
        
        for key, value in spec.items():
            if key not in result:
                result[key] = value
        
        return result
    
    def _set_nested_value(self, data: Dict, path: List[str], value: Any):
        """Set a nested value in a dictionary, handling array indices."""
        current = data
        
        for i, key in enumerate(path[:-1]):
            # Check for array index in key (e.g., "events[0]")
            if '[' in key and key.endswith(']'):
                base_key, index_str = key[:-1].split('[')
                index = int(index_str)
                
                if base_key not in current:
                    current[base_key] = []
                
                # Ensure it's a list
                if not isinstance(current[base_key], list):
                    current[base_key] = [current[base_key]] if current[base_key] else []

                # Extend array if needed
                while len(current[base_key]) <= index:
                    current[base_key].append({})
                
                current = current[base_key][index]
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]
        
        # Handle the last key
        last_key = path[-1]
        if '[' in last_key and last_key.endswith(']'):
            base_key, index_str = last_key[:-1].split('[')
            index = int(index_str)
            
            if base_key not in current:
                current[base_key] = []
            
            # Ensure it's a list
            if not isinstance(current[base_key], list):
                current[base_key] = [current[base_key]] if current[base_key] else []

            while len(current[base_key]) <= index:
                current[base_key].append(None)
            
            current[base_key][index] = value
        else:
            current[last_key] = value
    
    def _compare_json_tool(self, input_data: str) -> str:
        """
        Tool to compare two JSON objects and identify differences.
        
        Args:
            input_data: JSON string containing 'actual' and 'expected'
            
        Returns:
            Comparison report as string
        """
        try:
            data = json.loads(input_data)
            actual = data['actual']
            expected = data['expected']
            
            differences = self._find_differences(actual, expected)
            
            if not differences:
                return "No differences found. Validation successful!"
            else:
                return json.dumps({
                    "status": "differences_found",
                    "differences": differences
                }, indent=2)
        except Exception as e:
            return f"Error comparing JSON: {str(e)}"
    
    def _find_differences(self, actual: Any, expected: Any, path: str = "") -> List[Dict]:
        """
        Recursively find differences between actual and expected values.
        
        Args:
            actual: Actual value
            expected: Expected value
            path: Current path in the JSON structure
            
        Returns:
            List of differences
        """
        differences = []
        
        if type(actual) != type(expected):
            differences.append({
                "path": path or "root",
                "issue": "Type mismatch",
                "actual_type": str(type(actual).__name__),
                "expected_type": str(type(expected).__name__),
                "actual": actual,
                "expected": expected
            })
            return differences
        
        if isinstance(actual, dict):
            # Check for missing keys
            all_keys = set(actual.keys()) | set(expected.keys())
            
            for key in all_keys:
                current_path = f"{path}.{key}" if path else key
                
                if key not in actual:
                    differences.append({
                        "path": current_path,
                        "issue": "Missing in actual",
                        "expected": expected[key]
                    })
                elif key not in expected:
                    differences.append({
                        "path": current_path,
                        "issue": "Extra in actual",
                        "actual": actual[key]
                    })
                else:
                    differences.extend(
                        self._find_differences(actual[key], expected[key], current_path)
                    )
        
        elif isinstance(actual, list):
            if len(actual) != len(expected):
                differences.append({
                    "path": path or "root",
                    "issue": "Array length mismatch",
                    "actual_length": len(actual),
                    "expected_length": len(expected)
                })
            
            for i, (actual_item, expected_item) in enumerate(zip(actual, expected)):
                current_path = f"{path}[{i}]"
                differences.extend(
                    self._find_differences(actual_item, expected_item, current_path)
                )
        
        else:
            # Compare primitive values
            if actual != expected:
                differences.append({
                    "path": path or "root",
                    "issue": "Value mismatch",
                    "actual": actual,
                    "expected": expected
                })
        
        return differences
    
    def validate_jolt_spec(
        self,
        jolt_spec: Dict[str, Any],
        input_json: Dict[str, Any],
        expected_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a JOLT specification against input and expected output.
        
        Args:
            jolt_spec: The JOLT specification to validate
            input_json: Input JSON data
            expected_output: Expected output JSON data
            
        Returns:
            Validation report as a dictionary
        """
        # Step 1: Apply JOLT transformation
        transform_input = json.dumps({
            "jolt_spec": jolt_spec,
            "input_json": input_json
        })
        transformed_result_str = self._execute_jolt_tool(transform_input)
        
        try:
            transformed_result = json.loads(transformed_result_str)
        except:
            transformed_result = transformed_result_str
        
        # Step 2: Compare with expected output
        compare_input = json.dumps({
            "actual": transformed_result,
            "expected": expected_output
        })
        comparison_result = self._compare_json_tool(compare_input)
        
        # Step 3: Use LLM to provide analysis
        prompt = f"""You are a JOLT specification validator. Analyze the following validation results:

JOLT SPECIFICATION:
{json.dumps(jolt_spec, indent=2)}

INPUT JSON:
{json.dumps(input_json, indent=2)}

EXPECTED OUTPUT:
{json.dumps(expected_output, indent=2)}

ACTUAL TRANSFORMATION RESULT:
{json.dumps(transformed_result, indent=2) if isinstance(transformed_result, dict) else transformed_result}

COMPARISON RESULT:
{comparison_result}

Please provide:
1. A summary of whether the validation passed or failed
2. Details of any differences found
3. Recommendations for fixing any issues
"""
        
        # Get LLM analysis
        llm_response = self.llm.invoke(prompt)
        llm_analysis = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        
        # Create a structured report
        report = {
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "jolt_spec": jolt_spec,
            "input": input_json,
            "expected_output": expected_output,
            "actual_output": transformed_result,
            "comparison": comparison_result,
            "validation_result": llm_analysis,
            "validation_passed": "No differences found" in comparison_result
        }
        
        return report
    
    def validate_from_files(
        self,
        jolt_spec_file: str,
        input_file: str,
        expected_output_file: str
    ) -> Dict[str, Any]:
        """
        Validate JOLT specification from files.
        
        Args:
            jolt_spec_file: Path to JOLT specification file
            input_file: Path to input JSON file
            expected_output_file: Path to expected output JSON file
            
        Returns:
            Validation report
        """
        with open(jolt_spec_file, 'r') as f:
            jolt_spec = json.load(f)
        
        with open(input_file, 'r') as f:
            input_json = json.load(f)
        
        with open(expected_output_file, 'r') as f:
            expected_output = json.load(f)
        
        return self.validate_jolt_spec(jolt_spec, input_json, expected_output)
    
    def save_validation_report(self, report: Dict[str, Any], output_file: str):
        """
        Save validation report to a file.
        
        Args:
            report: Validation report
            output_file: Path to save the report
        """
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Validation report saved to: {output_file}")


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    # Sample data
    jolt_spec = [
        {
            "operation": "shift",
            "spec": {
                "user": {
                    "firstName": "fullName",
                    "email": "contact.email"
                },
                "timestamp": "eventtime"
            }
        }
    ]
    
    input_json = {
        "user": {
            "firstName": "John",
            "lastName": "Doe",
            "email": "john.doe@example.com"
        },
        "timestamp": "2024-01-01T12:00:00Z"
    }
    
    expected_output = {
        "fullName": "John Doe",
        "contact": {
            "email": "john.doe@example.com"
        },
        "eventtime": "2024-01-01T12:00:00Z"
    }
    
    # Create validator
    validator = JoltValidator()
    
    # Validate
    report = validator.validate_jolt_spec(jolt_spec, input_json, expected_output)
    
    # Save report
    validator.save_validation_report(report, "validation_report.json")
    
    print("\nValidation Report:")
    print(json.dumps(report, indent=2))
