# Quick curl commands for testing MCP JOLT Server

## Basic Test (one-liner)
curl -X POST http://localhost:8081/api/v1/transform/jolt -H "Content-Type: application/json" -d '{"input_json": {"category_uid": 2, "class_uid": 2004, "class_name": "Detection Finding", "metadata": {"product": {"name": "Acme Security Defender"}}}, "jolt_spec": [{"operation": "shift", "spec": {"category_uid": "events[0].category_id", "class_uid": "events[0].baseeventid", "class_name": "events[0].class_name", "metadata": {"product": {"name": "events[0].product_name"}}}}]}'

## With jq for pretty output
curl -X POST http://localhost:8081/api/v1/transform/jolt -H "Content-Type: application/json" -d '{"input_json": {"category_uid": 2, "class_uid": 2004, "class_name": "Detection Finding", "metadata": {"product": {"name": "Acme Security Defender"}}}, "jolt_spec": [{"operation": "shift", "spec": {"category_uid": "events[0].category_id", "class_uid": "events[0].baseeventid", "class_name": "events[0].class_name", "metadata": {"product": {"name": "events[0].product_name"}}}}]}' | jq '.'

## Health check
curl -X POST http://localhost:8081/api/v1/transform/jolt -H "Content-Type: application/json" -d '{"input_json": {}, "jolt_spec": [{"operation": "shift", "spec": {}}]}' && echo -e "\n✅ Server is running!" || echo "❌ Server not responding"
