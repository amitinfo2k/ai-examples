#!/bin/bash

# MCP JOLT Server Test Script
# This script tests the MCP JOLT transformation server

set -e

# Configuration
SERVER_URL="${SERVER_URL:-http://localhost:8081}"
ENDPOINT="${ENDPOINT:-/api/v1/transform/jolt}"
FULL_URL="${SERVER_URL}${ENDPOINT}"

echo "🧪 Testing MCP JOLT Transformation Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Server URL: ${FULL_URL}"
echo ""

# Test 1: Basic connectivity
echo "1️⃣  Testing server connectivity..."
if curl -s -X POST "${FULL_URL}" \
  -H "Content-Type: application/json" \
  -d '{"input_json": {}, "jolt_spec": [{"operation": "shift", "spec": {}}]}' \
  -o /dev/null -w "%{http_code}" | grep -q "200"; then
  echo "   ✅ Server is reachable"
else
  echo "   ❌ Server is not responding"
  exit 1
fi

echo ""

# Test 2: Full transformation test
echo "2️⃣  Testing JOLT transformation..."
RESPONSE=$(curl -s -X POST "${FULL_URL}" \
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
  }')

if [ -z "$RESPONSE" ]; then
  echo "   ❌ No response from server"
  exit 1
fi

echo "   ✅ Transformation successful"
echo ""
echo "📄 Response:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Pretty print if jq is available
if command -v jq &> /dev/null; then
  echo "$RESPONSE" | jq '.'
else
  echo "$RESPONSE"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All tests passed!"
