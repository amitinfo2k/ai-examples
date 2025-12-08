---
description: Verify Agent-to-Agent (A2A) Protocol Compliance
---

This workflow verifies that the Jolt Platform services are compliant with the Google A2A protocol (ADK).

1. **Verify Agent Cards**:
   Check that both Generator and Validator expose their Agent Cards.
   ```bash
   kubectl port-forward svc/jolt-generator-service 8081:80 -n jolt-platform &
   PID_GEN=$!
   kubectl port-forward svc/jolt-validator-service 8080:80 -n jolt-platform &
   PID_VAL=$!
   sleep 5
   
   echo "Checking Generator Agent Card..."
   curl -s http://localhost:8081/.well-known/agent.json | jq .
   
   echo "Checking Validator Agent Card..."
   curl -s http://localhost:8080/.well-known/agent.json | jq .
   
   kill $PID_GEN $PID_VAL
   ```

2. **Run a Test Workflow**:
   Trigger a workflow that requires refinement (e.g., using an input that produces an initial mismatch) and verify the A2A interaction logs in the UI.
   
   - Go to the Streamlit UI (http://localhost:8501).
   - Select "Workflow" page.
   - Run a workflow.
   - Check the "A2A Messages" tab after completion.
   - Ensure the following sequence is present:
     - 🔍 **DISCOVERY** | Validator ➡️ Generator
     - ❌ **ERROR_REPORT** | Validator ➡️ Generator
     - 🛠️ **PATCH_PROPOSAL** | Generator ➡️ Validator
     - ✅ **VERIFICATION_RESULT** | Validator ➡️ Generator (if successful)

3. **Check Logs**:
   Verify logs for ADK Task lifecycle events.
   ```bash
   kubectl logs -l app=jolt-validator -n jolt-platform --tail=100 | grep "ADK"
   kubectl logs -l app=jolt-generator -n jolt-platform --tail=100 | grep "Task"
   ```
