# Workflow Progress Tracking - Quick Start Guide

## Problem Statement

You reported that workflow progress is not visible in the UI, and workflows sometimes timeout without showing any logs. This makes it impossible to:
- See what stage the workflow is in
- Debug issues when workflows fail
- Know if the workflow is making progress or stuck

## Solution Implemented

I've implemented a **real-time progress tracking system** that:
1. ✅ Shows live progress percentage (0-100%)
2. ✅ Displays current workflow step
3. ✅ Shows real-time logs as they happen
4. ✅ Handles long-running workflows (up to 5 minutes)
5. ✅ Provides detailed error messages
6. ✅ Prevents timeout issues

## What Changed

### 1. New Task Store
**File**: `orchestrator/models/task_store.py`

A thread-safe in-memory store that tracks:
- Current workflow status (pending, generating, validating, etc.)
- Progress percentage
- Live logs
- Current step description
- Results and errors

### 2. Updated Orchestrator
**File**: `orchestrator/main.py`

**Changes**:
- Workflows now run in **background** (non-blocking)
- POST `/workflow/generate-and-validate` returns immediately with a task ID
- New GET `/workflow/status/{task_id}` endpoint for progress polling
- Each workflow step updates the task store with progress

**Workflow Stages**:
```
Stage 1: Generation (10% → 30%)
  ├─ Initialize generator
  └─ Generate JOLT spec

Stage 2: File Reading (30% → 40%)
  ├─ Read input file
  └─ Read output file

Stage 3: Validation Loop (40% → 100%)
  ├─ Attempt 1 (40% → 60%)
  ├─ Attempt 2 (60% → 80%)
  └─ Attempt 3 (80% → 100%)
```

### 3. Updated Frontend UI
**File**: `frontend/pages/2_🚀_Workflow.py`

**Changes**:
- Starts workflow and gets task ID
- Polls for progress every 1 second
- Shows:
  - Progress bar (visual percentage)
  - Current step text ("Generating Jolt Specification...")
  - Live logs (last 10 lines)
- Stops polling when workflow completes
- Shows final status (success, failed, or needs review)

### 4. Verbose Agent Logging
**File**: `agents/generator/crew_agent.py`

Enabled `verbose=True` to show detailed agent activity in logs.

## How to Test

### Option 1: Run Normally

1. **Start the orchestrator** (if not already running):
   ```bash
   cd /home/amit.wankhede@GSLAB.COM/Work/PoC/GenAI/ai-examples/a2a-examples/jolt_platform_agentic_ui
   python -m uvicorn orchestrator.main:app --host 0.0.0.0 --port 8088 --reload
   ```

2. **Start the frontend** (in a new terminal):
   ```bash
   cd /home/amit.wankhede@GSLAB.COM/Work/PoC/GenAI/ai-examples/a2a-examples/jolt_platform_agentic_ui
   streamlit run frontend/pages/2_🚀_Workflow.py
   ```

3. **Run a workflow**:
   - Open the Streamlit UI in your browser
   - Navigate to "🚀 Workflow" page
   - Click "▶️ Run Complete Workflow"

4. **Observe the progress**:
   - Watch the progress bar move from 0% to 100%
   - See the status text update ("Generating...", "Validating...", etc.)
   - View live logs appearing in the text area
   - See the final result when complete

### Option 2: Test API Directly

```bash
# Start workflow
TASK_RESPONSE=$(curl -X POST http://localhost:8088/workflow/generate-and-validate \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "generate_spec",
    "input_file_path": "input.json",
    "output_file_path": "output.json",
    "auth_mapping": {
      "user_id": "test",
      "gdrive_token": "valid_token",
      "target_directory_id": "mock"
    }
  }')

echo "Response: $TASK_RESPONSE"

# Extract task_id (you'll see it in the response)
# Then poll for status:
TASK_ID="<paste-task-id-here>"

# Check status multiple times
curl http://localhost:8088/workflow/status/$TASK_ID | jq
sleep 2
curl http://localhost:8088/workflow/status/$TASK_ID | jq
sleep 2
curl http://localhost:8088/workflow/status/$TASK_ID | jq
```

## What You'll See

### In the UI

**Before (Old Behavior)**:
```
🤖 Running workflow...
[Static spinner, no feedback]
[After 120 seconds: ⏱️ Timeout!]
```

**After (New Behavior)**:
```
📍 Generating Jolt Specification (30%)
[Progress bar: ████████░░░░░░░░░░░░░░░░]

Recent Logs:
Starting Generation Phase...
Generated Jolt Spec with 2 operation(s)
Reading files for validation...
Starting Validation Phase...
--- Validation Attempt 1/4 ---
Validation Result: ❌ FAILED
❌ Validation Failed. Found 3 error(s). Triggering Refinement...
Generated refined Jolt Spec
--- Validation Attempt 2/4 ---
Validation Result: ✅ PASSED
✅ Validation Successful!
```

### In the Logs (Terminal)

With verbose mode enabled, you'll see detailed agent logs:
```
INFO:     127.0.0.1:52148 - "POST /workflow/generate-and-validate HTTP/1.1" 200 OK
[2025-12-02 19:36:45][INFO]: Task started: abc-123
[2025-12-02 19:36:46][DEBUG]: Agent working on task...
[2025-12-02 19:36:47][DEBUG]: Generated Jolt spec: [{"operation": "shift", ...}]
[2025-12-02 19:36:48][INFO]: Validation attempt 1...
```

## Configuration Options

### Change Timeout Duration

**File**: `frontend/pages/2_🚀_Workflow.py`  
**Line**: ~87
```python
max_polls = 300  # Change this value (seconds)
```

Default: 5 minutes (300 seconds)

### Change Polling Interval

**File**: `frontend/pages/2_🚀_Workflow.py`  
**Line**: ~93
```python
time.sleep(1)  # Change this value (seconds)
```

Default: 1 second

### Change Log Display Count

**File**: `frontend/pages/2_🚀_Workflow.py`  
**Line**: ~110
```python
recent_logs = logs[-10:]  # Change number of logs shown
```

Default: Last 10 log entries

## Troubleshooting

### Problem: "No progress showing"

**Check**:
1. Is the orchestrator running? Check `http://localhost:8088/health`
2. Are there errors in the orchestrator terminal?
3. Check browser console for JavaScript errors

### Problem: "Workflow timeout after 5 minutes"

**Solutions**:
1. Increase `max_polls` in the frontend
2. Check if Gemini API is responding slowly
3. Review orchestrator logs for stuck steps

### Problem: "Task not found"

**Cause**: Task was cleaned up or orchestrator restarted  
**Solution**: Tasks are stored in-memory. If orchestrator restarts, tasks are lost. For production, implement persistent storage.

## Files Modified

| File | Purpose | Changes |
|------|---------|---------|
| `orchestrator/models/task_store.py` | **NEW** | Task tracking store |
| `orchestrator/main.py` | Updated | Background workflow execution |
| `frontend/pages/2_🚀_Workflow.py` | Updated | Progress polling UI |
| `agents/generator/crew_agent.py` | Updated | Verbose logging enabled |
| `WORKFLOW_PROGRESS.md` | **NEW** | Full documentation |

## Next Steps

1. **Test the changes** (see "How to Test" above)
2. **Review the logs** during workflow execution
3. **Provide feedback** if any issues occur

## Need Help?

If you encounter any issues:
1. Check the orchestrator logs for errors
2. Check the browser console in developer tools
3. Verify the orchestrator is accessible at `http://localhost:8088/health`
4. Review `WORKFLOW_PROGRESS.md` for detailed troubleshooting

---

**Status**: ✅ Ready to test  
**Breaking Changes**: None (backward compatible)  
**Dependencies**: No new packages required
