# Thread Cleanup Usage Guide

## 🧹 GoGain AI Thread Cleanup Utility

This script helps you clean up OpenAI conversation threads to start fresh every time you test the pain assessment system.

## Quick Commands

### List all existing threads
```bash
python3 thread_cleanup.py list
```

### Clean a specific user's thread (recommended for testing)
```bash
python3 thread_cleanup.py clean-user test_user_01
```

### Clean all threads (use with caution)
```bash
python3 thread_cleanup.py clean-all
```

### Clean everything (threads + memory caches)
```bash
python3 thread_cleanup.py clean-everything
```

### Just clean memory caches
```bash
python3 thread_cleanup.py clean-memory
```

## Common Usage Scenarios

### 🔄 Before Testing Pain Assessment
```bash
# Clean your test user's conversation history
python3 thread_cleanup.py clean-user test_user_01

# Now run your pain assessment tests
python3 test_pain_conversations.py
```

### 🧪 Clean Testing Environment
```bash
# List what's there
python3 thread_cleanup.py list

# Clean specific test users
python3 thread_cleanup.py clean-user shoulder_test_user
python3 thread_cleanup.py clean-user knee_test_user

# Or clean everything for a fresh start
python3 thread_cleanup.py clean-everything
```

### 📊 Before Running Conversation Flow Tests
```bash
# Clean the demo users
python3 thread_cleanup.py clean-user demo_user_shoulder
python3 thread_cleanup.py clean-user demo_user_knee
python3 thread_cleanup.py clean-user demo_user_back

# Run the conversation flow demonstration
python3 show_conversation_flow.py
```

## What Gets Cleaned

The script cleans up:

✅ **OpenAI Thread History** (if API key is available)
- Conversation messages stored in OpenAI threads
- Thread metadata and settings

✅ **Local Thread Files**
- `thread_history_*.json` files
- Files in `thread_sessions/` directory
- Files in `sessions/`, `threads/`, `.threads/` directories

✅ **Memory Caches**
- In-memory workflow caches from `app/api/chat.py`
- Any temporary conversation state

✅ **Temporary Files** (with clean-everything)
- `lastconv.txt`
- `*.log` files
- `temp_*.json` files

## Safety Features

⚠️ **Confirmation Required**: All destructive operations require explicit confirmation

🔒 **Graceful Handling**: Works even without OpenAI API key (file-based cleanup only)

📝 **Verbose Output**: Shows exactly what's being cleaned

## Examples

### Clean before pain assessment testing:
```bash
$ python3 thread_cleanup.py clean-user test_user_01
⚠️  WARNING: This will delete conversation history for user 'test_user_01'!
Are you sure? Type 'yes' to confirm: yes
🧹 Cleaning threads for user: test_user_01
🧹 Cleaning thread: test_user_01
🗑️  Deleted file: thread_history_test_user_01.json
✅ Successfully cleaned thread for user: test_user_01
```

### List all threads:
```bash
$ python3 thread_cleanup.py list
🔍 Found 3 threads:
  1. test_user_01
  2. shoulder_test_user
  3. knee_test_user
```

## Integration with Testing

Add to your testing workflow:

```bash
#!/bin/bash
# testing_workflow.sh

echo "🧹 Cleaning test environment..."
python3 thread_cleanup.py clean-user test_user_01

echo "🧪 Running pain assessment tests..."
python3 test_flexible_ai_assessment.py

echo "🗣️ Testing conversation flows..."
python3 show_conversation_flow.py

echo "✅ Testing complete!"
```

This ensures every test starts with a clean slate, preventing conversation history from affecting your pain assessment infinite loop fixes!