# /Users/mac/Desktop/work/GoGain/GOGAIN-AI-SERVICE/interactive_chat.py
import requests
import json
import argparse
import os
import readline  # Optional: Improves input editing experience (arrows, history)
from datetime import datetime
from typing import List, Dict, Any
import re

# --- Configuration ---
DEFAULT_API_URL = "http://localhost:8000/api/v1/chat"
DEFAULT_USER_ID = "test_user_01"
LOG_FILE = "chat_conversations.log"
REQUEST_TIMEOUT = 60  # seconds - increased for agent workflows

# Stream-friendly default headers
STREAM_HEADERS = {
    "Accept": "text/event-stream",
    "Accept-Encoding": "identity",  # disable compression to reduce client buffering
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
}

# --- Argument Parsing ---
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive chat client for the GoGain AI Service."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"URL of the chat API endpoint (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=DEFAULT_USER_ID,
        help=f"User ID for the chat session (default: {DEFAULT_USER_ID})",
    )
    return parser.parse_args()

# --- Conversation Management ---
class ConversationManager:
    def __init__(self, log_file: str):
        self.session_history: List[Dict[str, str]] = []
        self.log_file = log_file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def add_message(self, role: str, message: str):
        """Add a message to current session history"""
        self.session_history.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        self._log_to_file(role, message)
    
    def get_session_history(self) -> List[Dict[str, str]]:
        """Get current session conversation history"""
        return self.session_history.copy()
    
    def _log_to_file(self, role: str, message: str):
        """Log conversation to file"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                timestamp = datetime.now().isoformat()
                f.write(f"[{timestamp}] [{self.session_id}] {role.upper()}: {message}\n")
        except Exception as e:
            print(f"Warning: Could not log to file: {e}")

# --- Server-Sent Events Parser ---
def parse_streaming_response(response) -> str:
    """
    Parse Server-Sent Events in real-time and return final response.
    """
    tokens = []
    final_response = ""
    
    try:
        # Process streaming response line by line
        # Use small chunking and unicode decoding for immediate flush
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                try:
                    data_str = line[6:]  # Remove 'data: ' prefix
                    event_data = json.loads(data_str)
                    
                    if event_data.get('type') == 'token':
                        token = event_data.get('content', '')
                        tokens.append(token)
                        print(token, end='', flush=True)  # Real-time streaming
                    elif event_data.get('type') == 'complete':
                        final_response = event_data.get('final_response', '')
                        break
                        
                except json.JSONDecodeError:
                    continue
                    
        # Return final response if available, otherwise join tokens        
        return final_response if final_response else ''.join(tokens)
        
    except Exception as e:
        return f"[Streaming Error: {str(e)}]"

# --- API Interaction ---
def send_message(api_url: str, user_id: str, message: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """
    Sends a message to the chat API and returns the AI's response text.
    Handles both Server-Sent Events and JSON responses.
    """
    payload = {
        "message": message,
        "user_id": user_id
    }
    
    # Add conversation history if provided
    if conversation_history:
        payload["conversation_history"] = conversation_history
        
    # Prefer no proxies for localhost to avoid buffering; allow opt-in via env
    use_proxies = os.getenv("CHAT_USE_PROXIES", "0") == "1"

    try:
        # Use a session to control environment proxies
        with requests.Session() as session:
            session.trust_env = use_proxies  # disable env proxies by default for better streaming

            response = session.post(
                api_url,
                headers=STREAM_HEADERS,
                json=payload,
                timeout=REQUEST_TIMEOUT,
                stream=True,
            )
            response.raise_for_status()

            # Handle streaming response strictly by content type
            ctype = response.headers.get('content-type', '')
            if ctype.startswith('text/event-stream'):
                return parse_streaming_response(response)

            # Fallback to JSON parsing for non-streaming responses
            try:
                response_data = response.json()
                return response_data.get("response", "[Error: 'response' field missing in JSON]")
            except json.JSONDecodeError:
                # If neither SSE nor JSON, return first 500 chars as fallback
                # DO NOT access response.text before trying SSE; here it's safe (non-streaming)
                return f"[Unparsed Response]: {response.text[:500]}..."

    except requests.exceptions.ConnectionError:
        return f"[Error: Could not connect to API at {api_url}. Is the server running?]"
    except requests.exceptions.Timeout:
        return f"[Error: Request timed out after {REQUEST_TIMEOUT} seconds.]"
    except requests.exceptions.HTTPError as e:
        # response may not be defined if HTTPError was thrown before assignment
        try:
            snippet = response.text[:500]
        except Exception:
            snippet = str(e)
        return f"[Error: HTTP Error {getattr(response, 'status_code', 'unknown')}. Response: {snippet}]"
    except requests.exceptions.RequestException as e:
        return f"[Error: An unexpected request error occurred: {e}]"
    except Exception as e:
        return f"[Error: An unexpected client-side error occurred: {e}]"

# --- Main Interactive Loop ---
def run_chat(api_url: str, user_id: str):
    """Runs the main interactive chat loop."""
    # Initialize conversation manager
    conversation = ConversationManager(LOG_FILE)
    
    print("--- GoGain AI Interactive Chat Client (Enhanced) ---")
    print(f" API Target: {api_url}")
    print(f" User ID:    {user_id}")
    print(f" Session ID: {conversation.session_id}")
    print(f" Log File:   {LOG_FILE}")
    print(" Features:")
    print("   - Server-Sent Events streaming support")
    print("   - Session-based conversation logging")
    print("   - Current session message retention")
    print("-------------------------------------------")
    print("Type 'quit' or 'exit' to end the chat.")
    print("Type 'history' to see current session messages.")
    print("Type 'clear' to clear current session history.")
    print("-------------------------------------------")

    while True:
        try:
            user_input = input("You: ")
            
            if user_input.strip().lower() in ["quit", "exit"]:
                print("Exiting chat. Session logged.")
                conversation._log_to_file("SYSTEM", "Session ended")
                break
                
            if user_input.strip().lower() == "history":
                print("\n--- Current Session History ---")
                for i, msg in enumerate(conversation.get_session_history(), 1):
                    print(f"{i}. [{msg['role'].upper()}]: {msg['message']}")
                print("--- End History ---\n")
                continue
                
            if user_input.strip().lower() == "clear":
                conversation.session_history.clear()
                print("Session history cleared.\n")
                continue

            if not user_input.strip():
                continue # Skip empty input

            # Add user message to session history
            conversation.add_message("user", user_input)
            
            # Send message with current session history
            print("AI: ", end="", flush=True)  # Show AI prompt immediately
            ai_output = send_message(api_url, user_id, user_input, 
                                   conversation.get_session_history())
            
            # Always just add a newline after streaming - don't print response again
            print()  # Just add newline after streaming
            
            # Add AI response to session history
            conversation.add_message("ai", ai_output)

        except (EOFError, KeyboardInterrupt): # Handle Ctrl+D or Ctrl+C
            print("\nExiting chat. Session logged.")
            conversation._log_to_file("SYSTEM", "Session interrupted")
            break

# --- Entry Point ---
if __name__ == "__main__":
    args = parse_arguments()
    run_chat(args.url, args.user)
