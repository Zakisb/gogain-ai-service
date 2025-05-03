# /Users/mac/Desktop/work/GoGain/GOGAIN-AI-SERVICE/interactive_chat.py
import requests
import json
import argparse
import os
import readline # Optional: Improves input editing experience (arrows, history)

# --- Configuration ---
DEFAULT_API_URL = "http://localhost:8000/api/v1/chat"
DEFAULT_USER_ID = "test_user_01"

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

# --- API Interaction ---
def send_message(api_url: str, user_id: str, message: str):
    """
    Sends a message to the chat API and returns the AI's response text.
    Requires the server API ('app/api/chat.py') to accept 'user_id' 
    in its ChatRequest model and implement persistent memory based on it.
    """
    payload = {
        "message": message,
        "user_id": user_id
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        try:
            response_data = response.json()
            ai_response = response_data.get("response", "[Error: 'response' field missing in JSON]")
            return ai_response
        except json.JSONDecodeError:
            return f"[Error: Failed to decode API response. Response text: {response.text[:1000]}]" # Show first 1000 chars

    except requests.exceptions.ConnectionError:
        return f"[Error: Could not connect to API at {api_url}. Is the server running?]"
    except requests.exceptions.Timeout:
        return "[Error: Request timed out.]"
    except requests.exceptions.HTTPError as e:
        return f"[Error: HTTP Error {response.status_code}. Response: {response.text[:1000]}]"
    except requests.exceptions.RequestException as e:
        return f"[Error: An unexpected request error occurred: {e}]"
    except Exception as e:
         return f"[Error: An unexpected client-side error occurred: {e}]"

# --- Main Interactive Loop ---
def run_chat(api_url: str, user_id: str):
    """Runs the main interactive chat loop."""
    print("--- GoGain AI Interactive Chat Client ---")
    print(f" API Target: {api_url}")
    print(f" User ID:    {user_id}")
    print(" Server Requirements:")
    print("   - API endpoint must accept 'user_id' in request body.")
    print("   - Server must implement persistent memory linked to 'user_id'.")
    print("-------------------------------------------")
    print("Type 'quit' or 'exit' to end the chat.")
    print("-------------------------------------------")

    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ["quit", "exit"]:
                print("Exiting chat.")
                break

            if not user_input.strip():
                continue # Skip empty input

            ai_output = send_message(api_url, user_id, user_input)
            print(f"AI:  {ai_output}")

        except (EOFError, KeyboardInterrupt): # Handle Ctrl+D or Ctrl+C
            print("\nExiting chat.")
            break

# --- Entry Point ---
if __name__ == "__main__":
    args = parse_arguments()
    run_chat(args.url, args.user)