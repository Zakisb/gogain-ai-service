import os
import sys

# Add the project root to the Python path to allow imports from the 'app' module.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.core.memory import get_mem0_client

def retrieve_user_memories(user_id: str):
    """
    Retrieves and prints all memories for a given user ID.
    """
    # Set dummy keys if not present, for script execution.
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "dummy_key_for_testing"
    if not os.getenv("MEM0_API_KEY"):
        os.environ["MEM0_API_KEY"] = "dummy_key_for_testing"

    mem0_client = get_mem0_client()

    if not mem0_client:
        print("Failed to initialize Mem0Client. Check your environment variables (OPENAI_API_KEY, MEM0_API_KEY, QDRANT_URL).")
        return

    try:
        print(f"\nFetching memories for user_id: {user_id}...")

        # Use the get_memories function with a high limit to fetch all memories
        memories_result = mem0_client.get_memories(user_id=user_id, limit=1000)

        if memories_result.get("success"):
            memories = memories_result.get("memories", [])
            if not memories:
                print(f"No memories found for user_id: {user_id}")
            else:
                print(f"Found {len(memories)} memories:")
                for i, mem in enumerate(memories):
                    print(f"\n--- Memory {i+1} ---")
                    print(f"  ID:      {mem.get('id')}")
                    content = mem.get('memory') or mem.get('content')
                    print(f"  Content: {content}")
                    metadata = mem.get('metadata', {})
                    if metadata:
                        print("  Metadata:")
                        for key, value in metadata.items():
                            print(f"    - {key}: {value}")
        else:
            print(f"An error occurred: {memories_result.get('error')}")

    except KeyboardInterrupt:
        print("\n\nScript interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Ensure user_id is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python retrieve_user_memories.py <user_id>")
        sys.exit(1)

    user_id = sys.argv[1]
    retrieve_user_memories(user_id)
