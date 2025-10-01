# test_qdrant.py
import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("❌ QDRANT_URL is not set in .env or environment")

    client = QdrantClient(url=qdrant_url)

    try:
        response = client.get_collections()
        print("✅ Connected to Qdrant!")
        print("Collections:", response)
    except Exception as e:
        print("❌ Error connecting to Qdrant:", e)

if __name__ == "__main__":
    main()
