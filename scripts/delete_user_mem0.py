#!/usr/bin/env python3
"""
Delete all Mem0 memories for a given user ID (backed by local Qdrant).

Usage:
  python scripts/delete_user_mem0.py --user-id <USER_ID> [--yes] [--dry-run] [--collection mem0]

Notes:
- Loads environment vars from .env (expects at least OPENAI_API_KEY and QDRANT_URL for Mem0/Qdrant).
- Tries Mem0 deletion first, then verifies and hard-deletes remaining points directly in Qdrant.
- Use --dry-run to only show how many points would be deleted.
"""

import argparse
import os
import sys
from typing import Optional
from pathlib import Path


def load_env() -> None:
    try:
        from dotenv import load_dotenv, find_dotenv
        env_path = find_dotenv()
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
    except Exception:
        # dotenv is optional; proceed with process env
        pass


def make_mem0_client():
    # Ensure repo root is on sys.path for 'app' package imports
    try:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    except Exception:
        pass

    # Instantiate directly to avoid feature toggle on MEM0_API_KEY in get_mem0_client()
    from app.core.memory import Mem0Client
    return Mem0Client()


def count_qdrant_user_points(collection: str, user_id: str) -> int:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    client = QdrantClient(url=qdrant_url)
    flt = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
    try:
        resp = client.count(collection, count_filter=flt, exact=True)
        return int(getattr(resp, "count", 0))
    except Exception:
        return 0


def hard_delete_qdrant_user_points(collection: str, user_id: str) -> int:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    client = QdrantClient(url=qdrant_url)
    flt = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
    # Use filtered delete
    op = client.delete(collection_name=collection, points_selector=flt)
    # Best-effort: recount after delete
    remaining = count_qdrant_user_points(collection, user_id)
    # We cannot reliably get deleted count from op; return best guess
    return -1 if remaining < 0 else remaining


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Delete all Mem0 memories for a user")
    parser.add_argument("--user-id", required=True, help="User ID to delete memories for")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="Only show how many items would be deleted")
    parser.add_argument("--collection", default="mem0", help="Qdrant collection name (default: mem0)")
    args = parser.parse_args(argv)

    load_env()

    user_id = str(args.user_id)
    collection = args.collection

    if args.dry_run:
        count = count_qdrant_user_points(collection, user_id)
        print(f"[DRY-RUN] Would delete {count} memories for user '{user_id}' from collection '{collection}'.")
        return 0

    if not args.yes:
        confirm = input(f"Delete ALL memories for user '{user_id}' from Mem0/Qdrant? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            return 1

    # Step 1: Try Mem0-driven deletion (iterates through get_all -> delete)
    try:
        mem0 = make_mem0_client()
    except Exception as e:
        print(f"⚠️  Could not initialize Mem0 client ({e}). Will attempt direct Qdrant deletion only.")
        mem0 = None

    if mem0 is not None:
        result = mem0.delete_all_memories(user_id)
        if result.get("success"):
            print(f"✅ Mem0 delete_all_memories: {result.get('message', 'completed')}")
        else:
            print(f"⚠️  Mem0 deletion reported an error: {result.get('error')}")

    # Step 2: Verify in Qdrant and hard-delete any remnants by user_id filter
    pre = count_qdrant_user_points(collection, user_id)
    if pre > 0:
        print(f"ℹ️  {pre} points still present for user '{user_id}' in '{collection}', applying hard delete…")
        hard_delete_qdrant_user_points(collection, user_id)
        post = count_qdrant_user_points(collection, user_id)
        if post == 0:
            print(f"✅ Successfully removed all remaining points for user '{user_id}' from '{collection}'.")
            return 0
        else:
            print(f"⚠️  {post} points still present after deletion. Check Qdrant logs/config.")
            return 2
    else:
        print(f"✅ No points found for user '{user_id}' in '{collection}'. Nothing else to delete.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
