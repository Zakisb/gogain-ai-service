#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and print a user profile using app.core.database.get_user_profile"
    )
    parser.add_argument(
        "user_id",
        type=int,
        help="Numeric user ID to look up",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    load_dotenv(dotenv_path=project_root / ".env")

    from app.core.database import get_user_profile

    profile: Any = get_user_profile(args.user_id)
    if profile:
        print(json.dumps(profile, indent=2, sort_keys=True, default=str))
    else:
        print(f"No profile found for user_id={args.user_id}")


if __name__ == "__main__":
    main()
