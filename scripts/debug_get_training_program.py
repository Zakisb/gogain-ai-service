#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and print the latest training program snapshot for a user."
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

    from app.core.database import get_user_training_program_overview

    program = get_user_training_program_overview(args.user_id)
    if program:
        print(json.dumps(program, indent=2, sort_keys=True, default=str))
    else:
        print(f"No training program found for user_id={args.user_id}")


if __name__ == "__main__":
    main()
