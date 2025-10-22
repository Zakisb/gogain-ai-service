#!/usr/bin/env python3
"""
Tiny Qdrant inspector: prints collections, configs, payload schemas, aliases,
and basic cluster info using only the HTTP API (no repo code modifications).

Usage (inside container):
  QDRANT_URL=http://localhost:6333 python3 scripts/inspect_qdrant.py

Defaults to http://localhost:6333 if QDRANT_URL is not set.
"""

import os
import json
import sys
from urllib import request, error


def fetch_json(base_url: str, path: str):
    url = f"{base_url.rstrip('/')}{path}"
    try:
        with request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        return {"error": f"HTTP {e.code}", "url": url, "body": body}
    except Exception as e:
        return {"error": str(e), "url": url}


def main():
    base = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")

    # Probe basic service info (best‑effort; not all versions expose root info)
    root_info = fetch_json(base, "/")

    # Collections list
    col_list = fetch_json(base, "/collections")
    collections = []
    names = []
    try:
        names = [c.get("name") for c in (col_list.get("result", {}).get("collections", []) or []) if c.get("name")]
    except Exception:
        names = []

    # Per‑collection details
    for name in names:
        info = fetch_json(base, f"/collections/{name}")
        aliases = fetch_json(base, f"/collections/{name}/aliases")
        schema = fetch_json(base, f"/collections/{name}/points/payload/schema")
        collections.append({
            "name": name,
            "info": info,
            "aliases": aliases,
            "payload_schema": schema,
        })

    # Global aliases (best‑effort)
    all_aliases = fetch_json(base, "/aliases")

    # Cluster info (best‑effort; may be single‑node)
    cluster_status = fetch_json(base, "/cluster/status")
    cluster_info = fetch_json(base, "/cluster/info")

    out = {
        "base_url": base,
        "service": root_info,
        "collections_list": col_list,
        "collections": collections,
        "aliases": all_aliases,
        "cluster_status": cluster_status,
        "cluster_info": cluster_info,
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())

