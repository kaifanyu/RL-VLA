"""Probe upstream readiness without opening a policy session or allocating a GPU."""

import os
import sys
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


def main() -> int:
    try:
        port = int(os.environ.get("OPENPI_PORT", "8000"))
        if not 1 <= port <= 65535:
            raise ValueError("OPENPI_PORT must be between 1 and 65535")
        # Local readiness must work even when the environment has HTTP proxy settings.
        opener = build_opener(ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/healthz", timeout=5) as response:
            return 0 if response.status == 200 and response.read(16).strip() == b"OK" else 1
    except (OSError, URLError, ValueError) as error:
        print(f"OpenPI is not ready: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
