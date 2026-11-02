"""Command line interface for neuroSush Studio."""

from __future__ import annotations

import argparse
import ipaddress
import socket
import sys
import threading
import time
import warnings
import webbrowser

import uvicorn

from studio.config import config


def _is_loopback(host: str) -> bool:
    if host in ("localhost", "ip6-localhost", "ip6-loopback", "127.0.0.1", "::1"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback
    except ValueError:
        try:
            for info in socket.getaddrinfo(host, None):
                ip = ipaddress.ip_address(info[4][0])
                if not ip.is_loopback:
                    return False
            return True
        except Exception:
            return False


def main() -> int:
    """Run the neuroSush Studio server from the command line."""
    parser = argparse.ArgumentParser(description="Run neuroSush Studio local web server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to (default: 8765)")
    parser.add_argument("--open", action="store_true", help="Open browser on start")
    parser.add_argument("--token", default=None, help="Authentication token")
    args = parser.parse_args()

    token = args.token or config.auth_token

    if not _is_loopback(args.host) and not token:
        msg = (
            f"Warning: Binding to non-loopback host {args.host!r} without an authentication token."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        print(msg, file=sys.stderr)

    if args.open:
        open_host = "localhost" if args.host == "0.0.0.0" else args.host
        url = f"http://{open_host}:{args.port}/"

        def _open_browser() -> None:
            time.sleep(1.0)
            webbrowser.open(url)

        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run("studio.app:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
