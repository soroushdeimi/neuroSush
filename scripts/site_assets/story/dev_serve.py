"""Serve the standalone story and existing shared assets."""

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8776
    print(f"Story: http://127.0.0.1:{port}/story/story.html", flush=True)
    ThreadingHTTPServer(
        ("127.0.0.1", port),
        partial(SimpleHTTPRequestHandler, directory=str(root)),
    ).serve_forever()
