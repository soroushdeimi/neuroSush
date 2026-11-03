"""Serve the standalone story and existing shared assets."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from functools import partial
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8776
    print(f'Story: http://127.0.0.1:{port}/story/story.html', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), partial(SimpleHTTPRequestHandler, directory=str(root))).serve_forever()
