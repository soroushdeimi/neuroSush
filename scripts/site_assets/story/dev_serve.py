"""Serve the standalone story and existing shared assets."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from functools import partial

if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    print('Story: http://127.0.0.1:8766/story/story.html', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8766), partial(SimpleHTTPRequestHandler, directory=str(root))).serve_forever()
