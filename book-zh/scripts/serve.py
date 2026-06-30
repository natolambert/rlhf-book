#!/usr/bin/env python3
"""Serve the built HTML site locally for previewing.

The production site (rlhfbook.com) rewrites extensionless "pretty" URLs such as
``/c/05-reward-models`` to the underlying ``.html`` file, and references assets
with root-absolute paths (``/assets/...``, ``/pagefind/...``). Opening the build
over ``file://`` therefore breaks navigation and asset loading. This static
server, rooted at ``build/html``, restores both: absolute paths resolve and
extensionless URLs fall back to ``<path>.html``.

Usage:
    uv run python book/scripts/serve.py [--dir build/html] [--port 8000]

Stdlib only; no dependencies.
"""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver


class CleanURLHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that maps ``/foo`` to ``/foo.html`` when needed."""

    def send_head(self):  # noqa: D102 - inherited behavior, extended below
        path = self.translate_path(self.path.split("?", 1)[0].split("#", 1)[0])
        if not os.path.exists(path) and not path.endswith(".html"):
            candidate = path.rstrip("/") + ".html"
            if os.path.isfile(candidate):
                # Rewrite the request path so the base handler serves the .html.
                query = self.path[len(self.path.split("?", 1)[0]):]
                self.path = self.path.split("?", 1)[0].rstrip("/") + ".html" + query
        return super().send_head()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="build/html", help="Directory to serve")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port to listen on (default 8000, or $PORT)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Address to bind (default 127.0.0.1)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(
            f"{args.dir!r} not found. Run `make html files` first to build the site."
        )

    os.chdir(args.dir)

    class Server(socketserver.TCPServer):
        allow_reuse_address = True

    with Server((args.host, args.port), CleanURLHandler) as httpd:
        url = f"http://{args.host}:{args.port}/"
        print(f"Serving {args.dir} at {url} (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
