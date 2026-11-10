#!/usr/bin/env python
"""Serves the agent's saved games (data/agent_games/*.json) to the real
OpenFront client, so they can be watched with the full game graphics.

The dev client (``npm run dev`` in engine/openfront-ai/openfront, on
http://localhost:9000) loads a finished game by fetching
``http://localhost:8787/game/<gameID>`` — the archive API openfront.io
runs in production — and replays it locally. This script stands in for
that API. Dev client builds report gitCommit "DEV", which skips the
client's record-version check, so liveGame.ts's records load as-is.

Usage:
    python scripts/07_serve_replays.py [--port 8787]
Then open http://localhost:9000/game/<gameID> (printed below for each game).
"""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openfront_il.utils import paths

CLIENT_URL = "http://localhost:9000"
_GAME_PATH = re.compile(r"^/game/([A-Za-z0-9]{8})/?$")


class ReplayHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        # The client runs on a different origin (localhost:9000).
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server naming
        # The client's GET sets Content-Type: application/json, which makes
        # the browser send a CORS preflight first.
        self._send(204, b"")

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        match = _GAME_PATH.match(self.path)
        if match:
            record = paths.AGENT_GAMES_DIR / f"{match.group(1)}.json"
            if record.exists():
                self._send(200, record.read_bytes())
            else:
                self._send(404, b'{"error":"not found"}')
            return
        if self.path in ("/", "/index.html"):
            self._send(200, _index_page().encode(), "text/html; charset=utf-8")
            return
        # Everything else the client might ask the real API for (accounts,
        # cosmetics, ...) simply doesn't exist here.
        self._send(404, b'{"error":"not found"}')

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        if "/game/" in self.path:
            super().log_message(format, *args)


def _games() -> list[tuple[str, dict[str, object]]]:
    games = []
    for record_path in sorted(
        paths.AGENT_GAMES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        try:
            info = json.loads(record_path.read_text(encoding="utf-8"))["info"]
        except (OSError, ValueError, KeyError):
            continue
        games.append((record_path.stem, info))
    return games


def _index_page() -> str:
    rows = "".join(
        f'<li><a href="{CLIENT_URL}/game/{game_id}">{game_id}</a> - '
        f"{info['config']['gameMap']}, {info['num_turns']} ticks, "
        f"winner: {info.get('winner') or 'none'}</li>"
        for game_id, info in _games()
    )
    return f"<h1>Agent games</h1><ul>{rows or '<li>none yet</li>'}</ul>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    games = _games()
    print(f"serving {len(games)} agent game(s) from {paths.AGENT_GAMES_DIR}")
    for game_id, info in games[:10]:
        print(f"  {CLIENT_URL}/game/{game_id}   ({info['config']['gameMap']})")
    print(f"full list: http://localhost:{args.port}/   (Ctrl+C to stop)")
    ThreadingHTTPServer(("localhost", args.port), ReplayHandler).serve_forever()


if __name__ == "__main__":
    main()
