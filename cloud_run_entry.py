"""HTTP service that exposes basic Meridian runtime diagnostics.

The previous revision of this file only returned a plain text string. Cloud Run
deployments succeeded, but the response did not provide any insight into the
container environment, which made it difficult for Advantage Medical
Professionals to confirm the service was configured with their datasets.

This module now exposes a couple of lightweight JSON endpoints that report the
Meridian version, configured dataset locations, and suggestions for next steps.
These diagnostics are intentionally minimal—they do *not* execute a full MMM
pipeline—but they give operators immediate feedback that the container booted
with the expected configuration and has access to the referenced files.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import platform
from pathlib import Path
from typing import Any, Dict
from urllib import parse

try:
  from meridian import __version__ as _MERIDIAN_VERSION
except Exception:  # pragma: no cover - defensive import guard
  _MERIDIAN_VERSION = None


_DATA_PATH_ENV = "MERIDIAN_DATA_PATH"
_CONFIG_PATH_ENV = "MERIDIAN_CONFIG_PATH"


def _collect_path_metadata(path_value: str | None) -> Dict[str, Any]:
  """Return filesystem metadata for a configured path."""

  info: Dict[str, Any] = {"configured": bool(path_value)}
  if not path_value:
    info["message"] = (
        "Set the environment variable to point at your aggregated MMM dataset."
    )
    return info

  path = Path(path_value)
  info["path"] = path_value
  info["exists"] = path.exists()
  if path.is_file():
    info["size_bytes"] = path.stat().st_size
  elif path.is_dir():
    info["contains"] = sorted(p.name for p in path.iterdir())[:25]
  else:
    info["message"] = "Path does not point to a regular file or directory."

  return info


def _status_payload() -> Dict[str, Any]:
  """Builds a JSON-serialisable object describing the runtime state."""

  return {
      "service": "Meridian Cloud Run adapter",
      "python_version": platform.python_version(),
      "meridian_version": _MERIDIAN_VERSION,
      "environment": {
          _DATA_PATH_ENV: _collect_path_metadata(os.getenv(_DATA_PATH_ENV)),
          _CONFIG_PATH_ENV: _collect_path_metadata(os.getenv(_CONFIG_PATH_ENV)),
      },
      "next_steps": [
          "Load your Advantage Medical Professionals aggregated MMM dataset",
          "Configure authenticated endpoints that trigger model refresh jobs",
          "Connect upstream ingestion jobs (HubSpot, web analytics, etc.) that"
          " populate the aggregated data referenced above",
      ],
  }


class MeridianDiagnosticsHandler(BaseHTTPRequestHandler):
  """Serve a couple of useful routes for Cloud Run diagnostics."""

  def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK):
    body = json.dumps(payload, indent=2).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self):  # pylint: disable=invalid-name
    parsed = parse.urlparse(self.path)
    if parsed.path in ("/", "/healthz"):
      self._send_json({"status": "ok", "detail": "Meridian service is running."})
      return

    if parsed.path == "/status":
      self._send_json(_status_payload())
      return

    self._send_json({"error": "Not Found", "path": parsed.path}, HTTPStatus.NOT_FOUND)

  def do_POST(self):  # pylint: disable=invalid-name
    parsed = parse.urlparse(self.path)
    if parsed.path == "/refresh":
      self._send_json(
          {
              "message": (
                  "Model refresh endpoint is not yet implemented. Extend this"
                  " handler to trigger your Meridian modeling jobs."
              )
          },
          status=HTTPStatus.NOT_IMPLEMENTED,
      )
      return

    self._send_json({"error": "Not Found", "path": parsed.path}, HTTPStatus.NOT_FOUND)

  def log_message(self, format, *args):  # pylint: disable=redefined-builtin
    # Silence the default request logging to keep Cloud Run logs cleaner.
    return


def main():
  port = int(os.environ.get("PORT", "8080"))
  server_address = ("", port)
  httpd = ThreadingHTTPServer(server_address, MeridianDiagnosticsHandler)
  httpd.serve_forever()


if __name__ == "__main__":
  main()
