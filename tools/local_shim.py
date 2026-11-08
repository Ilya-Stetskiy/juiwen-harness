#!/usr/bin/env python3
"""Translating shim in front of the local llama-server (lab tool, stdlib only).

The harness picks the thinking mode from the model id (`-noreason` suffix), as the competition gateway does. llama-server
has one model and chooses thinking through a request field, so this shim maps:
    local-35b          -> thinking on  (temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5)
    local-35b-noreason -> thinking off (chat_template_kwargs.enable_thinking=false, temperature 0.7, top_p 0.8, ...)
Sampling values are only filled in when the request does not set them. Nothing is logged: no prompts or answers on disk.

usage: local_shim.py [--listen 127.0.0.1:18444] [--upstream http://127.0.0.1:8011]
"""
import argparse
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REAL = "qwen3.6-35b-a3b"
SAMPLING = {"think": {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5, "top_k": 20},
            "nothink": {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5, "top_k": 20}}
UPSTREAM = "http://127.0.0.1:8011"


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # no access log
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send(200, b'{"status": "ok"}')
        try:
            with urllib.request.urlopen(UPSTREAM + self.path, timeout=30) as r:
                return self._send(r.status, r.read())
        except Exception as e:  # noqa: BLE001
            return self._send(502, json.dumps({"error": str(e)}).encode())

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        try:
            body = json.loads(raw)
            mode = "nothink" if str(body.get("model", "")).endswith("-noreason") else "think"
            body["model"] = REAL
            for k, v in SAMPLING[mode].items():
                body.setdefault(k, v)
            if mode == "nothink":
                body["chat_template_kwargs"] = {"enable_thinking": False}
            data = json.dumps(body).encode()
        except Exception:  # noqa: BLE001 - forward untouched
            data = raw
        req = urllib.request.Request(UPSTREAM + self.path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return self._send(r.status, r.read(), r.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as e:
            return self._send(e.code, e.read())
        except Exception as e:  # noqa: BLE001
            return self._send(502, json.dumps({"error": str(e)}).encode())


def main() -> None:
    global UPSTREAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:18444")
    ap.add_argument("--upstream", default=UPSTREAM)
    a = ap.parse_args()
    UPSTREAM = a.upstream.rstrip("/")
    host, port = a.listen.rsplit(":", 1)
    ThreadingHTTPServer((host, int(port)), H).serve_forever()


if __name__ == "__main__":
    main()
