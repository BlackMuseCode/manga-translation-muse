#!/usr/bin/env python3
"""Standalone Kaggle Model Proxy client.

Mints a short-lived proxy token with your own Kaggle API token and calls
an OpenAI-compatible chat-completions endpoint on it.

No Muse connector, no vault, no stored credentials: the Kaggle token is
read from --key-file (raw KGAT_... text or legacy kaggle.json) or the
KAGGLE_API_TOKEN environment variable, used only in-process, and never
printed or written to disk.

Usage:
  python model_proxy.py check --key-file ~/kaggle.txt
      Mint a token and print the proxy base URI (not the token itself).
  from model_proxy import get_token, post   # in translate_sheets.py
"""
import argparse
import json
import os
import urllib.request
from pathlib import Path

API_BASE = "https://api.kaggle.com/v1"


def kaggle_token(key_file=None):
    """Return the caller's Kaggle API token (KGAT access token)."""
    if key_file:
        raw = Path(key_file).read_text(encoding="utf-8-sig").strip()
        if raw.startswith("{"):
            return json.loads(raw)["key"]  # legacy kaggle.json
        return raw
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise SystemExit(
            "no Kaggle token: pass --key-file <kaggle.txt> or set KAGGLE_API_TOKEN")
    return token


def _call(service, method, token, body=None):
    req = urllib.request.Request(
        f"{API_BASE}/{service}/{method}",
        data=json.dumps(body or {}).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def get_token(key_file=None):
    """Mint a Model Proxy token. Returns (base_uri, token).

    The token lasts about 2 hours; on HTTP 401, mint a fresh one.
    """
    tok = _call("models.ModelProxyApiService",
                "CreateDefaultModelProxyToken",
                kaggle_token(key_file), {})
    return tok["baseUri"].rstrip("/"), tok["token"]


def post(base, key, body, timeout=300):
    """POST an OpenAI-compatible chat-completions body to the proxy."""
    req = urllib.request.Request(
        base + "/openapi/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["check"],
                    help="check: mint a token and show the proxy base URI")
    ap.add_argument("--key-file", default=None,
                    help="raw KGAT token file or legacy kaggle.json")
    args = ap.parse_args(argv)
    base, _key = get_token(args.key_file)
    print("proxy base URI:", base)
    print("token minted OK (about 2h validity; not printed).")


if __name__ == "__main__":
    main()
