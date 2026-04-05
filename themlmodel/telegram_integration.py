from __future__ import annotations

import json
import os
from typing import Any
from urllib import parse, request

TELEGRAM_TOKEN_ENV_VAR = "TELEGRAM_BOT_TOKEN"
TELEGRAM_API_BASE = "https://api.telegram.org"


def get_telegram_bot_token(env: dict[str, str] | None = None) -> str:
    if env is None:
        env = os.environ
    token = env.get(TELEGRAM_TOKEN_ENV_VAR, "").strip()
    if not token:
        raise ValueError(f"Missing Telegram bot token. Set {TELEGRAM_TOKEN_ENV_VAR}.")
    return token


def send_telegram_message(
    chat_id: str | int,
    text: str,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    if token is None:
        token = get_telegram_bot_token()
    endpoint = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"
    payload = parse.urlencode({"chat_id": str(chat_id), "text": text}).encode("utf-8")
    req = request.Request(endpoint, data=payload, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)
