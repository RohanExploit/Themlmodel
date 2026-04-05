import unittest
from unittest.mock import patch
from urllib import error

from themlmodel.telegram_integration import (
    TELEGRAM_TOKEN_ENV_VAR,
    get_telegram_bot_token,
    send_telegram_message,
)


class _FakeResponse:
    def __init__(self, payload: str):
        self._payload = payload

    def read(self):
        return self._payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TelegramIntegrationTests(unittest.TestCase):
    def test_get_telegram_bot_token_from_env(self):
        token = get_telegram_bot_token({TELEGRAM_TOKEN_ENV_VAR: "abc123"})
        self.assertEqual(token, "abc123")

    def test_get_telegram_bot_token_requires_value(self):
        with self.assertRaises(ValueError):
            get_telegram_bot_token({})

    def test_send_telegram_message_uses_expected_endpoint_and_payload(self):
        with patch(
            "themlmodel.telegram_integration.request.urlopen",
            return_value=_FakeResponse('{"ok": true, "result": {"message_id": 1}}'),
        ) as mocked_urlopen:
            response = send_telegram_message(chat_id=42, text="hello", token="bot-token")

        self.assertTrue(response["ok"])
        called_req = mocked_urlopen.call_args.args[0]
        self.assertEqual(called_req.full_url, "https://api.telegram.org/botbot-token/sendMessage")
        self.assertEqual(called_req.data, b"chat_id=42&text=hello")
        headers = {k.lower(): v for k, v in called_req.header_items()}
        self.assertEqual(headers.get("content-type"), "application/x-www-form-urlencoded")

    def test_send_telegram_message_wraps_network_errors(self):
        with patch(
            "themlmodel.telegram_integration.request.urlopen",
            side_effect=error.URLError("network down"),
        ):
            with self.assertRaises(RuntimeError):
                send_telegram_message(chat_id=42, text="hello", token="bot-token")


if __name__ == "__main__":
    unittest.main()
