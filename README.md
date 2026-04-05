# Themlmodel

Minimal from-scratch segmentation baseline focused on improving IoU and pixel accuracy without using any pretrained/open-source model weights.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Telegram bot integration

The package includes a Telegram integration layer in `themlmodel.telegram_integration`.

Set your bot token using an environment variable (do not hardcode it in code):

```bash
export TELEGRAM_BOT_TOKEN="your-token"
```

Then call:

```python
from themlmodel import send_telegram_message

send_telegram_message(chat_id=123456, text="Training finished ✅")
```

Security note: If a token was shared publicly, rotate/revoke it in BotFather and replace it with a new one.
