import asyncio
import sys

import requests

from src.telegram import TelegramClient


def main():
    """
    Implements primitive CLI.
    """
    cmd = ""

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

    code = 0

    if cmd == "healthcheck":
        code = healthcheck()
    elif cmd == "set-webhook":
        code = asyncio.run(set_webhook(sys.argv))
    elif cmd == "delete-webhook":
        code = asyncio.run(delete_webhook())
    elif cmd == "get-webhook":
        code = asyncio.run(get_webhook())
    else:
        code = 2

    if code == 2:
        print_usage()

    sys.exit(code)


def print_usage() -> None:
    """
    Prints CLI usage information.
    """
    print(
        "Usage: python -m src.cli\n"
        "  healthcheck\n"
        "  set-webhook <url> [secret_token]\n"
        "  delete-webhook\n"
        "  get-webhook"
    )


def healthcheck() -> int:
    """
    Makes HTTP request to the health endpoint on an expected address.
    Returns 0 if successful, otherwise returns 1.
    """
    try:
        resp = requests.get("http://127.0.0.1:8000/health", timeout=3)
        resp.raise_for_status()
    except Exception:
        return 1

    return 0


async def set_webhook(argv: list[str]) -> int:
    """
    Sets Telegram webhook URL.
    """
    if len(argv) < 3:
        return 2

    url = argv[2]
    secret_token = argv[3] if len(argv) > 3 else None
    response = None

    async with TelegramClient() as client:
        response = await client.set_webhook(url=url, secret_token=secret_token)

    print(response)

    return 0


async def delete_webhook() -> int:
    """
    Deletes current Telegram webhook.
    """
    response = None

    async with TelegramClient() as client:
        response = await client.delete_webhook()

    print(response)

    return 0


async def get_webhook() -> int:
    """
    Gets current Telegram webhook info.
    """
    response = None

    async with TelegramClient() as client:
        response = await client.get_webhook_info()

    print(response)

    return 0


if __name__ == "__main__":
    main()
