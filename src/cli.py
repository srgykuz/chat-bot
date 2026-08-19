import sys

import requests


def main():
    """
    Implements primitive CLI.
    """
    arg = ""

    if len(sys.argv) > 1:
        arg = sys.argv[1]

    if arg == "healthcheck":
        healthcheck()
    else:
        raise Exception("See the source for usage.")


def healthcheck() -> None:
    """
    Makes HTTP request to the health endpoint on an expected address.
    Returns None if successful, otherwise raises an exception.
    """
    resp = requests.get("http://127.0.0.1:8000/health", timeout=3)
    resp.raise_for_status()


if __name__ == "__main__":
    main()
