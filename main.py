import argparse

from app.config import settings
from app.preflight import PreflightError, validate_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Georgian document RAG utilities")
    parser.add_argument("command", choices=["preflight"])
    args = parser.parse_args()

    try:
        validate_runtime(settings)
    except PreflightError as exc:
        print(f"Preflight failed: {exc}")
        return 1

    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
