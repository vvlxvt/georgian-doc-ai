import argparse
from pathlib import Path

from app.config import settings
from app.database import SessionLocal
from app.dependencies import create_document_pipeline
from app.preflight import PreflightError, validate_runtime


def run_preflight() -> int:
    try:
        validate_runtime(settings)
    except PreflightError as exc:
        print(f"Preflight failed: {exc}")
        return 1

    print("Preflight passed.")
    return 0


def run_index(path: str) -> int:
    pdf_path = Path(path)

    if not pdf_path.is_file():
        print(f"Preflight failed: '{path}' is not a readable file")
        return 1

    try:
        validate_runtime(settings)
    except PreflightError as exc:
        print(f"Preflight failed: {exc}")
        return 1

    try:
        with SessionLocal() as session:
            document_id = create_document_pipeline(session).process(str(pdf_path))
    except Exception as exc:
        print(f"Indexing failed for '{pdf_path.name}': {exc}")
        return 1

    print(f"Indexed '{pdf_path.name}' as document {document_id}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Georgian document RAG utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preflight", help="validate all runtime dependencies")

    index_parser = subparsers.add_parser(
        "index",
        help="ingest a PDF into SQLite and Qdrant",
    )
    index_parser.add_argument("path", help="path to the PDF file to index")

    args = parser.parse_args()

    if args.command == "preflight":
        return run_preflight()

    return run_index(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
