import sys

import pytest

import main
from app.preflight import PreflightError


class FakePipeline:
    def __init__(self, session):
        self.session = session
        self.processed = []

    def process(self, path):
        self.processed.append(path)
        return "document-id"


class FakeSession:
    def __init__(self):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True
        return False


def test_run_index_reports_missing_file(capsys):
    assert main.run_index("does-not-exist.pdf") == 1

    assert "not a readable file" in capsys.readouterr().out


def test_run_index_stops_when_preflight_fails(tmp_path, monkeypatch, capsys):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    def failing_preflight(config):
        raise PreflightError("Ollama model 'x' is not installed")

    monkeypatch.setattr(main, "validate_runtime", failing_preflight)

    assert main.run_index(str(pdf_path)) == 1
    assert "not installed" in capsys.readouterr().out


def test_run_index_processes_pdf_and_closes_session(tmp_path, monkeypatch, capsys):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    session = FakeSession()
    pipelines = []

    def build_pipeline(session_arg):
        pipeline = FakePipeline(session_arg)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(main, "validate_runtime", lambda config: None)
    monkeypatch.setattr(main, "SessionLocal", lambda: session)
    monkeypatch.setattr(main, "create_document_pipeline", build_pipeline)

    assert main.run_index(str(pdf_path)) == 0

    assert pipelines[0].processed == [str(pdf_path)]
    assert session.closed is True
    assert "document-id" in capsys.readouterr().out


def test_run_index_reports_pipeline_failure(tmp_path, monkeypatch, capsys):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    class BrokenPipeline:
        def __init__(self, session):
            pass

        def process(self, path):
            raise RuntimeError("Qdrant refused the upsert")

    monkeypatch.setattr(main, "validate_runtime", lambda config: None)
    monkeypatch.setattr(main, "SessionLocal", lambda: FakeSession())
    monkeypatch.setattr(main, "create_document_pipeline", BrokenPipeline)

    assert main.run_index(str(pdf_path)) == 1
    assert "Indexing failed" in capsys.readouterr().out


def test_main_dispatches_preflight(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "preflight"])
    monkeypatch.setattr(main, "run_preflight", lambda: 0)

    assert main.main() == 0


def test_main_dispatches_index(tmp_path, monkeypatch):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    received = {}
    monkeypatch.setattr(sys, "argv", ["main.py", "index", str(pdf_path)])

    def fake_run_index(path):
        received["path"] = path
        return 0

    monkeypatch.setattr(main, "run_index", fake_run_index)

    assert main.main() == 0
    assert received["path"] == str(pdf_path)


def test_main_requires_a_command(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    with pytest.raises(SystemExit):
        main.main()


def test_run_preflight_passes_when_runtime_is_valid(monkeypatch, capsys):
    monkeypatch.setattr(main, "validate_runtime", lambda config: None)

    assert main.run_preflight() == 0
    assert "Preflight passed" in capsys.readouterr().out


def test_run_preflight_fails_cleanly(monkeypatch, capsys):
    def failing(config):
        raise PreflightError("Qdrant is unavailable")

    monkeypatch.setattr(main, "validate_runtime", failing)

    assert main.run_preflight() == 1
    assert "Qdrant is unavailable" in capsys.readouterr().out
