from pathlib import Path

from storage.database import _configured_db_path


def test_database_path_can_be_configured_for_modal_volume(monkeypatch):
    monkeypatch.delenv("UNDERWRITING_DB_PATH", raising=False)
    monkeypatch.setenv("DATABASE_URL", "sqlite:////data/underwriting.db")

    assert _configured_db_path() == Path("/data/underwriting.db")


def test_modal_entrypoint_uses_asgi_app_and_volume():
    modal_entrypoint = Path("modal_app.py").read_text(encoding="utf-8")

    assert "@modal.asgi_app()" in modal_entrypoint
    assert "modal.Volume.from_name" in modal_entrypoint
    assert "sqlite:///" in modal_entrypoint
    assert "from app.main import app as web_app" in modal_entrypoint


def test_modal_image_adds_local_files_after_build_steps():
    modal_entrypoint = Path("modal_app.py").read_text(encoding="utf-8")

    first_local_add = min(
        modal_entrypoint.index(".add_local_dir"),
        modal_entrypoint.index(".add_local_python_source"),
    )
    assert modal_entrypoint.index(".pip_install_from_requirements") < first_local_add
    assert modal_entrypoint.index(".env({") < first_local_add
