"""Modal deployment entrypoint for the Agentic Underwriter FastAPI app."""

from __future__ import annotations

import os

import modal


APP_NAME = os.getenv("MODAL_APP_NAME", "agentic-underwriter")
DATA_MOUNT_PATH = "/data"
DATA_VOLUME_NAME = os.getenv("MODAL_DATA_VOLUME", "agentic-underwriter-data")

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_requirements("requirements.txt")
    .env({
        "DATABASE_URL": f"sqlite:///{DATA_MOUNT_PATH}/underwriting.db",
        "RAG_RETRIEVAL_MODE": "lexical",
        "RAG_EMBEDDINGS_ENABLED": "false",
        "LLM_STRUCTURED_OUTPUT_ENABLED": "false",
        "TRACE_BACKEND": "logging",
    })
    .add_local_dir("app/externaldata/docs", "/root/app/externaldata/docs")
    .add_local_python_source(
        "app",
        "evals",
        "models",
        "observability",
        "storage",
        "tools",
        "workflows",
    )
)


@app.function(
    image=image,
    volumes={DATA_MOUNT_PATH: data_volume},
    min_containers=1,
    max_containers=1,
    timeout=120,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    """Serve the existing FastAPI app on Modal as an ASGI endpoint."""
    from app.main import app as web_app

    @web_app.middleware("http")
    async def commit_modal_volume_after_request(request, call_next):
        response = await call_next(request)
        data_volume.commit()
        return response

    return web_app
