import logging

from fastapi import FastAPI

from squad_system.api.endpoints import health_router, infer_router
from squad_system.utils.logs import initialize_logging

logger = logging.getLogger(__name__)


def get_fastapi_app() -> FastAPI:
    initialize_logging("config/logging.yml")

    api = FastAPI()
    api.include_router(health_router)
    api.include_router(infer_router)

    return api
