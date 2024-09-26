import logging

from fastapi import APIRouter, Depends, HTTPException

from squad_system.api.auth import create_api_key_bearer
from squad_system.api.data_model import InferenceRequest, InferenceResponse
from squad_system.api.handlers import inference_request_handler

logger = logging.getLogger(__name__)

health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.get("")
def get_health() -> dict[str, str]:
    try:
        if inference_request_handler.health_check():
            return {"status": "OK"}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Unexpected error")


infer_router = APIRouter(
    prefix="/infer", tags=["infer"], dependencies=[Depends(create_api_key_bearer())]
)


@infer_router.post(path="", response_model=InferenceResponse, status_code=200)
def infer(infer_request: InferenceRequest) -> InferenceResponse:
    try:
        logger.info(f"Received request: {infer_request}")
        infer_response = inference_request_handler.infer(infer_request)
        logger.info(f"Returning response: {infer_response}")
        return infer_response
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Unexpected error")
