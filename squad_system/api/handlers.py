import logging
from typing import Optional

from squad_system.api.data_model import InferenceRequest, InferenceResponse
from squad_system.nlp.inference.pipeline import (
    InferencePipeline,
    get_inference_pipeline,
)

logger = logging.getLogger(__name__)


class InferenceRequestHandler:
    def __init__(self, inference_pipeline: InferencePipeline):
        self.inference_pipeline = inference_pipeline

    def infer(self, request: InferenceRequest) -> Optional[InferenceResponse]:
        try:
            questions = request.questions
            context = request.context
            answers = []

            for question in questions:
                answer = self.inference_pipeline.infer(question, context)
                answers.append(answer)

            return InferenceResponse(
                questions=questions, context=context, answers=answers
            )
        except Exception as e:
            logger.exception(e)
            raise e

    def health_check(self) -> bool:
        return True


def get_inference_request_handler() -> InferenceRequestHandler:
    inference_pipeline = get_inference_pipeline()
    return InferenceRequestHandler(inference_pipeline)


inference_request_handler = get_inference_request_handler()
