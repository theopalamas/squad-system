from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    questions: list[str] = Field(
        default=[], description="List of questions to be answered based on the context."
    )
    context: str = Field(
        default="", description="Context that might contain answers to the questions."
    )


class InferenceResponse(BaseModel):
    questions: list[str] = Field(
        default=[], description="List of questions to be answered based on the context."
    )
    context: str = Field(
        default="", description="Context that might contain answers to the questions."
    )
    answers: list[str] = Field(
        default=[],
        description="List of answers to the questions. Empty string for unanswerable questions.",
    )
