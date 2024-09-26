from pydantic import BaseModel, Field


class SquadItem(BaseModel):
    question: str = Field(description="Question to be answered.")
    context: str = Field(description="Context for answering the question.")
    answer: str = Field(description="Answer text.")
    answer_start: int = Field(
        description="Index of the answer's first character inside the context."
    )
    is_impossible: bool = Field(
        description="Whether the question can be answered based on the context."
    )


class ClassificationAnswerableItem(BaseModel):
    token_ids: list[int] = Field(
        description="Token ids as encoded by the tokenizer. The question and context are encoded together."
    )
    answerable: int = Field(
        description="Label for question-context pair. 0 for not answerable / 1 for answerable"
    )
    attention_mask: list[int] = Field(
        "List of indices specifying which tokens should be attended to by the model."
    )


class ClassificationIndicesItem(BaseModel):
    token_ids: list[int] = Field(
        description="Token ids as encoded by the tokenizer. The question and context are encoded together."
    )
    answer_start: int = Field(
        description="Index of the answer's first token inside the context."
    )
    answer_end: int = Field(
        description="Index of the answer's last token inside the context."
    )
    attention_mask: list[int] = Field(
        "List of indices specifying which tokens should be attended to by the model."
    )
