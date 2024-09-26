from tqdm import tqdm
from transformers import AutoTokenizer

from squad_system.nlp.dataset.data_model import (
    ClassificationAnswerableItem,
    ClassificationIndicesItem,
    SquadItem,
)


def preprocess_for_classification_answerable(
    data: list[SquadItem], pretrained_tokenizer_name: str
) -> list[ClassificationAnswerableItem]:
    """
    Preprocesses and encodes the raw Squad data for the 'classification_answerable' task
    """
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_tokenizer_name, clean_up_tokenization_spaces=True
    )

    classification_answerable_data = []
    for squad_item in tqdm(data, desc="Preprocessing for answerable classification"):
        encoding = tokenizer.encode_plus(
            text=squad_item.question,
            text_pair=squad_item.context,
            truncation="only_second",
            max_length=384,
            padding="max_length",
            return_offsets_mapping=True,
            return_attention_mask=True,
        )

        classification_answerable_data.append(
            ClassificationAnswerableItem(
                token_ids=encoding["input_ids"],
                answerable=0 if squad_item.is_impossible else 1,
                attention_mask=encoding["attention_mask"],
            )
        )

    return classification_answerable_data


def preprocess_for_classification_indices(
    data: list[SquadItem], pretrained_tokenizer_name: str, allow_unanswerable: bool
) -> list[ClassificationIndicesItem]:
    """
    Preprocesses and encodes the raw Squad data for the 'classification_indices' task
    """
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_tokenizer_name, clean_up_tokenization_spaces=True
    )

    classification_indices_data = []
    for squad_item in tqdm(
        data, desc=f"Preprocessing for indices classification ({allow_unanswerable=})"
    ):
        if not squad_item.is_impossible or (
            squad_item.is_impossible and allow_unanswerable
        ):
            encoding = tokenizer.encode_plus(
                text=squad_item.question,
                text_pair=squad_item.context,
                truncation="only_second",
                max_length=384,
                padding="max_length",
                return_offsets_mapping=True,
                return_attention_mask=True,
            )

            # Taken from https://huggingface.co/docs/transformers/tasks/question_answering
            end_char = squad_item.answer_start + len(squad_item.answer)
            sequence_ids = encoding.sequence_ids()
            offset = encoding["offset_mapping"]

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if (
                offset[context_start][0] > end_char
                or offset[context_end][1] < squad_item.answer_start
            ):
                answer_start = 0
                answer_end = 0
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= squad_item.answer_start:
                    idx += 1
                answer_start = idx - 1

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                answer_end = idx + 1

            # If the question cannot be answered
            if squad_item.is_impossible:
                answer_start = 0
                answer_end = 0

            classification_indices_data.append(
                ClassificationIndicesItem(
                    token_ids=encoding["input_ids"],
                    answer_start=answer_start,
                    answer_end=answer_end,
                    attention_mask=encoding["attention_mask"],
                )
            )

    return classification_indices_data
