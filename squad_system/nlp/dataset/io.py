import json
import os
import pickle

from squad_system.nlp.dataset.data_model import (
    ClassificationAnswerableItem,
    ClassificationIndicesItem,
    SquadItem,
)


def load_data_raw(data_path: str) -> list[SquadItem]:
    with open(data_path, "r") as f:
        dataset = json.load(f)

    squad_items = []
    for topic in dataset["data"]:
        for paragraph in topic["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                is_impossible = qa["is_impossible"]
                if is_impossible:
                    squad_items.append(
                        SquadItem(
                            question=question,
                            context=context,
                            answer="",
                            answer_start=-1,
                            is_impossible=is_impossible,
                        )
                    )
                else:
                    for answer in qa["answers"]:
                        squad_items.append(
                            SquadItem(
                                question=question,
                                context=context,
                                answer=answer["text"],
                                answer_start=answer["answer_start"],
                                is_impossible=is_impossible,
                            )
                        )

    return squad_items


def save_data_preprocessed(
    preprocessed_data: list[ClassificationAnswerableItem]
    | list[ClassificationIndicesItem],
    data_path: str,
) -> None:
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "wb") as out_file:
        pickle.dump(preprocessed_data, out_file)


def load_data_preprocessed(
    data_path: str,
) -> list[ClassificationAnswerableItem] | list[ClassificationIndicesItem]:
    with open(data_path, "rb") as in_file:
        return pickle.load(in_file)


def load_data_raw_for_inference(data_path):
    with open(data_path, "r") as f:
        dataset = json.load(f)
    raw_data = []
    for topic in dataset["data"]:
        for paragraph in topic["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                id = qa["id"]
                raw_data.append({"id": id, "context": context, "question": question})
    return raw_data
