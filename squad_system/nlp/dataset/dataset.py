from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset

from squad_system.nlp.dataset.data_model import (
    ClassificationAnswerableItem,
    ClassificationIndicesItem,
)


class ClassificationAnswerableDataset(Dataset):
    def __init__(
        self, classification_answerable_data: list[ClassificationAnswerableItem]
    ) -> None:
        self.classification_answerable_data = classification_answerable_data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item_dict = self.classification_answerable_data[idx].dict()
        return {
            "token_ids": torch.tensor(item_dict["token_ids"], dtype=torch.int64),
            "answerable": torch.tensor(item_dict["answerable"], dtype=torch.int64),
            "attention_mask": torch.tensor(
                item_dict["attention_mask"], dtype=torch.int64
            ),
        }

    def __len__(self) -> int:
        return len(self.classification_answerable_data)


class ClassificationIndicesDataset(Dataset):
    def __init__(
        self, classification_indices_data: list[ClassificationIndicesItem]
    ) -> None:
        self.classification_indices_data = classification_indices_data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item_dict = self.classification_indices_data[idx].dict()
        return {
            "token_ids": torch.tensor(item_dict["token_ids"], dtype=torch.int64),
            "answer_start": torch.tensor(item_dict["answer_start"], dtype=torch.int64),
            "answer_end": torch.tensor(item_dict["answer_end"], dtype=torch.int64),
            "attention_mask": torch.tensor(
                item_dict["attention_mask"], dtype=torch.int64
            ),
        }

    def __len__(self) -> int:
        return len(self.classification_indices_data)


def get_dataloader(
    preprocessed_data: list[ClassificationAnswerableItem]
    | list[ClassificationIndicesItem],
    task: Literal["classification_answerable", "classification_indices"],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """
    Creates the dataloader for the appropriate task.
    :param preprocessed_data: Preprocessed data for the task
    :param task: Task to get dataloader for (either "classification_answerable" or "classification_indices")
    :param batch_size: Dataloader batch_size
    :param shuffle: Whether to shuffle data
    :param num_workers: Datalaoder workers
    """
    datasets = {
        "classification_answerable": ClassificationAnswerableDataset,
        "classification_indices": ClassificationIndicesDataset,
    }
    return DataLoader(
        datasets[task](preprocessed_data),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
    )
