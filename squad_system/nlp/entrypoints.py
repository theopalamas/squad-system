import json
import os
from typing import Literal

from tqdm import tqdm

from squad_system.nlp.dataset.io import (
    load_data_preprocessed,
    load_data_raw,
    load_data_raw_for_inference,
    save_data_preprocessed,
)
from squad_system.nlp.dataset.preprocess import (
    preprocess_for_classification_answerable,
    preprocess_for_classification_indices,
)
from squad_system.nlp.inference.pipeline import get_inference_pipeline
from squad_system.nlp.train.config import load_config
from squad_system.nlp.train.trainer import get_trainer
from squad_system.nlp.utils import get_device, seed_all


def preprocess_data(
    data_path_train: str,
    data_path_val: str,
    preprocessed_data_path: str,
    pretrained_lm_name: str,
    task: Literal["classification_answerable", "classification_indices"],
    allow_unanswerable: bool = True,
) -> None:
    """
    Preprocess entrypoint. This is used to load and preprocess the Squad data
    :param data_path_train: The train raw squad file
    :param data_path_val: The dev raw squad file
    :param preprocessed_data_path: Output path for saving the perprocessed data
    :param pretrained_lm_name: The language model used for tokenization (must be from transformers library)
    :param task: Which task to preprocess data for
    :param allow_unanswerable: If preprocessing for 'classification_indices', should we allow unanswerable questions in the dataset?
    """
    data_raw_train = load_data_raw(data_path_train)
    data_raw_val = load_data_raw(data_path_val)

    if task == "classification_answerable":
        preprocessed_data_train = preprocess_for_classification_answerable(
            data_raw_train, pretrained_lm_name
        )
        preprocessed_data_val = preprocess_for_classification_answerable(
            data_raw_val, pretrained_lm_name
        )
    if task == "classification_indices":
        preprocessed_data_train = preprocess_for_classification_indices(
            data_raw_train, pretrained_lm_name, allow_unanswerable
        )
        preprocessed_data_val = preprocess_for_classification_indices(
            data_raw_val, pretrained_lm_name, allow_unanswerable
        )

    save_data_preprocessed(
        preprocessed_data_train, f"{preprocessed_data_path}/train.pickle"
    )
    save_data_preprocessed(
        preprocessed_data_val, f"{preprocessed_data_path}/val.pickle"
    )


def train(cfg_path: str, preprocessed_data_path: str, checkpoint_path: str) -> None:
    """
    Training entrypoint. This is used to train the models
    :param cfg_path: yml configuration for training
    :param preprocessed_data_path: Preprocesse data path
    :param checkpoint_path: Output path for saving model checkpoints
    """
    cfg = load_config(cfg_path, checkpoint_path)
    seed_all(cfg.seed)

    preprocessed_data_train = load_data_preprocessed(
        f"{preprocessed_data_path}/train.pickle"
    )
    preprocessed_data_val = load_data_preprocessed(
        f"{preprocessed_data_path}/val.pickle"
    )

    trainer = get_trainer(cfg, preprocessed_data_train, preprocessed_data_val)
    trainer.train()


def infer(
    data_path: str,
    checkpoint_file_answerable: str,
    cfg_file_answerable: str,
    checkpoint_file_indices: str,
    cfg_file_indices: str,
    results_file_path: str,
) -> None:
    """
    Inference entrypoint. This is used to make batch predictions on the dev split
    of the SQUAD2.0 dataset or any other dataset with the same format.
    :param data_path: Dataset for prediction
    :param checkpoint_file_answerable: Model path for 'classification_answerable' model
    :param cfg_file_answerable: Config file for 'classification_answerable' model
    :param checkpoint_file_indices: Model path for 'classification_indices' model
    :param cfg_file_indices: Config file for 'classification_indices' model
    :param results_file_path: Prediction output file
    """
    device = get_device()
    inference_pipeline = get_inference_pipeline(
        checkpoint_file_answerable,
        cfg_file_answerable,
        checkpoint_file_indices,
        cfg_file_indices,
        device,
    )

    data_raw = load_data_raw_for_inference(data_path)

    predicted_data = {}

    for data_raw_sample in tqdm(data_raw, desc="Inferring..."):
        answer = inference_pipeline.infer(
            data_raw_sample["question"], data_raw_sample["context"]
        )
        predicted_data[data_raw_sample["id"]] = answer

    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    with open(results_file_path, "w") as results_file:
        json.dump(predicted_data, results_file)
