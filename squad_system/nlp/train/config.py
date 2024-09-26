from typing import Literal

import yaml
from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    seed: int = Field(
        default=42,
        description="Seed for random process initialization. Setting a seed leads to reproducible results.",
    )
    task: Literal["classification_answerable", "classification_indices"] = Field(
        description="Task for model training."
    )
    epochs: int = Field(default=5, gt=0, description="Training epochs.")
    batch_size: int = Field(default=16, gt=0, description="Batch size.")
    num_workers: int = Field(
        default=0, ge=0, description="Dataloader number of workers."
    )
    pretrained_lm_name: str = Field(
        default="distilbert-base-uncased",
        description="Language model used for tokenization and training.",
    )
    downstream_head_in_size: int = Field(
        default=512, description="Dense layer size used for the downstream task."
    )
    downstream_head_out_size: int = Field(
        description="Output size for downstream head."
    )
    lm_dropout_prob: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Dropout probabilities for the last language model layer.",
    )
    b1: float = Field(default=0.9, description="Beta1 for AdamW optimizer.")
    b2: float = Field(default=0.999, description="Beta2 for AdamW optimizer.")
    eps: float = Field(default=1e-8, description="Epsilon for AdamW optimizer.")
    learning_rate: float = Field(default=3e-5, description="Learning rate.")
    checkpoint_path: str = Field(
        default="out", description="Output path for storing checkpoints."
    )


def load_config(cfg_path: str, checkpoint_path: str) -> TrainConfig:
    """
    Loads training config from yml file
    :param cfg_path: yml config file path
    :param checkpoint_path: Output path for saving model checkpoints
    :return:
    """
    with open(cfg_path, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    cfg_nlp = {}
    cfg_train = cfg.get("train", {})
    cfg_model = cfg.get("model", {})
    cfg_optim = cfg.get("optimizer", {})
    cfg_task_params = cfg_model.pop("task_params", {})

    cfg_nlp.update(cfg_train)
    cfg_nlp.update(cfg_model)
    cfg_nlp.update(cfg_task_params[cfg_train["task"]])
    cfg_nlp.update(cfg_optim)
    cfg_nlp["checkpoint_path"] = checkpoint_path

    return TrainConfig(**cfg_nlp)
