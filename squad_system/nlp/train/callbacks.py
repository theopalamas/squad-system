import json
import os

import torch

from squad_system.nlp.train.config import TrainConfig
from squad_system.nlp.train.state_model import TrainerState


class CheckpointCallback:
    """
    Callback for saving the model, optimizer and config at the end of each epoch
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        os.makedirs(self.cfg.checkpoint_path, exist_ok=True)
        self.checkpoints_model_path = f"{self.cfg.checkpoint_path}/checkpoints_model"
        os.makedirs(self.checkpoints_model_path, exist_ok=True)
        self.checkpoints_optimizers_path = (
            f"{self.cfg.checkpoint_path}/checkpoints_optimizers"
        )
        os.makedirs(self.checkpoints_optimizers_path, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        trainer_state: TrainerState,
    ) -> None:
        model_file = (
            f"{self.checkpoints_model_path}/model_ckpt_{trainer_state.epochs_train}.pt"
        )
        optimizer_file = f"{self.checkpoints_optimizers_path}/optimizer_ckpt_{trainer_state.epochs_train}.pt"
        config_file = f"{self.cfg.checkpoint_path}/train_config.json"
        state_file = f"{self.cfg.checkpoint_path}/trainer_state.json"

        torch.save(model.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optimizer_file)
        with open(config_file, "w") as cfg_file:
            json.dump(self.cfg.model_dump(), cfg_file)
        with open(state_file, "w") as st_file:
            json.dump(trainer_state.model_dump(), st_file)
