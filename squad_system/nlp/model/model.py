import torch
import torch.nn as nn
from transformers import AutoModel

from squad_system.nlp.train.config import TrainConfig


class SquadModel(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.language_model = AutoModel.from_pretrained(cfg.pretrained_lm_name)
        self.language_model_dropout = nn.Dropout(cfg.lm_dropout_prob)
        self.downstream_head = nn.Linear(
            cfg.downstream_head_in_size, cfg.downstream_head_out_size
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        language_model_out = self.language_model(
            input_ids=x, attention_mask=attention_mask
        )
        if self.cfg.task == "classification_answerable":
            language_model_last_hidden_state = self.language_model_dropout(
                language_model_out.last_hidden_state[:, 0, :]
            )
        if self.cfg.task == "classification_indices":
            language_model_last_hidden_state = self.language_model_dropout(
                language_model_out.last_hidden_state
            )
        logits = self.downstream_head(language_model_last_hidden_state)
        return logits
