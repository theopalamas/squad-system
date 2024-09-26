import json

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from squad_system.nlp.model.model import SquadModel
from squad_system.nlp.train.config import TrainConfig


def load_model(
    checkpoint_file: str, cfg_file: str, device: str
) -> [TrainConfig, Tokenizer, SquadModel]:
    """
    Loads the model and config for inference. Additionally creates the appropriate tokenizer
    :param checkpoint_file: Model checkpoint path saved from a training session
    :param cfg_file: Config file path saved from a training session
    :param device: Device to load to
    """
    model_state = torch.load(checkpoint_file, map_location=device, weights_only=True)
    with open(cfg_file, "r") as f:
        cfg = json.load(f)
    cfg = TrainConfig(**cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_lm_name, clean_up_tokenization_spaces=True
    )
    model = SquadModel(cfg)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return cfg, tokenizer, model
