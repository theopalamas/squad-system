import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """
    Sets seed for all packages for reroducibility reasons
    :param seed: seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def get_device() -> str:
    """
    Decides on which device to run the models, giving precedence to gpu if available
    :return: The device string (cpu, cuda)
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
