import torch
from datetime import datetime
from pathlib import Path


def find_best_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def create_path(save_path):
    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True, exist_ok=True)


def cur_time():
    return (datetime.now()).strftime("%b-%d %H:%M")
