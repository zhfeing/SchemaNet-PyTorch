import torch.nn as nn


def get_dropout(drop_prob: float = None):
    return nn.Dropout(drop_prob) if drop_prob is not None else nn.Identity()
