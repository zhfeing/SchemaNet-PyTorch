from typing import Any, Dict

from models.layers import get_patch_embedding, get_pos_encoding
from models.transformer import Transformer
from .vit import ViT
from .deit import DeiT


def get_vit(model_cfg: Dict[str, Any], num_classes: int) -> ViT:
    embed_dim = model_cfg["transformer"]["embed_dim"]

    patch_embed = get_patch_embedding(embed_dim=embed_dim, **model_cfg["patch_embed"])
    pos_encoding = get_pos_encoding(
        num_tokens=patch_embed.num_patches + 1,
        embed_dim=embed_dim,
        **model_cfg["pos_encoding"]
    )
    transformer = Transformer(**model_cfg["transformer"])
    vit = ViT(patch_embed, pos_encoding, transformer, num_classes)
    return vit


def get_deit(model_cfg: Dict[str, Any], num_classes: int) -> DeiT:
    embed_dim = model_cfg["transformer"]["embed_dim"]

    patch_embed = get_patch_embedding(embed_dim=embed_dim, **model_cfg["patch_embed"])
    pos_encoding = get_pos_encoding(
        num_tokens=patch_embed.num_patches + 2,
        embed_dim=embed_dim,
        **model_cfg["pos_encoding"]
    )
    transformer = Transformer(**model_cfg["transformer"])
    deit = DeiT(patch_embed, pos_encoding, transformer, num_classes)
    return deit
