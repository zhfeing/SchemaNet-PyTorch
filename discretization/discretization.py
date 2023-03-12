from typing import Tuple

import logging

import torch
import torch.nn as nn


class Discretization(nn.Module):
    def __init__(
        self,
        size: int,
        dim: int,
        detach_input_seq: bool = True,
        uniform_range: Tuple[float, float] = [-1, 1]
    ):
        """
        Feature discretization with pre-defined visual vocabulary set (a.k.a., codebook)
        Args:
            size: vocabulary size
            dim: visual word dimension
            detach_input_seq: if `True`, input sequence for encoding will be detached
        """
        super().__init__()
        self.logger = logging.getLogger("discretization")
        self.size = size
        self.dim = dim
        self.detach_input_seq = detach_input_seq

        self.logger.info("Creating discretization with size: %d, dimension: %d", size, dim)

        self.vocabulary = nn.Embedding(size, dim)
        self._reset_parameters(uniform_range)
        self.activate()

    def _reset_parameters(self, uniform_range: Tuple[float, float]):
        self.logger.info("Initializing with Uniform[%.2f, %.2f]", uniform_range[0], uniform_range[1])
        nn.init.uniform_(self.vocabulary.weight, uniform_range[0], uniform_range[1])

    def initial_vocabulary(self, vocabulary_fp: str):
        self.logger.info("Loading from external vocabulary...")
        vocabulary: torch.Tensor = torch.load(vocabulary_fp, map_location="cpu")
        if vocabulary.shape[0] > self.size:
            self.logger.warning("Too much external vocabulary, using random picked vocabulary...")
            rand_perm = torch.randperm(vocabulary.shape[0])
            vocabulary = vocabulary[rand_perm][:self.size]
        with torch.no_grad():
            self.vocabulary.weight.copy_(vocabulary)

    def deactivate(self):
        self.logger.debug("Deactivated discretization!")
        self._activate = False

    def activate(self):
        self.logger.debug("Activated discretization!")
        self._activate = True

    def encode(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        if self.detach_input_seq:
            seq = seq.detach()
        n, bs = seq.shape[:2]
        # [n, bs, dim] -> [n * bs, dim]
        seq = seq.reshape(n * bs, self.dim)
        # [n * bs, self.size] distance matrix
        ingredients = torch.cdist(seq, self.vocabulary.weight).argmin(dim=1)
        if self._activate:
            seq = self.vocabulary(ingredients)
        seq = seq.reshape(n, bs, self.dim)
        ingredients = ingredients.reshape(n, bs)
        return seq, ingredients

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Args:
            seq: [n, bs, dim]
        Return:
            encoded sequence [n, bs, dim], and matched ingredients for each input token [n, bs]
        """
        t = seq.shape[2]
        assert int(t) == self.dim, f"dimension {seq.shape[2]} not match to {self.dim}"
        return self.encode(seq)


