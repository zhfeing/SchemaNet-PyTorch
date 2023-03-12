import torch
import torch.nn as nn

from models.layers import get_activation_fn


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, identity_proj: bool = False):
        super().__init__()
        if identity_proj:
            assert in_dim == out_dim
        self.linear = nn.Identity() if identity_proj else nn.Linear(in_dim, out_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        if isinstance(self.linear, nn.Linear):
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.normal_(self.linear.bias)

    def forward(self, edges: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edges: shape: [bs, n, n]
            x: shape: [bs, n, dim]
        """
        # [bs, n, n] [bs, n, dim] -> [bs, n, dim]
        adj = edges + edges.transpose(1, 2)
        In = torch.zeros_like(adj)
        In.diagonal(dim1=1, dim2=2).fill_(1)
        feat = torch.bmm(adj / 2 + In, feat)
        return self.linear(feat)


class Layer(nn.Module):
    def __init__(self, emb_dim: int, activation: str, identity_proj: bool = False):
        super().__init__()
        self.g_conv = GraphConv(emb_dim, emb_dim, identity_proj)
        self.norm = nn.LayerNorm(emb_dim)
        self.activation = get_activation_fn(activation)

    def forward(self, edges: torch.Tensor, feat: torch.Tensor, feat_mask: torch.BoolTensor = None):
        feat = self.g_conv(edges, feat)
        if feat_mask is not None:
            feat.masked_fill_(feat_mask[..., None], 0)
        feat = self.activation(self.norm(feat))
        return feat


class GNN(nn.Module):
    def __init__(
        self,
        num_codes: int,
        embed_dim: int,
        num_layers: int,
        identity_proj: bool = False,
        activation: str = "relu"
    ):
        super().__init__()
        self.num_codes = num_codes
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=num_codes + 1,
            embedding_dim=embed_dim,
            padding_idx=num_codes
        )
        layers = [Layer(embed_dim, activation, identity_proj) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.trunc_normal_(self.embedding.weight[:self.num_codes])

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        ingredients: torch.LongTensor,
        feat_mask: torch.BoolTensor = None
    ):
        """
        Args:
            nodes: [bs, n]
            edges: [bs, n, n]
        """
        # code embedding [bs, n, dim]
        feat: torch.Tensor = self.embedding(ingredients)
        for layer in self.layers:
            feat = layer(edges, feat, feat_mask)
        # weighted average pooling
        feat = feat * nodes[..., None]
        feat = feat.mean(dim=1)
        feat = self.fc(feat)
        return feat
