import torch
from torch import nn
from .embedding import EmbeddingLayer
from .interaction import InnerProductLayer

class LR_Layer(nn.Module):
    def __init__(self, feature_map, output_activation=None, use_bias=True):
        super(LR_Layer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True) if use_bias else None
        self.output_activation = output_activation
        # A trick for quick one-hot encoding in LR
        self.embedding_layer = EmbeddingLayer(feature_map, 1, use_pretrain=False)

    def forward(self, X):
        embed_weights = self.embedding_layer(X)
        output = embed_weights.sum(dim=1)
        if self.bias is not None:
            output += self.bias
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output

class FM_Layer(nn.Module):
    def __init__(self, feature_map, output_activation=None, use_bias=True):
        super(FM_Layer, self).__init__()
        self.inner_product_layer = InnerProductLayer(feature_map.num_fields, output="product_sum_pooling")
        self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=use_bias)
        self.output_activation = output_activation

    def forward(self, X, feature_emb):
        lr_out = self.lr_layer(X)
        dot_sum = self.inner_product_layer(feature_emb)
        output = dot_sum + lr_out
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output