from typing import Optional
from dataclasses import dataclass
from mlx.utils import tree_flatten, tree_unflatten, tree_map
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear
from mlx.nn.layers.normalization import LayerNorm

import mlx.core as mx
import mlx.nn as nn

from transformers import BertModel, AutoTokenizer
import torch
import numpy


@dataclass
class ModelArgs:
    intermediate_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    vocab_size: int = 30522
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 512


class TransformerEncoderLayer(Module):
    """
    A transformer encoder layer with (the original BERT) post-normalization.
    """

    def __init__(self, dims: int, num_heads: int, mlp_dims: Optional[int] = None):
        super().__init__()
        mlp_dims = mlp_dims or dims * 4
        self.attention = nn.MultiHeadAttention(dims, num_heads)
        self.ln1 = LayerNorm(dims)
        self.ln2 = LayerNorm(dims)
        self.linear1 = Linear(dims, mlp_dims)
        self.linear2 = Linear(mlp_dims, dims)

    def __call__(self, x, mask):
        y = self.attention(x, x, x, mask)
        y = self.ln1(x) + y

        y = self.linear1(y)
        y = mx.maximum(y, 0)
        y = self.linear2(y)
        x = self.ln2(x) + y

        return x


class TransformerEncoder(Module):
    def __init__(
        self, num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None
    ):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(dims, num_heads, mlp_dims)
            for i in range(num_layers)
        ]

    def __call__(self, x, mask):
        for l in self.layers:
            x = l(x, mask)

        return x


class BertEmbeddings(nn.Module):
    def __init__(self, config: ModelArgs):
        self.word_embeddings = nn.Embedding(config.vocab_size, config.intermediate_size)
        self.token_type_embeddings = nn.Embedding(2, config.intermediate_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.intermediate_size
        )
        self.norm = nn.LayerNorm(config.intermediate_size)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(mx.array([[0, 1, 2]]))
        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = position + words + token_types
        print(self.norm(embeddings))
        return self.norm(embeddings)


class Bert(nn.Module):
    def __init__(self, config: ModelArgs):
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(
            num_layers=config.num_hidden_layers,
            dims=config.intermediate_size,
            num_heads=config.num_attention_heads,
        )
        self.pooler = nn.Linear(config.intermediate_size, config.vocab_size)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        x = self.embeddings(input_ids, token_type_ids)
        y = self.encoder(x, attention_mask)
        return mx.tanh(self.pooler(y[:, 0]))


def replace_key(key: str) -> str:
    key = key.replace(".layer.", ".layers.")
    key = key.replace(".self.key.", ".key_proj.")
    key = key.replace(".self.query.", ".query_proj.")
    key = key.replace(".self.value.", ".value_proj.")
    key = key.replace(".attention.output.LayerNorm.", ".ln1.")
    key = key.replace(".output.LayerNorm.", ".ln2.")
    key = key.replace(".intermediate.dense.", ".linear1.")
    key = key.replace(".output.dense.", ".linear2.")
    key = key.replace(".attention.linear2.", ".attention.out_proj.")
    key = key.replace(".LayerNorm.", ".norm.")
    key = key.replace("pooler.dense.", "pooler.")
    return key


if __name__ == "__main__":
    model = Bert(ModelArgs())

    mlx_tensors = dict(
        [(key, array.shape) for key, array in tree_flatten(model.parameters())]
    )

    m = BertModel.from_pretrained("bert-base-uncased")
    t = AutoTokenizer.from_pretrained("bert-base-uncased")

    torch_tensors = {
        replace_key(key): tensor.numpy()
        for key, tensor in m.state_dict().items()
    }

    # weights = mx.load("bert-base-uncased.npz")
    weights = tree_unflatten(list(torch_tensors.items()))
    weights = tree_map(lambda p: mx.array(p), weights)

    model.update(weights)

    tokens = t("test", return_tensors="np")
    tokens = {key: mx.array(v) for key, v in tokens.items()}

    numpy.array(model(**tokens))

    torch_tokens = t("test", return_tensors="pt")
    print(m.embeddings(**{k: v for k, v in torch_tokens.items() if k != "attention_mask"}))
    #print(m(**torch_tokens).pooler_output.detach().numpy())
