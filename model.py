from typing import Optional
from dataclasses import dataclass
from mlx.utils import tree_flatten, tree_unflatten, tree_map
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear
from mlx.nn.layers.normalization import LayerNorm

import mlx.core as mx
import mlx.nn as nn
import math

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


class MultiHeadAttention(Module):
    """
    Minor update to the MultiHeadAttention module to ensure that the
    projections use bias.
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                f"The input feature dimensions should be divisible by the number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        self.query_proj = Linear(query_input_dims, dims, True)
        self.key_proj = Linear(key_input_dims, dims, True)
        self.value_proj = Linear(value_input_dims, value_dims, True)
        self.out_proj = Linear(value_dims, value_output_dims, True)

    def __call__(self, queries, keys, values, mask=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        # Dimensions are [batch x num heads x sequence x hidden dim]
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat)


class TransformerEncoderLayer(Module):
    """
    A transformer encoder layer with (the original BERT) post-normalization.
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        mlp_dims = mlp_dims or dims * 4
        self.attention = MultiHeadAttention(dims, num_heads)
        self.ln1 = LayerNorm(dims, eps=layer_norm_eps)
        self.ln2 = LayerNorm(dims, eps=layer_norm_eps)
        self.linear1 = Linear(dims, mlp_dims)
        self.linear2 = Linear(mlp_dims, dims)

    def __call__(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        add_and_norm = self.ln1(x + attention_out)

        ff = self.linear1(add_and_norm)
        ff_relu = mx.maximum(ff, 0)
        ff_out = self.linear2(ff_relu)
        x = self.ln2(ff_out + add_and_norm)

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
        for l in self.layers[:1]:
            x = l(x, mask)

        return x


class BertEmbeddings(nn.Module):
    def __init__(self, config: ModelArgs):
        self.word_embeddings = nn.Embedding(config.vocab_size, config.intermediate_size)
        self.token_type_embeddings = nn.Embedding(2, config.intermediate_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.intermediate_size
        )
        self.norm = nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(mx.array([[0, 1, 2]]))
        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = position + words + token_types
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
        return y
        # return mx.tanh(self.pooler(y[:, 0]))


def replace_key(key: str) -> str:
    key = key.replace(".layer.", ".layers.")
    key = key.replace(".self.key.", ".key_proj.")
    key = key.replace(".self.query.", ".query_proj.")
    key = key.replace(".self.value.", ".value_proj.")
    key = key.replace(".attention.output.dense.", ".attention.out_proj.")
    key = key.replace(".attention.output.LayerNorm.", ".ln1.")
    key = key.replace(".output.LayerNorm.", ".ln2.")
    key = key.replace(".intermediate.dense.", ".linear1.")
    key = key.replace(".output.dense.", ".linear2.")
    key = key.replace(".LayerNorm.", ".norm.")
    key = key.replace("pooler.dense.", "pooler.")
    return key


def test_correctly_loaded(torch_tensors: dict[str, numpy.array], m: BertModel) -> None:
    for i in range(12):
        keys = {
            f"encoder.layer.{i}.attention.self.query.weight": f"encoder.layers.{i}.attention.query_proj.weight",
            f"encoder.layer.{i}.attention.self.query.bias": f"encoder.layers.{i}.attention.query_proj.bias",
            f"encoder.layer.{i}.attention.self.key.weight": f"encoder.layers.{i}.attention.key_proj.weight",
            f"encoder.layer.{i}.attention.self.key.bias": f"encoder.layers.{i}.attention.key_proj.bias",
            f"encoder.layer.{i}.attention.self.value.weight": f"encoder.layers.{i}.attention.value_proj.weight",
            f"encoder.layer.{i}.attention.self.value.bias": f"encoder.layers.{i}.attention.value_proj.bias",
            f"encoder.layer.{i}.attention.output.dense.weight": f"encoder.layers.{i}.attention.out_proj.weight",
            f"encoder.layer.{i}.attention.output.dense.bias": f"encoder.layers.{i}.attention.out_proj.bias",
            f"encoder.layer.{i}.attention.output.LayerNorm.weight": f"encoder.layers.{i}.ln1.weight",
            f"encoder.layer.{i}.attention.output.LayerNorm.bias": f"encoder.layers.{i}.ln1.bias",
            f"encoder.layer.{i}.intermediate.dense.weight": f"encoder.layers.{i}.linear1.weight",
            f"encoder.layer.{i}.intermediate.dense.bias": f"encoder.layers.{i}.linear1.bias",
            f"encoder.layer.{i}.output.dense.weight": f"encoder.layers.{i}.linear2.weight",
            f"encoder.layer.{i}.output.dense.bias": f"encoder.layers.{i}.linear2.bias",
            f"encoder.layer.{i}.output.LayerNorm.weight": f"encoder.layers.{i}.ln2.weight",
            f"encoder.layer.{i}.output.LayerNorm.bias": f"encoder.layers.{i}.ln2.bias",
        }

        for orig, new in keys.items():
            assert numpy.allclose(torch_tensors[new], m.state_dict()[orig].numpy())


if __name__ == "__main__":
    model = Bert(ModelArgs())

    mlx_tensors = dict(
        [(key, array.shape) for key, array in tree_flatten(model.parameters())]
    )

    m = BertModel.from_pretrained("bert-base-uncased")
    m.eval()
    t = AutoTokenizer.from_pretrained("bert-base-uncased")

    torch_tensors = {
        replace_key(key): tensor.numpy() for key, tensor in m.state_dict().items()
    }

    # weights = mx.load("bert-base-uncased.npz")
    weights = tree_unflatten(list(torch_tensors.items()))
    weights = tree_map(lambda p: mx.array(p), weights)

    model.update(weights)

    tokens = t("test", return_tensors="np")
    tokens = {key: mx.array(v) for key, v in tokens.items()}

    embeddings = numpy.array(model(**tokens))

    torch_tokens = t("test", return_tensors="pt")
    torch_embeddings = m.embeddings(
        **{k: v for k, v in torch_tokens.items() if k != "attention_mask"}
    )

    first_attn_torch = (
        m.encoder.layer[0](torch_embeddings, torch_tokens["attention_mask"])[0]
        .detach()
        .numpy()
    )

    first_attn_mlx = numpy.array(model(**tokens))

    print(first_attn_mlx[0][0][:10])
    print(first_attn_torch[0][0][:10])

    # torch_output = m(**torch_tokens).last_hidden_state.detach().numpy()

    # print(embeddings[0, :10])
    # print(torch_embeddings)
    # print(torch_output[0, :10])
    # print(numpy.allclose(embeddings, torch_embeddings))
    # print(numpy.abs(embeddings - torch_embeddings).sum())
    # print(m(**torch_tokens).pooler_output.detach().numpy())
