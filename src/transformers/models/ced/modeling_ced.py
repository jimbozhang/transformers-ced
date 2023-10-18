# coding=utf-8
# Copyright 2023 MIT and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch CED (Ced) model."""

import collections
import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ced import CedConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "CedConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "xiaomi/ced-tiny"
_EXPECTED_OUTPUT_SHAPE = [1, 1214, 768]

# Audio classification docstring
_SEQ_CLASS_CHECKPOINT = "xiaomi/ced-tiny"
_SEQ_CLASS_EXPECTED_OUTPUT = "'Speech'"
_SEQ_CLASS_EXPECTED_LOSS = 0.17


CED_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xiaomi/ced-tiny",
    # See all CED models at https://huggingface.co/models?filter=ced
]


# Copied from
# transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer.ASTEmbeddings
# with AST->Ced
class CedEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config: CedConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = CedPatchEmbeddings(config)

        frequency_out_dimension, time_out_dimension = self.get_shape(config)
        num_patches = frequency_out_dimension * time_out_dimension
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 2, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def get_shape(self, config):
        # see Karpathy's cs231n blog on how to calculate the output dimensions
        # https://cs231n.github.io/convolutional-networks/#conv
        frequency_out_dimension = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
        time_out_dimension = (config.max_length - config.patch_size) // config.time_stride + 1

        return frequency_out_dimension, time_out_dimension

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        batch_size = input_values.shape[0]
        embeddings = self.patch_embeddings(input_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


# Copied from
# transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer.ASTPatchEmbeddings
# with AST->Ced
class CedPatchEmbeddings(nn.Module):
    """
    This class turns `input_values` into the initial `hidden_states` (patch embeddings) of shape `(batch_size,
    seq_length, hidden_size)` to be consumed by a Transformer.
    """

    def __init__(self, config):
        super().__init__()

        patch_size = config.patch_size
        frequency_stride = config.frequency_stride
        time_stride = config.time_stride

        self.projection = nn.Conv2d(
            1, config.hidden_size, kernel_size=(patch_size, patch_size), stride=(frequency_stride, time_stride)
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        input_values = input_values.unsqueeze(1)
        input_values = input_values.transpose(2, 3)
        embeddings = self.projection(input_values).flatten(2).transpose(1, 2)
        return embeddings


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->Ced
class CedSelfAttention(nn.Module):
    def __init__(self, config: CedConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->Ced
class CedSelfOutput(nn.Module):
    """
    The residual connection is defined in CedLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: CedConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->Ced
class CedAttention(nn.Module):
    def __init__(self, config: CedConfig) -> None:
        super().__init__()
        self.attention = CedSelfAttention(config)
        self.output = CedSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->Ced
class CedIntermediate(nn.Module):
    def __init__(self, config: CedConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->Ced
class CedOutput(nn.Module):
    def __init__(self, config: CedConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->Ced
class CedLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: CedConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = CedAttention(config)
        self.intermediate = CedIntermediate(config)
        self.output = CedOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in Ced, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in Ced, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->Ced
class CedEncoder(nn.Module):
    def __init__(self, config: CedConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([CedLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from
# transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer.ASTPreTrainedModel
# with AST->Ced,audio_spectrogram_transformer->ced
class CedPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CedConfig
    base_model_prefix = "ced"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module: CedEncoder, value: bool = False) -> None:
        if isinstance(module, CedEncoder):
            module.gradient_checkpointing = value


CED_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CedConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CED_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, max_length, num_mel_bins)`):
            Float values mel features extracted from the raw audio waveform. Raw audio waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `torch.FloatTensor`. See [`~ASTFeatureExtractor.__call__`]

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

Conv_Kernel = Union[int, Tuple[int, int]]


def to_2tuple(x: Any) -> Tuple[Any, Any]:
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class AudioPatchEmbed(nn.Module):
    def __init__(self,
                 input_size: Conv_Kernel = 224,
                 patch_size: Conv_Kernel = 16,
                 patch_stride: Conv_Kernel = 16,
                 in_chans: int = 1,
                 embed_dim: int = 768,
                 norm_layer: Optional[Callable] = None,
                 flatten: bool = False):
        super().__init__()
        self.input_size = to_2tuple(input_size)
        self.patch_size = to_2tuple(patch_size)
        self.patch_stride = to_2tuple(patch_stride)
        self.grid_size = (self.input_size[0] // self.patch_stride[0],
                          self.input_size[1] // self.patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, 'b c f t -> b (f t) c')
        x = self.norm(x)
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        causal: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.causal = causal

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # if mask is not None:
        # # Mask is a tensor of shape [B, T, T]
        # # Different from self.causal == True, the mask might be something like:
        # # [False, False, True]
        # # [False, False, True]
        # # [True, True, True]
        # # We use -inf to pad here, since if we would pad by any number, the entries at rows only containing
        # # [True, True, True] would lead to weights such as: [0.33,0.33,0.33], which is not correct
        # mask_value = torch.as_tensor(-float('inf'))
        # print(mask.shape, attn.shape)
        # attn = attn.masked_fill(mask, mask_value)
        if self.causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]
            mask = torch.ones(i, j, device=q.device,
                              dtype=torch.bool).triu(j - i + 1)
            attn = attn.masked_fill(mask, mask_value)
        attn = attn.softmax(dim=-1)
        # Only for the case that a mask with all True entries on a row is passed.
        # attn = torch.nan_to_num(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: Callable = nn.GELU,
                 drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        attention_type: Callable = Attention,
        attention_kwargs={},
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attention_type(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   attn_drop=attn_drop,
                                   proj_drop=drop,
                                   **attention_kwargs)
        self.ls1 = nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)
        self.ls2 = nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# Taken from timm
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


@add_start_docstrings(
    "The bare Ced Model transformer outputting raw hidden-states without any specific head on top.",
    CED_START_DOCSTRING,
)
# Copied from
# transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer.ASTModel
# with AST->Ced,AUDIO_SPECTROGRAM_TRANSFORMER->CED
class CedModel(CedPreTrainedModel):
    def __init__(self, config: CedConfig) -> None:
        super().__init__(config)
        self.config = config

        # Allowed length in number of frames, otherwise the positional embedding will throw an error
        self.maximal_allowed_length = self.config.target_length

        self.init_bn = torch.nn.BatchNorm2d(config.n_mels, momentum=0.01)

        self.patch_embed = AudioPatchEmbed(input_size=(config.n_mels,
                                                       config.target_length),
                                           embed_dim=config.embed_dim,
                                           patch_size=config.patch_size,
                                           flatten=False,
                                           patch_stride=config.patch_stride)

        self.time_pos_embed = nn.Parameter(
            torch.randn(1, config.embed_dim, 1, self.patch_embed.grid_size[1]) * .02)
        self.freq_pos_embed = nn.Parameter(
            torch.randn(1, config.embed_dim, self.patch_embed.grid_size[0], 1) * .02)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)
               ]  # stochastic depth decay rule
        self.pos_drop = nn.Dropout(p=config.drop_rate)
        self.blocks = nn.Sequential(*[
            Block(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                init_values=config.init_values,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                attention_type=Attention,
            ) for i in range(config.depth)
        ])
        self.norm = norm_layer(config.embed_dim)
        self.outputlayer = nn.Sequential(nn.LayerNorm(config.embed_dim),
                                         nn.Linear(config.embed_dim, config.outputdim))
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_input_embeddings(self) -> CedPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        b, c, f, t = x.shape
        x = x + self.time_pos_embed[:, :, :, :t]
        x = x + self.freq_pos_embed[:, :, :, :]  # Just to support __getitem__ in posembed

        # x = rearrange(x, 'b c f t -> b (f t) c')
        x = torch.permute(torch.flatten(x, 2, 3), (0, 2, 1))

        if self.config.pooling == 'token':
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.token_pos_embed
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == 'token':
            x = x[:, 0]
            return self.outputlayer(x).sigmoid()
        elif self.config.pooling == 'mean':
            x = x.mean(1)
            return self.outputlayer(x).sigmoid()
        elif self.config.pooling == 'logit':
            x = x.mean(1)
            return self.outputlayer(x)
        elif self.config.pooling == 'dm':
            # Unpack using the frequency dimension, which is constant
            # 'b (f t) d -> b f t d', f=self.patch_embed.grid_size[0])
            x = torch.reshape(x, (x.shape[0], self.patch_embed.grid_size[0], -1, x.shape[3]))

            # First poolin frequency, then sigmoid the (B T D) output
            x = self.outputlayer(x.mean(1)).sigmoid()
            return x.mean(1)
        else:
            return x.mean(1)

    def forward_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.unsqueeze(x, 1)

        x = torch.permute(x, (0, 2, 1, 3))
        x = self.init_bn(x)
        x = torch.permute(x, (0, 2, 1, 3))

        if x.shape[-1] > self.maximal_allowed_length:
            splits = x.split(self.target_length, -1)

            if splits[-1].shape[-1] < self.target_length:
                if self.pad_last:
                    pad = torch.zeros(*x.shape[:-1],
                                      self.target_length,
                                      device=x.device)
                    pad[..., :splits[-1].shape[-1]] = splits[-1]
                    splits = torch.stack((*splits[:-1], pad), dim=0)
                else:
                    splits = torch.stack(splits[:-1], dim=0)
            else:
                splits = torch.stack(splits[:-1], dim=0)
            n_splits = len(splits)
            x = torch.flatten(splits, 0, 1)  # spl b c f t-> (spl b) c f t
            x = self.forward_head(self.forward_features(x))
            x = torch.reshape(x, (n_splits, -1, self.outputdim))  # (spl b) d -> spl b d, spl=n_splits

            if self.eval_avg == 'mean':
                x = x.mean(0)
            elif self.eval_avg == 'max':
                x = x.max(0)[0]
            else:
                raise ValueError(
                    f'Unknown Eval average function ({self.eval_avg})')

        else:
            x = self.forward_features(x)
            x = self.forward_head(x)
        return x

    def forward(self, x: torch.Tensor):
        return self.forward_spectrogram(x)

    @add_start_docstrings_to_model_forward(CED_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def _forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_values is None:
            raise ValueError("You have to specify input_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(input_values)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        pooled_output = (sequence_output[:, 0] + sequence_output[:, 1]) / 2

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Copied from
# transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer.ASTMLPHead
# with AST->Ced
class CedMLPHead(nn.Module):
    def __init__(self, config: CedConfig):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

    def forward(self, hidden_state):
        hidden_state = self.layernorm(hidden_state)
        hidden_state = self.dense(hidden_state)
        return hidden_state


@add_start_docstrings(
    """
    CED model with an audio classification head on top (a linear layer on top of the pooled
    output) e.g. for datasets like AudioSet, Speech Commands v2.
    """,
    CED_START_DOCSTRING,
)
# Copied from
# transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer.ASTForAudioClassification
# with
# AST->Ced,AUDIO_SPECTROGRAM_TRANSFORMER->CED,audio_spectrogram_transformer->ced
class CedForAudioClassification(CedPreTrainedModel):
    def __init__(self, config: CedConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.ced = CedModel(config)

        # Classifier head
        self.classifier = CedMLPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CED_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_SEQ_CLASS_CHECKPOINT,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the audio classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ced(
            input_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
