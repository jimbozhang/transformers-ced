# coding=utf-8
# Copyright 2023 Google AI and The HuggingFace Inc. team. All rights reserved.
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
""" Audio Spectogram Transformer (Ced) model configuration"""


from ...configuration_utils import PretrainedConfig
from ...utils import logging

from typing import Optional


logger = logging.get_logger(__name__)

CED_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xiaomi/ced-tiny": "https://huggingface.co/xiaomi/ced-tiny/resolve/main/config.json",
}


class CedConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CedModel`]. It is used to instantiate an Ced
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Ced
    [xiaomi/ced-tiny](https://huggingface.co/xiaomi/ced-tiny)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        patch_size (`int`, *optional*, defaults to `16`):
            The size (resolution) of each patch.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        frequency_stride (`int`, *optional*, defaults to 10):
            Frequency stride to use when patchifying the spectrograms.
        time_stride (`int`, *optional*, defaults to 10):
            Temporal stride to use when patchifying the spectrograms.
        max_length (`int`, *optional*, defaults to 1024):
            Temporal dimension of the spectrograms.
        num_mel_bins (`int`, *optional*, defaults to 128):
            Frequency dimension of the spectrograms (number of Mel-frequency bins).

    Example:

    ```python
    >>> from transformers import CedConfig, CedModel

    >>> # Initializing a Ced xiaomi/ced-tiny style configuration
    >>> configuration = CedConfig()

    >>> # Initializing a model (with random weights) from the xiaomi/ced-tiny style configuration
    >>> model = CedModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "ced"

    def __init__(
        self,
        outputdim=527,
        patch_size=16,
        patch_stride=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        init_bn: bool = True,
        norm_layer=None,
        act_layer=None,
        init_values=None,
        target_length=1012,
        pooling='mean',
        wavtransforms=None,
        spectransforms=None,
        time_patch_out: Optional[float] = None,
        freq_patch_out: Optional[float] = None,
        eval_avg='mean',
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert pooling in ('mean', 'token', 'dm', 'logit')
        self.outputdim = outputdim
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.n_mels = kwargs.get('n_mels', 64)
        self.n_fft = kwargs.get('n_fft', 512)
        self.hop_size = kwargs.get('hop_size', 160)
        self.win_size = kwargs.get('win_size', 512)
        self.f_min = kwargs.get('f_min', 0)
        self.f_max = kwargs.get('f_max', 8000)
        self.center = kwargs.get('center', True)
        self.pad_last = kwargs.get('pad_last', True)
        self.eval_avg = eval_avg
        self.time_patch_out = time_patch_out
        self.freq_patch_out = freq_patch_out
        self.target_length = target_length
        self.n_mels = kwargs.get('n_mels', 64)
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop_rate
        self.init_values = init_values
