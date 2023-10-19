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


from typing import Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CED_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xiaomi/ced-tiny": "https://huggingface.co/xiaomi/ced-tiny/resolve/main/config.json",
}


class CedConfig(PretrainedConfig):
    r"""
    TODO: Add docstring
    ```"""

    def __init__(
        self,
        name="ced-tiny",
        outputdim=527,
        patch_size=16,
        patch_stride=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=None,
        target_length=1012,
        pooling="mean",
        time_patch_out: Optional[float] = None,
        freq_patch_out: Optional[float] = None,
        eval_avg="mean",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if name == "ced-tiny":
            patch_size = 16
            embed_dim = 192
            depth = 12
            num_heads = 3
            mlp_ratio = 4
        elif name == "ced-mini":
            patch_size = 16
            embed_dim = 256
            depth = 12
            num_heads = 4
            mlp_ratio = 4
        elif name == "ced-small":
            patch_size = 16
            embed_dim = 384
            depth = 12
            num_heads = 6
            mlp_ratio = 4
        elif name == "ced-base":
            patch_size = 16
            embed_dim = 768
            depth = 12
            num_heads = 12
            mlp_ratio = 4
        else:
            logger.warning(f"Unknown model name {name}. Using user custom config.")

        assert pooling in ("mean", "token", "dm", "logit")
        self.outputdim = outputdim
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.n_mels = kwargs.get("n_mels", 64)
        self.n_fft = kwargs.get("n_fft", 512)
        self.hop_size = kwargs.get("hop_size", 160)
        self.win_size = kwargs.get("win_size", 512)
        self.f_min = kwargs.get("f_min", 0)
        self.f_max = kwargs.get("f_max", 8000)
        self.center = kwargs.get("center", True)
        self.pad_last = kwargs.get("pad_last", True)
        self.eval_avg = eval_avg
        self.time_patch_out = time_patch_out
        self.freq_patch_out = freq_patch_out
        self.target_length = target_length
        self.n_mels = kwargs.get("n_mels", 64)
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop_rate
        self.init_values = init_values
