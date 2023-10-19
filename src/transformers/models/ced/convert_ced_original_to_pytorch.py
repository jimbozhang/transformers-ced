# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert CED checkpoints from the original repository. URL: https://github.com/RicherMans/CED"""


import argparse
from pathlib import Path

import torch

from transformers import CedConfig, CedForAudioClassification
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def create_ced_config(model_name):
    config = CedConfig()

    if model_name == "ced-tiny":
        config.patch_size = 16
        config.embed_dim = 192
        config.depth = 12
        config.num_heads = 3
        config.mlp_ratio = 4
    elif model_name == "ced-mini":
        config.patch_size = 16
        config.embed_dim = 256
        config.depth = 12
        config.num_heads = 4
        config.mlp_ratio = 4
    elif model_name == "ced-small":
        config.patch_size = 16
        config.embed_dim = 384
        config.depth = 12
        config.num_heads = 6
        config.mlp_ratio = 4
    elif model_name == "ced-base":
        config.patch_size = 16
        config.embed_dim = 768
        config.depth = 12
        config.num_heads = 12
        config.mlp_ratio = 4
    else:
        raise NotImplementedError(f"Model not supported: {model_name}")

    return config


def remove_keys(state_dict):
    ignore_keys = []
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(name):
    name = name.replace("init_bn.1", "init_bn")
    if not name.startswith("outputlayer"):
        name = f"ced_model.{name}"
    return name


@torch.no_grad()
def convert_ced_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our CED structure.
    """
    config = create_ced_config(model_name)

    model_name_to_url = {
        "ced-tiny": ("https://zenodo.org/record/8275347/files/audiotransformer_tiny_mAP_4814.pt?download=1"),
        "ced-mini": ("https://zenodo.org/record/8275347/files/audiotransformer_mini_mAP_4896.pt?download=1"),
        "ced-small": ("https://zenodo.org/record/8275319/files/audiotransformer_small_mAP_4958.pt?download=1"),
        "ced-base": ("https://zenodo.org/record/8275347/files/audiotransformer_base_mAP_4999.pt?download=1"),
    }

    state_dict = torch.hub.load_state_dict_from_url(model_name_to_url[model_name], map_location="cpu")
    remove_keys(state_dict)
    new_state_dict = {rename_key(key): val for key, val in state_dict.items()}

    model = CedForAudioClassification(config)
    model.load_state_dict(new_state_dict)

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        logger.info(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

        # test
        model_test = CedForAudioClassification(config).from_pretrained(pytorch_dump_folder_path)
        pass

    if push_to_hub:
        logger.info("Pushing model and feature extractor to the hub...")
        model.push_to_hub(f"xiaomi/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="ced_mini",
        type=str,
        help="Name of the CED model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_ced_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
