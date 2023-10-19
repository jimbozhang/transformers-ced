# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch CED (Ced) model. """

import inspect
import unittest

from huggingface_hub import hf_hub_download

from transformers import CedConfig
from transformers.testing_utils import require_torch, require_torchaudio, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_torchaudio_available

from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import CedForAudioClassification, CedModel
    from transformers.models.ced.modeling_ced import (
        CED_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


if is_torchaudio_available():
    import torchaudio

    from transformers import ASTFeatureExtractor


class CedModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        patch_size=2,
        max_length=24,
        num_mel_bins=16,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        scope=None,
        frequency_stride=2,
        time_stride=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.max_length = max_length
        self.num_mel_bins = num_mel_bins
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope
        self.frequency_stride = frequency_stride
        self.time_stride = time_stride

        # in Ced, the seq length equals the number of patches + 2 (we add 2 for the [CLS] and distillation tokens)
        frequency_out_dimension = (self.num_mel_bins - self.patch_size) // self.frequency_stride + 1
        time_out_dimension = (self.max_length - self.patch_size) // self.time_stride + 1
        num_patches = frequency_out_dimension * time_out_dimension
        self.seq_length = num_patches + 2

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.max_length, self.num_mel_bins])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, input_values, labels

    def get_config(self):
        return CedConfig(
            patch_size=self.patch_size,
            max_length=self.max_length,
            num_mel_bins=self.num_mel_bins,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            frequency_stride=self.frequency_stride,
            time_stride=self.time_stride,
        )

    def create_and_check_model(self, config, input_values, labels):
        model = CedModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict


@require_torch
class CedModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Ced does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (
            CedModel,
            CedForAudioClassification,
        )
        if is_torch_available()
        else ()
    )
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    @unittest.skip(reason="Ced does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in CED_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CedModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on some audio from AudioSet
def prepare_audio():
    filepath = hf_hub_download(
        repo_id="nielsr/audio-spectogram-transformer-checkpoint", filename="sample_audio.flac", repo_type="dataset"
    )

    audio, sampling_rate = torchaudio.load(filepath)

    return audio, sampling_rate


@require_torch
@require_torchaudio
class CedModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return ASTFeatureExtractor.from_pretrained("xiaomi/ced-tiny") if is_torchaudio_available() else None

    @slow
    def test_inference_audio_classification(self):
        feature_extractor = self.default_feature_extractor
        model = CedForAudioClassification.from_pretrained("xiaomi/ced-tiny").to(torch_device)

        feature_extractor = self.default_feature_extractor
        audio, sampling_rate = prepare_audio()
        audio = audio.squeeze().numpy()
        inputs = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 527))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-0.8760, -7.0042, -8.6602]).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))
