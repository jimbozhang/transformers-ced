# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""
Feature extractor class for Audio Spectrogram Transformer.
"""

from typing import Optional

import torch
import torchaudio.transforms as audio_transforms

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...utils import logging

logger = logging.get_logger(__name__)


class CedFeatureExtractor(SequenceFeatureExtractor):
    def __init__(self,
                 f_min: int = 0,
                 sample_rate: int = 16000,
                 win_size: int = 512,
                 center: bool = True,
                 n_fft: int = 512,
                 f_max: Optional[int] = None,
                 hop_size: int = 160,
                 n_mels: int = 64):
        self.mel_spectrogram = audio_transforms.MelSpectrogram(f_min=f_min,
                                                               sample_rate=sample_rate,
                                                               win_length=win_size,
                                                               center=center,
                                                               n_fft=n_fft,
                                                               f_max=f_max,
                                                               hop_length=hop_size,
                                                               n_mels=n_mels),
        self.amplitude_to_db = audio_transforms.AmplitudeToDB(top_db=120)

    def __call__(self, x: torch.Tensor):
        x = self.mel_spectrogram(x)
        x = self.amplitude_to_db(x)
        return x
