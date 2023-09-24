<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CED

## Overview

The CED model was proposed in [CED: Consistent ensemble distillation for audio tagging](https://arxiv.org/abs/2308.11957) by Heinrich Dinkel, Yongqing Wang, Zhiyong Yan, Junbo Zhang and Yujun Wang.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

Augmentation and knowledge distillation (KD) are well-established techniques employed in audio classification tasks, aimed at enhancing performance and reducing model sizes on the widely recognized Audioset (AS) benchmark. Although both techniques are effective individually, their combined use, called consistent teaching, hasn't been explored before. This paper proposes CED, a simple training framework that distils student models from large teacher ensembles with consistent teaching. To achieve this, CED efficiently stores logits as well as the augmentation methods on disk, making it scalable to large-scale datasets. Central to CED's efficacy is its label-free nature, meaning that only the stored logits are used for the optimization of a student model only requiring 0.3\% additional disk space for AS. The study trains various transformer-based models, including a 10M parameter model achieving a 49.0 mean average precision (mAP) on AS.

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [Junbo Zhang](https://huggingface.co/jimbozhang).
The original code can be found [here](https://github.com/RicherMans/CED).


## CedConfig

[[autodoc]] CedConfig

## CedModel

[[autodoc]] CedModel
    - forward

## CedForAudioClassification

[[autodoc]] CedForAudioClassification
    - forward