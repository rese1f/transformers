# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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

from typing import Union
import os

from ...configuration_utils import PretrainedConfig
from ...utils import (
    logging,
)
from ..auto import CONFIG_MAPPING
from ..clip import CLIPVisionConfig


logger = logging.get_logger(__name__)


class AuroraConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AuroraForConditionalGeneration`]. It is used to instantiate an
    Aurora Series model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the [Reself/AuroraCap-7B-VID](https://huggingface.co/Reself/AuroraCap-7B-VID)
    model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to -200):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"full"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
            If `"full"`, the full vision features are used.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.

    Example:

    ```python
    >>> from transformers import AuroraForConditionalGeneration, AuroraConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Aurora Reself/AuroraCap-7B-VID style configuration
    >>> configuration = AuroraConfig(vision_config, text_config)

    >>> # Initializing a model from the Reself/AuroraCap-7B-VID style configuration
    >>> model = AuroraForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "aurora"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=-200,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="full",
        vision_feature_layer=-2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            # apple/DFN5B-CLIP-ViT-H-14-378
            vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=5120,
                hidden_size=1280,
                patch_size=14,
                image_size=378,
                num_hidden_layers=32,
                num_attention_heads=16,
                projection_dim=512,
            )

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
