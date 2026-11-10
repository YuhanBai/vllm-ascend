# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)

from vllm_ascend.models.longcat_25_config import LongCat25Config


class LongCat25ModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def is_deepseek_mla(self) -> bool:
        return True


_CONFIG_REGISTRY.setdefault(LongCat25Config.model_type, LongCat25Config)
MODEL_ARCH_CONFIG_CONVERTORS.setdefault(LongCat25Config.model_type, LongCat25ModelArchConfigConvertor)
