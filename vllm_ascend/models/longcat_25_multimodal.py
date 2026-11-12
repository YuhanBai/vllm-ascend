# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LongCat-2.5 image and video preprocessing and model registration."""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature
from transformers.image_transforms import to_pil_image
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions, ImageDummyOptions, VideoDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

from vllm_ascend.models.longcat_25 import AscendLongCat25ForCausalLM
from vllm_ascend.models.longcat_25_vision import LongCat25Vision, LongCat25VisionBridge

IMAGE_PLACEHOLDER = "<longcat_img_start><longcat_img_pad><longcat_img_end>"
VIDEO_PLACEHOLDER = "<longcat_video_start><longcat_video_place><longcat_video_end>"
MAX_IMAGE_ASPECT_RATIO = 200
VISION_MODULES = ("visual_model.", "visual_bridge_model.")
Frame = Image.Image | np.ndarray | torch.Tensor


def _resize_image(height: int, width: int, config: dict) -> tuple[int, int]:
    factor = config["patch_size"] * config["spatial_merge_size"]
    if max(height, width) > MAX_IMAGE_ASPECT_RATIO * min(height, width):
        raise ValueError(f"LongCat image aspect ratio must be at most {MAX_IMAGE_ASPECT_RATIO}.")
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > config["max_pixels"]:
        scale = math.sqrt(height * width / config["max_pixels"])
        h = max(factor, math.floor(height / scale / factor) * factor)
        w = max(factor, math.floor(width / scale / factor) * factor)
    elif h * w < config["min_pixels"]:
        scale = math.sqrt(config["min_pixels"] / (height * width))
        h = math.ceil(height * scale / factor) * factor
        w = math.ceil(width * scale / factor) * factor
    return h, w


def _image_patches(image: Frame, config: dict) -> tuple[torch.Tensor, tuple[int, int, int]]:
    return _media_patches([image], config)


def _media_patches(frames: Sequence[Frame], config: dict) -> tuple[torch.Tensor, tuple[int, int, int]]:
    images = [frame if isinstance(frame, Image.Image) else to_pil_image(frame) for frame in frames]
    h, w = _resize_image(images[0].height, images[0].width, config)
    patch, merge, temporal = (config[key] for key in ("patch_size", "spatial_merge_size", "temporal_patch_size"))
    mean = np.asarray(config["image_mean"])[:, None, None]
    std = np.asarray(config["image_std"])[:, None, None]
    pixels = []
    for image in images:
        resized = image.convert("RGB").resize((w, h), resample=Image.Resampling.BICUBIC)
        pixels.append((np.asarray(resized).transpose(2, 0, 1) / 255.0 - mean) / std)
    pixels.extend([pixels[-1]] * (-len(pixels) % temporal))
    grid_t = len(pixels) // temporal
    grid_h, grid_w = h // patch, w // patch
    grouped = np.stack(pixels).reshape(
        grid_t, temporal, 3, grid_h // merge, merge, patch, grid_w // merge, merge, patch
    )
    patches = grouped.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(grid_t * grid_h * grid_w, -1)
    return torch.from_numpy(np.ascontiguousarray(patches, dtype=np.float32)), (grid_t, grid_h, grid_w)


def _video_patches(frames, metadata: Mapping[str, object], config: dict):
    fps = float(metadata.get("fps", 0))
    indices = list(metadata.get("frames_indices", range(len(frames))))
    if not indices or len(indices) != len(frames) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("LongCat video requires decoded frames, matching frame indices, and a positive FPS.")
    # Use the decoded frame nearest each second. Explicit media I/O sampling
    # can already produce a 1 FPS stream; retaining one frame per bin is safe.
    selected: dict[int, int] = {}
    for offset, frame_index in enumerate(indices):
        timestamp = frame_index / fps
        second = round(timestamp)
        if second not in selected or abs(timestamp - second) < abs(indices[selected[second]] / fps - second):
            selected[second] = offset
    seconds = sorted(selected)
    limit = config["max_frame_num"]
    if len(seconds) > limit:
        seconds = [seconds[index] for index in np.linspace(0, len(seconds) - 1, limit, dtype=int)]
    patches, grid = _media_patches([frames[selected[second]] for second in seconds], config)
    return patches, grid, seconds[:: config["temporal_patch_size"]]


def _max_media_tokens(config: dict) -> int:
    return config["max_pixels"] // (config["patch_size"] * config["spatial_merge_size"]) ** 2


class LongCat25ProcessingInfo(BaseProcessingInfo):
    @property
    def vision_config(self) -> dict:
        return self.get_hf_config().visual_config

    @property
    def video_config(self) -> dict:
        return self.get_hf_config().video_config

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None} if self.video_config.get("enable") else {"image": None}

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(video_needs_metadata=True)

    def get_supported_mm_processor_kwargs(self) -> dict[str, set[str]]:
        return {"text_kwargs": set(), "images_kwargs": set(), "videos_kwargs": set()}

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int]:
        limits = {"image": _max_media_tokens(self.vision_config)}
        if "video" in self.supported_mm_limits:
            config = self.video_config
            groups = math.ceil(config["max_frame_num"] / config["temporal_patch_size"])
            limits["video"] = min(seq_len, groups * _max_media_tokens(config))
        return limits


class LongCat25DummyInputs(BaseDummyInputsBuilder[LongCat25ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0) + VIDEO_PLACEHOLDER * mm_counts.get("video", 0)

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions]
    ) -> MultiModalDataDict:
        config = self.info.vision_config
        side = math.isqrt(config["max_pixels"])
        factor = config["patch_size"] * config["spatial_merge_size"]
        side = side // factor * factor
        data: MultiModalDataDict = {
            "image": self._get_dummy_images(
                width=side,
                height=side,
                num_images=mm_counts.get("image", 0),
                overrides=cast(ImageDummyOptions | None, mm_options.get("image")),
            )
        }
        if num_videos := mm_counts.get("video", 0):
            config = self.info.video_config
            factor = config["patch_size"] * config["spatial_merge_size"]
            side = math.isqrt(config["max_pixels"]) // factor * factor
            tokens_per_group = (side // factor) ** 2
            temporal = config["temporal_patch_size"]
            groups = max(1, seq_len // num_videos // tokens_per_group)
            videos = self._get_dummy_videos(
                width=side,
                height=side,
                num_frames=min(config["max_frame_num"], groups * temporal),
                num_videos=num_videos,
                overrides=cast(VideoDummyOptions | None, mm_options.get("video")),
            )
            data["video"] = [(frames, {"fps": 1.0, "frames_indices": list(range(len(frames)))}) for frames in videos]
        return data


class LongCat25Processor(BaseMultiModalProcessor[LongCat25ProcessingInfo]):
    def _call_hf_processor(self, hf_data: Mapping[str, object], hf_kwargs: Mapping[str, object]) -> BatchFeature:
        images = cast(Sequence[Frame], hf_data.get("images", ()))
        videos = cast(Sequence[tuple[Sequence[Frame], Mapping[str, object]]], hf_data.get("videos", ()))
        features = BatchFeature()
        if images:
            patches, grids = zip(*(_image_patches(image, self.info.vision_config) for image in images))
            features.update(pixel_values=torch.cat(patches), image_grid_thw=torch.tensor(grids))
        if videos:
            patches, grids, seconds = zip(
                *(_video_patches(frames, metadata, self.info.video_config) for frames, metadata in videos)
            )
            features.update(
                pixel_values_videos=torch.cat(patches),
                video_grid_thw=torch.tensor(grids),
                video_timestamps=[torch.tensor(item, dtype=torch.long) for item in seconds],
            )
        return features

    def _get_mm_fields_config(self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]):
        fields = {}
        for modality, pixels_key in (("image", "pixel_values"), ("video", "pixel_values_videos")):
            grid_key = f"{modality}_grid_thw"
            grids = hf_inputs.get(grid_key)
            sizes = torch.empty(0, dtype=torch.long) if grids is None else grids.prod(-1)
            fields[pixels_key] = MultiModalFieldConfig.flat_from_sizes(modality, sizes)
            fields[grid_key] = MultiModalFieldConfig.batched(modality, keep_on_cpu=True)
        fields["video_timestamps"] = MultiModalFieldConfig.batched("video", keep_on_cpu=True)
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        config = self.info.vision_config
        pad_id = config["image_pad_token_id"]
        merge_area = config["spatial_merge_size"] ** 2

        def replacement(index: int) -> list[int]:
            grid = out_mm_kwargs["image"][index]["image_grid_thw"].data
            return [pad_id] * (math.prod(grid.tolist()) // merge_area)

        updates = [PromptReplacement(modality="image", target=[pad_id], replacement=replacement)]
        if "video" in self.info.supported_mm_limits:
            video = self.info.video_config
            marker = [video[key] for key in ("video_start_token_id", "video_place_token_id", "video_end_token_id")]
            tokenizer = self.info.get_tokenizer()

            def video_replacement(index: int) -> PromptUpdateDetails:
                item = out_mm_kwargs["video"][index]
                _, height, width = item["video_grid_thw"].data.tolist()
                per_group = height * width // video["spatial_merge_size"] ** 2
                tokens = [marker[0]]
                for second in item["video_timestamps"].data.tolist():
                    tokens.extend(
                        [video["image_start_token_id"], *([marker[1]] * per_group), video["image_end_token_id"]]
                    )
                    if "<TIMEIDX>" in video.get("frame_pattern", ""):
                        tokens.extend(tokenizer.encode(f"at second[{second}]", add_special_tokens=False))
                tokens.append(marker[2])
                return PromptUpdateDetails.select_token_id(tokens, marker[1])

            updates.append(PromptReplacement(modality="video", target=marker, replacement=video_replacement))
        return updates


@MULTIMODAL_REGISTRY.register_processor(
    LongCat25Processor, info=LongCat25ProcessingInfo, dummy_inputs=LongCat25DummyInputs
)
class AscendLongCat25ForConditionalGeneration(AscendLongCat25ForCausalLM, SupportsMultiModal):
    supports_multimodal_raw_input_only = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        if modality == "image":
            return IMAGE_PLACEHOLDER
        if modality == "video":
            return VIDEO_PLACEHOLDER
        raise ValueError(f"LongCat-2.5 does not support the {modality!r} modality in this adapter.")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        vision = config.visual_config
        quantizer = config.visual_quantizer_config
        if not vision["enable"] or not quantizer["enable"]:
            raise ValueError("LongCat-2.5 requires the visual encoder and RVQ checkpoint for image inference.")
        if not vllm_config.use_v2_model_runner or vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("LongCat-2.5 multimodal inference requires the V2 model runner and PP=1.")
        if vllm_config.model_config.enable_prompt_embeds:
            raise ValueError("LongCat-2.5 image inference requires token IDs for over-embedding.")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        modalities = {"image", "video"} if config.video_config.get("enable") else {"image"}
        with self._mark_tower_model(vllm_config, modalities):
            self.visual_model = LongCat25Vision(vision, quantizer)
            self.visual_bridge_model = LongCat25VisionBridge(vision, quantizer, config.hidden_size, config.rms_norm_eps)

    def get_language_model(self) -> AscendLongCat25ForCausalLM:
        return self

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        embeddings = super().embed_input_ids(input_ids)
        if multimodal_embeddings:
            if is_multimodal is None:
                raise ValueError("A multimodal embedding mask is required.")
            return _merge_multimodal_embeddings(embeddings, multimodal_embeddings, is_multimodal)
        return embeddings

    def embed_multimodal(self, **kwargs):
        grids_by_pixel_key = {"pixel_values": "image_grid_thw", "pixel_values_videos": "video_grid_thw"}
        result = []
        for key, pixels in kwargs.items():
            if key not in grids_by_pixel_key or pixels is None:
                continue
            grids = kwargs[grids_by_pixel_key[key]]
            grids = grids.tolist() if isinstance(grids, torch.Tensor) else [grid.tolist() for grid in grids]
            if not isinstance(pixels, torch.Tensor):
                pixels = torch.cat(pixels)
            for media, grid in zip(pixels.split([math.prod(grid) for grid in grids]), grids):
                result.append(self.visual_bridge_model(self.visual_model(media, grid)))
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = {name: param for name, param in self.named_parameters() if name.startswith(VISION_MODULES)}
        if not params:  # The tower is absent when the image limit is zero.
            return super().load_weights(weights)
        loaded_vision = set()
        quantizer_prefix = "model.visual_model.visual_quantizer.quantize."
        codebook_name = quantizer_prefix + "codebooks.0.embed"

        def text_weights():
            for name, weight in weights:
                if not name.removeprefix("model.").startswith(VISION_MODULES):
                    yield name, weight
                    continue
                if name == codebook_name:
                    target = "visual_model.visual_quantizer.codebook"
                    if weight.shape[0] == params[target].shape[0] + 1:
                        weight = weight[:-1]
                elif name.startswith(quantizer_prefix):
                    # The official checkpoint repeats the shared codebook and includes EMA training state.
                    continue
                else:
                    target = name.removeprefix("model.")
                param = params[target]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded_vision.add(target)

        loaded = super().load_weights(text_weights())
        if missing := params.keys() - loaded_vision:
            raise ValueError(f"LongCat-2.5 checkpoint is missing visual weights: {sorted(missing)}")
        return loaded | loaded_vision
