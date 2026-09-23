"""FlexMem's streaming-video adapter for :mod:`lmms_eval`.

The upstream project evaluates each video with a custom script. This adapter
keeps that loading, frame-sampling, prompt, and memory-reset path while
exposing lmms-eval's regular ``generate_until`` interface.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Sequence
from datetime import timedelta
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from accelerate import Accelerator, InitProcessGroupKwargs
from decord import VideoReader, cpu
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from loguru import logger as eval_logger
from PIL import Image

_DTYPE_ALIASES = {"float16", "fp16"}


class FlexMem(lmms):
    """Use FlexMem through ``lmms-eval --model flexmem``.

    Visual memory is stateful, so a batch contains exactly one video and memory
    is reset before every request. Multi-process lmms-eval runs are supported:
    each process loads the model on its local CUDA device.
    """

    def __init__(
        self,
        pretrained: str,
        model_base: str | None = None,
        config_path: str | None = None,
        conv_template: str = "qwen_1_5",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: int | str = 1,
        dtype: str = "float16",
        max_num_frames: int | None = None,
        sample_fps: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if int(batch_size) != 1:
            raise ValueError("FlexMem only supports batch_size=1 because visual memory is reset per video.")
        if dtype.lower() not in _DTYPE_ALIASES:
            raise ValueError("FlexMem's upstream loader and visual preprocessor require dtype=float16.")
        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError("FlexMem evaluation requires CUDA, but PyTorch cannot see a CUDA device.")
        if kwargs:
            eval_logger.warning("Ignoring unsupported FlexMem model arguments: {}", ", ".join(sorted(kwargs)))

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        self.accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes
        self._device = self._resolve_device(device)
        self.batch_size_per_gpu = 1
        self.conv_template = conv_template
        self.dtype = torch.float16

        settings = self._load_settings(config_path)
        self.max_num_frames = int(max_num_frames or settings["max_num_frames"])
        self.sample_fps = float(sample_fps or settings["sample_fps"])
        self.chunk_size = int(settings["chunk_size"])
        self._validate_settings(settings)
        if self.max_num_frames < 1 or self.sample_fps <= 0:
            raise ValueError("max_num_frames and sample_fps must be positive.")

        overwrite_config: dict[str, Any] = {
            "tokens_per_frame": int(settings["tokens_per_frame"]),
            "mm_newline_position": "grid",
            "delay_load": False,
            "_attn_implementation": "eager",
        }
        if overwrite_config["tokens_per_frame"] == 182:
            overwrite_config.update(
                {
                    "mm_spatial_pool_stride": 2,
                    "mm_spatial_pool_mode": "average",
                    "mm_pooling_position": "before",
                }
            )

        # The project’s original evaluation script explicitly uses eager HF
        # attention when it constructs the FlexMem streaming model.
        disable_torch_init()
        model_path = str(Path(pretrained).expanduser())
        model_name = get_model_name_from_path(model_path) + "_stream"
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            device_map=self._loading_device_map(device_map),
            overwrite_config=overwrite_config,
            attn_implementation="eager",
            chunk_size=self.chunk_size,
            topb=int(settings["topb"]),
            preblk=int(settings["preblk"]),
            preratio=int(settings["preratio"]),
            decratio=int(settings["decratio"]),
            tokens_per_frame=int(settings["tokens_per_frame"]),
        )
        self._model.eval()
        self._config = self._model.config

        if conv_template not in conv_templates:
            available = ", ".join(sorted(conv_templates))
            raise ValueError(f"Unknown conv_template '{conv_template}'. Available templates: {available}")
        eval_logger.info("Loaded FlexMem on {} (rank {}/{})", self._device, self.rank, self.world_size)

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def batch_size(self) -> int:
        return self.batch_size_per_gpu

    @property
    def max_length(self) -> int:
        return self._max_length

    def loglikelihood(self, requests: list[Instance]):
        raise NotImplementedError("FlexMem currently supports lmms-eval generate_until tasks only.")

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate one answer per request while preserving FlexMem's stream."""
        results: list[str] = []
        collator = utils.Collator([request.args for request in requests], self._collate, grouping=True)

        # A stateful visual memory prevents batching distinct videos. Collation
        # still gives deterministic ordering and groups equal generation options.
        for chunk in collator.get_batched(n=1, batch_fn=None):
            contexts, all_gen_kwargs, doc_to_visual, doc_ids, tasks, splits = zip(*chunk)
            context = contexts[0]
            gen_kwargs = dict(all_gen_kwargs[0])
            task, split, doc_id = tasks[0], splits[0], doc_ids[0]
            doc = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual[0](doc)
            answer = self._generate_one(context, visuals, gen_kwargs, self._request_id(doc, doc_id))
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
            results.append(answer)

        return collator.get_original(results)

    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError("FlexMem does not support multi-round lmms-eval tasks.")

    @staticmethod
    def _collate(item: tuple[Any, ...]) -> tuple[int, str]:
        context = item[0][0] if isinstance(item[0], list) else item[0]
        return -len(context), context

    def _generate_one(self, context: str | list[str], visuals: Any, gen_kwargs: dict[str, Any], request_id: str) -> str:
        context = context[0] if isinstance(context, list) else context
        video = self._visuals_to_tensor(visuals)
        prompt, question = self._build_prompt(context)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        question_ids = self.tokenizer(question, return_tensors="pt").input_ids.to(self.device)

        conversation = conv_templates[self.conv_template].copy()
        stop_str = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        generation_args = self._generation_args(gen_kwargs, stopping_criteria)

        self.model.memory.reset()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                question_ids=question_ids,
                is_parallel=False,
                qid=request_id,
                images=[video],
                modalities=["video"],
                **generation_args,
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return self._extract_answer(output, stop_str, gen_kwargs.get("until"))

    def _build_prompt(self, context: str) -> tuple[str, str]:
        # FlexMem represents a full video with one image token. lmms-eval's
        # video tasks normally have none; image-interleaved tasks can have many.
        # Replace all of theirs to avoid a feature/token count mismatch.
        question = context.replace(DEFAULT_IMAGE_TOKEN, "").strip()
        visual_prefix = DEFAULT_IMAGE_TOKEN
        if self.config.mm_use_im_start_end:
            visual_prefix = DEFAULT_IM_START_TOKEN + visual_prefix + DEFAULT_IM_END_TOKEN

        conversation = copy.deepcopy(conv_templates[self.conv_template])
        conversation.append_message(conversation.roles[0], f"{visual_prefix}\n{question}")
        conversation.append_message(conversation.roles[1], None)
        return conversation.get_prompt(), question

    def _visuals_to_tensor(self, visuals: Any) -> torch.Tensor:
        media = self._flatten_visuals(visuals)
        video_paths = [item for item in media if isinstance(item, (str, Path))]
        if video_paths:
            if len(video_paths) != 1 or len(media) != 1:
                raise ValueError("FlexMem expects one video path or a sequence of image frames per request.")
            frames = self._sample_video(Path(video_paths[0]))
        else:
            frames = self._sample_image_frames(media)

        pixels = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        return pixels.to(device=self.device, dtype=self.dtype, non_blocking=True)

    @staticmethod
    def _flatten_visuals(visuals: Any) -> list[Any]:
        if visuals is None:
            raise ValueError("FlexMem requires a video or image frames, but this request has no visual input.")
        if isinstance(visuals, (str, Path, Image.Image, np.ndarray, torch.Tensor)):
            return [visuals]
        if isinstance(visuals, Iterable):
            flattened: list[Any] = []
            for item in visuals:
                if isinstance(item, Iterable) and not isinstance(item, (str, Path, Image.Image, np.ndarray, torch.Tensor)):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            return flattened
        raise TypeError(f"Unsupported visual input type: {type(visuals)!r}")

    def _sample_video(self, path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {path}")
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        total_frames = len(reader)
        if total_frames < 1:
            raise ValueError(f"Video contains no decodable frames: {path}")
        fps = max(float(reader.get_avg_fps()), 1e-6)
        duration = total_frames / fps
        requested = min(int(duration * self.sample_fps), self.max_num_frames)
        frame_count = self._round_to_chunk_size(max(1, requested))
        frame_indices = np.linspace(0, total_frames - 1, frame_count, dtype=int)
        return reader.get_batch(frame_indices.tolist()).asnumpy()

    def _sample_image_frames(self, frames: Sequence[Any]) -> list[Any]:
        if not frames:
            raise ValueError("FlexMem received an empty frame sequence.")
        count = self._round_to_chunk_size(min(len(frames), self.max_num_frames))
        indices = np.linspace(0, len(frames) - 1, count, dtype=int)
        selected = [frames[index] for index in indices]
        for frame in selected:
            if not isinstance(frame, (Image.Image, np.ndarray, torch.Tensor)):
                raise TypeError(f"Unsupported video-frame type: {type(frame)!r}")
        return selected

    def _round_to_chunk_size(self, frame_count: int) -> int:
        return int(math.ceil(frame_count / self.chunk_size) * self.chunk_size)

    def _generation_args(self, gen_kwargs: dict[str, Any], stopping_criteria: KeywordsStoppingCriteria) -> dict[str, Any]:
        do_sample = bool(gen_kwargs.get("do_sample", False))
        args: dict[str, Any] = {
            "do_sample": do_sample,
            "max_new_tokens": int(gen_kwargs.get("max_new_tokens", 1024)),
            "use_cache": True,
            "stopping_criteria": [stopping_criteria],
        }
        if do_sample:
            args["temperature"] = float(gen_kwargs.get("temperature", 1.0))
            if gen_kwargs.get("top_p") is not None:
                args["top_p"] = float(gen_kwargs["top_p"])
            if gen_kwargs.get("top_k") is not None:
                args["top_k"] = int(gen_kwargs["top_k"])
        if gen_kwargs.get("num_beams") is not None:
            args["num_beams"] = int(gen_kwargs["num_beams"])
        return args

    @staticmethod
    def _extract_answer(output: str, stop_str: str | None, until: str | Sequence[str] | None) -> str:
        # The custom model returns the complete decoded conversation. This is
        # how the repository's original evaluator isolates the assistant turn.
        if "assistant\n" in output:
            output = output.rsplit("assistant\n", 1)[-1]
        if stop_str and output.endswith(stop_str):
            output = output[: -len(stop_str)]
        if until:
            terms = [until] if isinstance(until, str) else until
            for term in terms:
                if term:
                    output = output.split(term, 1)[0]
        return output.strip()

    def _resolve_device(self, device: str) -> torch.device:
        if device.startswith("cuda") and self.accelerator.num_processes > 1:
            return torch.device(f"cuda:{self.accelerator.local_process_index}")
        return torch.device(device)

    def _loading_device_map(self, requested: str) -> str:
        if self.accelerator.num_processes > 1 and requested == "auto":
            return str(self.device)
        return requested

    @staticmethod
    def _request_id(doc: Any, doc_id: Any) -> str:
        if isinstance(doc, dict):
            return str(doc.get("id", doc.get("question_id", doc_id)))
        return str(doc_id)

    @staticmethod
    def _load_settings(config_path: str | None) -> dict[str, Any]:
        path = Path(config_path).expanduser() if config_path else Path(files("flexmem_lmms_eval").joinpath("configs/flexmem.yaml"))
        if not path.is_file():
            raise FileNotFoundError(f"FlexMem configuration file does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            settings = yaml.safe_load(handle)
        if not isinstance(settings, dict):
            raise TypeError(f"FlexMem configuration must be a YAML mapping: {path}")
        return settings

    @staticmethod
    def _validate_settings(settings: dict[str, Any]) -> None:
        required = {"sample_fps", "max_num_frames", "tokens_per_frame", "chunk_size", "preblk", "preratio", "decratio", "topb"}
        missing = required - settings.keys()
        if missing:
            raise ValueError(f"FlexMem configuration is missing: {', '.join(sorted(missing))}")
        invalid = [key for key in required if float(settings[key]) <= 0]
        if invalid:
            raise ValueError(f"FlexMem configuration values must be positive: {', '.join(sorted(invalid))}")
