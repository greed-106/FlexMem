"""A single lmms-eval task for the locally cached MVBench release."""

from __future__ import annotations

import os
import re
import string
from functools import cache
from pathlib import Path
from typing import Any

import datasets
from lmms_eval.api.task import ConfigurableTask

DATA_LIST = {
    "object_interaction": "star/Charades_segment",
    "action_sequence": "star/Charades_segment",
    "action_prediction": "star/Charades_segment",
    "action_localization": "sta/sta_video_segment",
    "moving_count": "clevrer/video_validation",
    "fine_grained_pose": "nturgbd_convert",
    "character_order": "perception/videos",
    "object_shuffle": "perception/videos",
    "egocentric_navigation": "vlnqa",
    "moving_direction": "clevrer/video_validation",
    "episodic_reasoning": "tvqa/video_fps3_hq_segment",
    "fine_grained_action": "Moments_in_Time_Raw/videos",
    "scene_transition": "scene_qa/video",
    "state_change": "perception/videos",
    "moving_attribute": "clevrer/video_validation",
    "action_antonym": "ssv2_video_mp4",
    "unexpected_action": "FunQA_test/test",
    "counterfactual_inference": "clevrer/video_validation",
    "object_existence": "clevrer/video_validation",
    "action_count": "perception/videos",
}


def _video_root() -> Path:
    configured = os.environ.get("FLEXMEM_MVBENCH_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "mvbench_video"


@cache
def _media_files(folder: str) -> tuple[Path, ...]:
    return tuple(path for path in Path(folder).rglob("*") if path.is_file())


def _resolve_video(doc: dict[str, Any], subtask: str) -> Path:
    """Resolve source names in Arrow annotations to the downloaded clip names."""
    folder = _video_root() / DATA_LIST[subtask]
    requested = Path(str(doc["video"]))
    direct = folder / requested
    if direct.is_file():
        return direct

    stem = requested.stem
    candidates = tuple(path for path in _media_files(str(folder)) if path.stem == stem or path.stem.startswith(stem + "_"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1 and "start" in doc and "end" in doc:
        matches = []
        for candidate in candidates:
            try:
                start, end = map(float, candidate.stem[len(stem) + 1 :].rsplit("_", 1))
            except ValueError:
                continue
            if abs(start - float(doc["start"])) < 1e-5 and abs(end - float(doc["end"])) < 1e-5:
                matches.append(candidate)
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(f"Cannot resolve MVBench video for {subtask}: {requested} under {folder}")


def doc_to_visual(doc: dict[str, Any]) -> list[str]:
    return [doc["video_path"]]


def doc_to_text(doc: dict[str, Any]) -> str:
    options = "".join(f"({letter}) {candidate}\n" for letter, candidate in zip(string.ascii_uppercase, doc["candidates"]))
    return f"Question:{doc['question']}\nOption:\n{options}Only give the best option.\n"


def _parse_answer(answer: str) -> str:
    answer = answer.replace("\n", " ").replace("\t", " ").strip()
    answer = re.sub(r"(?!<=\d)(\.)(?!\d)", "", answer)
    punctuation = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
    for char in punctuation:
        answer = answer.replace(char, " " if f" {char}" not in answer and f"{char} " not in answer else "")
    answer = answer.strip(" '\"").lower()
    match = re.search(r"\b([a-e])\b", answer, re.IGNORECASE)
    return match.group(1).upper() if match else answer


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, Any]]:
    answer = string.ascii_uppercase[doc["candidates"].index(doc["answer"])]
    prediction = _parse_answer(results[0])
    return {"mvbench_accuracy": {"pred_answer": prediction, "gt_answer": answer, "score": int(prediction == answer)}}


def aggregate_results(results: list[dict[str, Any]]) -> float:
    answered = [result for result in results if result["pred_answer"]]
    return 100 * sum(result["score"] for result in answered) / len(answered) if answered else 0.0


class FlexMemMVBenchTask(ConfigurableTask):
    """Combine MVBench's 20 cached configurations into one lmms-eval task."""

    def __init__(self, config: dict[str, Any]) -> None:
        config = dict(config)
        config.pop("class", None)
        super().__init__(config=config)

    def download(self, dataset_kwargs: dict[str, Any] | None = None) -> None:
        cache_dir = os.environ.get("LMMS_EVAL_DATASETS_CACHE") or os.path.join(
            os.environ.get("HF_HOME", "~/.cache/huggingface"), "datasets"
        )
        rows = []
        for subtask in DATA_LIST:
            split = datasets.load_dataset(
                "OpenGVLab/MVBench",
                name=subtask,
                split="train",
                token=True,
                revision="video",
                cache_dir=cache_dir,
            )
            for index, doc in enumerate(split):
                rows.append(
                    {
                        "id": f"{subtask}:{index}",
                        "subtask": subtask,
                        "video_path": str(_resolve_video(doc, subtask)),
                        "question": doc["question"],
                        "candidates": list(doc["candidates"]),
                        "answer": doc["answer"],
                    }
                )
        self.dataset = datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})
        self.dataset_no_image = self.dataset
