from types import MethodType, SimpleNamespace

from PIL import Image

from flexmem_lmms_eval.model import FlexMem
from flexmem_lmms_eval.plugin import get_model_manifests


def test_plugin_manifests_register_simple_models():
    standard, fast = get_model_manifests()

    assert standard.model_id == "flexmem"
    assert standard.simple_class_path == "flexmem_lmms_eval.model.FlexMem"
    assert standard.aliases == ("flexmem_stream",)
    assert fast.model_id == "flexmem_fast"
    assert fast.simple_class_path == "flexmem_lmms_eval.fast_model.FlexMemFast"


def test_default_settings_are_complete():
    settings = FlexMem._load_settings(None)

    FlexMem._validate_settings(settings)
    assert settings["chunk_size"] == 8
    assert settings["tokens_per_frame"] == 210


def test_image_frame_sampling_keeps_chunk_alignment():
    adapter = object.__new__(FlexMem)
    adapter.chunk_size = 8
    adapter.max_num_frames = 10
    frames = [Image.new("RGB", (2, 2), color=(index, 0, 0)) for index in range(9)]

    sampled = adapter._sample_image_frames(frames)

    assert len(sampled) == 16
    assert sampled[0].getpixel((0, 0)) == (0, 0, 0)
    assert sampled[-1].getpixel((0, 0)) == (8, 0, 0)


def test_prompt_has_exactly_one_flexmem_visual_token():
    adapter = object.__new__(FlexMem)
    adapter._config = SimpleNamespace(mm_use_im_start_end=False)
    adapter.conv_template = "qwen_1_5"

    prompt, question = adapter._build_prompt("<image> What happens? <image>")

    assert prompt.count("<image>") == 1
    assert "<image>" not in question
    assert "What happens?" in question


def test_generate_until_accepts_lmms_eval_simple_request_batches():
    adapter = object.__new__(FlexMem)
    adapter.task_dict = {"task": {"test": [{"id": "one"}, {"id": "two"}]}}
    adapter.cache_hook = SimpleNamespace(add_partial=lambda *args: None)
    calls = []

    def generate_one(self, context, visuals, gen_kwargs, request_id):
        calls.append((context, visuals, gen_kwargs, request_id))
        return f"answer-{request_id}"

    adapter._generate_one = MethodType(generate_one, adapter)

    def visual_fn(doc):
        return [f"{doc['id']}.mp4"]

    requests = [
        SimpleNamespace(args=("short", {"max_new_tokens": 16}, visual_fn, 0, "task", "test")),
        SimpleNamespace(args=("a longer prompt", {"max_new_tokens": 16}, visual_fn, 1, "task", "test")),
    ]

    assert adapter.generate_until(requests) == ["answer-one", "answer-two"]
    assert {call[1][0] for call in calls} == {"one.mp4", "two.mp4"}
