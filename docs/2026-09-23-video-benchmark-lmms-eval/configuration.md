# 配置、抽帧与作者脚本对齐说明

## 结论

当前三个评测的**视觉记忆配置和抽帧实现**与相应 FlexMem 变体对齐；但 lmms-eval 任务的**提示词和生成参数**采用 lmms-eval 原生设置，因而不构成作者脚本的逐项复现。

作者只公开了标准 FlexMem 的 LongVideoBench 脚本和 FlexMem-fast 的 MLVU 脚本，没有公开 MVBench 或 Video-MME 的 evaluator。因此，后两者不存在可被严格逐项复刻的作者 benchmark 配置。

## 模型与视觉记忆配置

| 评测 | lmms-eval 模型 | 配置文件 | `sample_fps` | `max_num_frames` | `tokens_per_frame` | `chunk_size` | `preblk` | `preratio` | `decratio` | `topb` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MVBench | `flexmem` | `FlexMem/config.yaml` | 2 | 1024 | 210 | 8 | 12 | 4 | 2 | 16 |
| MLVU-dev | `flexmem_fast` | `FlexMem-fast/config.yaml` | 0.5 | 512 | 210 | 8 | 12 | 4 | 4 | 32 |
| Video-MME | `flexmem` | `FlexMem/config.yaml` | 2 | 1024 | 210 | 8 | 12 | 4 | 2 | 16 |

其中 `preblk` 是预填充记忆块数，`preratio` 与 `decratio` 分别是预填充和生成阶段的记忆压缩率，`topb` 是生成时检索的记忆块数。

## 抽帧

当前 `flexmem` / `flexmem_fast` 插件执行以下逻辑：

```python
requested = min(int(video_duration * sample_fps), max_num_frames)
frame_count = ceil(max(1, requested) / chunk_size) * chunk_size
indices = linspace(0, total_frames - 1, frame_count, dtype=int)
```

这与作者脚本的主路径一致：先按视频时长与 `sample_fps` 确定帧数、以 `max_num_frames` 截断、向上补齐到 8 帧 chunk 的倍数，再在整段视频均匀采样。插件对不足一帧的极短视频保证至少采样 1 帧；作者脚本没有这个保护分支。补齐到 chunk 倍数可能使最终帧数超过 `max_num_frames`，且短视频中可能出现重复 index。

作者 shell 脚本中的 `FRAMES=99` / `--for_get_frames_num 99` 没有进入上述计算，不是生效的抽帧参数。

## 当前 lmms-eval 任务协议

| 评测 | task | 样本数 | 提示词/输入 | 生成参数 |
| --- | --- | ---: | --- | --- |
| MVBench | `flexmem_mvbench` | 4000 | lmms-eval 原生 MVBench MCQ 格式：问题、`(A)` 选项、`Only give the best option.` | `max_new_tokens=16`，`temperature=0`，`top_p=1`，`num_beams=1`，`do_sample=false` |
| MLVU-dev | `mlvu_dev` | 2174 | 数据集原始 `question`（含选项）后追加 `Only give the best option. Best option: (` | 同上 |
| Video-MME | `videomme` | 2700 | 无字幕视频与四选项 MCQ；要求直接给出选项字母 | 同上 |

Video-MME 的 `videomme_w_subtitle` 会把本地 SRT 内容加入输入，属于不同协议；本阶段不使用它。

## 与作者脚本的差异

| 项目 | 作者公开脚本 | 当前 lmms-eval 设置 | 对齐结论 |
| --- | --- | --- | --- |
| MLVU-dev 内存/抽帧 | FlexMem-fast YAML，0.5 FPS、512 帧上限等 | 相同 | 对齐；极短视频保护是插件额外分支 |
| MLVU-dev 提示词 | 重建 `A. ...` 选项，要求只输出字母 | 数据集题目加 lmms-eval 后缀 | 不对齐 |
| MLVU-dev 生成 | greedy，`temperature=1`、`top_p=1`、`top_k=1`、`max_new_tokens=1024` | greedy，`temperature=0`、`max_new_tokens=16`、`num_beams=1` | 不对齐 |
| MVBench | 未发布作者 evaluator | 标准 FlexMem YAML 加 lmms-eval 原生 MCQ 协议 | 只有视觉记忆配置可对齐；不能声称脚本复现 |
| Video-MME | 未发布作者 evaluator | 标准 FlexMem YAML、无字幕 lmms-eval 协议 | 只有视觉记忆配置可对齐；不能声称脚本复现 |

作者所有公开 evaluator 都在每个样本推理前调用 `model.memory.reset()`；当前两个 lmms-eval 模型插件也在每个请求前重置 memory。

## 运行命令

运行前应明确设置共享离线缓存：

```bash
export HF_HOME=/mnt/hdd1/ymj/VLM/datasets/benchmarks/hf-home
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

MVBench：

```bash
uv run lmms-eval --model flexmem \
  --model_args pretrained=/ABS/PATH/TO/LLaVA-Video-7B-Qwen2,batch_size=1 \
  --tasks flexmem_mvbench \
  --include_path flexmem_lmms_eval/tasks/mvbench \
  --batch_size 1 --output_path outputs/mvbench
```

MLVU-dev：

```bash
uv run lmms-eval --model flexmem_fast \
  --model_args pretrained=/ABS/PATH/TO/LLaVA-Video-7B-Qwen2,batch_size=1 \
  --tasks mlvu_dev --batch_size 1 --output_path outputs/mlvu-dev
```

Video-MME：

```bash
uv run lmms-eval --model flexmem \
  --model_args pretrained=/ABS/PATH/TO/LLaVA-Video-7B-Qwen2,batch_size=1 \
  --tasks videomme --batch_size 1 --output_path outputs/videomme
```

正式运行必须由 SQLite 队列的固定 manifest 以绝对 `uv` 路径提交，并由 Supervisor 托管消费者；以上命令仅说明任务接口，不应绕过调度器直接用于长程 GPU 评测。
