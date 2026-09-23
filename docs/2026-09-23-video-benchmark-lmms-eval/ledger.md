# 视频 benchmark 的 lmms-eval 适配账本

## 目标

在本仓库的 `uv` 环境中，通过 lmms-eval 评测本地缓存的 MVBench、MLVU-dev 与 Video-MME。FlexMem 使用 batch size 1，并在每个样本前重置视觉记忆。

## 当前状态

- 代码：已完成 lmms-eval 的 `flexmem` 与 `flexmem_fast` 模型插件，以及单任务 `flexmem_mvbench` 的本地路径适配。
- 数据检查：MVBench 的 4000 条 Arrow 标注均已映射到本地媒体；MLVU-dev 为 2174 条；Video-MME 为 2700 条。
- 实际 GPU 评测：尚未执行。当前没有记录结果、耗时或准确率，不能将本文档视为实验结果。
- 权重获取：`LLaVA-Video-7B-Qwen2` 正在由独立 Supervisor 实例托管下载，尚未完成校验；不能作为已可用模型记录。
- 阻塞：需要确认可用 GPU、完成并校验 `LLaVA-Video-7B-Qwen2` 本地权重，然后以 Supervisor 托管的 SQLite 队列提交任务。

## 本阶段完成项

- 以单个 lmms-eval Python task 组合 MVBench 的 20 个官方子配置，避免维护 20 份项目 YAML。
- 对切片名与 Arrow 中源视频名不一致的本地 MVBench 媒体进行确定性解析；验证 4000 条均存在。
- 完成 `pytest`（5 项）与 `ruff check`。
- 新增仓库级 `AGENTS.md`，提供可迁移的 uv、缓存、SQLite/Supervisor、评测公平性和实验文档规范；项目专用路径与任务配置不写入其中。
- 在通用规范中记录当前机器的共享数据根目录 `/mnt/hdd1/ymj/VLM`、已知 benchmark 缓存位置及其只读/产物隔离约束。

## 未完成项

- 将 SQLite 消费者交由 Supervisor 实际部署，并建立本次实验的不可变任务 manifest。
- 分别执行三个 benchmark 并把输出、队列数据库、日志和结果摘要归档到本目录对应的实验产物目录。
- 根据本次配置是否追求“lmms-eval 标准协议”或“最大程度贴近作者脚本”，决定是否另做 MLVU 严格对齐任务。

## 关键决策

- MVBench 仅通过 lmms-eval 执行，不运行作者路径；作者仓库没有发布 MVBench evaluator。
- Video-MME 使用无字幕任务 `videomme`。`videomme_w_subtitle` 是另一种输入协议，不能与无字幕分数混写。
- 当前采用 lmms-eval 原生的 MCQ 生成设置，而不是把三个任务强行改成作者仅为 LongVideoBench/MLVU 发布的生成设置。完整差异见 [configuration.md](configuration.md)。

## 代码与环境证据

- 基线 commit：`1ebc4ef66e162752bbaa3b56e9593300764423e3`；当前工作树包含本实验的未提交适配改动。
- Python：由 `uv` 和 `uv.lock` 管理，要求 Python 3.10；lmms-eval 固定为 `0.7.3`。
- 数据缓存：`/mnt/hdd1/ymj/VLM/datasets/benchmarks/hf-home`。
- 权重下载：Supervisor 配置与日志位于 `artifacts/experiments/video-benchmark-lmms-eval/acquisition/`；运行时 socket、PID 与锁位于本地目录 `/var/tmp/flexmem-1000/acquisition/`，避免将 SQLite/锁状态放到共享盘。下载固定为 `lmms-lab/LLaVA-Video-7B-Qwen2@013210b3aff822f1558b166d39c1046dd109520f`，使用 `https://hf-mirror.com` 传输并续传至 `/mnt/hdd1/ymj/VLM/models/lmms-lab/LLaVA-Video-7B-Qwen2`。完成后仍需进行文件大小和 SHA256 校验。
