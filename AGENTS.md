# 可迁移的研究项目协作规范

本文件面向机器学习研究、论文复现和长程评测项目。它不固定模型、数据集、GPU 编号、缓存路径或 benchmark；这些项目/实验专用值必须写入实验 manifest 与账本，而不是写入本文件。

## 开始工作前

1. 阅读本文件与当前实验目录 `docs/YYYY-MM-DD-<experiment-name>/ledger.md`。
2. 查询任务队列、Supervisor、已有日志和结果；不得重复提交已成功的任务。
3. 检查工作树，保留用户或其他工作者的无关改动。
4. GPU 任务开始前确认用户授权的物理 GPU 空闲；默认每张卡同时最多一个队列任务。

上下文恢复或新会话开始后，重复以上检查。

## Python、CUDA 与数据环境

- 用 `uv` 管理 Python 版本、依赖和锁文件。正式命令统一使用 `uv run --locked --no-sync ...`。
- 普通 Python 包可以使用可靠的区域镜像；CUDA PyTorch 必须来自与服务器驱动匹配的官方 PyTorch wheel index，避免误装 CPU 包。
- 不修改系统 CUDA Toolkit。仅在编译 CUDA 扩展时，通过 manifest 显式设置 `CUDA_HOME`、`PATH` 和 `LD_LIBRARY_PATH`。
- 不依赖 `source ~/.bashrc` 作为自动化环境初始化。队列任务、Supervisor 和子进程必须从 manifest 获得完整环境。
- Hugging Face、数据集和模型缓存应在实验 manifest 中显式设置，并在同一项目内共用，避免重复下载和占满磁盘。离线运行时显式设置相应的离线变量。
- 下载大型公开文件时固定来源、revision、路径、字节数和 SHA256；镜像只可作为传输渠道，使用前要验证其与来源文件一致。

## 当前机器共享数据目录

- 共享数据根目录是 `/mnt/hdd1/ymj/VLM`。它用于多个项目共用的模型权重、数据集和 benchmark 缓存；操作前先检查目标路径，不得覆盖、删除、移动或重命名已有共享数据。
- 已知 benchmark Hugging Face 缓存位于 `/mnt/hdd1/ymj/VLM/datasets/benchmarks/hf-home`。使用该缓存的离线任务在 manifest 中显式设置：

  ```bash
  HF_HOME=/mnt/hdd1/ymj/VLM/datasets/benchmarks/hf-home
  HF_HUB_OFFLINE=1
  HF_DATASETS_OFFLINE=1
  ```

- 共享根目录不是队列数据库、锁文件、临时目录或实验日志的默认位置。SQLite 状态放在已验证的本地目录；原始实验产物放在项目的 `artifacts/experiments/`，除非实验账本明确批准另一处专用产物目录。
- 新增大型共享数据前，先确认来源、许可、容量、固定 revision 和校验信息，并在 acquisition manifest 或实验账本中记录；不要在共享根目录中留下来源不明的临时下载或重复副本。

## 模型权重与大型文件获取

- 下载前明确记录上游仓库/发布页、许可证、固定 revision（commit 或版本号）、目标目录、预计容量、所需文件清单，以及每个大文件的字节数和 SHA256/LFS OID。模型按发布者与仓库名等稳定层级分类存入共享模型目录；不得将不同模型的文件混放。
- 传输镜像只承担下载渠道的角色，不能替代来源标识。优先以固定 revision、响应中的 commit/ETag 与发布的大小、哈希核验镜像内容；若上游不可访问，必须在 acquisition manifest 说明原因和已完成的替代验证，不能把镜像说成原始来源。
- 大型、耗时或无人值守的下载一律由独立 Supervisor 实例托管，不以前台终端、`nohup` 或临时后台进程维持。每个下载实例使用仅当前用户可访问的本地 socket、PID、锁和临时状态目录；配置、原始日志和 acquisition manifest 存入相应实验的 `artifacts/experiments/<experiment-name>/acquisition/`。
- Supervisor 下载命令使用绝对 `uv` 路径和显式环境（包括镜像端点、HF/缓存目录和 `PATH`），固定 revision、文件清单和目标目录；应启用下载器的断点续传。下载本身是有界任务，默认 `autorestart=false`、`startretries=0`：失败后先查看日志、检查来源与磁盘，再由人明确决定是否重试。
- 接管一个已在下载的任务时，先核对其 PID/进程组、目标目录和缓存；停止原进程并确认退出后，才启动 Supervisor 中同一份可续传命令，绝不可并发运行两份针对同一目标目录的下载。
- 下载完成不等于可用。先核验所需文件齐全、字节数、每个权重分片的 SHA256，以及索引/配置对分片的引用，再写入完成状态。保留最终 manifest 以便复现；推理无关且存在反序列化风险的训练状态或 checkpoint 元数据不要默认下载。

## SQLite GPU 队列与 Supervisor

- 正式长程训练、评测和消融使用 SQLite 单消费者队列。任务必须以绝对 `uv` 可执行文件开头的 argv 数组描述，禁止 `shell=True`、`eval`、`bash -c` 和隐式环境继承。
- manifest 是不可变的：创建队列后不得追加或修改任务。成功任务跳过；失败、超时和中断默认不自动重试。
- SQLite 数据库和锁文件必须存放在确认支持 SQLite 锁与 `flock` 的本地目录，且仅运行用户可写；不能放在共享网络文件系统。
- 调度器要记录任务 ID、完整 argv、资源分配、进程身份、起止时间、退出码、状态和日志位置。恢复时发现旧进程或不确定的 claimed 窗口，必须拒绝重复派发。
- 持久消费者一律由 Supervisor 托管。建议设置 `autostart=false`、`autorestart=false`、`startretries=0`、`stopasgroup=true`、`killasgroup=true`；恢复前先检查数据库、日志和遗留进程组。
- Supervisor 保证进程脱离终端，不保证容器重建后的状态保留，也不能替代对独立任务进程组的清理和检查。
- 若主机没有 Supervisor，先使用项目认可的用户级包管理方式安装；不要用 `nohup`、终端后台任务或临时 systemd 替代。

## 评测与公平比较

- 使用评测框架时，所有比较对象应固定数据版本、划分、提示词、评分器、生成参数、随机种子、图像/视频预处理和输入协议（例如字幕策略）。
- 抽帧、最大帧数、视觉 token 预算、分辨率和推理 batch size 都是实验变量，必须显式记录。要将“抽帧消融”与“模型结构、记忆压缩或检索策略消融”分开。
- 若论文作者未公开某 benchmark 的 evaluator，不得声称逐项复现作者脚本；应准确表述为采用论文模型配置的扩展评测。
- 模型适配、数据路径适配和 benchmark task 应优先做成小而可安装/可复用的组件，避免修改第三方评测框架的安装目录。

## 实验文档与产物

- 每个新实验项目首次启动时创建稳定目录 `docs/YYYY-MM-DD-<experiment-name>/`。跨日继续时更新同一目录，禁止按自然日拆分同一实验。
- 每个实验至少有 `ledger.md`，记录目标、状态、完成项、运行项、失败项、关键决策、精确配置、产物位置、阻塞原因和下一步。
- 文档必须区分“代码支持”“本阶段实际执行”和“已删除”。未运行的配置、推断或计划不得写成结果。
- 文档存放说明、图表和简要总结；原始日志、队列数据库、manifest、输出、源码快照和测试报告存放于 `artifacts/experiments/<experiment-name>/`。
- 每次代码/配置改动、任务提交、完成、失败或资源阻塞后更新账本。阶段完成后另写总结，说明成功、失败、原因、尝试和结论。
- 实验、复现和教学文档使用中文；图片复制到文档目录的 `images/` 并使用相对路径引用。

## 修改、验证与 Git

- 采用最小必要改动；不重构无关代码，不覆盖已有未提交改动。
- 修改后运行与风险相称的单元测试、静态检查、数据检查或 smoke test。调度器至少验证资源分配、环境传递、成功/失败/超时/中断/恢复和 Supervisor 生命周期。
- 只有用户明确要求时才创建 commit；推送也需要当轮明确授权。
- 由 Codex 创建或改写的提交必须显式使用 `Codex <codex@openai.com>` 作为 author 和 committer；不要依赖主机自动推断的用户名、主机名或全局 Git 配置。提交时以仓库本地配置或 `git -c user.name=Codex -c user.email=codex@openai.com` 显式传入该身份。该元数据不改变远端认证账户。
