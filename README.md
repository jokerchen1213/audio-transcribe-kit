# cc-transcribe —— 本地离线语音转写（会议 / 通话录音 → 汇总 + 带说话人转录稿）

把录音转成同目录 `_md/<同名>.md`：front-matter → `## 结构化汇总`（占位，由 `audio-transcribe` skill 填写）→ `## 转录稿`（`<!-- T hh:mm:ss -->` 时间标记 + `**说话人N**：`）。**全程离线、不上传、原件只读、不需要 HuggingFace 账号。**

- **2026-10-22 起默认走 Windows 路线**（见下「Windows 路线」节）；`--local` 走 Mac 本地。**2026-10-31 起 Windows 不可用时自动回退 Mac 本地**（律师裁定），`--win-only` 保留旧的"报 3 不降级"。
- 识别：Qwen3-ASR-1.7B 8-bit（MLX，Apple Silicon GPU）
- 说话人分离：本地默认 Nemotron-3-Diarization（Apple MPS，独立 venv，见「Mac 本地 Nemotron」节）；sherpa-onnx（pyannote-segmentation-3.0 ONNX + 3D-Speaker 中文声纹，纯 CPU、无 torch）保留为退路与回归对照
- 后端可插拔：`--asr` / `--diarizer` 切换；产物登记进同目录 `_转换索引.md`（六列格式与 cc-convert 共用一份索引：`原件 | 页数 | MD | 状态 | 引擎 | 日期`；音频无页码，`页数` 列记时长）

## 安装

```bash
cd <仓库根>                    # 在 cases 下即 _工具/cc-transcribe
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# 系统依赖：ffmpeg / ffprobe（brew install ffmpeg）
```

核心依赖已钉版：`mlx-qwen3-asr==0.4.4`、`sherpa-onnx==1.13.8`、`numpy`、`soundfile`（**不含 torch**）。

首跑自动下载模型，此后全离线：

| 模型 | 体积 | 落点 |
|---|---|---|
| Qwen3-ASR-1.7B 8-bit 权重 | 2.17 GB | `~/.cache/huggingface/hub/models--moona3k--mlx-qwen3-asr-1.7b-8bit` |
| Qwen3-ForcedAligner-0.6B（词级时间戳） | 1.84 GB | `~/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B` |
| sherpa 分割 + 声纹模型 | 约 46 MB | `~/.cache/cc-transcribe/sherpa/` |

两个 HF 模型合计约 4.0 GB，实际字节存在 `~/.cache/huggingface/hub/blobs/`（内容寻址），上表模型目录是指向 blobs 的符号链接视图，统计占用时不要重复计数。

## 模型下载不了？（境内 HF 不可达）

1. `export HF_ENDPOINT=https://hf-mirror.com` 后重跑；
2. 或全部走 ModelScope 本地化（ASR 与强制对齐器都要落地，否则对齐器仍会去 HF 拉）：先 `pip install modelscope`；`modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir <目录1>` 拉原权重，`git clone https://github.com/moona3k/mlx-qwen3-asr`（`scripts/convert.py` **不随 pip 包分发**，pip 装进来的只有库模块 `mlx_qwen3_asr/convert.py`）后用其 `scripts/convert.py --quantize 8 --encoder-bits 8` 转换（参数以上游仓库 README 为准）；另 `modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir <目录2>`（对齐器为官方原始权重，同一加载器按「本地目录含 config.json 优先、否则 `snapshot_download`」解析，**无需转换**）；最后把 `backends.toml` `[asr.qwen3-mlx]` 的 `model` 指向 <目录1>、`aligner` 指向 <目录2>（本地目录优先，完全免网络）。

两条退路都**不需要 HuggingFace 账号**。已缓存权重但断网时，首次 `snapshot_download` 的 HEAD 请求会先超时（约 10 s）再回退到本地缓存；设 `HF_HUB_OFFLINE=1` 可跳过这次探测。sherpa 模型走 GitHub Releases 直链，同样免账号；下载失败时报错会打印直链，手动下好放进缓存目录即可。

## 用法

```bash
./cc-transcribe <文件或目录>... [选项]
```

| 参数 | 说明 |
|---|---|
| `paths`（位置参数） | 一个或多个音频文件 / 目录；目录递归收集 `.m4a .mp3 .mp4 .wav .amr .wma .aac .flac .ogg .opus .mov .webm`，跳过 `_md/` 与隐藏文件 |
| `--asr NAME` | 识别后端，默认取 `backends.toml` 的 `[default] asr` |
| `--diarizer NAME` | 分离后端，默认取 `[default] diarizer`；`none` 关闭 |
| `--speakers N` | 说话人数（已知时强烈建议传，见「已知边界」） |
| `--min-speakers N` / `--max-speakers N` | 人数区间；sherpa 只支持「指定人数」或「阈值自动」，本区间仅用于事后校验并提示 |
| `--context "词 词"` | 热词 / 专名，直传识别后端 |
| `--no-diarize` | 等价 `--diarizer none` |
| `--force` | 已有 `_md` 也重转 |
| `--dry-run` | 只列清单不执行（不加载模型、不触网） |
| `--json` | 额外输出 `_md/<同名>.json`（meta + paragraphs + turns），供调试 |
| `--config PATH` | 指定 `backends.toml` 路径 |
| `--local` | 强制 Mac 本地路线（无视 `[default] runner = "win"`） |
| `--win-only` | 只用 Windows：不可用时退出码 3、不回退本机（默认会自动回退 Mac 本地） |
| `--no-second-opinion` | Windows 路线不跑 MOSS 第二意见 |

不带 `--dry-run` 时也先打印清单：`→ 待转 / 重转`、`· 跳过（已有 _md）`、`· 跳过（已有 _md；原件更新，--force 重转）`。单个文件失败不拖垮整批，退出码非 0。

产物 front-matter 字段：`source / duration / recorded_at / transcribed_at / asr / diarizer / speakers / speakers_hint / quality / context / chunks`（`speaker_map` 由下游改名时补写）。`recorded_at` 从文件名日期（`YYYYMMDD`、`YYYY-MM-DD`、录音笔 `_YYMMDD_HHMMSS`，以及无年份的 `MM-DD ` 开头——年份取原件 mtime 年份，终端打印推得来源）推得，都不命中时回退到原件 mtime。超过 2 小时的录音按 30 分钟切块顺序识别，块数记在 `chunks`。

## Windows 路线（2026-10-22 起默认）

`backends.toml` `[default] runner = "win"` 时，Mac 只把录音转成 16 kHz wav，经 ssh（主机别名 `win`）送到 Windows 3090 的 WSL，一次远端作业跑完 `remote/win/run_job.sh <作业目录> qwen3 nemotron diarizen [moss]`，取回 `out.*.json` 后在 Mac 用原有拼接层写出 `_md`；Windows 侧作业目录 `D:\asr-jobs\<作业号>\` 在 `finally` 中删除。代码：`remote.py`（预检 / 作业编排）、`remote/win/`（四个引擎脚本 + `run_job.sh`）、`remote/asr-lock.yaml`（模型仓库、revision、权重 sha256）、`second_opinion.py`（MOSS 分歧表）。

- **默认组合**（2026-10-22 比测后律师裁定，报告 `~/claude-skills/_analysis/2026-10-22-转写引擎比测.md`）：文字 Qwen3-ASR-1.7B（`-hf`，transformers 5.19 原生）；说话人 Nemotron-3-Diarization，检出满 8 人或 `--speakers` > 8 时改用 DiariZen v2；MOSS-Transcribe-Diarize 只作第二意见，分歧写 `_md/<同名>.分歧.md`，主稿 front-matter 记 `disagreements` / `suspected_gaps`。
- **预检**（开工前一次）：ssh 可达、有人登录、显存空闲 ≥ `min_free_vram_mb`、计划任务 `ocr-glm-server` 不在运行、引擎脚本存在。任一不满足 → 打印 `⚠️ Windows 不可用（原因），改在 Mac 本机转写`，本地后端加载一次，后续文件都走本地（无 MOSS 第二意见）；批处理中途掉线或远端作业锁占用（`run_job.sh` 退出码 75）时，当前文件与其余文件同样改走本地（产物只在 Mac 写，无双写风险）。加 `--win-only` 时保留旧行为：stderr `WINDOWS_UNAVAILABLE: <原因>`、退出码 **3**、不降级。
- **部署**：引擎脚本随本仓库经群晖 Drive 同步到 Windows，WSL 经目录联接 `C:\cc` 读 `/mnt/c/cc/cc-transcribe/remote/win/`；`run_job.sh` 设 `PYTHONDONTWRITEBYTECODE=1`，不在同步目录里产生 `__pycache__`。WSL 环境 `/opt/asr/{core,diarizen}`、模型 `/opt/asr/models/<名>@<sha7>` 由比测阶段装好（运行时全离线）。
- **升级**：并排装候选版本 → 用 Mac `~/.local/share/cc-transcribe-bench/v1/`（AliMeeting 8 场 + 插入片段 + 回归基线 `baseline-2026-10-22.json`）回归 → 指标不退化才改 `asr-lock.yaml` 与引擎脚本。模型更新须在安装期临时开 TUN 走 HuggingFace（钉 revision），装完运行时离线。
- 速度（只记录，不作门槛）：含第二意见约录音时长的一半；不含约 0.2 倍。

## Mac 本地 Nemotron

`--local` 路线默认分离器为 `nemotron`（`backends/nemotron_diar.py`）：以子进程调用独立 venv 里的 `remote/mac/diar_nemotron_mps.py`，与 Windows 引擎同一算法（模型卡 30.4 s 离线流式 + speaker_cache），同版本权重在 M3 Pro MPS 上与 Windows CUDA 的 DER 差 ≤ 0.08 个百分点，约 20 秒 / 小时音频。主 `.venv` 仍不装 torch。

```bash
uv venv --python 3.12 ~/.venvs/cct-nemotron
uv pip install --python ~/.venvs/cct-nemotron/bin/python -r remote/mac/requirements-nemotron.txt
# 模型（只要三个文件）：从 Windows /opt/asr/models/nemotron-diar@f667ed7/ 拷 config.json、processor_config.json、model.safetensors，
# 或安装期开 TUN 后 hf download nvidia/Nemotron-3-Diarization --revision f667ed73aee57d40cc39428eb768b4fd87a0a29e
# 放到 ~/.cache/cc-transcribe/nemotron/nemotron-diar@f667ed7/；model.safetensors 的 sha256 须等于 remote/asr-lock.yaml 所记
```

检出满 8 人、`--speakers` > 8、或 venv / 模型缺失与运行出错时，自动改用 sherpa 并在终端提示，`diarizer` 字段写明原因。

## 后端与扩展

注册表 `backends/__init__.py`、协议 `backends/base.py`（`ASRBackend` / `Diarizer` 两个 Protocol + `Word` / `Segment` / `SpeakerTurn` / `ASRResult` 四个数据类；`ASRBackend.transcribe` 返回 `ASRResult(segments, language, truncated, finish_reason, model_label)`）、配置 `backends.toml`。**新增后端 = 一个适配器文件 + 注册表一行 + 配置一节**，不动主流程；后端独门参数走 `backends.toml` 各自小节，不进协议。

本期实现两个真实后端：`qwen3-mlx`（识别）与 `sherpa`（分离）；`funasr` 只在 `backends.toml` 留注释示例，**未实现**（实现它才需要 torch，故核心 venv 不装）。可插拔性由 `tests/fake_backends.py`（实现同一协议、返回固定数据的假后端）验证。

拼接层与后端无关：词按时间中点归入说话人段，再按「换说话人 / 停顿 > 2 秒 / 超 300 字」切段。Qwen3 的词级时间戳由强制对齐器给出，中文是**逐字、且标点被剥掉**的，适配器按全文指针回贴标点，因此转录稿带标点。说话人编号按首次出现顺序重编为连续的 1、2、3……（sherpa 阈值聚类原始 cluster id 有空洞）。

## 选型依据（摘自 spec §1）

### 识别模型：Qwen3-ASR-1.7B（8-bit MLX）

| 测试集 | Qwen3-ASR-1.7B | Whisper-large-v3 | Fun-ASR-Nano |
|---|---|---|---|
| WenetSpeech meeting（CER） | 5.88 | 19.11 | 6.60 |
| WenetSpeech net | 4.97 | 9.86 | 6.01 |
| AISHELL-2 | 2.71 | 5.06 | 2.75 |
| Fleurs-zh | 2.41 | 4.09 | 2.56 |

来源：Qwen3-ASR 与 Fun-ASR 各自 GitHub README 官方评测（2026-09-20 读取）。中文会议 / 对话场景 Qwen3-ASR-1.7B 在共同公开集上全部略胜；Whisper 差 2–4 倍且长音频有幻觉 / 重复问题，**不备 Whisper**；FireRedASR 精度更高但仅 CUDA，Mac 排除。Mac 运行时选 `mlx-qwen3-asr`（Apache 2.0，与官方逐 token 对齐校验、原生 MLX 强制对齐器），8-bit 输出与 fp16 逐字一致。

### 说话人分离：sherpa-onnx 而非 pyannote 原版 / cam++

- pyannote 是开源分离的公认基准；sherpa-onnx 把其分割模型转成 ONNX 托管在 GitHub Releases，**免 HF 账号**、纯 CPU、无 torch。
- 排除 pyannote.audio 原版：HF 门禁模型，需账号 + 接受条款 + token。
- 排除 FunASR cam++ 作主干：7.2M 声纹 + 聚类、无重叠处理、自动判人数易过切（官方博客 227 秒会议切出 11 人）；保留为备用后端。

### HuggingFace 账号需求核实

| 环节 | 需要账号 |
|---|---|
| Qwen3-ASR 原始权重（HF / ModelScope 双托管） | 否 |
| mlx-qwen3-asr 8-bit 转换权重（公开仓库，`snapshot_download` 匿名） | 否 |
| Qwen3-ForcedAligner-0.6B | 否 |
| sherpa-onnx 分离模型（GitHub Releases） | 否 |
| FunASR 全家（ModelScope `hub="ms"`） | 否 |
| pyannote.audio 原版 | **是**（本设计不用） |

## 基准（本机实测）

环境：Apple M3 Pro / 18 GB / macOS，ffmpeg 8.1，Python 3.12（uv venv），`mlx-qwen3-asr` 0.4.4、`mlx` 0.32.2、`sherpa-onnx` 1.13.8。样本：sherpa 官方公开中文 4 人样本 `0-four-speakers-zh.wav`，时长 56.9 s。以下三行来自**同一次暖跑的慢测会话**（模型已在缓存）中三个彼此独立的用例：识别、分离各自单跑，端到端是另一次完整转写，**因此第三行不等于前两行之和**。只记录实测值，不设门槛。

| 环节 | 耗时 | 内存 |
|---|---|---|
| qwen3-mlx 识别 | 7.4 s | maxrss 1595 MB |
| sherpa 分离（`num_threads` = CPU 核数一半） | 4.3 s（得 10 个说话人段） | — |
| 端到端（ffmpeg 转码 + 识别 + 分离 + 拼接 + 写盘） | 9.1 s | — |

- **经验系数**：9.1 s ÷ 56.9 s ≈ **0.16**，即 1 小时录音约 10 分钟（暖跑、端到端 × 63）。**仅此一个 56.9 s 样本的单次测量**，长录音与多说话人场景未外推验证，按需自行复测更新。
- **冷跑**：首次运行（含下载约 4.0 GB 的 ASR + 对齐器权重）为 92.8 s；sherpa 模型冷跑含下载约 10 s。
- **内存两个口径**：pytest 进程 `maxrss` 在多次暖跑中为 1.6–3.9 GB；`/usr/bin/time -l` 的 peak memory footprint 约 7.9 GB——后者含 Metal / GPU 共享内存与 mmap 的权重页，不是常驻物理内存，两个数都不等于「本工具要吃掉 7.9 GB 内存」。
- 暖跑识别耗时在多次运行中为 5.7–7.4 s，上表取与 sherpa、端到端同一次运行的读数。

## 已知边界

- **分割模型是 pyannote 3.0（2023），非 2025 的 community-1**；对中文线上会议的错误率无公开数据。
- **多人会议务必传 `--speakers N`**：同一 56.9 s 四人样本，`--speakers 4` 精确分出 4 人；不传人数、走默认阈值 0.5 聚类时过切成 7 人。人数未知时可先 `--speakers` 试几个值，或调 `backends.toml` 的 `threshold`。
- **`--speakers N` 只约束聚类数，不保证归属准确**：中途到场者可能被并入既有说话人（实测六人会议中一名后到者并入店长）。多人会议建议 `--speakers N` 与不传各跑一次比对（第二次用 `--force` 输出到同一 `_md`，或先备份），汇总时按内容复核归属。不做启发式合并 / 拆分。
- **热词串宜短（≤ 6 个词）**：`--context` 过长时，嘈杂 / 重叠片段易被 ASR 原样回吐成独立段落；工具会剔除去空白标点后与热词串一致、为其 ≥ 8 字子串或其重复拼接的段落（终端打印 `已剔除 N 段热词回吐`），含其他实质内容的段落不动。
- 重叠说话按词的时间中点归属，不做重叠分离；落在所有说话人段之外的词，1 秒内就近吸附，否则不标说话人。
- 识别 `truncated` 或字数异常低（< 0.3 字/秒）时 front-matter 标 `quality: suspect` 并打印告警；分离失败时降级为无说话人转录稿并标 `quality: diarization-failed`——两种情况产物仍写出，需人工复核。
- 汇总部分工具**永不代写**，只留 `<!-- SUMMARY: pending -->` 占位，由会话模型按 `audio-transcribe` skill 填写。

## 测试

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt   # 跑测试才需要（pytest / pyyaml）
.venv/bin/python -m pytest -q                                      # 快，全假后端，不触网不加载模型
RUN_SLOW=1 .venv/bin/python -m pytest -q tests/test_slow_real_backends.py -s   # 真实模型 + 网络，打印上表 bench 行
```

慢测跑真实 Qwen3-ASR 与 sherpa（首次含模型下载），`-s` 才能看到 `[bench]` 输出。

## 仓库

独立 git 仓库，**仅本地**，未建远程、未推送。若要比照 cc-convert 推私有仓，先过脱敏检查再由使用者指示建远程。
