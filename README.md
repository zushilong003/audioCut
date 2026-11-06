# audiocut — 有声小说音频处理 Pipeline

一条命令完成：**音频转码 → Demucs 人声分离 → 人声降噪 → Whisper 带词级时间戳转写 → 语句级分段（VAD 边缘裁剪）→ ECAPA 声纹聚类 + 性别鉴别 → 按角色分组导出音频 + 文本**。

无大模型调用，全部本地运行；支持 NVIDIA GPU（CUDA 12.8 已验证）加速，无 GPU 时自动回退 CPU。

---

## 目录结构

```
audioCut/
├── requirements.txt               # pip 锁文件
├── README.md
├── models/                        # 本地模型根目录
│   ├── faster-whisper-small/      #   Whisper 转写模型
│   ├── spkrec-ecapa-voxceleb/     #   ECAPA-TDNN 声纹模型
│   ├── hub/checkpoints/           #   Demucs 权重 (设 TORCH_HOME=models 后下载到这里)
│   └── torch-2.11.0+cu128-*.whl   #   CUDA 版 torch 离线安装包 (可选)
└── src/
    ├── config/                    # ← 配置集中存放
    │   ├── __init__.py            #   re-export config 包
    │   ├── config.py              #   所有常量 + set_run_dir()
    │   └── .env                   #   路径 / 参数配置
    ├── audio/                     # ← 输入输出目录
    │   ├── input/                 #   原始有声小说放这里
    │   │   └── story.m4a / story.wav / story.mp3
    │   └── output/                #   每次 run 自动建一个时间戳子目录
    │       ├── backup_input/      #     原始非 wav 输入归档
    │       └── 20261105_013219_6831/ #  年月日_时分秒_毫秒
    │           ├── separated/     #       Step1/1.5: 分离与降噪后的人声轨
    │           │   ├── vocals.wav
    │           │   └── vocals_denoised.wav
    │           ├── segments/      #       Step2-3: Whisper 转写 + 分段切片
    │           │   ├── seg_0001_*.wav
    │           │   └── segments.json      # 含词级时间戳
    │           ├── roles/         #       Step6: 角色音频 + 对应文本 (一一对应)
    │           │   ├── 角色A01-男.wav / 角色A01-男.txt
    │           │   ├── 角色A02-男.wav / 角色A02-男.txt
    │           │   └── ...
    │           ├── embeddings.json      # Step4: 每句 ECAPA 声纹向量 (缓存)
    │           └── roles_analysis.json  # Step5: 声纹聚类结果 (分析参考)
    └── audiocut/
        ├── __init__.py
        ├── main.py                # Pipeline 主入口 (含格式统一 + run 目录创建)
        ├── demucs_separator.py    # Step1: Demucs two-stems 人声分离
        ├── speech_denoiser.py     # Step1.5: noisereduce 人声降噪
        ├── transcriber.py         # Step2: faster-whisper 带词级时间戳转写
        ├── segmenter.py           # Step3: 分段 + VAD 边缘裁剪 + 切片落盘
        ├── speaker_verifier.py    # Step4-5: ECAPA-TDNN 声纹向量 + 聚类
        ├── gender_detector.py     # Step6: 基频 F0 性别鉴别
        └── role_extractor.py      # Step6: 聚类 + 性别 + 命名导出
```

---

## Pipeline 流程

```
 story.m4a / story.mp3 / story.wav
               │
    [ffmpeg 统一为 story.wav]
               │
   ┌───────────▼────────────┐
   │  Step 1: Demucs        │  人声分离 (htdemucs_ft, --shifts 5, 有 GPU 自动用 GPU)
   │  输出: vocals.wav      │
   └───────────┬────────────┘
               │
   ┌───────────▼────────────┐
   │  Step 1.5: 降噪        │  noisereduce 非平稳频谱降噪
   │  输出: vocals_denoised.wav
   └───────────┬────────────┘
               │
   ┌───────────▼────────────┐
   │  Step 2: faster-whisper │  词级时间戳转写 + VAD 过滤 (cuda/float16)
   │  models/faster-whisper-small/
   └───────────┬────────────┘
               │
   ┌───────────▼────────────┐
   │  Step 3: Segmenter     │  语句切分 (句末标点/停顿) + VAD 边缘裁剪
   │  输出: segments/*.wav  │  裁剪用 Whisper 词级时间戳判定:
   │        segments.json   │  含词起点的语音段保留, 之外判为音乐/噪声残留
   └───────────┬────────────┘
               │
   ┌───────────▼────────────┐
   │  Step 4-5: 声纹聚类    │  ECAPA-TDNN 向量 + 层次聚类 → 音色分组
   │  (ENABLE_VOICE_ANALYSIS)│  按音色簇总时长降序编号, 写入 roles_analysis.json
   └───────────┬────────────┘
               │
   ┌───────────▼────────────┐
   │  Step 6: 角色导出      │  5s <= 时长 <= 19s + 基频 F0 性别鉴别
   │  输出: 角色{字母}{NN}-{性别}.wav/.txt
   │  字母 A-Z=不同角色(按时长降序), NN=该角色第几个片段
   └────────────────────────┘
```

---

## 环境要求

| 项 | 要求 |
|----|------|
| Python | **3.12.8** |
| 包管理 | [uv](https://docs.astral.sh/uv/) (推荐) 或 pip |
| 系统 | Windows 10+ |
| 外部工具 | **ffmpeg** (必须在 PATH 里) |
| GPU (可选) | NVIDIA + CUDA 12.8，torch/torchaudio `+cu128` 版；无 GPU 自动回退 CPU |

### 核心依赖

| 包 | 版本 | 用途 |
|----|------|------|
| demucs | 4.1.0 | 人声分离 |
| faster-whisper | 1.2.1 | 带词级时间戳语音转写 |
| ctranslate2 | 4.8.5 | Whisper 推理后端 |
| torch / torchaudio | 2.11.0+cu128 | 深度学习底座 (CUDA 版) |
| speechbrain | 1.1.2 | ECAPA-TDNN 声纹提取 |
| scikit-learn | 1.9.1 | 声纹层次聚类 |
| noisereduce | 3.0.3 | 人声轨降噪 |
| scipy | 1.17.1 | 重采样 |
| soundfile | 0.14.0 | wav 读写 |
| python-dotenv | 1.2.4 | .env 加载 |

完整锁版本见 [requirements.txt](./requirements.txt)。

---

## 快速开始

### 1. 安装

```powershell
cd audioCut
uv venv --python 3.12.8 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
uv pip check
```
运行：
需要改成story.m4a / story.wav / story.mp3 放进 src/audio/input/ 目录下
.venv\Scripts\activate
python -m audiocut.main

GPU 用户注意：`torch==2.11.0+cu128` 不在默认 PyPI 源，需从 PyTorch 官方源安装（wheel 约 2.6GB，已备份在 `models/` 下可离线安装）：

```powershell
uv pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### 2. 模型目录

- Whisper：`models/faster-whisper-small/`（`model.bin` + `tokenizer.json` 等），缺失时首次运行自动下载
- 声纹：`models/spkrec-ecapa-voxceleb/`（`hyperparams.yaml` + `embedding_model.ckpt`），缺失时自动下载，国内建议 `$env:HF_ENDPOINT="https://hf-mirror.com"`
- Demucs：设 `TORCH_HOME` 指向项目 `models`，权重会下载到 `models/hub/checkpoints/`

### 3. 放音频

把有声小说放进 `src/audio/input/`，命名为 `story.m4a` / `story.wav` / `story.mp3`。

### 4. 运行

```powershell
$env:PYTHONPATH="src"; $env:TORCH_HOME=(Resolve-Path .\models).Path; python -m audiocut.main
```

---

## 关键机制

### 人声后处理降噪 (Step 1.5)

Demucs 输出的人声轨仍可能带轻微背景噪声/音乐残留，用 noisereduce 做非平稳频谱降噪（`DENOISE_PROP_DECREASE=0.8`），降噪后再送 Whisper 与分段。注意：压在台词下面的垫乐属于分离问题，频谱降噪无法在人声帧上消除它。

### VAD 边缘裁剪 (Step 3)

语句片段首尾可能残留静音或短暂音乐（Whisper 词时间戳会把末尾词拉长盖住音乐）。裁剪规则：

1. Silero VAD 在片段内找语音段
2. 用 Whisper 词级时间戳交叉验证：只保留「包含词起点」的首尾语音段
3. 段外内容判为音乐/噪声残留，裁掉；边缘保留 `TRIM_EDGE_PAD_MS`(150ms) 余量

纯 VAD 会把响亮音乐误判为语音，所以必须结合词级时间戳，避免误裁真实台词。

### 声纹聚类 (Step 4-5)

[speaker_verifier.py](src/audiocut/speaker_verifier.py)：每句提 ECAPA-TDNN 192 维向量，层次聚类（阈值 `SPK_CLUSTER_THRESHOLD=0.65`）后按质心相似度 `SPK_CLUSTER_MERGE=0.75` 合并同音色簇，按簇总时长降序编号，结果写入 `roles_analysis.json`。可通过 `ENABLE_VOICE_ANALYSIS=false` 关闭（关闭后退化为顺序命名 `角色NN.wav`）。

### 性别鉴别 (Step 6)

[gender_detector.py](src/audiocut/gender_detector.py)：对每个音色簇取最长片段，用自相关法 (ACF) 估计基频 F0 中位数，`< GENDER_F0_BOUNDARY`(170Hz) 判男，否则判女。无需额外依赖（用 scipy/numpy）。

### 角色导出 (Step 6)

导出规则：`5s <= 语句时长 <= 19s`（含两端，见 `ROLE_CLIP_MIN_DURATION` / `ROLE_CLIP_MAX_DURATION`）。按声纹聚类结果分组命名：

```
角色{字母}{序号}-{性别}.wav
角色{字母}{序号}-{性别}.txt
```

- **字母 A-Z**：不同音色角色，按簇总时长降序分配（A 最常出现）
- **序号 01-99**：该角色下的第几个音频片段，按在原音频中出现顺序编号
- **性别 男/女**：基频 F0 判定

例：`角色A01-男.wav`、`角色A02-男.wav` 是同一位男性角色的两个片段；`角色C01-女.wav` 是另一位女性角色的第一个片段。音频与文本同名一一对应。

---

## 可调参数 (.env)

| 键 | 默认 | 说明 |
|----|------|------|
| `INPUT_AUDIO_DIR` | `src/audio/input` | 原始音频目录 |
| `OUTPUT_DIR` | `src/audio/output` | 所有产物根目录 |
| `WHISPER_MODEL_DIR` | `models` | Whisper 本地模型根目录 |
| `WHISPER_MODEL_NAME` | `faster-whisper-small` | 模型目录名 |
| `DEMUCS_MODEL` | `htdemucs_ft` | Demucs two-stems 模型 (ft 版分离更干净) |
| `DEMUCS_SHIFTS` | `5` | 随机平移平均次数, 越大越干净越慢 |
| `ENABLE_DENOISE` | `true` | 是否对人声轨做 noisereduce 降噪 |
| `DENOISE_STATIONARY` | `false` | 降噪模式 (非平稳更适合音乐残留) |
| `DENOISE_PROP_DECREASE` | `0.8` | 噪声衰减比例 |
| `TRIM_SEGMENT_EDGES` | `true` | 是否用 VAD+词级时间戳裁剪片段首尾 |
| `TRIM_EDGE_PAD_MS` | `150` | 裁剪后保留的边缘余量 (毫秒) |
| `MIN_CLIP_DURATION` | `4` | 语句原子最短秒数 |
| `MAX_CLIP_DURATION` | `12` | 语句原子最长秒数, 超过强制在逗号处切 |
| `ROLE_CLIP_MIN_DURATION` | `5` | 角色导出时长下限 (含) |
| `ROLE_CLIP_MAX_DURATION` | `19` | 角色导出时长上限 (含) |
| `SPEAKER_GAP_THRESHOLD` | `0.45` | 词间静音超过它视为换人/停顿边界 |
| `ENABLE_VOICE_ANALYSIS` | `true` | 声纹聚类总开关 (关闭退化为 角色NN.wav 顺序命名) |
| `SPK_MODEL_DIR` | `models` | 声纹模型根目录 |
| `SPK_MODEL_NAME` | `speechbrain/spkrec-ecapa-voxceleb` | 声纹模型名 |
| `SPK_CLUSTER_THRESHOLD` | `0.65` | 语句归入音色簇的相似度阈值 |
| `SPK_CLUSTER_MERGE` | `0.75` | 簇质心合并阈值 |
| `GENDER_F0_BOUNDARY` | `170` | 性别分界 Hz: F0 中位数 < 该值判男, 否则判女 |
| `F0_ACF_PEAK_THRESHOLD` | `0.35` | 自相关峰低于该值视为清音, 不计入 F0 |
| `ROLE_LETTERS` | `ABCDEFGHIJKLMNOPQRSTUVWXYZ` | 角色字母池 (按簇时长降序分配) |

---

## 故障排查

| 现象 | 可能原因 | 解决 |
|------|----------|------|
| `RuntimeError: demucs 分离失败` | ffmpeg 没装好或模型首次下载失败 | `ffmpeg -version` 检查；换网络重跑 |
| Demucs 权重下载慢/失败 | 默认缓存目录在 C 盘且未走镜像 | 设 `TORCH_HOME=models` 后重跑 |
| Whisper 速度很慢 | 没用上 GPU | 确认 `torch.cuda.is_available()`，安装 `+cu128` 版 torch |
| `Could not parse CUDA device string 'cuda'` | speechbrain 对 device 字符串的解析警告 | 无害提示，自动回退到 device 0 |
| 片段末尾仍有轻微音乐 | 垫乐压在台词下，属分离极限 | 调大 `DENOISE_PROP_DECREASE` 或接受；裁剪无法处理人声帧内的音乐 |
| 输出角色片段偏少 | 时长窗口不合适 | 调 `ROLE_CLIP_MIN/MAX_DURATION` |

---

## License

仅项目源代码。模型权重 (Demucs / Whisper / ECAPA) 遵循各自许可证。
