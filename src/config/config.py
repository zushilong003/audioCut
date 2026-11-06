"""配置模块 — 从 .env 加载所有配置项."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录 (src/config/config.py → 上推 3 层到 audioCut/)
#   config.py → parent=src/config → parent=src → parent=audioCut
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# .env 与 config.py 同目录
ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_FILE)


def _get(key: str, default: str = "") -> str:
    val = os.environ.get(key, default).strip()
    return val


def _get_float(key: str, default: float) -> float:
    try:
        return float(_get(key, str(default)))
    except ValueError:
        return default


# ── Whisper ──────────────────────────────────────────────
WHISPER_MODEL_DIR = PROJECT_ROOT / _get("WHISPER_MODEL_DIR", "models")
WHISPER_MODEL_NAME = _get("WHISPER_MODEL_NAME", "small")

# ── Demucs ──────────────────────────────────────────────
DEMUCS_MODEL = _get("DEMUCS_MODEL", "htdemucs_ft")
# shifts: 多次随机平移平均, 越大分离越干净, 耗时线性增加
DEMUCS_SHIFTS = int(_get("DEMUCS_SHIFTS", "5"))

# ── 人声后处理降噪 ───────────────────────────────────────
# 对 Demucs 人声轨再做一次非平稳频谱降噪, 去背景音乐残差与环境噪声
ENABLE_DENOISE = _get("ENABLE_DENOISE", "true").lower() in {"1", "true", "yes", "on"}
DENOISE_STATIONARY = _get("DENOISE_STATIONARY", "false").lower() in {"1", "true", "yes", "on"}
DENOISE_PROP_DECREASE = _get_float("DENOISE_PROP_DECREASE", 0.8)

# ── 分段边缘裁剪 ──────────────────────────────────────────
# 用 Silero VAD 裁掉语句片段首尾的静音/音乐残留
TRIM_SEGMENT_EDGES = _get("TRIM_SEGMENT_EDGES", "true").lower() in {"1", "true", "yes", "on"}
TRIM_EDGE_PAD_MS = int(_get("TRIM_EDGE_PAD_MS", "150"))

# ── 路径 ──────────────────────────────────────────────
INPUT_AUDIO_DIR = PROJECT_ROOT / _get("INPUT_AUDIO_DIR", "audio/input")
OUTPUT_DIR = PROJECT_ROOT / _get("OUTPUT_DIR", "audio/output")

for d in (INPUT_AUDIO_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 三个产物子目录 —— 只是导入期的占位值; set_run_dir 会在每次 run 开始时
# 把它们重映射到 <OUTPUT_DIR>/<run_name>/separated|segments|roles
SEPARATED_DIR = OUTPUT_DIR / "separated"
SEGMENTS_DIR = OUTPUT_DIR / "segments"
ROLES_DIR = OUTPUT_DIR / "roles"


def set_run_dir(run_name: str) -> Path:
    """在 OUTPUT_DIR 下创建 run 子目录, 并把 SEPARATED_DIR / SEGMENTS_DIR / ROLES_DIR 重映射进去.

    Args:
        run_name: 形如 "20261104_190723_0391" 的时间戳目录名 (年月日_时分秒_毫秒).

    Returns:
        本次 run 的根目录 (OUTPUT_DIR / run_name).
    """
    global SEPARATED_DIR, SEGMENTS_DIR, ROLES_DIR

    run_root = OUTPUT_DIR / run_name

    SEPARATED_DIR = run_root / "separated"
    SEGMENTS_DIR = run_root / "segments"
    ROLES_DIR = run_root / "roles"

    for d in (SEPARATED_DIR, SEGMENTS_DIR, ROLES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 同步更新 src.config 包命名空间 —— `import config` 拿到的是包对象,
    # 而 `from .config import *` 在包加载时就把常量拷贝过去了, 后续改全局变量对包不可见.
    # 所以这里需要手动把包命名空间里的名字也改掉, 让其他模块能看到新值.
    import sys as _sys
    _pkg = _sys.modules.get("config")
    if _pkg is not None:
        _pkg.SEPARATED_DIR = SEPARATED_DIR
        _pkg.SEGMENTS_DIR = SEGMENTS_DIR
        _pkg.ROLES_DIR = ROLES_DIR

    return run_root

# ── 音频片段参数 ──────────────────────────────────────────
# 只保留时长落在 [MIN_CLIP_DURATION, MAX_CLIP_DURATION] 内的音色片段
MIN_CLIP_DURATION = _get_float("MIN_CLIP_DURATION", 4.0)
MAX_CLIP_DURATION = _get_float("MAX_CLIP_DURATION", 12.0)
# 两词间隔超过该值视为换人/明显停顿, 用于语句切分边界
SPEAKER_GAP_THRESHOLD = _get_float("SPEAKER_GAP_THRESHOLD", 0.45)
# 同角色相邻语句合并时允许的最大间隔 (秒), 超过就不合并, 避免拼出空档很长的音频
ROLE_JOIN_GAP = _get_float("ROLE_JOIN_GAP", 1.0)

# ── 角色片段导出 ──────────────────────────────────────────
# 只导出时长落在 [ROLE_CLIP_MIN_DURATION, ROLE_CLIP_MAX_DURATION] 内的语句片段 (含端点)
ROLE_CLIP_MIN_DURATION = _get_float("ROLE_CLIP_MIN_DURATION", 5.0)
ROLE_CLIP_MAX_DURATION = _get_float("ROLE_CLIP_MAX_DURATION", 19.0)

# Whisper 中文常输出繁体, 转写后统一转简体 (需要 opencc-python-reimplemented)
ZH_CONVERT_TO_SIMPLIFIED = _get("ZH_CONVERT_TO_SIMPLIFIED", "true").lower() in {"1", "true", "yes", "on"}

# ── 切分时换人检测 (ECAPA 滑窗, 切分阶段优化, 避免一片多人) ──
ENABLE_SPEAKER_SPLIT = _get("ENABLE_SPEAKER_SPLIT", "true").lower() in {"1", "true", "yes", "on"}
# 只对时长 >= 该值的语句原子做换人检测
SPEAKER_CHECK_MIN_DURATION = _get_float("SPEAKER_CHECK_MIN_DURATION", 4.0)
# 相邻滑窗声纹相似度低于该值判为换人 (ECAPA 在 1s 窗上同人基线约 0.36, 取 0.18)
SPK_CHANGE_THRESHOLD = _get_float("SPK_CHANGE_THRESHOLD", 0.18)

# ── 声纹 (ECAPA-TDNN) 参数 ───────────────────────────────
# 声纹分析总开关: 开启后 Step4-5 执行 ECAPA 聚类, 角色导出按音色分组命名
ENABLE_VOICE_ANALYSIS = _get("ENABLE_VOICE_ANALYSIS", "true").lower() in {"1", "true", "yes", "on"}
SPK_MODEL_DIR = PROJECT_ROOT / _get("SPK_MODEL_DIR", "models")
SPK_MODEL_NAME = _get("SPK_MODEL_NAME", "speechbrain/spkrec-ecapa-voxceleb")
# 余弦相似度阈值: 片段与角色质心相似度 >= 该值才允许输出
SPK_THRESHOLD = _get_float("SPK_THRESHOLD", 0.60)
# 相邻语句声纹相似度 >= 该值才允许合并为同一片段 (防止跨说话人拼接)
SPK_JOIN_THRESHOLD = _get_float("SPK_JOIN_THRESHOLD", 0.60)
# 声纹聚类: 语句与簇质心余弦 >= 该值才归入该音色簇
SPK_CLUSTER_THRESHOLD = _get_float("SPK_CLUSTER_THRESHOLD", 0.65)
# 聚类后处理: 两个簇的质心相似度 >= 该值则合并 (同一个音色被切碎时并回来)
SPK_CLUSTER_MERGE = _get_float("SPK_CLUSTER_MERGE", 0.75)
# 句内换人判定: 某个语音块比片段主体更像紧邻的上一句/下一句时, 至少要高出该值
SPK_MARGIN = _get_float("SPK_MARGIN", 0.10)
# 片段内部细粒度滑窗校验: 窗口里对手角色的相似度下限 / 相对优势阈值
# (两个相似音色的绝对相似度可能都在 0.5 附近, 只要谁更像就足以判定换人)
SPK_WINDOW_FLOOR = _get_float("SPK_WINDOW_FLOOR", 0.45)
SPK_WINDOW_MARGIN = _get_float("SPK_WINDOW_MARGIN", 0.05)
# 片段内部块一致性: 语音块与片段主体的相似度低于该值, 且更像紧邻的上一句/下一句,
# 判定句中混入了第二个音色
SPK_SELF_THRESHOLD = _get_float("SPK_SELF_THRESHOLD", 0.55)
# 主播模式: auto=按声纹聚类结果判断 / multi=强制按角色分别提取 / single=强制单音色
SINGLE_PERFORMER_MODE = _get("SINGLE_PERFORMER_MODE", "auto").lower()

# ── 性别鉴别 (基频 F0) ───────────────────────────────────
# 自相关 F0 估计参数
F0_FRAME_MS = int(_get("F0_FRAME_MS", "40"))
F0_HOP_MS = int(_get("F0_HOP_MS", "10"))
F0_MIN_HZ = _get_float("F0_MIN_HZ", 70.0)
F0_MAX_HZ = _get_float("F0_MAX_HZ", 400.0)
# 单帧自相关峰低于该值视为清音/无声, 不计入 F0 中位数
F0_ACF_PEAK_THRESHOLD = _get_float("F0_ACF_PEAK_THRESHOLD", 0.35)
# 性别分界: F0 中位数 < 该值判男, >= 该值判女 (男声多 < 165Hz, 女声多 > 180Hz)
GENDER_F0_BOUNDARY = _get_float("GENDER_F0_BOUNDARY", 170.0)

# ── 角色导出命名 ─────────────────────────────────────────
# 按音色聚类命名: 角色{字母}{序号}-{性别}.wav, 同一字母=同一角色, 序号为该角色第几个片段
# 字母分配按簇总时长降序 (A 最常出现), 序号按片段在原音频中出现顺序编号
ROLE_LETTERS = _get("ROLE_LETTERS", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
