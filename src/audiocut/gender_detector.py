"""性别鉴别模块 — 基于基频 F0 的说话人性别判定 (无额外依赖, 用 scipy/numpy).

原理:
1) 对音频分帧加窗, 用自相关法 (ACF) 估计每帧基频 F0.
2) 过滤清音/无声帧 (自相关峰不够显著), 对浊音帧的 F0 取中位数.
3) 中位数 < MALE_F0_MAX_HZ 判男, > FEMALE_F0_MIN_HZ 判女, 中间带按阈值分界.

有声小说旁白多为男声 (F0 80-165Hz), 女声 F0 普遍 180-320Hz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

import config

TARGET_SR = 16000


def _resample(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return wav.astype("float32")
    g = gcd(sr, TARGET_SR)
    return resample_poly(wav, TARGET_SR // g, sr // g).astype("float32")


def _estimate_f0_frame(frame: np.ndarray, sr: int, fmin: float, fmax: float) -> float:
    """单帧自相关 F0 估计, 返回 Hz; 清音/无声返回 0.0."""
    # 去直流 + 汉宁窗
    frame = frame - frame.mean()
    win = np.hanning(len(frame)).astype("float32")
    framed = frame * win

    # 自相关 (用 FFT 加速)
    n = len(framed)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    spec = np.fft.rfft(framed, fft_size)
    acf = np.fft.irfft(spec * np.conj(spec), fft_size)[:n]
    acf /= acf[0] + 1e-12

    # 只在 [fmin, fmax] 对应时延范围找峰
    min_lag = max(1, int(sr / fmax))
    max_lag = min(n - 1, int(sr / fmin))
    if max_lag <= min_lag:
        return 0.0

    region = acf[min_lag:max_lag + 1]
    peak_idx = int(np.argmax(region))
    peak_val = region[peak_idx]

    # 峰不够显著 → 清音/无声
    if peak_val < config.F0_ACF_PEAK_THRESHOLD:
        return 0.0

    lag = peak_idx + min_lag

    # 抛物线插值提高精度
    if 0 < peak_idx < len(region) - 1:
        a, b, c = region[peak_idx - 1], region[peak_idx], region[peak_idx + 1]
        denom = (a - 2 * b + c)
        if abs(denom) > 1e-12:
            lag += 0.5 * (a - c) / denom

    return float(sr / lag)


def median_f0(path: Path) -> float:
    """计算音频的浊音帧 F0 中位数 (Hz); 无有效浊音返回 0.0."""
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = _resample(wav, sr)
    sr = TARGET_SR

    if len(wav) < int(0.3 * sr):
        return 0.0

    frame_len = int(config.F0_FRAME_MS * sr / 1000)
    hop_len = int(config.F0_HOP_MS * sr / 1000)
    fmin, fmax = config.F0_MIN_HZ, config.F0_MAX_HZ

    f0s: list[float] = []
    for start in range(0, len(wav) - frame_len, hop_len):
        frame = wav[start:start + frame_len]
        f0 = _estimate_f0_frame(frame, sr, fmin, fmax)
        if f0 > 0:
            f0s.append(f0)

    if not f0s:
        return 0.0
    return float(np.median(f0s))


def detect_gender(path: Path) -> tuple[str, float]:
    """鉴别单段音频的性别.

    Returns:
        ("男" | "女", f0_median_hz). 无有效浊音时 f0=0, 按男声兜底 (旁白多为男).
    """
    f0 = median_f0(path)
    if f0 <= 0:
        return "男", 0.0
    if f0 < config.GENDER_F0_BOUNDARY:
        return "男", f0
    return "女", f0


def detect_cluster_gender(segment_paths: list[Path]) -> tuple[str, float]:
    """对一个音色簇判定性别: 取簇内最长片段的 F0 中位数 (最长片段基频最稳).

    若最长片段 F0 无效, 依次尝试次长片段.
    """
    if not segment_paths:
        return "男", 0.0

    # 按时长降序
    def _dur(p: Path) -> float:
        try:
            info = sf.info(str(p))
            return float(info.frames) / max(1, info.samplerate)
        except Exception:
            return 0.0

    ordered = sorted(segment_paths, key=_dur, reverse=True)
    for p in ordered:
        gender, f0 = detect_gender(p)
        if f0 > 0:
            return gender, f0
    # 全部无效 → 兜底男声
    return "男", 0.0
