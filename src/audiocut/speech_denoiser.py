"""人声降噪模块 — 对 Demucs 人声轨做非平稳频谱降噪, 去背景音乐残差与环境噪声."""
from __future__ import annotations

from pathlib import Path

import noisereduce as nr
import soundfile as sf

import config


def denoise_vocals(src: Path, dest: Path | None = None) -> Path:
    """对 vocals.wav 做频谱降噪, 输出 vocals_denoised.wav.

    Args:
        src: Demucs 输出的人声轨.
        dest: 降噪输出路径, 默认与 src 同目录的 vocals_denoised.wav.

    Returns:
        降噪后的音频路径.
    """
    if dest is None:
        dest = src.parent / "vocals_denoised.wav"

    print(f"[Denoise] 降噪: {src.name} (prop_decrease={config.DENOISE_PROP_DECREASE})")
    data, sr = sf.read(str(src), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)

    reduced = nr.reduce_noise(
        y=data,
        sr=sr,
        stationary=config.DENOISE_STATIONARY,
        prop_decrease=config.DENOISE_PROP_DECREASE,
    )
    sf.write(str(dest), reduced, sr)
    print(f"[Denoise] 完成: {dest}")
    return dest
