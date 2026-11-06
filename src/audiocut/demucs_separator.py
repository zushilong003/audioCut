"""Demucs 人声分离模块 — 调用 Demucs two-stems 模型提取纯净人声."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import config


def separate_vocals(input_audio: Path, output_dir: Path | None = None) -> Path:
    """对输入音频执行 Demucs two-stems 分离, 输出 vocals.wav.

    Args:
        input_audio: 原始音频文件路径.
        output_dir: 分离后的输出目录, 默认 config.SEPARATED_DIR (vocals.wav 直接落在该目录).

    Returns:
        分离后的人声文件路径 (.wav).
    """
    if output_dir is None:
        output_dir = config.SEPARATED_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Demucs] 开始分离: {input_audio.name}")

    # Demucs 推荐用 two-stems 模式 (--two-stems vocals) 直接输出人声轨
    # 调用命令行入口比 Python API 更稳 (避免 torchaudio backend 问题)
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-n",
        config.DEMUCS_MODEL,
        "--shifts",
        str(config.DEMUCS_SHIFTS),
        "-o",
        str(output_dir),
        str(input_audio),
    ]

    print(f"[Demucs] 执行: {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if result.returncode != 0:
        raise RuntimeError(f"Demucs 分离失败:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

    # Demucs 输出结构: output_dir/model_name/input_stem/vocals.wav
    vocals_candidate = (
        output_dir / config.DEMUCS_MODEL / input_audio.stem / "vocals.wav"
    )

    if not vocals_candidate.exists():
        # 递归搜索一下
        vocals_candidate = next(output_dir.rglob("vocals.wav"), None)
        if vocals_candidate is None:
            raise FileNotFoundError(f"Demucs 输出中找不到 vocals.wav, 输出目录: {output_dir}")

    # 拷贝到 output_dir 根层, 方便后续模块直接用
    dest = output_dir / "vocals.wav"
    if vocals_candidate != dest:
        shutil.copy2(vocals_candidate, dest)
        # 清理 Demucs 产生的多余层级目录
        extra_root = output_dir / config.DEMUCS_MODEL
        if extra_root.exists():
            shutil.rmtree(extra_root, ignore_errors=True)

    print(f"[Demucs] 完成: {dest}, 耗时 {time.perf_counter() - t0:.1f}s")
    return dest
