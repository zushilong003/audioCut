"""Whisper 转写模块 — 使用 faster-whisper-small 对人声轨做带时间戳的转写."""
from __future__ import annotations

import time
from pathlib import Path

import torch
from faster_whisper import WhisperModel

import config


_OPENCC = None


def _to_simplified(text: str) -> str:
    """繁体转简体: Whisper 对中文常输出繁体, 用 OpenCC 统一转简体."""
    global _OPENCC
    if not config.ZH_CONVERT_TO_SIMPLIFIED:
        return text
    if _OPENCC is None:
        from opencc import OpenCC

        _OPENCC = OpenCC("t2s")
    return _OPENCC.convert(text)


def _load_model() -> WhisperModel:
    """加载本地 faster-whisper 模型.

    优先用 config.WHISPER_MODEL_DIR 下指定名称的本地目录,
    目录不存在时退回官方模型名 (首次运行会自动下载).
    """
    local_path = config.WHISPER_MODEL_DIR / config.WHISPER_MODEL_NAME
    if local_path.exists():
        print(f"[Whisper] 使用本地模型: {local_path}")
        model_source = str(local_path)
    else:
        print(f"[Whisper] 本地未找到 {local_path}, 使用官方模型名: {config.WHISPER_MODEL_NAME}")
        model_source = config.WHISPER_MODEL_NAME

    # faster-whisper 的 device 参数: 无 GPU 时用 "cpu", 否则 "cuda"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[Whisper] device={device}, compute_type={compute_type}")

    return WhisperModel(model_source, device=device, compute_type=compute_type)


def transcribe_with_segments(vocals_wav: Path) -> list[dict]:
    """对人声轨做带时间戳的转写, 返回分段列表.

    Returns:
        列表元素 dict:
            {
                "start": float,    # 起始秒
                "end": float,      # 结束秒
                "text": str,       # 文本
                "avg_logprob": float,  # 置信度参考
                "no_speech_prob": float,
            }
    """
    print(f"[Whisper] 开始转写: {vocals_wav.name}")

    t_load = time.perf_counter()
    model = _load_model()
    print(f"[Whisper] 模型加载完成, 耗时 {time.perf_counter() - t_load:.1f}s")

    t0 = time.perf_counter()
    # word_timestamps=True 拿到词级时间, 后续才能按句末标点/说话人停顿精确切分
    segments_iter, info = model.transcribe(
        str(vocals_wav),
        language="zh",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        condition_on_previous_text=False,
        word_timestamps=True,
    )

    print(f"[Whisper] 检测到语言: {info.language}, 概率: {info.language_probability:.3f}")

    segments: list[dict] = []
    for seg in segments_iter:
        words = [
            {
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
                "word": _to_simplified(w.word),
                "probability": float(getattr(w, "probability", 0.0)),
            }
            for w in (getattr(seg, "words", None) or [])
        ]
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": _to_simplified(seg.text.strip()),
            "avg_logprob": getattr(seg, "avg_logprob", 0.0),
            "no_speech_prob": getattr(seg, "no_speech_prob", 0.0),
            "words": words,
        })

    print(f"[Whisper] 转写完成, 共 {len(segments)} 段, 耗时 {time.perf_counter() - t0:.1f}s")
    return segments
