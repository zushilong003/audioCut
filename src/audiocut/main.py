"""有声小说音频处理 pipeline 主入口."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import config
from .demucs_separator import separate_vocals
from .speech_denoiser import denoise_vocals
from .transcriber import transcribe_with_segments
from .segmenter import build_segments
from .role_extractor import export_role_clips
from .speaker_verifier import cluster_voices, compute_embeddings


PREFERRED_NAMES = ("story.m4a", "story.wav", "story.mp3")
SUPPORTED_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma", ".mp4"}


def _find_input_audio() -> Path | None:
    """在 INPUT_AUDIO_DIR 里找输入音频.

    优先级: story.m4a > story.wav > story.mp3, 找不到则退化为目录里第一个音频.
    """
    for name in PREFERRED_NAMES:
        p = config.INPUT_AUDIO_DIR / name
        if p.exists():
            return p

    for p in config.INPUT_AUDIO_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            return p
    return None


def _ensure_story_wav(src: Path) -> Path:
    """把输入音频统一为 input/story.wav, 后续所有步骤都用它.

    - 如果本身就是 story.wav, 直接返回.
    - 否则用 ffmpeg 转成 wav, 44.1kHz / 单声道 / 16bit PCM.
    """
    dest = config.INPUT_AUDIO_DIR / "story.wav"

    if src.resolve() == dest.resolve():
        print(f"[PreCheck] 已是 story.wav, 跳过转换")
        return dest

    print(f"[PreCheck] {src.name} -> story.wav (ffmpeg 转码)")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ar", "44100",
        "-ac", "1",
        "-sample_fmt", "s16",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 转换失败:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

    # 转换成功后把原文件挪到 output/backup_input/, 避免下次被 _find_input_audio 误拾取
    backup_dir = config.OUTPUT_DIR / "backup_input"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_dst = backup_dir / src.name
    if backup_dst.exists():
        backup_dst.unlink()
    shutil.move(str(src), str(backup_dst))
    print(f"[PreCheck] 原始文件已归档 -> {backup_dst}")

    return dest


def _make_run_name() -> str:
    """生成形如 20261104_190723_0391 的时间戳目录名 (年月日_时分秒_毫秒)."""
    now = datetime.now()
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{now.strftime('%f')[:4]}"


@contextmanager
def _stage(timings: list[tuple[str, float]], name: str):
    """记录单个阶段的开始/结束耗时."""
    print(f"\n[Stage] ▶ {name} ...")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        timings.append((name, dt))
        print(f"[Stage] ✔ {name} 耗时 {dt:.1f}s")


def _print_timings(timings: list[tuple[str, float]], total: float) -> None:
    print("\n" + "=" * 60)
    print("⏱ 各阶段耗时")
    for name, dt in timings:
        share = (dt / total * 100) if total > 0 else 0.0
        print(f"   {name:<28} {dt:8.1f}s  ({share:4.1f}%)")
    print(f"   {'总计':<28} {total:8.1f}s")
    print("=" * 60)


def run(input_audio: Path | None = None) -> None:
    timings: list[tuple[str, float]] = []
    total_start = time.perf_counter()

    # ── 0. 定位 + 统一转码 ──
    with _stage(timings, "0. 定位输入 + 统一转码"):
        if input_audio is None:
            input_audio = _find_input_audio()
        if input_audio is None:
            print(f"[ERROR] 在 {config.INPUT_AUDIO_DIR} 下没找到音频文件.")
            print(f"        请把有声小说命名为 story.m4a / story.wav / story.mp3 放进 input/,")
            print(f"        或用 --audio D:\\path\\to\\book.mp4 指定.")
            sys.exit(1)

        if not input_audio.exists():
            print(f"[ERROR] 指定的音频不存在: {input_audio}")
            sys.exit(1)

        story_wav = _ensure_story_wav(input_audio)

    # ── 0.5 创建本次 run 的时间戳输出目录 ──
    run_root = config.set_run_dir(_make_run_name())

    print("=" * 60)
    print("📦 有声小说处理 pipeline 启动")
    print(f"   标准输入: {story_wav}")
    print(f"   Run 目录: {run_root}")
    print(f"   Python:   {sys.version}")
    print("=" * 60)

    # ── Step 1: Demucs 人声分离 ──
    with _stage(timings, "1. Demucs 人声分离"):
        vocals_wav = separate_vocals(story_wav)

    # ── Step 1.5: 人声降噪 (频谱降噪去音乐残留/环境噪声) ──
    if config.ENABLE_DENOISE:
        with _stage(timings, "1.5 人声降噪"):
            vocals_wav = denoise_vocals(vocals_wav)

    # ── Step 2: Whisper 带时间戳转写 ──
    with _stage(timings, "2. Whisper 转写"):
        whisper_segments = transcribe_with_segments(vocals_wav)

    # ── Step 3: 二次分段 + 落盘 ──
    with _stage(timings, "3. 语句级分段"):
        all_segments = build_segments(vocals_wav, whisper_segments)
    if not all_segments:
        print("[ERROR] 分段结果为空, pipeline 中止")
        sys.exit(1)

    # ── Step 4-5: 声纹分析 (ECAPA 聚类, 用于角色分组命名) ──
    analysis_path: Path | None = None
    analysis: dict | None = None
    if config.ENABLE_VOICE_ANALYSIS:
        # ── Step 4: 声纹分析 (ECAPA-TDNN 向量) ──
        with _stage(timings, "4. ECAPA 声纹向量"):
            embeddings = compute_embeddings(
                all_segments,
                cache_path=run_root / "embeddings.json",
            )

        # ── Step 5: 声纹聚类 (直接按音色分组并编号) ──
        with _stage(timings, "5. 声纹聚类"):
            analysis = cluster_voices(all_segments, embeddings)

            if config.SINGLE_PERFORMER_MODE == "single":
                analysis["single_performer"] = True
                print("[Pipeline] 按配置强制单音色模式 (SINGLE_PERFORMER_MODE=single)")
            elif config.SINGLE_PERFORMER_MODE == "multi":
                analysis["single_performer"] = False
                print("[Pipeline] 按配置强制多音色模式 (SINGLE_PERFORMER_MODE=multi)")

            analysis["summary"] = (
                f"{len(analysis['roles'])} 个音色, 共 {len(all_segments)} 句对白"
                + ("（声纹判定为同一音色）" if analysis["single_performer"] else "（声纹判定为多个音色）")
            )

            analysis_path = run_root / "roles_analysis.json"
            analysis_path.write_text(
                json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[Pipeline] 分析结果已写入: {analysis_path}")
    else:
        print("[Pipeline] 声纹分析已跳过 (ENABLE_VOICE_ANALYSIS=false, 代码保留)")

    # ── Step 6: 按语句时长导出角色音频 + 对应文本 (有聚类结果则按音色分组命名) ──
    with _stage(timings, "6. 角色音频导出"):
        export_role_clips(all_segments, analysis=analysis)

    total = time.perf_counter() - total_start

    print("\n" + "=" * 60)
    print("✅ Pipeline 完成!")
    print(f"   Run 目录:   {run_root}")
    if analysis_path is not None:
        print(f"   分析结果:   {analysis_path}")
    print(f"   分离人声:   {config.SEPARATED_DIR}")
    print(f"   分段切片:   {config.SEGMENTS_DIR}")
    print(f"   角色音频:   {config.ROLES_DIR}")
    print("=" * 60)

    _print_timings(timings, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="有声小说音频处理 pipeline")
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="指定输入音频文件 (默认从 audio/input 里自动找 story.m4a/wav/mp3)",
    )
    args = parser.parse_args()
    run(args.audio)


if __name__ == "__main__":
    main()
