"""语句级分段模块 — 基于 Whisper 词级时间戳把对白切成完整的句子.

设计原则:
1) 只在「句末标点」或「说话人停顿」处断开, 保证每段是完整语句, 不会半句腰斩.
2) 单段最长不超过 MAX_CLIP_DURATION, 超长时优先在逗号处切.
3) 每段都是一个连续发声块, 天然只属于一个说话人, 避免一段里混两个角色.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from faster_whisper.vad import VadOptions, get_speech_timestamps

import config

# 句末标点 —— 在这些字符后必定断句
SENTENCE_END = set("。！？!?…")
# 次级停顿 —— 超长句时退而求其次在这些字符后切
CLAUSE_END = set("，,；;、：:")
# 两个字之间超过这么久没声音, 认为换人或明显停顿
SPEAKER_GAP = config.SPEAKER_GAP_THRESHOLD
# 单个「原子」允许的最长时长, 超过就强制在逗号处切
MAX_ATOM_DURATION = config.MAX_CLIP_DURATION
MIN_ATOM_DURATION = 0.4

# VAD 用于边缘裁剪: 小静音窗口让句中停顿也分段, 便于识别边缘的音乐残留
_VAD_OPTIONS = VadOptions(
    min_speech_duration_ms=250,
    min_silence_duration_ms=300,
    speech_pad_ms=50,
)


def _trim_speech_edges(
    chunk: np.ndarray,
    sr: int,
    words: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """用 Silero VAD 裁掉片段首尾的静音/音乐残留, 保留少量边缘余量.

    VAD 会把响亮音乐误判为语音, 所以结合 Whisper 词级时间戳:
    只保留「包含词起点」的首尾语音段, 之外的内容判为音乐/噪声残留.
    """
    if not config.TRIM_SEGMENT_EDGES or len(chunk) < int(0.5 * sr):
        return chunk
    x16 = resample_poly(chunk, 16000, sr).astype(np.float32)
    spans = get_speech_timestamps(x16, _VAD_OPTIONS, sampling_rate=16000)
    if not spans:
        return chunk

    if words:
        tol = 0.15 * 16000  # 词时间戳本身有误差, 放宽 150ms

        def _has_word(span: dict) -> bool:
            return any(
                span["start"] - tol <= ws * 16000 <= span["end"] + tol
                for ws, _ in words
            )

        first_idx = next((i for i, s in enumerate(spans) if _has_word(s)), None)
        last_idx = max((i for i, s in enumerate(spans) if _has_word(s)), default=None)
        if first_idx is not None and last_idx is not None:
            spans = spans[first_idx:last_idx + 1]

    pad16 = int(config.TRIM_EDGE_PAD_MS / 1000 * 16000)
    s16 = max(0, spans[0]["start"] - pad16)
    e16 = min(len(x16), spans[-1]["end"] + pad16)
    if e16 <= s16:
        return chunk
    scale = sr / 16000
    return chunk[int(s16 * scale):int(e16 * scale)]


def _iter_words(whisper_segments: list[dict]):
    """把 Whisper 的段级结果摊平成词级序列."""
    for seg in whisper_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        # 静音误识别直接丢
        if seg.get("no_speech_prob", 0.0) > 0.8 and len(text) < 4:
            continue

        words = seg.get("words") or []
        if words:
            for w in words:
                token = str(w.get("word", "")).strip()
                if not token:
                    continue
                yield {
                    "start": float(w.get("start", seg["start"])),
                    "end": float(w.get("end", seg["end"])),
                    "text": token,
                    "logprob": float(seg.get("avg_logprob", 0.0)),
                    "no_speech_prob": float(seg.get("no_speech_prob", 0.0)),
                }
        else:
            # 极少数情况没有词级时间戳, 整段当一个原子
            yield {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": text,
                "logprob": float(seg.get("avg_logprob", 0.0)),
                "no_speech_prob": float(seg.get("no_speech_prob", 0.0)),
            }


def _find_clause_cut(cur: list[dict]) -> int | None:
    """在累计词序列里找一个可切的逗号位置, 返回切点下标 (右侧起始)."""
    best = None
    base = cur[0]["start"]
    for i, w in enumerate(cur[:-1]):
        if w["text"][-1] not in CLAUSE_END:
            continue
        left_dur = w["end"] - base
        if 2.0 <= left_dur <= MAX_ATOM_DURATION:
            best = i + 1
    return best


def _build_atoms(whisper_segments: list[dict]) -> list[dict]:
    """把词序列切成语句原子."""
    words = list(_iter_words(whisper_segments))
    atoms: list[dict] = []
    cur: list[dict] = []

    def close() -> None:
        nonlocal cur
        if not cur:
            return
        text = re.sub(r"\s+", "", "".join(w["text"] for w in cur))
        atoms.append({
            "start": round(cur[0]["start"], 3),
            "end": round(cur[-1]["end"], 3),
            "text": text,
            "avg_logprob": float(np.mean([w["logprob"] for w in cur])),
            "no_speech_prob": float(np.max([w["no_speech_prob"] for w in cur])),
            # 词级时间戳 (含词文本): 边缘裁剪区分语音/音乐残留 + 换人劈分时重组文本
            "words": [(round(w["start"], 3), round(w["end"], 3), w["text"]) for w in cur],
        })
        cur = []

    for i, w in enumerate(words):
        cur.append(w)
        next_gap = (words[i + 1]["start"] - w["end"]) if i + 1 < len(words) else 1e9
        dur = w["end"] - cur[0]["start"]

        # 超长且此刻没有句末标点 -> 退到最近逗号处断开
        if dur >= MAX_ATOM_DURATION and w["text"][-1] not in SENTENCE_END:
            cut = _find_clause_cut(cur)
            if cut is not None:
                right = cur[cut:]
                cur = cur[:cut]
                close()
                cur = right
                continue

        if w["text"][-1] in SENTENCE_END or next_gap >= SPEAKER_GAP or dur >= MAX_ATOM_DURATION:
            close()

    close()

    # 过滤无效原子 + 把极短碎片并回相邻同人语句
    filtered: list[dict] = []
    for a in atoms:
        if len(a["text"]) < 2 or (a["end"] - a["start"]) < MIN_ATOM_DURATION:
            continue
        # 极短 (<1.2s) 且紧邻上一段 (<0.25s) -> 视为同一句的碎片, 合并
        if (
            filtered
            and (a["end"] - a["start"]) < 1.2
            and (a["start"] - filtered[-1]["end"]) < 0.25
            and (a["end"] - filtered[-1]["start"]) <= MAX_ATOM_DURATION
        ):
            prev = filtered[-1]
            prev["end"] = a["end"]
            prev["text"] = prev["text"] + a["text"]
            prev["avg_logprob"] = (prev["avg_logprob"] + a["avg_logprob"]) / 2
            prev["words"] = prev["words"] + a["words"]
            continue
        filtered.append(a)

    return filtered


def save_segments(
    vocals_wav: Path,
    segments: list[dict],
    output_dir: Path | None = None,
) -> list[dict]:
    """把语句原子切成独立 wav, 并写 segments.json 索引."""
    if output_dir is None:
        output_dir = config.SEGMENTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    samples, sr = sf.read(str(vocals_wav))
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    saved: list[dict] = []
    for idx, seg in enumerate(segments, 1):
        start_sample = max(0, int(seg["start"] * sr))
        end_sample = min(len(samples), int(seg["end"] * sr))
        chunk = samples[start_sample:end_sample]
        if len(chunk) == 0:
            continue
        # 词级时间戳换算成片段内相对时间, 供边缘裁剪判定
        rel_words = [
            (max(0.0, w[0] - seg["start"]), w[1] - seg["start"])
            for w in (seg.get("words") or [])
            if w[1] > seg["start"] and w[0] < seg["end"]
        ]
        chunk = _trim_speech_edges(chunk, sr, rel_words)
        if len(chunk) == 0:
            continue

        fname = f"seg_{idx:04d}_{seg['start']:.2f}-{seg['end']:.2f}.wav"
        fpath = output_dir / fname
        sf.write(str(fpath), chunk, sr)

        enriched = dict(seg)
        enriched["audio_path"] = str(fpath)
        saved.append(enriched)

    print(f"[Segmenter] 保存了 {len(saved)} 个语句片段到 {output_dir}")

    index_path = output_dir / "segments.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(saved, f, ensure_ascii=False, indent=2)

    return saved


def _find_speaker_change(chunk: np.ndarray, sr: int, embedder, cosine_fn) -> float | None:
    """片段内声纹换人检测: 返回换人点(片段内相对秒), 无换人返回 None."""
    boundaries = _change_boundaries(chunk, sr, embedder, cosine_fn)
    return boundaries[0] if boundaries else None


def _split_atom_at_word(atom: dict, word_idx: int) -> tuple[dict, dict]:
    """在 words[word_idx] 起点把原子劈成两半, 文本按词重组."""
    words = atom["words"]
    left_w, right_w = words[:word_idx], words[word_idx:]
    boundary = round(right_w[0][0], 3)
    left = {
        **atom, "end": boundary, "words": left_w,
        "text": re.sub(r"\s+", "", "".join(w[2] for w in left_w)),
    }
    right = {
        **atom, "start": boundary, "words": right_w,
        "text": re.sub(r"\s+", "", "".join(w[2] for w in right_w)),
    }
    return left, right


def _split_mixed_speaker_atoms(atoms: list[dict], vocals_wav: Path) -> list[dict]:
    """切分时声纹一致性优化: 长原子内部检测到换人时, 在最近的词边界劈开.

    相邻 2s 滑窗的 ECAPA 声纹相似度 < SPK_CHANGE_THRESHOLD 判为换人,
    换人点吸附到最近的词起点; 劈开后递归复检, 保证单一片段不混入多人声音.
    """
    if not config.ENABLE_SPEAKER_SPLIT:
        return atoms
    if not any(a["end"] - a["start"] >= config.SPEAKER_CHECK_MIN_DURATION for a in atoms):
        return atoms

    from .speaker_verifier import SpeakerEmbedder, cosine

    samples, sr = sf.read(str(vocals_wav))
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    embedder = SpeakerEmbedder.get()
    min_part = 1.0  # 换人点靠边时也允许劈开: 短小块会被导出时长规则自然过滤

    def _recurse(atom: dict, depth: int) -> list[dict]:
        if depth >= 4 or atom["end"] - atom["start"] < config.SPEAKER_CHECK_MIN_DURATION:
            return [atom]
        chunk = samples[int(atom["start"] * sr):int(atom["end"] * sr)]
        change_rel = _find_speaker_change(chunk, sr, embedder, cosine)
        if change_rel is None:
            return [atom]
        change_abs = atom["start"] + change_rel
        words = atom.get("words") or []
        if len(words) < 2:
            return [atom]
        # 换人点吸附到最近的词起点
        k = min(range(1, len(words)), key=lambda i: abs(words[i][0] - change_abs))
        left, right = _split_atom_at_word(atom, k)
        if left["end"] - left["start"] < min_part or right["end"] - right["start"] < min_part:
            return [atom]
        print(f"[Segmenter] 检测到换人: {atom['start']:.1f}s 处在词「{right['words'][0][2]}」前劈开 "
              f"({left['end'] - left['start']:.1f}s + {right['end'] - right['start']:.1f}s)")
        return _recurse(left, depth + 1) + _recurse(right, depth + 1)

    result: list[dict] = []
    for atom in atoms:
        result.extend(_recurse(atom, 0))
    return result


def _change_boundaries(
    samples: np.ndarray, sr: int, embedder, cosine_fn, offset: float = 0.0
) -> list[float]:
    """滑窗声纹换人点检测 (返回秒, 已加 offset).

    1s 窗 / 0.25s 步长提声纹, 比较「前 1s 块均值」与「后 1s 块均值」的余弦相似度:
    单窗比对易被语调波动触发假阳性, 块均值比对只对持续的音色切换敏感.
    再要求换人后 1s 仍与之前不相似 (持续性校验), 排除瞬时噪声扰动.
    连续命中合并为一个换人点, 间距 >= 1.5s.
    """
    win, hop, block = 1.0, 0.25, 4  # block=4 步 => 前后各 1s
    embs, times = [], []
    t = 0.0
    dur = len(samples) / sr
    while t + win <= dur + 1e-6:
        w = samples[int(t * sr):int((t + win) * sr)]
        embs.append(embedder.embed_array(w, sr))
        times.append(t)
        t += hop
    if len(embs) < 3 * block:
        return []

    boundaries: list[float] = []
    for i in range(block, len(embs) - 2 * block + 1):
        before = np.mean(embs[i - block:i], axis=0)
        after = np.mean(embs[i:i + block], axis=0)
        far = np.mean(embs[i + block:i + 2 * block], axis=0)
        th = config.SPK_CHANGE_THRESHOLD
        if cosine_fn(before, after) < th and cosine_fn(before, far) < th:
            t_change = times[i] + offset
            if not boundaries or t_change - boundaries[-1] > 1.5:
                boundaries.append(round(float(t_change), 3))
    return boundaries


def _speaker_change_boundaries(vocals_wav: Path) -> list[float]:
    """整轨声纹扫描换人点 (绝对秒)."""
    from .speaker_verifier import SpeakerEmbedder, cosine

    samples, sr = sf.read(str(vocals_wav))
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    embedder = SpeakerEmbedder.get()
    boundaries = _change_boundaries(samples, sr, embedder, cosine)
    print(f"[Segmenter] 整轨声纹扫描: {len(boundaries)} 个换人点")
    return boundaries


def _split_segments_by_speaker(vocals_wav: Path, whisper_segments: list[dict]) -> list[dict]:
    """按声纹换人点劈开 Whisper 段: 换人点吸附到最近词起点, 保证段内同一说话人."""
    if not config.ENABLE_SPEAKER_SPLIT:
        return whisper_segments
    boundaries = _speaker_change_boundaries(vocals_wav)
    if not boundaries:
        return whisper_segments

    out: list[dict] = []
    for seg in whisper_segments:
        words = seg.get("words") or []
        if len(words) < 2:
            out.append(seg)
            continue
        cuts = set()
        for b in boundaries:
            if words[0]["start"] < b < words[-1]["end"]:
                k = min(range(1, len(words)), key=lambda i: abs(words[i]["start"] - b))
                if abs(words[k]["start"] - b) <= 1.2:
                    cuts.add(k)
        if not cuts:
            out.append(seg)
            continue
        prev = 0
        for c in sorted(cuts) + [len(words)]:
            w = words[prev:c]
            if w:
                out.append({
                    **seg,
                    "start": w[0]["start"],
                    "end": w[-1]["end"],
                    "text": "".join(str(x.get("word", "")) for x in w).strip(),
                    "words": w,
                })
            prev = c
    n_split = len(out) - len(whisper_segments)
    if n_split:
        print(f"[Segmenter] 按声纹换人劈开 {n_split} 处, 段数 {len(whisper_segments)} -> {len(out)}")
    return out


def build_segments(
    vocals_wav: Path,
    whisper_segments: list[dict],
) -> list[dict]:
    """端到端: 声纹换人劈分 + 语句切分 (同一人内部再按标点/停顿/长度切) + 切片落盘."""
    whisper_segments = _split_segments_by_speaker(vocals_wav, whisper_segments)
    atoms = _build_atoms(whisper_segments)
    atoms = _split_mixed_speaker_atoms(atoms, vocals_wav)  # 兜底: 漏检的换人再劈一次
    return save_segments(vocals_wav, atoms)
