"""声纹校验模块 — ECAPA-TDNN 声纹向量 + 余弦相似度阈值.

硬性规则落地方式:
1) 每个文本角色先用簇内语句算出声纹质心 (ECAPA 向量均值).
2) 任何片段写入前都要和该角色质心做余弦相似度复检, 低于阈值直接丢弃/裁剪,
   所以输出文件里不会混入别人的音色.
3) 只有相邻语句声纹也足够相似时, 才允许合并成一个片段; 声纹一变就断,
   切分边界由说话人决定, 与时长无关 (时长只做 <=19s 的上限保护).
"""
from __future__ import annotations

import json
import os
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

import config

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

TARGET_SR = 16000


class SpeakerEmbedder:
    """ECAPA-TDNN 声纹提取器 (惰性加载, 全局复用)."""

    _instance: "SpeakerEmbedder | None" = None

    def __init__(self) -> None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        model_dir = config.SPK_MODEL_DIR / config.SPK_MODEL_NAME.split("/")[-1]
        model_dir.mkdir(parents=True, exist_ok=True)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Speaker] 加载 ECAPA-TDNN: {model_dir} (device={self._device})")
        self._clf = EncoderClassifier.from_hparams(
            source=config.SPK_MODEL_NAME,
            savedir=str(model_dir),
            local_strategy=LocalStrategy.COPY_SKIP_CACHE,
            run_opts={"device": self._device},
        )
        print("[Speaker] ECAPA-TDNN 就绪")
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def get(cls) -> "SpeakerEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _resample(wav: np.ndarray, sr: int) -> np.ndarray:
        if sr == TARGET_SR:
            return wav.astype("float32")
        g = gcd(sr, TARGET_SR)
        return resample_poly(wav, TARGET_SR // g, sr // g).astype("float32")

    def embed_array(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = self._resample(wav, sr)
        if len(wav) < int(0.4 * TARGET_SR):
            raise ValueError("音频太短, 无法提取声纹")
        tensor = torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._clf.encode_batch(tensor).squeeze().cpu().numpy()
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm else emb

    def embed_file(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self._cache:
            return self._cache[key]
        wav, sr = sf.read(str(path), dtype="float32")
        emb = self.embed_array(wav, sr)
        self._cache[key] = emb
        return emb


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def compute_embeddings(
    segments: list[dict],
    cache_path: Path | None = None,
) -> np.ndarray:
    """为每个语句提取声纹向量, 支持缓存."""
    if cache_path and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) == len(segments):
            print(f"[Speaker] 复用声纹缓存: {cache_path.name}")
            return np.array([row["emb"] for row in cached], dtype=np.float32)

    embedder = SpeakerEmbedder.get()
    vectors: list[list[float]] = []
    print(f"[Speaker] 提取声纹向量: {len(segments)} 句")
    t0 = time.perf_counter()

    for idx, seg in enumerate(segments, 1):
        emb = embedder.embed_file(Path(seg["audio_path"]))
        vectors.append(emb.tolist())
        if idx % 20 == 0 or idx == len(segments):
            print(f"[Speaker]   {idx}/{len(segments)}  ({time.perf_counter() - t0:.1f}s)")

    print(f"[Speaker] 声纹向量提取完成, 耗时 {time.perf_counter() - t0:.1f}s")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump([{"segment_index": i, "emb": v} for i, v in enumerate(vectors)], f)

    return np.array(vectors, dtype=np.float32)


def cluster_voices(
    segments: list[dict],
    embeddings: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """直接用声纹聚类得到音色簇, 编号「音色1 / 音色2 ...」.

    1) 只用 >=2s 的句子做层次聚类 (短句声纹不稳, 单独处理), 余弦距离阈值 1-threshold,
       平均链接: 两个簇的平均相似度 >= threshold 才合并.
    2) 再按簇质心做一轮合并: 质心相似度 >= SPK_CLUSTER_MERGE 的簇视为同一音色, 合并后再算质心.
    3) 短句按「与各簇质心最像且 >= SPK_JOIN_THRESHOLD」归入已有簇, 否则弃用.
    4) 总语音不足 MIN_CLIP_DURATION 的簇直接丢弃 (凑不出 4s 片段).
    5) 簇按总时长从长到短编号, 时长越长的音色编号越靠前.
    """
    from sklearn.cluster import AgglomerativeClustering

    thr = config.SPK_CLUSTER_THRESHOLD if threshold is None else threshold
    merge_thr = config.SPK_CLUSTER_MERGE
    total = len(segments)
    durations = [max(0.0, s["end"] - s["start"]) for s in segments]
    print(f"[Speaker] 声纹聚类: {total} 句 "
          f"(层次聚类阈值 {thr} + 质心合并阈值 {merge_thr})")

    def _centroid(members: list[int]) -> np.ndarray:
        vec = np.array([embeddings[i] for i in members], dtype=np.float32).mean(axis=0)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    long_idx = [i for i in range(total) if durations[i] >= 2.0]
    labels = [-1] * total
    groups: list[list[int]] = []

    if len(long_idx) >= 2:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1.0 - thr,
        )
        sub_labels = model.fit_predict(embeddings[long_idx])
        groups = [
            [i for i, lb in zip(long_idx, sub_labels) if lb == k]
            for k in range(int(sub_labels.max()) + 1)
        ]
    elif long_idx:
        groups = [long_idx]
    groups = [g for g in groups if g]

    # 质心合并: 同一个音色被切碎成多个簇时, 按质心相似度并回来
    while len(groups) > 1:
        centroids = [_centroid(g) for g in groups]
        best_pair, best_sim = None, 0.0
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                s = cosine(centroids[a], centroids[b])
                if s > best_sim:
                    best_pair, best_sim = (a, b), s
        if best_pair is None or best_sim < merge_thr:
            break
        a, b = best_pair
        print(f"[Speaker] 合并两个音色簇 (质心相似度 {best_sim:.2f})")
        groups[a] = groups[a] + groups[b]
        groups.pop(b)

    centroids = [_centroid(g) for g in groups]
    for k, g in enumerate(groups):
        for i in g:
            labels[i] = k

    # 短句归到最像的簇; 相似度不够就弃用
    for i in range(total):
        if labels[i] != -1 or not centroids:
            continue
        sims = [cosine(embeddings[i], c) for c in centroids]
        best = int(np.argmax(sims))
        if sims[best] >= config.SPK_JOIN_THRESHOLD:
            labels[i] = best

    # 重分配两轮, 让边界句子回到最像的簇
    if centroids:
        for _ in range(2):
            for i in range(total):
                if labels[i] == -1:
                    continue
                sims = [cosine(embeddings[i], c) for c in centroids]
                labels[i] = int(np.argmax(sims))
            for k in range(len(centroids)):
                members = [j for j, lb in enumerate(labels) if lb == k]
                if members:
                    centroids[k] = _centroid(members)

    groups = [[i for i in range(total) if labels[i] == k] for k in range(len(centroids))]
    groups = [
        g for g in groups
        if sum(durations[i] for i in g) >= config.MIN_CLIP_DURATION
    ]
    groups.sort(key=lambda g: -sum(durations[i] for i in g))

    roles: list[dict] = []
    seg_rows: list[dict] = []
    for idx, members in enumerate(groups, 1):
        name = f"音色{idx}"
        centroid = _centroid(members)
        sims = [cosine(embeddings[i], centroid) for i in members]
        total_dur = sum(durations[i] for i in members)
        roles.append({
            "name": name,
            "description": "",
            "count": len(members),
            "centroid": [round(float(x), 6) for x in centroid],
            "voice_similarity": round(float(np.mean(sims)), 3),
            "total_duration": round(total_dur, 1),
        })
        seg_rows.extend({"segment_index": i, "role": name} for i in members)

        print(f"[Speaker]   {name}: {len(members)} 句, 总语音 {total_dur:.1f}s, "
              f"句-质心平均相似度 {np.mean(sims):.2f}")

    dropped = total - len(seg_rows)
    if dropped:
        print(f"[Speaker] 聚类弃用 {dropped} 句 (与各音色簇都不够像或所属簇太短, 不参与提取)")

    return {
        "roles": roles,
        "segments": seg_rows,
        "single_performer": len(roles) <= 1,
        "voice_threshold": config.SPK_THRESHOLD,
        "cluster_threshold": thr,
    }
