"""角色片段导出模块 — 按时长筛选并按声纹聚类结果命名.

命名规则:
  角色{字母}{序号}-{性别}.wav / .txt
  - 字母 A-Z: 不同音色角色, 按簇总时长降序分配 (A 最常出现)
  - 序号 01-99: 该角色下的第几个音频片段, 按在原音频中出现顺序编号
  - 性别 男/女: 由基频 F0 中位数判定
  例: 角色A01-男.wav / 角色A02-男.wav 为同一位男性角色的两个片段;
      角色C01-女.wav 为另一位女性角色的第一个片段.

若未提供聚类结果 (声纹分析关闭), 退化为顺序命名 角色NN.wav.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import config


def _resolve_audio(seg: dict) -> Path:
    src = Path(seg.get("audio_path", ""))
    if src.exists():
        return src
    fallback = config.SEGMENTS_DIR / src.name
    if fallback.exists():
        return fallback
    return src


def _duration_filter(segments: list[dict]) -> tuple[list[dict], int]:
    """按时长 [ROLE_CLIP_MIN_DURATION, ROLE_CLIP_MAX_DURATION] 过滤, 返回 (保留, 跳过数)."""
    min_d, max_d = config.ROLE_CLIP_MIN_DURATION, config.ROLE_CLIP_MAX_DURATION
    kept: list[dict] = []
    skipped = 0
    for seg in segments:
        duration = float(seg["end"]) - float(seg["start"])
        if not (min_d <= duration <= max_d):
            skipped += 1
            continue
        kept.append(seg)
    return kept, skipped


def export_role_clips(
    segments: list[dict],
    output_dir: Path | None = None,
    analysis: dict | None = None,
) -> list[Path]:
    """导出符合时长规则的语句片段, 按声纹聚类命名.

    Args:
        segments: build_segments 产出的语句列表, 每项含 start / end / text / audio_path.
        output_dir: 输出目录, 默认 config.ROLES_DIR.
        analysis: cluster_voices 返回的聚类结果 (含 roles 与 segments 映射).
                  为 None 时退化为顺序命名 角色NN.wav.

    Returns:
        生成的 wav 路径列表.
    """
    from .gender_detector import detect_cluster_gender

    if output_dir is None:
        output_dir = config.ROLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧文件
    for old in list(output_dir.glob("角色*.wav")) + list(output_dir.glob("角色*.txt")):
        old.unlink()

    min_d, max_d = config.ROLE_CLIP_MIN_DURATION, config.ROLE_CLIP_MAX_DURATION
    print(f"[RoleExport] 导出规则: {min_d}s <= 时长 <= {max_d}s")

    generated: list[Path] = []

    # ── 无聚类结果: 顺序命名 ──
    if not analysis or not analysis.get("roles"):
        kept, skipped = _duration_filter(segments)
        for seg in kept:
            src = _resolve_audio(seg)
            if not src.exists():
                skipped += 1
                continue
            idx = len(generated) + 1
            wav_path = output_dir / f"角色{idx:02d}.wav"
            txt_path = output_dir / f"角色{idx:02d}.txt"
            shutil.copy2(src, wav_path)
            txt_path.write_text(str(seg.get("text", "")).strip(), encoding="utf-8")
            generated.append(wav_path)
        print(f"[RoleExport] (无聚类) 共导出 {len(generated)} 个角色音频, 跳过 {skipped} 个")
        return generated

    # ── 有聚类结果: 按角色分组命名 ──
    roles = analysis["roles"]  # 已按 total_duration 降序
    seg_to_role: dict[int, str] = {}
    for row in analysis.get("segments", []):
        seg_to_role[int(row["segment_index"])] = row["role"]

    # 角色名 → 成员 segments
    role_members: dict[str, list[dict]] = {r["name"]: [] for r in roles}
    for idx, seg in enumerate(segments):
        role_name = seg_to_role.get(idx)
        if role_name in role_members:
            role_members[role_name].append(seg)

    letters = config.ROLE_LETTERS
    total_skipped = 0
    letter_idx = 0  # 只给实际导出的角色分配字母, 跳过空簇不占字母位

    for role in roles:
        role_name = role["name"]
        members = role_members.get(role_name, [])
        if not members:
            continue

        # 按时长过滤
        kept, skipped = _duration_filter(members)
        total_skipped += skipped
        if not kept:
            print(f"[RoleExport] {role_name}: 无符合时长的片段, 跳过")
            continue

        # 字母分配 (按角色顺序即总时长降序, 只给有输出的角色)
        letter = letters[letter_idx] if letter_idx < len(letters) else f"Z{letter_idx - len(letters) + 1:02d}"
        letter_idx += 1

        # 性别鉴别: 用簇内最长片段的 F0
        audio_paths = [_resolve_audio(s) for s in kept if _resolve_audio(s).exists()]
        gender, f0 = detect_cluster_gender(audio_paths) if audio_paths else ("男", 0.0)

        # 按出现顺序 (start) 编号
        kept.sort(key=lambda s: float(s["start"]))

        print(f"[RoleExport] {role_name} → 角色{letter} ({len(kept)} 片段, "
              f"总时长 {role.get('total_duration', 0):.1f}s, F0={f0:.0f}Hz, 性别={gender})")

        for seq, seg in enumerate(kept, 1):
            src = _resolve_audio(seg)
            if not src.exists():
                total_skipped += 1
                continue
            base = f"角色{letter}{seq:02d}-{gender}"
            wav_path = output_dir / f"{base}.wav"
            txt_path = output_dir / f"{base}.txt"
            shutil.copy2(src, wav_path)
            txt_path.write_text(str(seg.get("text", "")).strip(), encoding="utf-8")
            generated.append(wav_path)

            duration = float(seg["end"]) - float(seg["start"])
            print(f"[RoleExport]   {wav_path.name} ({duration:.1f}s, "
                  f"{seg['start']:.1f}-{seg['end']:.1f}s) -> {txt_path.name}")

    print(f"[RoleExport] 共导出 {len(generated)} 个角色音频, 跳过 {total_skipped} 个")
    return generated
