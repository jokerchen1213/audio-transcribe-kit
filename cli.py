#!/usr/bin/env python3
"""cc-transcribe：本地离线语音转写（Qwen3-ASR on MLX + sherpa 说话人分离，后端可插拔）。原件只读。

两条路线（`backends.toml` `[default] runner`）：
- `win`：Qwen3-ASR-hf + Nemotron/DiariZen 在 Windows 3090 上跑（经 remote.py），MOSS 作第二意见；
- 其它：Mac 本地，沿用原 ASR / 分离后端（`--local` 可强制）。
- 2026-10-31 起（律师裁定）：win 路线下 Windows 不可用（预检失败或中途掉线 / 忙）自动回退 Mac 本地；`--win-only` 保留旧行为（退出码 3）。
"""
from __future__ import annotations
import argparse, json, os, re, sys, tempfile, tomllib
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
import remote
import second_opinion
from audio import AUDIO_EXTS, AudioError, probe_duration, split_wav, to_wav16k
from backends import ASR_BACKENDS, DIARIZERS, get_asr, get_diarizer, list_backends
from backends.base import SpeakerTurn, Word
from config import load_config, resolve
from index import md_target, needs_transcribe, upsert
from merge import assign_speakers, build_paragraphs, drop_context_echo, shift_words
from writer import Meta, format_ts, render

CHUNK_S = 1800.0
LONG_AUDIO_S = 7200.0
PURE_DIR = "4.提交微法院的电子资料存档"   # 纯净区（cases/CLAUDE.md 规则 1）：连派生件也不写，与 cc-convert 同名常量
RESIDUAL_SPEECH_S = 5.0                  # Windows 路线：累计发声 < 5 秒的残标签丢弃

def parse_args(argv):
    p = argparse.ArgumentParser(prog="cc-transcribe", description=__doc__)
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--asr"); p.add_argument("--diarizer")
    p.add_argument("--speakers", type=int); p.add_argument("--min-speakers", type=int); p.add_argument("--max-speakers", type=int)
    p.add_argument("--context", default="")
    p.add_argument("--no-diarize", action="store_true")
    p.add_argument("--force", action="store_true"); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--config", type=Path)
    p.add_argument("--local", action="store_true", help="强制走 Mac 本地路线（无视 backends.toml 的 runner=win）")
    p.add_argument("--win-only", action="store_true",
                   help="只用 Windows：不可用时退出码 3、不回退本机（默认会自动回退 Mac 本机）")
    p.add_argument("--no-second-opinion", action="store_true", help="Windows 路线不跑 MOSS 第二意见")
    return p.parse_args(argv)

def _skip_pure(p: Path) -> bool:
    if PURE_DIR in p.parts:
        print(f"跳过（纯净区，不写派生件）：{p.name}"); return True
    return False

def list_audio(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        if p.is_dir():
            out += sorted(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in AUDIO_EXTS
                          and "_md" not in x.parts and not x.name.startswith("."))   # 跳过 macOS AppleDouble ._x 等隐藏文件
        elif p.suffix.lower() in AUDIO_EXTS:
            out.append(p)
        else:
            print(f"跳过（非音频）：{p}")
    return [x for x in out if not _skip_pure(x)]   # 纯净区文件：目录递归与单文件指名都跳过，每个文件提示一次

# 文件名日期：优先完整 YYYYMMDD / YYYY-MM-DD（两侧不能再是数字），其次录音笔形态 _YYMMDD_HHMMSS，
# 再次无年份的 MM-DD 开头（后接空格 / 下划线 / 非数字，年份取文件 mtime 年份）；
# 每个命中都经 date() 校验，非法日期（月 13、日 99）视同未命中，最终回退到文件 mtime。
_DATE_FULL = re.compile(r"(?<!\d)(20\d{2})[-_]?(\d{2})[-_]?(\d{2})(?!\d)")
_DATE_REC = re.compile(r"_(\d{2})(\d{2})(\d{2})_\d{6}(?!\d)")
_DATE_MD = re.compile(r"^(\d{2})-(\d{2})(?!\d)")

def _recorded_at(src: Path) -> str:
    for pat, century in ((_DATE_FULL, ""), (_DATE_REC, "20")):
        for m in pat.finditer(src.name):
            y, mo, d = m.groups()
            try:
                return date(int(century + y), int(mo), int(d)).isoformat()
            except ValueError:
                continue
    mtime = datetime.fromtimestamp(src.stat().st_mtime)
    if m := _DATE_MD.match(src.name):
        try:
            got = date(mtime.year, int(m.group(1)), int(m.group(2))).isoformat()
            print(f"recorded_at 由文件名月日 + 文件年份推得：{got}")   # 不增 front-matter 字段，只在终端标明来源
            return got
        except ValueError:
            pass
    return mtime.strftime("%Y-%m-%d")

def _all_words(asr, wav: Path, context: str, work: Path):
    dur = probe_duration(wav)
    if dur <= LONG_AUDIO_S:
        res = asr.transcribe(wav, context=context or None)
        parts = [(res, 0.0)]
    else:
        parts = [(asr.transcribe(p, context=context or None), off) for p, off in split_wav(wav, CHUNK_S, work / "parts")]
    words, truncated = [], False
    for res, off in parts:
        truncated |= res.truncated
        for s in res.segments:
            ws = s.words or [Word(s.text, s.start, s.end)]
            words += shift_words(ws, off)
    return words, truncated, parts[0][0].model_label, len(parts), dur

def _write_atomic(target: Path, text: str) -> None:
    """先写 .tmp 再 os.replace：中断不会留下 0 字节 / 半成品 _md（否则会被去重闸永久跳过）。"""
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    except OSError as e:
        raise RuntimeError(f"目标不可写：{target}（{e.strerror or e}）") from e
    finally:
        tmp.unlink(missing_ok=True)   # 含 KeyboardInterrupt：绝不留 .tmp

def _clean_turns(turns: list[SpeakerTurn], min_speech: float = RESIDUAL_SPEECH_S) -> tuple[list[SpeakerTurn], int]:
    """残标签清理（Windows 路线与本地 nemotron）：累计发声 < min(min_speech, 全部发声的 5%) 的说话人 turns 丢弃，
    剩余按首次出现从 0 重新编号；绝不剔光。相对阈值保护短录音（20 秒录音里只说 3 秒的人不是残标签），
    长录音上等同于 < min_speech。返回 (新 turns（按 start 排序）, 丢弃的说话人个数)。"""
    ordered = sorted(turns, key=lambda t: t.start)
    total: dict[int, float] = {}
    for t in ordered:
        total[t.speaker] = total.get(t.speaker, 0.0) + max(0.0, t.end - t.start)
    floor = min(min_speech, 0.05 * sum(total.values()))
    keep = {spk for spk, sec in total.items() if sec >= floor} or set(total)
    renumber: dict[int, int] = {}
    out: list[SpeakerTurn] = []
    for t in ordered:
        if t.speaker not in keep:
            continue
        spk = renumber.setdefault(t.speaker, len(renumber))
        out.append(SpeakerTurn(spk, t.start, t.end))
    return out, len(total) - len(keep)

@dataclass
class Capture:
    """“取得”阶段的结果：两条路线各产出一份，交给共用的“合并写出”阶段。"""
    words: list[Word]
    truncated: bool
    asr_label: str
    chunks: int
    dur: float
    turns: list[SpeakerTurn] = field(default_factory=list)
    diar_label: str | None = None
    quality: str = "ok"
    engine: str = ""                 # _转换索引.md 的引擎列
    engine_host: str | None = None
    asr_revision: str | None = None
    diarizer_revision: str | None = None
    speakers_dropped: int | None = None
    second_opinion: str | None = None
    disagreements: int | None = None
    suspected_gaps: int | None = None
    so_result: second_opinion.Result | None = None

def _capture_local(src: Path, ns, asr, diarizer, diar_name: str | None, work: Path, fallback=None) -> Capture:
    """Mac 本地取得：转码 → ASR（必要时切块）→ 分离。fallback 为 None（sherpa 等）时与改动前 run_one 前半段逐行一致。
    fallback 非 None 仅用于 nemotron（2026-10-22 起本地默认）：指定人数 > 8、模型满 8 人或 nemotron 出错时改用
    fallback()（sherpa），并做与 Windows 路线相同的残标签清理。"""
    wav = to_wav16k(src, work)
    words, truncated, model_label, chunks, dur = _all_words(asr, wav, ns.context, work)
    quality, turns, diar_label, dropped = "ok", [], None, None
    if diarizer is not None and fallback is None:
        try:
            turns = diarizer.diarize(wav, num_speakers=ns.speakers, min_speakers=ns.min_speakers, max_speakers=ns.max_speakers)
            diar_label = getattr(diarizer, "label", diar_name)
        except Exception as e:
            print(f"⚠️ 说话人分离失败，降级为无说话人转录稿：{e}")
            quality = "diarization-failed"
    elif diarizer is not None:
        reason = f"指定 {ns.speakers} 人 > 8" if ns.speakers is not None and ns.speakers > 8 else ""
        if not reason:
            try:
                turns = diarizer.diarize(wav, num_speakers=ns.speakers, min_speakers=ns.min_speakers, max_speakers=ns.max_speakers)
                diar_label = getattr(diarizer, "label", diar_name)
                if getattr(diarizer, "capped", False):
                    reason = "nemotron 满 8 人后切换"
            except Exception as e:
                print(f"⚠️ nemotron 分离失败，改用 sherpa：{e}")
                reason = "nemotron 失败后切换"
        if reason:
            try:
                fb = fallback()
                turns = fb.diarize(wav, num_speakers=ns.speakers, min_speakers=ns.min_speakers, max_speakers=ns.max_speakers)
                diar_name, diar_label = fb.name, f"{getattr(fb, 'label', fb.name)}（{reason}）"
            except Exception as e:
                print(f"⚠️ 说话人分离失败，降级为无说话人转录稿：{e}")
                turns, diar_label, quality = [], None, "diarization-failed"
        if turns:
            turns, dropped = _clean_turns(turns)
        if ns.speakers is not None and turns and len({t.speaker for t in turns}) != ns.speakers:
            print(f"⚠️ 指定 --speakers {ns.speakers} 人，最终分离出 {len({t.speaker for t in turns})} 人，请复核归属")
    engine = model_label.split(" / ", 1)[-1] + (f" + {diar_name}" if diar_label else "")
    return Capture(words=words, truncated=truncated, asr_label=model_label, chunks=chunks, dur=dur,
                   turns=turns, diar_label=diar_label, quality=quality, engine=engine, speakers_dropped=dropped)

def _capture_win(src: Path, ns, cfg: dict, work: Path) -> Capture:
    """Windows 取得：Mac 只 to_wav16k，其余全在一次远端作业里跑完（qwen3 + 两个分离器 [+ moss]）。"""
    wav = to_wav16k(src, work)
    job = {"language": "Chinese", "context": ns.context, "num_speakers": ns.speakers,
           "min_speakers": ns.min_speakers, "max_speakers": ns.max_speakers}
    engines = ["qwen3"]
    if not ns.no_diarize:
        engines += ["nemotron", "diarizen"]
    want_second = bool(cfg.get("second_opinion", True)) and not ns.no_second_opinion
    if want_second:
        engines.append("moss")
    results = remote.run_remote(wav, job, engines, cfg)

    qwen = results.get("qwen3")
    if not qwen:
        raise RuntimeError("Windows 作业没有返回 qwen3 结果")
    if not qwen.get("ok"):
        raise RuntimeError(f"qwen3 识别失败：{qwen.get('error') or '未知错误'}")   # 不降级：该文件报错失败

    words: list[Word] = []
    for seg in qwen.get("segments") or []:
        ws = [Word(str(w["text"]), float(w["start"]), float(w["end"])) for w in (seg.get("words") or [])]
        if not ws:   # 与 _all_words 同法：无词级时间戳的段用整段兜底
            ws = [Word(str(seg.get("text") or ""), float(seg["start"]), float(seg["end"]))]
        words += ws
    truncated = bool(qwen.get("truncated"))
    dur = probe_duration(wav)
    chunks = int(qwen.get("n_segments") or len(qwen.get("segments") or []))   # qwen3 实际段数
    revisions = qwen.get("model_revisions") or {}
    asr_rev = str(revisions.get("qwen3-asr-hf") or qwen.get("model_revision") or "")
    asr_short = f"@{asr_rev[:7]}" if asr_rev else ""
    asr_label = f"qwen3 / Qwen3-ASR-1.7B-hf{asr_short}"

    turns: list[SpeakerTurn] = []
    diar_label = diar_rev = diar_name = None
    quality, dropped = "ok", None
    if not ns.no_diarize:
        nemo = results.get("nemotron") or {}
        dia = results.get("diarizen") or {}

        def _ok(x):
            return bool(x.get("ok"))

        # 说话人选择：指定人数 > 8 → diarizen；nemotron 满 8 人 → diarizen；否则 nemotron。
        use_dia, reason = False, ""
        if ns.speakers is not None and ns.speakers > 8:
            use_dia, reason = True, f"指定 {ns.speakers} 人 > 8"
        elif _ok(nemo) and nemo.get("capped"):
            use_dia, reason = True, "nemotron 满 8 人后切换"
        if use_dia and not _ok(dia):
            if _ok(nemo):
                use_dia, reason = False, "diarizen 失败后切换"
        elif not use_dia and not _ok(nemo):
            if _ok(dia):
                use_dia, reason = True, "nemotron 失败后切换"
        chosen = dia if use_dia else nemo
        if _ok(chosen):
            diar_name = "diarizen" if use_dia else "nemotron"
            diar_rev = str(chosen.get("model_revision") or "")
            raw = [SpeakerTurn(int(t["speaker"]), float(t["start"]), float(t["end"]))
                   for t in chosen.get("turns") or []]
            turns, dropped = _clean_turns(raw)   # 残标签清理 + 按首次出现重编号
            diar_label = diar_name + (f"@{diar_rev[:7]}" if diar_rev else "") + (f"（{reason}）" if reason else "")
        else:
            errors = [f"{name}: {x.get('error')}" for name, x in (("nemotron", nemo), ("diarizen", dia)) if x.get("error")]
            print("⚠️ 说话人分离失败，降级为无说话人转录稿：" + ("；".join(errors) or "两个分离器均未产出"))
            quality = "diarization-failed"

    if ns.speakers is not None and not ns.no_diarize:
        got = len({t.speaker for t in turns})
        if got != ns.speakers:   # 只提醒，不改 quality
            print(f"⚠️ 指定 --speakers {ns.speakers} 人，最终分离出 {got} 人（--speakers 只约束聚类数，请复核归属）")

    so_label = so_result = None
    disagreements = suspected = None
    if want_second:
        moss = results.get("moss") or {}
        if moss.get("ok"):
            so_result = second_opinion.compare(words, moss.get("segments") or [], moss.get("chunks") or [])
            so_label = "moss" + (f"@{str(moss.get('model_revision') or '')[:7]}" if moss.get("model_revision") else "")
            disagreements, suspected = so_result.disagreements, so_result.suspected_gaps
        else:
            print(f"⚠️ MOSS 第二意见未运行（{moss.get('error') or '没有结果'}）")

    engine = f"Qwen3-ASR-1.7B-hf{asr_short}" + (f" + {diar_name}" if diar_label else "")
    return Capture(words=words, truncated=truncated, asr_label=asr_label, chunks=chunks, dur=dur,
                   turns=turns, diar_label=diar_label, quality=quality, engine=engine,
                   engine_host="win", asr_revision=asr_rev or None, diarizer_revision=diar_rev or None,
                   speakers_dropped=dropped, second_opinion=so_label,
                   disagreements=disagreements, suspected_gaps=suspected, so_result=so_result)

def _meta_for_json(meta: Meta) -> dict:
    """--json 用的 meta dict：新字段为 None 时整个键不输出（与改动前的 JSON 逐字节一致）。"""
    data = asdict(meta)
    for key in ("engine_host", "asr_revision", "diarizer_revision", "speakers_dropped",
                "second_opinion", "disagreements", "suspected_gaps"):
        if data.get(key) is None:
            data.pop(key, None)
    return data

def _emit(src: Path, ns, cap: Capture) -> Path:
    """合并写出（两条路线共用）：拼接 → 质检 → front-matter → _md / .json / _转换索引.md [→ 分歧.md]。"""
    labeled = assign_speakers(cap.words, cap.turns) if cap.turns else [(w, None) for w in cap.words]
    paras = build_paragraphs(labeled)
    if ns.context:
        kept = drop_context_echo(paras, ns.context)
        if len(kept) < len(paras):
            print(f"已剔除 {len(paras) - len(kept)} 段热词回吐")
        paras = kept
    text_len = sum(len(p.text) for p in paras)
    quality = cap.quality
    if cap.truncated or text_len < 0.3 * cap.dur:   # 警告不被 diarization-failed 掩盖；quality 只在仍为 ok 时降为 suspect
        print(f"⚠️ 识别结果可疑（truncated={cap.truncated}，{text_len} 字 / {cap.dur:.0f} 秒）")
        if quality == "ok":
            quality = "suspect"
    speakers = len({t.speaker for t in cap.turns})
    if speakers > 10:   # 两条路线都加：修正 Mac 现行方案切出几百个说话人仍标 ok
        print(f"⚠️ 说话人过多（{speakers} 人）：分离结果疑似过切，请用 --speakers 指定人数后复核")
        if quality == "ok":
            quality = "suspect"
    meta = Meta(source=src.name, duration_s=cap.dur, recorded_at=_recorded_at(src), transcribed_at=date.today().isoformat(),
                asr_label=cap.asr_label, diarizer_label=cap.diar_label, speakers=speakers,
                speakers_hint=str(ns.speakers) if ns.speakers else "auto", quality=quality,
                context=ns.context or "", chunks=cap.chunks,
                engine_host=cap.engine_host, asr_revision=cap.asr_revision, diarizer_revision=cap.diarizer_revision,
                speakers_dropped=cap.speakers_dropped, second_opinion=cap.second_opinion,
                disagreements=cap.disagreements, suspected_gaps=cap.suspected_gaps)
    md = md_target(src)
    _write_atomic(md, render(meta, paras))
    if ns.json:
        _write_atomic(md.with_suffix(".json"), json.dumps({"meta": _meta_for_json(meta), "paragraphs": [asdict(p) for p in paras],
            "turns": [asdict(t) for t in cap.turns]}, ensure_ascii=False, indent=1))
    upsert(src.parent / "_转换索引.md", src.name, format_ts(cap.dur), f"_md/{md.name}", "✅已转", cap.engine, meta.transcribed_at)
    if cap.so_result is not None:
        second_opinion.write_report(md, cap.so_result, cap.second_opinion or "moss", src.name)
    return md

def run_one_local(src: Path, ns, asr, diarizer, diar_name: str | None, fallback=None) -> Path:
    with tempfile.TemporaryDirectory(prefix="cc-transcribe-") as tmp:
        return _emit(src, ns, _capture_local(src, ns, asr, diarizer, diar_name, Path(tmp), fallback))

def run_one_win(src: Path, ns, cfg: dict) -> Path:
    with tempfile.TemporaryDirectory(prefix="cc-transcribe-") as tmp:
        return _emit(src, ns, _capture_win(src, ns, cfg, Path(tmp)))

def main(argv=None) -> int:
    ns = parse_args(argv)
    try:
        cfg = load_config(ns.config)
        asr_name, asr_cfg, diar_name, diar_cfg = resolve(cfg, ns.asr, "none" if ns.no_diarize else ns.diarizer)
    except (OSError, tomllib.TOMLDecodeError, KeyError) as e:
        print(f"错误：配置无效（{ns.config or '内置 backends.toml'}）：{e!r}", file=sys.stderr); return 2
    # 只校验名称、不实例化：--dry-run / 全部跳过时绝不加载模型、绝不触网。
    for kind, name, table in (("asr", asr_name, ASR_BACKENDS), ("diarizer", diar_name, DIARIZERS)):
        if name is not None and name not in table:
            print(f"错误：未注册的 {kind} 后端 '{name}'\n已注册：{list_backends()}", file=sys.stderr); return 2
    if ns.local and ns.win_only:
        print("错误：--local 与 --win-only 不能同用", file=sys.stderr); return 2
    route = "local" if ns.local else ("win" if str(cfg.get("default", {}).get("runner", "local")) == "win" else "local")
    win_cfg = cfg.get("runner", {}).get("win", {}) if route == "win" else {}
    files = list_audio(ns.paths)
    if not files:
        if ns.paths and all(PURE_DIR in p.parts for p in ns.paths):
            return 0   # 指名的全在纯净区：按设计不写任何派生件，不算错误
        print("没有可转写的音频文件", file=sys.stderr); return 1
    plan, rc = [], 0
    for f in files:
        go, why = needs_transcribe(f, ns.force)
        try:
            d = format_ts(probe_duration(f))
        except AudioError as e:
            print(f"✗ {f.name}：{e}", file=sys.stderr); rc = 1; continue
        plan.append((f, go, why, d))
        mark = {"new": "待转", "forced": "重转", "exists": "跳过（已有 _md）", "stale-hint": "跳过（已有 _md；原件更新，--force 重转）"}[why]
        print(f"{'→' if go else '·'} {f.name}  {d}  {mark}")
    todo = [f for f, go, _, _ in plan if go]
    if ns.dry_run or not todo:
        if ns.dry_run:   # 不做预检，只标路线
            print("（Windows 路线）" if route == "win" else "（Mac 本地）")
        return rc
    local = None   # (asr, diarizer, diar_name, fallback)，本地路线或回退时才加载

    def load_local():
        """加载 Mac 本地后端；失败返回 None（已打印原因）。"""
        nonlocal diar_name
        try:
            asr = get_asr(asr_name, asr_cfg)
            fallback = None
            if diar_name == "nemotron":   # 本地 nemotron 的退路：sherpa（满 8 人 / 指定 > 8 人 / nemotron 不可用）
                sherpa_cfg = cfg.get("diarizer", {}).get("sherpa", {})
                fallback = lambda: get_diarizer("sherpa", sherpa_cfg)
                try:
                    diarizer = get_diarizer(diar_name, diar_cfg)
                except (RuntimeError, ImportError, OSError) as e:
                    print(f"⚠️ nemotron 不可用，本次改用 sherpa：{e}")
                    diar_name, diarizer, fallback = "sherpa", get_diarizer("sherpa", sherpa_cfg), None
            else:
                diarizer = get_diarizer(diar_name, diar_cfg) if diar_name else None
        except (RuntimeError, ImportError, OSError) as e:
            print(f"✗ 后端加载失败：{e}", file=sys.stderr); return None
        return asr, diarizer, diar_name, fallback

    def fall_back(why) -> bool:
        """律师 2026-10-31 裁定：Windows 不可用时回退 Mac 本机。返回是否成功切到本地。"""
        nonlocal route, local
        print(f"⚠️ Windows 不可用（{why}），改在 Mac 本机转写（本地路线无 MOSS 第二意见；首次运行需下载约 4 GB 模型）")
        route = "local"
        local = load_local()
        return local is not None

    if route == "win":
        if ns.asr or ns.diarizer:
            print("⚠️ Windows 路线不使用 Mac 的 --asr / --diarizer（要用本地后端请加 --local）")
        try:
            remote.preflight(win_cfg)
        except remote.WindowsUnavailable as e:
            if ns.win_only:
                print(f"WINDOWS_UNAVAILABLE: {e}", file=sys.stderr); return 3   # --win-only：不降级
            if not fall_back(e):
                return 1
    else:
        local = load_local()
        if local is None:
            return 1
    for f in todo:
        try:
            if route == "win":
                try:
                    out = run_one_win(f, ns, win_cfg)
                except remote.WindowsUnavailable as e:   # 中途 75 / 掉线
                    if ns.win_only:
                        print(f"WINDOWS_UNAVAILABLE: {e}", file=sys.stderr); return 3
                    if not fall_back(e):
                        return 1
                    out = run_one_local(f, ns, *local)  # 产物只在 Mac 写，本文件与其余文件改在本机做是安全的
            else:
                out = run_one_local(f, ns, *local)
            print(f"✓ {f.name} → {out}")
        except Exception as e:   # 单文件失败不拖垮整批：报错、记 rc、继续下一个
            print(f"✗ {f.name}：{e}", file=sys.stderr); rc = 1
    return rc

if __name__ == "__main__":
    sys.exit(main())
