import json
from pathlib import Path
import pytest
import cli
from backends import ASR_BACKENDS, DIARIZERS
from index import md_target
from writer import H_SUMMARY, H_TRANSCRIPT, SUMMARY_PLACEHOLDER

@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setitem(ASR_BACKENDS, "fake-asr", "tests.fake_backends:FakeASR")
    monkeypatch.setitem(DIARIZERS, "fake-diar", "tests.fake_backends:FakeDiarizer")

def _main(args):
    """本文件的用例一律走 Mac 本地路线：内置 backends.toml 的 runner 已改为 win，
    这里统一补 --local（Windows 路线另见 tests/test_win_route.py）。"""
    return cli.main(list(args) + ["--local"])

def _src(tmp_path, synth_wav) -> Path:
    case = tmp_path / "20260101某案" / "4.录音资料"; case.mkdir(parents=True)
    src = case / "通话 张三.m4a"
    import subprocess; subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(synth_wav), "-c:a", "aac", str(src)], check=True)
    return src

def test_end_to_end_with_fakes(tmp_path, synth_wav, fake_registry, capsys):
    src = _src(tmp_path, synth_wav)
    rc = _main([str(src), "--asr", "fake-asr", "--diarizer", "fake-diar", "--speakers", "2", "--context", "张三 李四"])
    assert rc == 0
    md = md_target(src); text = md.read_text(encoding="utf-8")
    assert text.index(H_SUMMARY) < text.index(SUMMARY_PLACEHOLDER) < text.index(H_TRANSCRIPT)
    assert "**说话人1**：你好世界" in text and "**说话人2**：再见" in text
    assert 'context: "张三 李四"' in text and "speakers_hint: 2" in text and "quality: ok" in text
    idx = (src.parent / "_转换索引.md").read_text(encoding="utf-8")
    assert "| 通话 张三.m4a | 00:00:05 | _md/通话 张三.md | ✅已转 | fake-asr/1.0 + fake-diar |" in idx
    assert "✓" in capsys.readouterr().out

def test_skip_existing_and_force(tmp_path, synth_wav, fake_registry, capsys):
    src = _src(tmp_path, synth_wav)
    assert _main([str(src), "--asr", "fake-asr", "--diarizer", "fake-diar"]) == 0
    assert _main([str(src), "--asr", "fake-asr", "--diarizer", "fake-diar"]) == 0
    assert "跳过" in capsys.readouterr().out
    assert _main([str(src), "--asr", "fake-asr", "--diarizer", "fake-diar", "--force"]) == 0

def test_no_diarize_and_json(tmp_path, synth_wav, fake_registry):
    src = _src(tmp_path, synth_wav)
    assert _main([str(src), "--asr", "fake-asr", "--no-diarize", "--json"]) == 0
    text = md_target(src).read_text(encoding="utf-8")
    assert "diarizer: none" in text and "说话人" not in text
    j = json.loads(md_target(src).with_suffix(".json").read_text(encoding="utf-8"))
    assert j["paragraphs"][0]["speaker"] is None

def test_diarizer_failure_degrades(tmp_path, synth_wav, fake_registry, monkeypatch):
    import tests.fake_backends as fb
    monkeypatch.setattr(fb.FakeDiarizer, "diarize", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    src = _src(tmp_path, synth_wav)
    assert _main([str(src), "--asr", "fake-asr", "--diarizer", "fake-diar"]) == 0
    assert "quality: diarization-failed" in md_target(src).read_text(encoding="utf-8")

def test_dir_listing_dry_run(tmp_path, synth_wav, fake_registry, capsys):
    src = _src(tmp_path, synth_wav); (src.parent / "note.txt").write_text("x")
    assert _main([str(src.parent), "--asr", "fake-asr", "--diarizer", "fake-diar", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "通话 张三.m4a" in out and "note.txt" not in out and not md_target(src).exists()

def test_unknown_backend_exit_code(tmp_path, synth_wav, capsys):
    src = _src(tmp_path, synth_wav)
    assert _main([str(src), "--asr", "nope"]) == 2
    assert "qwen3-mlx" in capsys.readouterr().err

# ---- Fix round 1 ----
def _touch(tmp_path, name) -> Path:
    p = tmp_path / name; p.write_bytes(b""); return p

def _mtime_date(p: Path) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")

@pytest.mark.parametrize("name,expect", [
    ("x_202613990.m4a", None),                          # 非法日期 2026-13-99 → mtime
    ("录音_133926_20241021.m4a", "2024-10-21"),          # 完整 YYYYMMDD 优先于 _HHMMSS_ 误判
    ("tel_10020260101.m4a", None),                      # 20260101 嵌在更长数字串里 → mtime
    ("通话录音 某秘书 某仲裁委_241021_133926.m4a", "2024-10-21"),
    ("会议 2026-03-05.m4a", "2026-03-05"),
    ("通话 张三.m4a", None),
    ("13-45 x.mp3", None),                              # MM-DD 前缀但月 13 非法 → mtime
    ("0526.mp3", None),                                 # 无连字符，不命中 MM-DD 分支 → mtime
    ("05-261.mp3", None),                               # MM-DD 后紧跟数字 → 不命中
])
def test_recorded_at_parsing(tmp_path, name, expect):
    p = _touch(tmp_path, name)
    assert cli._recorded_at(p) == (expect or _mtime_date(p))

def _src_named(tmp_path, synth_wav, name) -> Path:
    case = tmp_path / "20260101某案" / "4.录音资料"; case.mkdir(parents=True, exist_ok=True)
    src = case / name
    import subprocess; subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(synth_wav), "-c:a", "aac", str(src)], check=True)
    return src

def test_one_file_failure_does_not_abort_batch(tmp_path, synth_wav, fake_registry, monkeypatch, capsys):
    import tests.fake_backends as fb
    a = _src_named(tmp_path, synth_wav, "a.m4a"); b = _src_named(tmp_path, synth_wav, "b.m4a")
    orig, calls = fb.FakeASR.transcribe, []
    def flaky(self, wav, *, context=None):
        calls.append(wav)
        if len(calls) == 2:
            raise ValueError("模型炸了")
        return orig(self, wav, context=context)
    monkeypatch.setattr(fb.FakeASR, "transcribe", flaky)
    rc = _main([str(a.parent), "--asr", "fake-asr", "--diarizer", "fake-diar"])
    out = capsys.readouterr()
    assert rc == 1 and len(calls) == 2
    assert md_target(a).exists() and not md_target(b).exists()
    idx = (a.parent / "_转换索引.md").read_text(encoding="utf-8")
    assert "| a.m4a |" in idx and "b.m4a" not in idx
    assert "✗ b.m4a：模型炸了" in out.err and "Traceback" not in out.err

def test_config_without_default_exits_2(tmp_path, synth_wav, fake_registry, capsys):
    src = _src(tmp_path, synth_wav)
    bad = tmp_path / "bad.toml"; bad.write_text('[asr.fake-asr]\nmodel = "x"\n', encoding="utf-8")
    assert _main([str(src), "--config", str(bad)]) == 2
    assert "配置无效" in capsys.readouterr().err and not md_target(src).exists()

def test_dry_run_and_all_skipped_never_instantiate(tmp_path, synth_wav, fake_registry, monkeypatch):
    src = _src(tmp_path, synth_wav)
    assert _main([str(src), "--asr", "fake-asr", "--diarizer", "fake-diar"]) == 0   # 先产出一份 _md
    def boom(*a, **k): raise AssertionError("不该实例化后端")
    monkeypatch.setattr(cli, "get_asr", boom); monkeypatch.setattr(cli, "get_diarizer", boom)
    fresh = _src_named(tmp_path / "other", synth_wav, "新.m4a")
    assert _main([str(fresh), "--asr", "fake-asr", "--diarizer", "fake-diar", "--dry-run"]) == 0
    assert not md_target(fresh).exists()
    assert _main([str(src), "--asr", "fake-asr", "--diarizer", "fake-diar"]) == 0     # 全部跳过

def test_list_audio_skips_dotfiles(tmp_path):
    d = tmp_path / "vol"; d.mkdir()
    (d / "rec.m4a").write_bytes(b""); (d / "._rec.m4a").write_bytes(b"")
    assert [p.name for p in cli.list_audio([d])] == ["rec.m4a"]

# ---- Final fix wave ----
def _src_at(path: Path, synth_wav) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    import subprocess; subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(synth_wav), "-c:a", "aac", str(path)], check=True)
    return path

def test_pure_dir_excluded(tmp_path, synth_wav, fake_registry, capsys):
    case = tmp_path / "20260101某案" / "3.诉讼资料"
    pure = _src_at(case / cli.PURE_DIR / "rec.m4a", synth_wav)
    ours = _src_at(case / "3.我方诉讼资料" / "rec2.m4a", synth_wav)
    assert _main([str(case.parent), "--asr", "fake-asr", "--diarizer", "fake-diar"]) == 0
    out = capsys.readouterr().out
    assert (ours.parent / "_md" / "rec2.md").exists()
    assert not list(pure.parent.rglob("_md")) and not list(pure.parent.rglob("_转换索引.md"))
    assert "跳过（纯净区，不写派生件）：rec.m4a" in out and "rec.m4a  00:00:05" not in out
    before = sorted(p.name for p in pure.parent.rglob("*"))
    assert _main([str(pure), "--asr", "fake-asr", "--diarizer", "fake-diar"]) == 0
    assert sorted(p.name for p in pure.parent.rglob("*")) == before
    assert "跳过（纯净区，不写派生件）：rec.m4a" in capsys.readouterr().out

def test_cross_format_same_stem_end_to_end(tmp_path, synth_wav, fake_registry):
    src = _src(tmp_path, synth_wav)
    docx_md = src.parent / "_md" / "通话 张三.md"; docx_md.parent.mkdir()
    docx_bytes = "---\nsource: ../通话 张三.docx\nengine: native-docx\n---\n\n<!-- SEC: 1 -->\n正文\n".encode("utf-8")
    docx_md.write_bytes(docx_bytes)
    assert _main([str(src), "--asr", "fake-asr", "--diarizer", "fake-diar", "--json"]) == 0
    assert docx_md.read_bytes() == docx_bytes
    md = src.parent / "_md" / "通话 张三.m4a.md"
    assert md_target(src) == md and md.exists() and 'source: "通话 张三.m4a"' in md.read_text(encoding="utf-8")
    assert (src.parent / "_md" / "通话 张三.m4a.json").exists()
    idx = (src.parent / "_转换索引.md").read_text(encoding="utf-8")
    assert "| 通话 张三.m4a | 00:00:05 | _md/通话 张三.m4a.md | ✅已转 |" in idx

def test_truncated_warning_not_hidden_by_diar_failure(tmp_path, synth_wav, fake_registry, monkeypatch, capsys):
    import tests.fake_backends as fb
    from backends.base import ASRResult
    monkeypatch.setattr(fb.FakeDiarizer, "diarize", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    orig = fb.FakeASR.transcribe
    def truncated(self, wav, *, context=None):
        r = orig(self, wav, context=context)
        return ASRResult(r.segments, r.language, True, "length", r.model_label)
    monkeypatch.setattr(fb.FakeASR, "transcribe", truncated)
    src = _src(tmp_path, synth_wav)
    assert _main([str(src), "--asr", "fake-asr", "--diarizer", "fake-diar"]) == 0
    assert "quality: diarization-failed" in md_target(src).read_text(encoding="utf-8")
    assert "⚠️ 识别结果可疑" in capsys.readouterr().out

def test_chunking_shifts_timestamps(tmp_path, synth_wav, fake_registry, monkeypatch):
    import re, tests.fake_backends as fb
    monkeypatch.setattr(cli, "LONG_AUDIO_S", 1.0); monkeypatch.setattr(cli, "CHUNK_S", 2.0)
    shared = fb.FakeASR({})
    monkeypatch.setattr(cli, "get_asr", lambda name, cfg: shared)
    src = _src(tmp_path, synth_wav)
    assert _main([str(src), "--asr", "fake-asr", "--diarizer", "fake-diar"]) == 0
    assert len(shared.calls) == 3 and len({c["wav"] for c in shared.calls}) == 3
    text = md_target(src).read_text(encoding="utf-8")
    assert "chunks: 3" in text
    stamps = [int(h) * 3600 + int(m) * 60 + int(s) for h, m, s in re.findall(r"<!-- T (\d\d):(\d\d):(\d\d) -->", text)]
    assert stamps and max(stamps) >= 2

# ---- 首战后修正 1 / 2 ----
def test_recorded_at_month_day_prefix_uses_mtime_year(tmp_path, capsys):
    p = _touch(tmp_path, "05-26 财务与信任危机.mp3")
    want = _mtime_date(p)[:4] + "-05-26"
    assert cli._recorded_at(p) == want
    assert f"recorded_at 由文件名月日 + 文件年份推得：{want}" in capsys.readouterr().out
    q = _touch(tmp_path, "05-26_x.mp3")   # 下划线相接同样命中
    assert cli._recorded_at(q) == _mtime_date(q)[:4] + "-05-26"

def test_context_echo_dropped_end_to_end(tmp_path, synth_wav, fake_registry, monkeypatch, capsys):
    import tests.fake_backends as fb
    from backends.base import ASRResult, Segment, Word
    ctx = "张三 李四 Tom 财务核对"
    def echoing(self, wav, *, context=None):
        words = [Word("你好", 0.0, 0.5), Word("世界", 0.6, 1.0),
                 Word("张三", 3.5, 3.7), Word("李四", 3.7, 3.9), Word("Tom", 3.9, 4.1), Word("财务核对", 4.1, 4.3),   # 回吐段
                 Word("再见", 6.6, 7.0)]
        return ASRResult([Segment("", 0.0, 7.0, words)], "Chinese", False, "eos", "fake-asr/1.0")
    monkeypatch.setattr(fb.FakeASR, "transcribe", echoing)
    src = _src(tmp_path, synth_wav)
    assert _main([str(src), "--asr", "fake-asr", "--no-diarize", "--context", ctx]) == 0
    text = md_target(src).read_text(encoding="utf-8")
    assert "张三李四Tom财务核对" not in text and "你好世界" in text and "再见" in text
    assert f'context: "{ctx}"' in text
    assert "已剔除 1 段热词回吐" in capsys.readouterr().out
