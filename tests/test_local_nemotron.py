"""Mac 本地 nemotron（2026-10-22 起本地默认分离器）的路由：满 8 人 / 指定 > 8 人 / 出错 / 不可用 → 退回 sherpa；残标签清理。
真实 nemotron 子进程不在此测（需要独立 venv 与 MPS）；这里用假后端占用 "nemotron" / "sherpa" 两个注册名。"""
import subprocess
import pytest
import cli
from backends import DIARIZERS, ASR_BACKENDS
from backends.base import SpeakerTurn
from index import md_target
from tests.fake_backends import FakeNemotron, FakeSherpa


@pytest.fixture
def reg(monkeypatch):
    monkeypatch.setitem(ASR_BACKENDS, "fake-asr", "tests.fake_backends:FakeASR")
    monkeypatch.setitem(DIARIZERS, "nemotron", "tests.fake_backends:FakeNemotron")
    monkeypatch.setitem(DIARIZERS, "sherpa", "tests.fake_backends:FakeSherpa")
    for k, v in dict(capped_next=False, raise_on_diarize=False, raise_on_init=False, calls=0).items():
        setattr(FakeNemotron, k, v)
    FakeSherpa.calls = 0


def _src(tmp_path, synth_wav):
    case = tmp_path / "20260101某案" / "4.录音资料"; case.mkdir(parents=True)
    src = case / "通话 张三.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(synth_wav), "-c:a", "aac", str(src)], check=True)
    return src


def _run(src, *extra):
    return cli.main([str(src), "--local", "--asr", "fake-asr", *extra])


def test_default_local_is_nemotron_and_cleans_residual(tmp_path, synth_wav, reg):
    src = _src(tmp_path, synth_wav)
    assert _run(src) == 0
    text = md_target(src).read_text(encoding="utf-8")
    assert "diarizer: nemotron@test (mps)" in text and "speakers: 2" in text and "speakers_dropped: 1" in text
    assert FakeNemotron.calls == 1 and FakeSherpa.calls == 0
    assert "+ nemotron |" in (src.parent / "_转换索引.md").read_text(encoding="utf-8")


def test_capped_falls_back_to_sherpa(tmp_path, synth_wav, reg):
    FakeNemotron.capped_next = True
    src = _src(tmp_path, synth_wav)
    assert _run(src) == 0
    text = md_target(src).read_text(encoding="utf-8")
    assert "diarizer: sherpa / fake（nemotron 满 8 人后切换）" in text
    assert FakeSherpa.calls == 1
    assert "+ sherpa |" in (src.parent / "_转换索引.md").read_text(encoding="utf-8")


def test_more_than_8_speakers_skips_nemotron(tmp_path, synth_wav, reg):
    src = _src(tmp_path, synth_wav)
    assert _run(src, "--speakers", "9") == 0
    assert FakeNemotron.calls == 0 and FakeSherpa.calls == 1
    assert "指定 9 人 > 8" in md_target(src).read_text(encoding="utf-8")


def test_nemotron_runtime_error_falls_back(tmp_path, synth_wav, reg, capsys):
    FakeNemotron.raise_on_diarize = True
    src = _src(tmp_path, synth_wav)
    assert _run(src) == 0
    assert "nemotron 失败后切换" in md_target(src).read_text(encoding="utf-8")
    assert "改用 sherpa" in capsys.readouterr().out


def test_nemotron_unavailable_uses_sherpa(tmp_path, synth_wav, reg, capsys):
    FakeNemotron.raise_on_init = True
    src = _src(tmp_path, synth_wav)
    assert _run(src) == 0
    text = md_target(src).read_text(encoding="utf-8")
    assert "diarizer: sherpa / fake" in text and "speakers_dropped" not in text   # 退回 sherpa 走原路径，不清理
    assert "nemotron 不可用" in capsys.readouterr().out


def test_explicit_sherpa_keeps_old_path(tmp_path, synth_wav, reg):
    src = _src(tmp_path, synth_wav)
    assert _run(src, "--diarizer", "sherpa") == 0
    text = md_target(src).read_text(encoding="utf-8")
    assert FakeNemotron.calls == 0 and "speakers_dropped" not in text and "diarizer: sherpa / fake" in text


def test_clean_turns_relative_floor_and_never_empty():
    # 长录音：绝对 5 秒生效
    long = [SpeakerTurn(0, 0, 600), SpeakerTurn(1, 600, 900), SpeakerTurn(2, 900, 903)]
    out, dropped = cli._clean_turns(long)
    assert dropped == 1 and {t.speaker for t in out} == {0, 1}
    # 短录音：3 秒的人不是残标签
    short = [SpeakerTurn(0, 0, 12), SpeakerTurn(1, 12, 15)]
    assert cli._clean_turns(short) == (short, 0)
    # 全部很短：不剔光
    tiny = [SpeakerTurn(0, 0, 0.3)]
    assert cli._clean_turns(tiny)[0] == tiny
