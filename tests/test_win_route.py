"""Windows 路线：cli 路由 / 取得阶段（全部 mock 远端，不连 Windows、不加载模型）。"""
import json
import subprocess
from pathlib import Path

import pytest
import yaml

import cli
import remote
from backends import ASR_BACKENDS, DIARIZERS
from index import md_target

FIX = Path(__file__).resolve().parent / "fixtures" / "win"
_REAL_PREFLIGHT = remote.preflight          # win fixture 会把它换掉，这里留一份真身


def _fx(name: str) -> dict:
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


def _results(qwen=None, nemo=None, dia=None, moss=None) -> dict:
    return {"qwen3": qwen if qwen is not None else _fx("qwen3"),
            "nemotron": nemo if nemo is not None else _fx("nemotron"),
            "diarizen": dia if dia is not None else _fx("diarizen"),
            "moss": moss if moss is not None else _fx("moss")}


@pytest.fixture
def src(tmp_path, synth_wav) -> Path:
    case = tmp_path / "20260101某案" / "4.录音资料"; case.mkdir(parents=True)
    s = case / "通话 张三.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(synth_wav), "-c:a", "aac", str(s)], check=True)
    return s


@pytest.fixture
def win(monkeypatch):
    """内置 backends.toml 已是 runner=win；把远端边界换成夹具。"""
    state = {"preflight": [], "calls": [], "results": _results()}

    def fake_preflight(cfg):
        state["preflight"].append(cfg)

    def fake_run_remote(wav, job, engines, cfg):
        state["calls"].append({"job": job, "engines": engines, "cfg": cfg})
        return state["results"]

    monkeypatch.setattr(remote, "preflight", fake_preflight)
    monkeypatch.setattr(remote, "run_remote", fake_run_remote)
    return state


def _fm(md) -> dict:
    text = Path(md).read_text(encoding="utf-8")
    _, fm_text, _ = text.split("---", 2)
    return yaml.safe_load(fm_text)


def _index(src) -> str:
    return (src.parent / "_转换索引.md").read_text(encoding="utf-8")


# ---- 说话人路由四种情形 ----------------------------------------------------

def test_win_route_normal_uses_nemotron(src, win):
    assert cli.main([str(src)]) == 0
    fm = _fm(md_target(src))
    assert fm["engine_host"] == "win"
    assert fm["asr"] == "qwen3 / Qwen3-ASR-1.7B-hf@bcd2b5b"
    assert fm["asr_revision"] == "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
    assert fm["diarizer"] == "nemotron@f667ed7"
    assert fm["diarizer_revision"] == "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
    assert fm["speakers"] == 4 and fm["speakers_dropped"] == 0
    assert fm["second_opinion"] == "moss@704aa4a"
    assert isinstance(fm["disagreements"], int) and isinstance(fm["suspected_gaps"], int)
    assert fm["quality"] == "ok" and fm["chunks"] == 1
    call = win["calls"][0]
    assert call["engines"] == ["qwen3", "nemotron", "diarizen", "moss"]
    assert call["job"] == {"language": "Chinese", "context": "", "num_speakers": None,
                          "min_speakers": None, "max_speakers": None}
    assert len(win["preflight"]) == 1
    assert "| 通话 张三.m4a | 00:00:05 | _md/通话 张三.md | ✅已转 | Qwen3-ASR-1.7B-hf@bcd2b5b + nemotron |" in _index(src)


def test_win_route_capped_switches_to_diarizen(src, win):
    nemo = _fx("nemotron"); nemo["capped"] = True
    win["results"] = _results(nemo=nemo)
    assert cli.main([str(src)]) == 0
    fm = _fm(md_target(src))
    assert fm["diarizer"] == "diarizen@027b3e2（nemotron 满 8 人后切换）"
    assert "+ diarizen |" in _index(src)


def test_win_route_speakers_gt_8_uses_diarizen(src, win, capsys):
    assert cli.main([str(src), "--speakers", "10"]) == 0
    fm = _fm(md_target(src))
    assert fm["diarizer"] == "diarizen@027b3e2（指定 10 人 > 8）"
    assert win["calls"][0]["job"]["num_speakers"] == 10
    assert "指定 --speakers 10 人，最终分离出 4 人" in capsys.readouterr().out


def test_win_route_chosen_diarizer_fails_uses_other(src, win):
    nemo = _fx("nemotron"); nemo["ok"] = False; nemo["error"] = "boom"
    win["results"] = _results(nemo=nemo)
    assert cli.main([str(src)]) == 0
    assert _fm(md_target(src))["diarizer"] == "diarizen@027b3e2（nemotron 失败后切换）"

    nemo = _fx("nemotron"); nemo["capped"] = True
    dia = _fx("diarizen"); dia["ok"] = False; dia["error"] = "CUDA OOM"
    win["results"] = _results(nemo=nemo, dia=dia)
    assert cli.main([str(src), "--force"]) == 0
    assert _fm(md_target(src))["diarizer"] == "nemotron@f667ed7（diarizen 失败后切换）"


def test_win_route_both_diarizers_fail(src, win):
    for name in ("nemotron", "diarizen"):
        fx = _fx(name); fx["ok"] = False; fx["error"] = "boom"
        if name == "nemotron":
            nemo = fx
        else:
            dia = fx
    win["results"] = _results(nemo=nemo, dia=dia)
    assert cli.main([str(src)]) == 0
    text = md_target(src).read_text(encoding="utf-8")
    fm = _fm(md_target(src))
    assert fm["quality"] == "diarization-failed" and fm["diarizer"] == "none" and fm["speakers"] == 0
    assert "说话人" not in text.split("## 转录稿")[1]


# ---- 残标签清理与重编号 ----------------------------------------------------

def test_win_route_residual_speaker_dropped_and_renumbered(src, win):
    nemo = _fx("nemotron")
    nemo["turns"] = [{"speaker": 5, "start": 0.0, "end": 4.0},     # 4 s < min(5 s, 总发声 5%) → 丢弃
                     {"speaker": 9, "start": 5.0, "end": 105.0},   # 100 s，首次出现 → 0
                     {"speaker": 3, "start": 106.0, "end": 240.0}]  # 134 s → 1（总发声 238 s，5% = 11.9 s，阈值取 5 s）
    win["results"] = _results(nemo=nemo)
    assert cli.main([str(src)]) == 0
    fm = _fm(md_target(src))
    assert fm["speakers"] == 2 and fm["speakers_dropped"] == 1
    text = md_target(src).read_text(encoding="utf-8")
    assert "**说话人3**" not in text and "**说话人2**" in text


def test_clean_turns_unit():
    from backends.base import SpeakerTurn
    turns = [SpeakerTurn(4, 0.0, 60.0), SpeakerTurn(9, 60.0, 120.0), SpeakerTurn(4, 120.0, 200.0),
             SpeakerTurn(7, 200.0, 203.0)]
    out, dropped = cli._clean_turns(turns)
    assert dropped == 1                      # speaker 7 只有 3 s（总发声 203 s，阈值 min(5, 10.15) = 5 s）
    assert [t.speaker for t in out] == [0, 1, 0]
    assert [(t.start, t.end) for t in out] == [(0.0, 60.0), (60.0, 120.0), (120.0, 200.0)]


def test_clean_turns_renumbers_by_first_appearance():
    from backends.base import SpeakerTurn
    out, dropped = cli._clean_turns([SpeakerTurn(7, 0.0, 8.0), SpeakerTurn(2, 9.0, 18.0)])
    assert dropped == 0 and [t.speaker for t in out] == [0, 1]


# ---- 质检：> 10 人降为 suspect ---------------------------------------------

def test_more_than_10_speakers_warns_and_marks_suspect(src, win, capsys):
    nemo = _fx("nemotron")
    nemo["turns"] = [{"speaker": i, "start": i * 10.0, "end": i * 10.0 + 6.0} for i in range(12)]
    win["results"] = _results(nemo=nemo)
    assert cli.main([str(src)]) == 0
    out = capsys.readouterr().out
    assert "说话人过多（12 人）" in out
    assert _fm(md_target(src))["quality"] == "suspect"


# ---- qwen3 失败 / 预检 / 75 -------------------------------------------------

def test_qwen3_failure_fails_file(src, win, capsys):
    qwen = _fx("qwen3"); qwen["ok"] = False; qwen["error"] = "CUDA out of memory"
    win["results"] = _results(qwen=qwen)
    assert cli.main([str(src)]) == 1
    assert "qwen3 识别失败：CUDA out of memory" in capsys.readouterr().err
    assert not md_target(src).exists()


def test_missing_qwen3_result_fails_file(src, win, capsys):
    win["results"] = {"nemotron": _fx("nemotron")}
    assert cli.main([str(src)]) == 1
    assert "没有返回 qwen3 结果" in capsys.readouterr().err


def test_preflight_failure_win_only_returns_3(src, win, capsys, monkeypatch):
    def boom(cfg):
        raise remote.WindowsUnavailable("显卡空闲显存 8000 MB 不足 12000 MB（显卡被占用）")

    monkeypatch.setattr(remote, "preflight", boom)
    assert cli.main([str(src), "--win-only"]) == 3
    err = capsys.readouterr().err
    assert "WINDOWS_UNAVAILABLE: 显卡空闲显存 8000 MB 不足 12000 MB（显卡被占用）" in err
    assert not md_target(src).exists() and not win["calls"]


def test_preflight_ssh_failure_win_only_returns_3_at_subprocess_level(src, win, capsys, monkeypatch):
    """预检失败走真实 preflight/_ps，只在 subprocess 层 mock：ssh 起不来（其余 ffmpeg/ffprobe 照常）。"""
    real_run = subprocess.run

    def fake_run(cmd, *a, **k):
        if cmd and str(cmd[0]).endswith("ssh"):
            raise FileNotFoundError("ssh 不存在")
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    monkeypatch.setattr(remote, "preflight", _REAL_PREFLIGHT)
    assert cli.main([str(src), "--win-only"]) == 3
    err = capsys.readouterr().err
    assert "WINDOWS_UNAVAILABLE: Windows 不在线或 ssh 不通" in err
    assert not md_target(src).exists()


def test_remote_busy_75_mid_run_win_only_returns_3(src, win, capsys, monkeypatch):
    def busy(wav, job, engines, cfg):
        raise remote.WindowsUnavailable("Windows 正在跑别的转写作业")

    monkeypatch.setattr(remote, "run_remote", busy)
    assert cli.main([str(src), "--win-only"]) == 3
    assert "WINDOWS_UNAVAILABLE: Windows 正在跑别的转写作业" in capsys.readouterr().err


# ---- 默认回退 Mac 本机（律师 2026-10-31 裁定） ------------------------------

@pytest.fixture
def fake_local(monkeypatch, tmp_path):
    """本地后端全换成桩：不加载模型；run_one_local 只记账并写一个占位产物路径。"""
    state = {"loaded": 0, "ran": []}

    def fake_get_asr(name, cfg):
        state["loaded"] += 1; return object()

    def fake_run_one_local(src, ns, asr, diarizer, diar_name, fallback=None):
        state["ran"].append(src.name); return tmp_path / f"{src.stem}.md"

    monkeypatch.setattr(cli, "get_asr", fake_get_asr)
    monkeypatch.setattr(cli, "get_diarizer", lambda name, cfg: object())
    monkeypatch.setattr(cli, "run_one_local", fake_run_one_local)
    return state


def test_preflight_failure_falls_back_to_local(src, win, fake_local, capsys, monkeypatch):
    def boom(cfg):
        raise remote.WindowsUnavailable("Windows 不在线或 ssh 不通（timeout）")

    monkeypatch.setattr(remote, "preflight", boom)
    assert cli.main([str(src)]) == 0
    out = capsys.readouterr().out
    assert "Windows 不可用（Windows 不在线或 ssh 不通（timeout））" in out and "改在 Mac 本机转写" in out
    assert fake_local["ran"] == [src.name] and not win["calls"]


def test_mid_run_unavailable_falls_back_for_current_and_rest(src, win, fake_local, capsys, monkeypatch):
    second = src.parent / "通话 李四.m4a"; second.write_bytes(src.read_bytes())

    def busy(wav, job, engines, cfg):
        raise remote.WindowsUnavailable("Windows 正在跑别的转写作业")

    monkeypatch.setattr(remote, "run_remote", busy)
    assert cli.main([str(src.parent)]) == 0
    assert sorted(fake_local["ran"]) == sorted([src.name, second.name])
    assert fake_local["loaded"] == 1                       # 本地后端只加载一次
    assert capsys.readouterr().out.count("改在 Mac 本机转写") == 1


def test_local_and_win_only_conflict(src, win):
    assert cli.main([str(src), "--local", "--win-only"]) == 2


# ---- --dry-run / --local / --no-second-opinion / --no-diarize ---------------

def test_dry_run_no_preflight_and_route_marker(src, win, capsys, monkeypatch):
    def boom(cfg):
        raise AssertionError("--dry-run 不该预检")

    monkeypatch.setattr(remote, "preflight", boom)
    assert cli.main([str(src), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "待转" in out and "（Windows 路线）" in out
    assert not md_target(src).exists() and not win["calls"]


def test_local_flag_uses_mac_backends(src, monkeypatch, capsys):
    monkeypatch.setitem(ASR_BACKENDS, "fake-asr", "tests.fake_backends:FakeASR")
    monkeypatch.setitem(DIARIZERS, "fake-diar", "tests.fake_backends:FakeDiarizer")

    def boom(*a, **k):
        raise AssertionError("--local 不该碰 Windows")

    monkeypatch.setattr(remote, "preflight", boom)
    monkeypatch.setattr(remote, "run_remote", boom)
    assert cli.main([str(src), "--local", "--asr", "fake-asr", "--diarizer", "fake-diar"]) == 0
    text = md_target(src).read_text(encoding="utf-8")
    assert "diarizer: fake-diar" in text and "engine_host" not in text
    assert "| 通话 张三.m4a | 00:00:05 | _md/通话 张三.md | ✅已转 | fake-asr/1.0 + fake-diar |" in _index(src)


def test_no_second_opinion(src, win):
    assert cli.main([str(src), "--no-second-opinion"]) == 0
    assert win["calls"][0]["engines"] == ["qwen3", "nemotron", "diarizen"]
    fm = _fm(md_target(src))
    assert "second_opinion" not in fm and "disagreements" not in fm and "suspected_gaps" not in fm


def test_win_route_no_diarize(src, win):
    assert cli.main([str(src), "--no-diarize"]) == 0
    assert win["calls"][0]["engines"] == ["qwen3", "moss"]
    fm = _fm(md_target(src))
    assert fm["diarizer"] == "none" and fm["speakers"] == 0


def test_win_route_ignores_asr_flag_with_warning(src, win, capsys):
    assert cli.main([str(src), "--asr", "qwen3-mlx"]) == 0
    assert "Windows 路线不使用 Mac 的 --asr / --diarizer" in capsys.readouterr().out
