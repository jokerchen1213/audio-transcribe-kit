"""remote.py：ssh/scp 通道的编排逻辑（subprocess 全部 mock，不连任何主机）。"""
import json
import re
import subprocess
from pathlib import Path

import pytest

import remote


@pytest.fixture
def cfg() -> dict:
    return {"host": "win", "jobs_dir_win": r"D:\asr-jobs",
            "engines_dir_wsl": "/mnt/c/cc/cc-transcribe/remote/win",
            "min_free_vram_mb": 12000, "second_opinion": True, "timeout_s_per_audio_hour": 3600}


def _preflight_text(**kw) -> str:
    lines = {"user": r"WIN-HOST\user", "explorer": "1", "free_mb": " 21067 ",
             "glm": "Ready", "runjob": "1"}
    lines.update({k: str(v) for k, v in kw.items()})
    return "\n".join(f"{k}={v}" for k, v in lines.items())


# ---- 路径换算 --------------------------------------------------------------

def test_win_to_wsl():
    assert remote.win_to_wsl(r"D:\asr-jobs\20261022-101010-ab12") == "/mnt/d/asr-jobs/20261022-101010-ab12"
    assert remote.win_to_wsl("D:/asr-jobs/x/") == "/mnt/d/asr-jobs/x"
    assert remote.win_to_wsl(r"E:\a") == "/mnt/e/a"
    with pytest.raises(ValueError):
        remote.win_to_wsl("relative/path")


# ---- preflight -------------------------------------------------------------

def _patch_ps(monkeypatch, text_or_exc):
    calls = []

    def fake(script, host=remote.HOST_DEFAULT, timeout=120):
        calls.append({"script": script, "host": host, "timeout": timeout})
        if isinstance(text_or_exc, Exception):
            raise text_or_exc
        return text_or_exc

    monkeypatch.setattr(remote, "_ps", fake)
    return calls


def test_preflight_ok(monkeypatch, cfg):
    calls = _patch_ps(monkeypatch, _preflight_text())
    remote.preflight(cfg)                      # 不抛
    assert calls[0]["host"] == "win"
    assert calls[0]["timeout"] == 15           # 卡要求 15 秒内可连
    assert "run_job.sh" in calls[0]["script"] and "ocr-glm-server" in calls[0]["script"]


@pytest.mark.parametrize("field,value,needle", [
    ("user", "", "无人登录"),
    ("explorer", "0", "explorer.exe"),
    ("free_mb", "8000", "不足 12000 MB"),
    ("glm", "Running", "ocr-glm-server"),
    ("runjob", "0", "run_job.sh"),
])
def test_preflight_failures(monkeypatch, cfg, field, value, needle):
    _patch_ps(monkeypatch, _preflight_text(**{field: value}))
    with pytest.raises(remote.WindowsUnavailable) as e:
        remote.preflight(cfg)
    assert needle in str(e.value)


def test_preflight_ssh_unreachable(monkeypatch, cfg):
    _patch_ps(monkeypatch, remote.WindowsUnavailable("Windows 响应超时（15 秒）"))
    with pytest.raises(remote.WindowsUnavailable) as e:
        remote.preflight(cfg)
    assert "不在线或 ssh 不通" in str(e.value)


def test_preflight_requires_engines_dir(monkeypatch, cfg):
    cfg.pop("engines_dir_wsl")
    with pytest.raises(remote.WindowsUnavailable, match="engines_dir_wsl"):
        remote.preflight(cfg)


# ---- _ps / _scp 都带超时、PowerShell 走 -EncodedCommand ----------------------

def test_ps_uses_encoded_command_and_timeout(monkeypatch):
    seen = {}

    def fake_run(cmd, capture_output=True, timeout=None):
        seen["cmd"] = cmd
        seen["timeout"] = timeout
        return subprocess.CompletedProcess(cmd, 0, stdout="a=1\n<Objs x>\n#< CLIXML noise\n".encode(), stderr=b"")

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    out = remote._ps("echo hi", "win", timeout=42)
    assert seen["timeout"] == 42
    assert "-EncodedCommand" in seen["cmd"][-1] and "ConnectTimeout=" in " ".join(seen["cmd"])
    assert out == "a=1"                       # CLIXML 噪声被剔除


def test_ps_timeout_becomes_windows_unavailable(monkeypatch):
    def fake_run(cmd, capture_output=True, timeout=None):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    with pytest.raises(remote.WindowsUnavailable, match="超时"):
        remote._ps("echo hi", "win", timeout=5)


def test_scp_has_timeout(monkeypatch):
    seen = {}

    def fake_run(cmd, capture_output=True, timeout=None):
        seen["cmd"], seen["timeout"] = cmd, timeout
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    remote._scp("in.wav", "win:D:/asr-jobs/x/in.wav", timeout=600)
    assert seen["timeout"] == 600
    assert "ConnectTimeout=" in " ".join(seen["cmd"]) and "BatchMode=yes" in seen["cmd"]


# ---- run_remote 编排 -------------------------------------------------------

def _patch_pipeline(monkeypatch, cfg, results_rc=0, outputs=("out.qwen3.json", "out.nemotron.json", "out.moss.json"),
                    payload=None, run_exc=None, ps_text=""):
    seen = {"ps": [], "scp": [], "run": [], "list": []}

    def fake_ps(script, host=remote.HOST_DEFAULT, timeout=120):
        seen["ps"].append(script)
        if run_exc is not None and "New-Item" in script:
            raise run_exc
        return ps_text

    def fake_scp(source, target, timeout):
        seen["scp"].append((source, target))
        if source.split("/")[-1].startswith("out."):
            name = source.split("/")[-1]
            Path(target).write_text(json.dumps(payload or {"ok": True, "engine": name.split(".")[1]}),
                                    encoding="utf-8")

    def fake_run(host, engines_dir, job_wsl, engines, timeout, job_win):
        seen["run"].append({"host": host, "engines_dir": engines_dir, "job_wsl": job_wsl,
                            "engines": engines, "timeout": timeout, "job_win": job_win})
        return results_rc, "stdout", "stderr"

    def fake_list(host, job_win):
        seen["list"].append(job_win)
        return list(outputs)

    monkeypatch.setattr(remote, "_ps", fake_ps)
    monkeypatch.setattr(remote, "_scp", fake_scp)
    monkeypatch.setattr(remote, "_run_job", fake_run)
    monkeypatch.setattr(remote, "_list_outputs", fake_list)
    monkeypatch.setattr(remote, "probe_duration", lambda p: 3600.0)
    return seen


def test_run_remote_roundtrip(monkeypatch, cfg, tmp_path):
    seen = _patch_pipeline(monkeypatch, cfg)
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"RIFF")
    got = remote.run_remote(wav, {"language": "Chinese"}, ["qwen3", "nemotron", "moss"], cfg)
    assert set(got) == {"qwen3", "nemotron", "moss"}
    assert got["qwen3"]["ok"] is True
    job = seen["run"][0]
    assert job["job_wsl"].startswith("/mnt/d/asr-jobs/") and job["engines_dir"].endswith("/remote/win")
    assert job["engines"] == ["qwen3", "nemotron", "moss"]
    assert job["timeout"] == 3600                     # 1 h 音频 × 3600 s
    job_id = job["job_wsl"].rsplit("/", 1)[-1]
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}", job_id)
    # 建目录 → 上传 in.wav / job.json → 取回 → finally 删目录
    assert any("New-Item" in s for s in seen["ps"])
    uploads = [(s, t) for s, t in seen["scp"] if s.endswith("in.wav") or s.endswith("job.json")]
    assert len(uploads) == 2
    assert all(t.startswith("win:D:/asr-jobs/") for _, t in uploads)
    downloads = [(s, t) for s, t in seen["scp"] if s.startswith("win:") and "/out." in s]
    assert len(downloads) == 3
    assert all(not t.startswith("win:") for _, t in downloads)   # 取回到本地临时目录
    assert any("Remove-Item" in s and job_id in s for s in seen["ps"])


def test_run_remote_timeout_is_min_30_minutes(monkeypatch, cfg, tmp_path):
    seen = _patch_pipeline(monkeypatch, cfg)
    monkeypatch.setattr(remote, "probe_duration", lambda p: 60.0)
    wav = tmp_path / "in.wav"; wav.write_bytes(b"RIFF")
    remote.run_remote(wav, {}, ["qwen3"], cfg)
    assert seen["run"][0]["timeout"] == 1800


def test_run_remote_75_busy(monkeypatch, cfg, tmp_path):
    seen = _patch_pipeline(monkeypatch, cfg, results_rc=75, outputs=())
    wav = tmp_path / "in.wav"; wav.write_bytes(b"RIFF")
    with pytest.raises(remote.WindowsUnavailable, match="Windows 正在跑别的转写作业"):
        remote.run_remote(wav, {}, ["qwen3"], cfg)
    assert any("Remove-Item" in s for s in seen["ps"])      # 75 也清理


def test_run_remote_no_outputs(monkeypatch, cfg, tmp_path):
    seen = _patch_pipeline(monkeypatch, cfg, results_rc=1, outputs=())
    wav = tmp_path / "in.wav"; wav.write_bytes(b"RIFF")
    with pytest.raises(RuntimeError, match="没有产出 out"):
        remote.run_remote(wav, {}, ["qwen3"], cfg)
    assert any("Remove-Item" in s for s in seen["ps"])


def test_run_remote_failure_still_cleans_up(monkeypatch, cfg, tmp_path):
    seen = _patch_pipeline(monkeypatch, cfg)
    wav = tmp_path / "in.wav"; wav.write_bytes(b"RIFF")

    def boom(*a, **k):
        raise RuntimeError("scp 失败")

    monkeypatch.setattr(remote, "_scp", boom)
    with pytest.raises(RuntimeError, match="scp 失败"):
        remote.run_remote(wav, {}, ["qwen3"], cfg)
    assert any("Remove-Item" in s for s in seen["ps"])


# ---- 超时 / Ctrl-C 杀进程组 -------------------------------------------------

def test_run_job_timeout_kills_group(monkeypatch):
    killed = []
    monkeypatch.setattr(remote, "_kill_job", lambda host, job_win: killed.append(job_win))

    def fake_run(cmd, capture_output=True, timeout=None):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="超时"):
        remote._run_job("win", "/mnt/c/cc/remote/win", "/mnt/d/asr-jobs/x", ["qwen3"], 60, r"D:\asr-jobs\x")
    assert killed == [r"D:\asr-jobs\x"]


def test_run_job_keyboard_interrupt_kills_group(monkeypatch):
    killed = []
    monkeypatch.setattr(remote, "_kill_job", lambda host, job_win: killed.append(job_win))

    def fake_run(cmd, capture_output=True, timeout=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    with pytest.raises(KeyboardInterrupt):
        remote._run_job("win", "/mnt/c/cc/remote/win", "/mnt/d/asr-jobs/x", ["qwen3"], 60, r"D:\asr-jobs\x")
    assert killed == [r"D:\asr-jobs\x"]


def test_run_job_command_shape(monkeypatch):
    seen = {}

    def fake_run(cmd, capture_output=True, timeout=None):
        seen["cmd"], seen["timeout"] = cmd, timeout
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    rc, out, err = remote._run_job("win", "/mnt/c/cc/cc-transcribe/remote/win/", "/mnt/d/asr-jobs/abc",
                                   ["qwen3", "moss"], 120, r"D:\asr-jobs\abc")
    assert rc == 0 and seen["timeout"] == 120
    assert "wsl.exe -d Ubuntu -u root -- bash /mnt/c/cc/cc-transcribe/remote/win/run_job.sh" in seen["cmd"][-1]
    assert seen["cmd"][-1].endswith("/mnt/d/asr-jobs/abc qwen3 moss")


def test_kill_job_reads_pgid_and_kills(monkeypatch):
    seen = []

    def fake_ps(script, host=remote.HOST_DEFAULT, timeout=120):
        seen.append(script)
        return "pgid=12345\n" if "Get-Content" in script else ""

    monkeypatch.setattr(remote, "_ps", fake_ps)
    remote._kill_job("win", r"D:\asr-jobs\abc")
    assert any("kill -TERM -- -12345" in s for s in seen)
    assert any("kill -KILL -- -12345" in s for s in seen)


def test_kill_job_without_pgid_is_noop(monkeypatch):
    seen = []

    def fake_ps(script, host=remote.HOST_DEFAULT, timeout=120):
        seen.append(script)
        return "pgid="

    monkeypatch.setattr(remote, "_ps", fake_ps)
    remote._kill_job("win", r"D:\asr-jobs\abc")
    assert not any("kill -" in s for s in seen)


def test_kill_job_tolerates_ps_errors(monkeypatch):
    def fake_ps(script, host=remote.HOST_DEFAULT, timeout=120):
        raise remote.WindowsUnavailable("掉了")

    monkeypatch.setattr(remote, "_ps", fake_ps)
    remote._kill_job("win", r"D:\asr-jobs\abc")     # 不抛
