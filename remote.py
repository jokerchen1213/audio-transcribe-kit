"""remote.py — Windows 3090 远程转写通道（Mac 侧）。

方案 A（律师 2026-10-22 裁定）：Qwen3（`-hf`）出文字 + Nemotron 分说话人；Nemotron
满 8 人或用户指定人数 > 8 时改用 DiariZen；MOSS 只作第二意见。Mac 只做 `to_wav16k`，
把 16 kHz 单声道 wav 与 `job.json` 经 scp 送到 Windows 的 `D:\\asr-jobs\\<作业号>\\`
（Drive 外），由 Windows 上的 WSL 跑 `remote/win/run_job.sh`（一次作业跑完所有引擎），
再把 `out.*.json` 取回。

约定：
- Windows 侧单作业串行由 `run_job.sh` 的 `flock -n /var/tmp/asr-jobs/.lock` 保证；
  拿不到锁时脚本退出码 75，本模块翻成 `WindowsUnavailable("Windows 正在跑别的转写作业")`。
- 超时（`timeout_s_per_audio_hour` × 音频小时数，最少 30 分钟）或 Ctrl-C：先按作业目录
  里的 `pgid` 结束 WSL 进程组，再清理 Windows 作业目录。
- 作业目录无论成败都在 `finally` 里删除（Drive 外，不留垃圾）。
- 所有 ssh / scp 调用都带超时；PowerShell 一律走 `-EncodedCommand`，输出 UTF-8。

`_ps()` 的 ssh + `-EncodedCommand` + UTF-8 写法照搬 cc-convert 的
`second_opinion.py`（同一台机器、同一套 ssh 约定，只复用写法，不跨仓库 import）。
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from audio import probe_duration

HOST_DEFAULT = "win"
JOBS_DIR_WIN_DEFAULT = r"D:\asr-jobs"
MIN_FREE_VRAM_MB_DEFAULT = 12000
TIMEOUT_PER_HOUR_DEFAULT = 3600
MIN_TIMEOUT_S = 1800          # 最少 30 分钟
GLM_TASK = "ocr-glm-server"
SSH_CONNECT_TIMEOUT = 8
PREFLIGHT_TIMEOUT = 15        # 卡要求 15 秒内可连


class WindowsUnavailable(Exception):
    """Windows 不可用（不在线 / 无人登录 / 显存不足 / 正在跑 OCR / 引擎未部署 / 正在跑别的作业）。"""


# ---- ssh / scp 基础通道 ----------------------------------------------------

def _ps(script: str, host: str = HOST_DEFAULT, timeout: int = 120) -> str:
    """在 Windows 上跑一段 PowerShell（-EncodedCommand，UTF-8 输出），返回 stdout 文本。"""
    full = ('[Console]::OutputEncoding=[Text.Encoding]::UTF8; $ProgressPreference="SilentlyContinue"\n'
            + script)
    b64 = base64.b64encode(full.encode("utf-16-le")).decode()
    cmd = ["ssh", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}", "-o", "BatchMode=yes", host,
           f"powershell -NoProfile -EncodedCommand {b64}"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise WindowsUnavailable(f"Windows 响应超时（{timeout} 秒）") from e
    except OSError as e:
        raise WindowsUnavailable(f"ssh 无法执行：{e}") from e
    out = r.stdout.decode("utf-8", "replace").replace("\x00", "")
    err = r.stderr.decode("utf-8", "replace").replace("\x00", "")
    if r.returncode != 0:
        raise WindowsUnavailable(f"Windows 命令失败（rc={r.returncode}）：{(err or out).strip()[:300]}")
    # 剔除 PowerShell 远程 shell 的 CLIXML 噪声（cc-convert 同法）
    return "\n".join(l for l in out.splitlines() if l.strip() and not l.startswith(("#< CLIXML", "<Objs")))


def _kv(text: str) -> dict[str, str]:
    return dict(l.split("=", 1) for l in text.splitlines() if "=" in l)


def _values(text: str, key: str) -> list[str]:
    return [l.split("=", 1)[1] for l in text.splitlines() if l.startswith(key + "=")]


def _scp(source: str, target: str, timeout: int) -> None:
    """一次 scp（source / target 已写成 host:path 或本地路径），带超时。"""
    cmd = ["scp", "-q", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}", "-o", "BatchMode=yes",
           source, target]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"scp 超时（{timeout} 秒）：{source} → {target}") from e
    except OSError as e:
        raise RuntimeError(f"scp 无法执行：{e}") from e
    if r.returncode != 0:
        raise RuntimeError(f"scp 失败（rc={r.returncode}）：{r.stderr.decode('utf-8', 'replace').strip()[:200]}")


def win_to_wsl(path: str) -> str:
    """`D:\\asr-jobs\\x` → `/mnt/d/asr-jobs/x`（只处理盘符绝对路径）。"""
    m = re.match(r"^([A-Za-z]):[\\/]*(.*)$", str(path))
    if not m:
        raise ValueError(f"不是 Windows 盘符绝对路径：{path}")
    rest = m.group(2).replace("\\", "/").strip("/")
    return "/mnt/" + m.group(1).lower() + ("/" + rest if rest else "")


def _remote(host: str, win_path: str) -> str:
    """scp 远端写法：host:D:/a/b（正斜杠，OpenSSH for Windows 两种协议都认）。"""
    return host + ":" + win_path.replace("\\", "/")


# ---- 预检 ------------------------------------------------------------------

def _preflight_script(engines_dir_wsl: str) -> str:
    run_job = engines_dir_wsl.rstrip("/") + "/run_job.sh"
    return ("'user=' + ((Get-CimInstance Win32_ComputerSystem).UserName)\n"
            "'explorer=' + @(Get-Process explorer -ErrorAction SilentlyContinue).Count\n"
            "'free_mb=' + (nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)\n"
            f"'glm=' + (Get-ScheduledTask -TaskName '{GLM_TASK}' -ErrorAction SilentlyContinue).State\n"
            f"'runjob=' + $(wsl.exe -d Ubuntu -u root -- test -f {run_job}; "
            "if ($LASTEXITCODE -eq 0) { 1 } else { 0 })\n")


def preflight(cfg: dict) -> None:
    """派活前检查 Windows 是否可用；不满足抛 WindowsUnavailable（一句中文原因）。

    检查项（按卡）：ssh 15 秒内可连 → 有人登录（explorer.exe 在跑）→ 显卡空闲显存
    ≥ min_free_vram_mb → 计划任务 ocr-glm-server 不在 Running → WSL 里 run_job.sh 存在。
    """
    host = str(cfg.get("host", HOST_DEFAULT))
    min_free = int(cfg.get("min_free_vram_mb", MIN_FREE_VRAM_MB_DEFAULT))
    engines_dir = str(cfg.get("engines_dir_wsl", "")).rstrip("/")
    if not engines_dir:
        raise WindowsUnavailable("runner.win.engines_dir_wsl 未配置")
    try:
        info = _kv(_ps(_preflight_script(engines_dir), host, timeout=PREFLIGHT_TIMEOUT))
    except WindowsUnavailable as e:
        raise WindowsUnavailable(f"Windows 不在线或 ssh 不通（{e}）") from e

    if not info.get("user", "").strip():
        raise WindowsUnavailable("Windows 无人登录（重启后未登录桌面，需人工登录）")
    if info.get("explorer", "0").strip() in ("", "0"):
        raise WindowsUnavailable("Windows 没有人登录桌面（explorer.exe 未运行）")
    free = int(re.sub(r"\D", "", info.get("free_mb", "0")) or 0)
    if free < min_free:
        raise WindowsUnavailable(f"显卡空闲显存 {free} MB 不足 {min_free} MB（显卡被占用）")
    if info.get("glm", "").strip().lower() == "running":
        raise WindowsUnavailable(f"Windows 正在跑计划任务 {GLM_TASK}（显存被 OCR 服务占用）")
    if info.get("runjob", "").strip() != "1":
        raise WindowsUnavailable(f"WSL 里没有 {engines_dir}/run_job.sh（Windows 引擎未部署）")


# ---- 作业执行 --------------------------------------------------------------

def _run_job(host: str, engines_dir_wsl: str, job_wsl: str, engines: list[str], timeout: int,
             job_win: str) -> tuple[int, str, str]:
    """ssh 到 Windows 跑 run_job.sh。超时 / Ctrl-C 先按 pgid 杀 WSL 进程组再抛。"""
    remote_cmd = (f"wsl.exe -d Ubuntu -u root -- bash {engines_dir_wsl.rstrip('/')}/run_job.sh "
                  f"{job_wsl} " + " ".join(engines))
    cmd = ["ssh", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}", "-o", "BatchMode=yes",
           "-o", "ServerAliveInterval=30", host, remote_cmd]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_job(host, job_win)
        raise RuntimeError(f"Windows 转写超时（{timeout} 秒），已结束远端进程组")
    except KeyboardInterrupt:
        _kill_job(host, job_win)
        raise
    return (r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace"))


def _kill_job(host: str, job_win: str) -> None:
    """按作业目录里的 `pgid` 结束 WSL 进程组（先 TERM，1 秒后 KILL）。尽力而为，绝不抛。"""
    pgid = ""
    try:
        info = _kv(_ps(f"'pgid=' + (Get-Content -LiteralPath '{job_win}\\pgid' "
                       "-ErrorAction SilentlyContinue)", host, timeout=30))
        pgid = re.sub(r"\D", "", info.get("pgid", "") or "")
    except Exception:
        return
    if not pgid:
        return
    for sig in ("TERM", "KILL"):
        try:
            _ps(f"wsl.exe -d Ubuntu -u root -- kill -{sig} -- -{pgid}", host, timeout=30)
        except Exception:
            pass
        time.sleep(1)


def _list_outputs(host: str, job_win: str) -> list[str]:
    out = _ps(f"Get-ChildItem -LiteralPath '{job_win}' -Filter 'out.*.json' | "
              "ForEach-Object { 'file=' + $_.Name }", host, timeout=30)
    return [n for n in _values(out, "file") if n]


def run_remote(wav16k: Path, job: dict, engines: list[str], cfg: dict) -> dict[str, dict]:
    """把一次转写作业送到 Windows 跑完，返回 {引擎名: out.<引擎>.json 解析结果}。

    退出码 75 → WindowsUnavailable；没有产出 → RuntimeError；超时 / Ctrl-C → 杀进程组后抛。
    """
    host = str(cfg.get("host", HOST_DEFAULT))
    jobs_dir = str(cfg.get("jobs_dir_win", JOBS_DIR_WIN_DEFAULT)).rstrip("\\/")
    engines_dir = str(cfg.get("engines_dir_wsl", "")).rstrip("/")
    if not engines_dir:
        raise WindowsUnavailable("runner.win.engines_dir_wsl 未配置")
    per_hour = float(cfg.get("timeout_s_per_audio_hour", TIMEOUT_PER_HOUR_DEFAULT))
    duration = probe_duration(Path(wav16k))
    timeout = max(MIN_TIMEOUT_S, int(per_hour * duration / 3600.0))

    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    job_win = jobs_dir + "\\" + job_id
    job_wsl = win_to_wsl(job_win)
    tmp = Path(tempfile.mkdtemp(prefix="cc-transcribe-win-"))
    job_json = tmp / "job.json"
    job_json.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    try:
        _ps(f"New-Item -ItemType Directory -Force -Path '{job_win}' | Out-Null", host, timeout=30)
        _scp(str(wav16k), _remote(host, job_win + "\\in.wav"), timeout=600)
        _scp(str(job_json), _remote(host, job_win + "\\job.json"), timeout=120)

        rc, out, err = _run_job(host, engines_dir, job_wsl, engines, timeout, job_win)
        if rc == 75:
            raise WindowsUnavailable("Windows 正在跑别的转写作业")

        names = _list_outputs(host, job_win)
        if not names:
            tail = " / ".join(((err or out).strip().splitlines() or ["<无输出>"])[-4:])
            raise RuntimeError(f"Windows 作业没有产出 out.*.json（rc={rc}）：{tail}")
        got: dict[str, dict] = {}
        for name in names:
            local = tmp / name
            _scp(_remote(host, job_win + "\\" + name), str(local), timeout=120)
            got[name.split(".")[1]] = json.loads(local.read_text(encoding="utf-8"))
        return got
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        try:   # 作业目录无论成败都删；删不掉不影响本次结果
            _ps(f"Remove-Item -Recurse -Force -LiteralPath '{job_win}' -ErrorAction SilentlyContinue",
                host, timeout=30)
        except Exception:
            pass
