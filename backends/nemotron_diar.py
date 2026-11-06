"""Mac 本地 Nemotron-3-Diarization 说话人分离（Apple MPS）。

模型与 torch / transformers 装在独立 venv（见 remote/mac/requirements-nemotron.txt），主 .venv 不装 torch；
本后端以子进程调用 remote/mac/diar_nemotron_mps.py，只交换 wav 路径与 JSON。
模型只能分到 8 人：检出满 8 人时 `capped` 为 True，由 cli 按规则退回 sherpa（`--speakers` > 8 同理）。
不接受人数参数（模型无此输入）；传入的人数只用于 cli 的路由与提醒。
"""
from __future__ import annotations
import json, subprocess, tempfile
from pathlib import Path
from backends.base import SpeakerTurn

SCRIPT = Path(__file__).resolve().parent.parent / "remote" / "mac" / "diar_nemotron_mps.py"
DEFAULT_PYTHON = Path.home() / ".venvs" / "cct-nemotron" / "bin" / "python"
DEFAULT_MODEL = Path.home() / ".cache" / "cc-transcribe" / "nemotron" / "nemotron-diar@f667ed7"


class NemotronDiarizer:
    name = "nemotron"

    def __init__(self, cfg: dict):
        self.python = Path(cfg.get("python", DEFAULT_PYTHON)).expanduser()
        self.model_dir = Path(cfg.get("model_dir", DEFAULT_MODEL)).expanduser()
        self.device = cfg.get("device", "mps")
        self.timeout_s = int(cfg.get("timeout_s", 1800))
        if not self.python.exists():
            raise RuntimeError(f"Nemotron 独立 venv 不存在：{self.python}（安装见 remote/mac/requirements-nemotron.txt）")
        if not (self.model_dir / "model.safetensors").exists():
            raise RuntimeError(f"Nemotron 模型不在：{self.model_dir}（获取方式见 README「Windows 路线」节）")
        self.label = f"nemotron@{self.model_dir.name.split('@')[-1]} ({self.device})"
        self.capped = False

    def diarize(self, wav16k: Path, *, num_speakers=None, min_speakers=None, max_speakers=None) -> list[SpeakerTurn]:
        with tempfile.TemporaryDirectory(prefix="cct-nemo-") as tmp:
            out = Path(tmp) / "out.json"
            proc = subprocess.run([str(self.python), str(SCRIPT), str(wav16k), str(out), str(self.model_dir), self.device],
                                  capture_output=True, text=True, timeout=self.timeout_s)
            if not out.exists():
                raise RuntimeError(f"Nemotron 子进程未产出结果（退出码 {proc.returncode}）：{proc.stderr[-500:]}")
            res = json.loads(out.read_text(encoding="utf-8"))
        if not res.get("ok"):
            raise RuntimeError(f"Nemotron 失败：{res.get('error')}")
        self.capped = bool(res.get("capped"))
        self.label = f"nemotron@{self.model_dir.name.split('@')[-1]} ({res.get('device', self.device)})"
        return [SpeakerTurn(int(t["speaker"]), float(t["start"]), float(t["end"])) for t in res["turns"]]
