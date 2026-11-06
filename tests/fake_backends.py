from pathlib import Path
from backends.base import ASRResult, Segment, SpeakerTurn, Word

class FakeASR:
    name = "fake-asr"
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.calls: list[dict] = []
    def transcribe(self, wav16k: Path, *, context: str | None = None) -> ASRResult:
        self.calls.append({"wav": wav16k, "context": context})
        words = [Word("你好", 0.0, 0.5), Word("世界", 0.6, 1.0), Word("再见", 3.0, 3.4)]
        seg = Segment("你好世界再见", 0.0, 3.4, words)
        return ASRResult([seg], "Chinese", False, "eos", "fake-asr/1.0")

class FakeDiarizer:
    name = "fake-diar"
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
    def diarize(self, wav16k: Path, *, num_speakers=None, min_speakers=None, max_speakers=None):
        return [SpeakerTurn(0, 0.0, 1.2), SpeakerTurn(1, 2.8, 3.5)]


# ---- Mac 本地 nemotron 路由测试用（tests/test_local_nemotron.py）----
class FakeNemotron:
    name = "nemotron"
    label = "nemotron@test (mps)"
    capped_next = False
    raise_on_diarize = False
    raise_on_init = False
    calls = 0

    def __init__(self, cfg=None):
        if FakeNemotron.raise_on_init:
            raise RuntimeError("Nemotron 独立 venv 不存在")
        self.capped = False

    def diarize(self, wav16k, *, num_speakers=None, min_speakers=None, max_speakers=None):
        FakeNemotron.calls += 1
        if FakeNemotron.raise_on_diarize:
            raise RuntimeError("MPS 出错")
        self.capped = FakeNemotron.capped_next
        # 说话人 7 只发声 0.05 秒（< 全部发声 1.95 秒的 5%）：残标签，应被清理
        return [SpeakerTurn(0, 0.0, 1.2), SpeakerTurn(7, 1.5, 1.55), SpeakerTurn(1, 2.8, 3.5)]


class FakeSherpa:
    name = "sherpa"
    label = "sherpa / fake"
    calls = 0

    def __init__(self, cfg=None):
        pass

    def diarize(self, wav16k, *, num_speakers=None, min_speakers=None, max_speakers=None):
        FakeSherpa.calls += 1
        return [SpeakerTurn(0, 0.0, 1.2), SpeakerTurn(1, 2.8, 3.5)]
