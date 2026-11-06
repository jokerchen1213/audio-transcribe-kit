import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shutil, subprocess
import pytest

@pytest.fixture
def synth_wav(tmp_path: Path) -> Path:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg 缺失")
    out = tmp_path / "synth.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
                    "-ac", "2", "-ar", "44100", str(out)], check=True)
    return out

# ---- 慢测试门控 + 公开样本 fixture（sherpa 官方 4 人中文样本，不用任何真实录音）----
import os, urllib.request
FIXTURE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav"
FIXTURE_DIR = ROOT / "tests" / "fixtures"

def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="RUN_SLOW=1 才跑真实后端")
    for it in items:
        if "slow" in it.keywords:
            it.add_marker(skip)

@pytest.fixture(scope="session")
def four_speakers_wav() -> Path:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    p = FIXTURE_DIR / "0-four-speakers-zh.wav"
    if not p.exists():   # 先下到 .part 再原子改名，避免中断留下被 exists() 永远接受的残缺 wav
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(FIXTURE_URL, tmp)
        os.replace(tmp, p)
    return p
