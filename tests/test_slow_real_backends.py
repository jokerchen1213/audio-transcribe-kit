import resource, time
from pathlib import Path
import pytest
from backends import get_asr
from config import load_config, resolve

@pytest.mark.slow
def test_qwen3_mlx_transcribes_chinese_with_word_timestamps(four_speakers_wav: Path):
    cfg = load_config(); name, acfg, _, _ = resolve(cfg, "qwen3-mlx", "none")
    asr = get_asr(name, acfg)
    t0 = time.time(); res = asr.transcribe(four_speakers_wav, context="测试"); dt = time.time() - t0
    text = "".join(s.text for s in res.segments)
    assert any("一" <= c <= "鿿" for c in text) and len(text) > 30
    words = [w for s in res.segments for w in s.words]
    assert words and all(w.end >= w.start for w in words) and words[-1].end <= 57.5
    assert any(w.text and w.text[-1] in "，。？！" for w in words)   # 标点已从整段文本回填到词级条目
    assert res.model_label.startswith("qwen3-mlx / ")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(f"\n[bench] qwen3-mlx 56.9s audio: {dt:.1f}s wall, {rss:.0f} MB maxrss")

from backends import get_diarizer

@pytest.mark.slow
def test_sherpa_finds_four_speakers(four_speakers_wav: Path):
    cfg = load_config(); _, _, dname, dcfg = resolve(cfg, None, "sherpa")
    d = get_diarizer(dname, dcfg)
    t0 = time.time(); turns = d.diarize(four_speakers_wav, num_speakers=4); dt = time.time() - t0
    assert {t.speaker for t in turns} == {0, 1, 2, 3}
    assert all(t.end > t.start for t in turns) and turns == sorted(turns, key=lambda t: t.start)
    print(f"\n[bench] sherpa 56.9s audio: {dt:.1f}s wall, {len(turns)} turns")

import cli
from index import md_target

@pytest.mark.slow
def test_real_end_to_end(four_speakers_wav: Path, tmp_path: Path):
    src = tmp_path / "四人样本.wav"; src.write_bytes(four_speakers_wav.read_bytes())
    t0 = time.time(); assert cli.main([str(src), "--speakers", "4", "--json"]) == 0; dt = time.time() - t0
    text = md_target(src).read_text(encoding="utf-8")
    assert "speakers: 4" in text and "quality: ok" in text and "**说话人4**" in text
    print(f"\n[bench] end-to-end 56.9s audio: {dt:.1f}s wall")
