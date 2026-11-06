from pathlib import Path
import subprocess
import soundfile as sf
import pytest
import audio
from audio import AUDIO_EXTS, AudioError, probe_duration, split_wav, to_wav16k

def test_probe_duration(synth_wav):
    assert abs(probe_duration(synth_wav) - 5.0) < 0.1

def test_to_wav16k_mono(synth_wav, tmp_path):
    out = to_wav16k(synth_wav, tmp_path / "work")
    info = sf.info(str(out))
    assert info.samplerate == 16000 and info.channels == 1 and out.suffix == ".wav"

def test_split_wav_offsets(synth_wav, tmp_path):
    wav = to_wav16k(synth_wav, tmp_path / "w")
    parts = split_wav(wav, 2.0, tmp_path / "parts")
    assert [round(o) for _, o in parts] == [0, 2, 4]
    assert abs(probe_duration(parts[0][0]) - 2.0) < 0.1

def test_bad_file_raises(tmp_path):
    bad = tmp_path / "x.m4a"; bad.write_bytes(b"nope")
    with pytest.raises(AudioError):
        to_wav16k(bad, tmp_path / "w")

def test_exts():
    assert {".m4a", ".mp3", ".mp4", ".wav", ".amr", ".wma", ".aac", ".flac", ".ogg", ".opus"} <= AUDIO_EXTS

def test_probe_duration_invalid_output_raises_audioerror(tmp_path, monkeypatch):
    # ffprobe 对部分容器会 0 退出但输出 "N/A"：float() 会抛 ValueError，须转成 AudioError 而非逃逸。
    def fake_run(cmd, capture_output=True, text=True):
        return subprocess.CompletedProcess(cmd, 0, stdout="N/A\n", stderr="")
    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    with pytest.raises(AudioError):
        probe_duration(tmp_path / "whatever.wav")

def test_split_wav_nonpositive_chunk_raises(tmp_path):
    with pytest.raises(AudioError):
        split_wav(tmp_path / "whatever.wav", 0, tmp_path / "parts")
    with pytest.raises(AudioError):
        split_wav(tmp_path / "whatever.wav", -1.0, tmp_path / "parts")

def test_to_wav16k_cleans_partial_output_on_failure(tmp_path, monkeypatch):
    # monkeypatch subprocess.run：模拟 ffmpeg 头部已写出半截文件后以非零码失败，
    # 验证 to_wav16k 删除残留输出而不是留下半成品。
    def fake_run(cmd, capture_output=True, text=True):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"partial")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fake ffmpeg failure")
    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    src = tmp_path / "src.wav"; src.write_bytes(b"whatever")
    work = tmp_path / "work"
    with pytest.raises(AudioError):
        to_wav16k(src, work)
    assert not (work / "src.16k.wav").exists()

def test_split_wav_cleans_partial_outputs_on_failure(synth_wav, tmp_path, monkeypatch):
    # 真实转出一个 16k wav，第一片正常切出、第二片 monkeypatch 成失败，
    # 验证中途失败时已写的片段（含刚失败的半截文件）被清理，out_dir 不留残留。
    wav = to_wav16k(synth_wav, tmp_path / "w")
    real_run = subprocess.run

    def fake_run(cmd, capture_output=True, text=True):
        if "part001" in cmd[-1]:
            out_path = Path(cmd[-1])
            out_path.write_bytes(b"partial")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fake ffmpeg failure")
        return real_run(cmd, capture_output=capture_output, text=text)

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    parts_dir = tmp_path / "parts"
    with pytest.raises(AudioError):
        split_wav(wav, 2.0, parts_dir)
    assert list(parts_dir.iterdir()) == []
