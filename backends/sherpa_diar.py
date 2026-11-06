"""sherpa-onnx 离线说话人分离：pyannote-segmentation-3.0（ONNX）+ 3D-Speaker 中文声纹 + 快速聚类。CPU。"""
from __future__ import annotations
import os
from pathlib import Path
import soundfile as sf
from backends.base import SpeakerTurn
from models import ensure_sherpa_models

def _renumber(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """sherpa 阈值聚类返回的 cluster id 有空洞（如 0,1,2,4,7）；按首次出现顺序重编为 0..n-1。turns 须已按 start 排序。"""
    ids: dict[int, int] = {}
    return [SpeakerTurn(ids.setdefault(t.speaker, len(ids)), t.start, t.end) for t in turns]

class SherpaDiarizer:
    name = "sherpa"
    label = "sherpa / pyannote-seg-3.0 + eres2net-zh"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.threshold = float(cfg.get("threshold", 0.5))
        self.min_on = float(cfg.get("min_duration_on", 0.3))
        self.min_off = float(cfg.get("min_duration_off", 0.5))
        self.threads = int(cfg.get("num_threads", max(1, (os.cpu_count() or 2) // 2)))
        self._seg, self._emb = ensure_sherpa_models(cfg["segmentation_url"], cfg["embedding_url"],
                                                    Path(cfg["cache_dir"]) if cfg.get("cache_dir") else None)

    def _build(self, num_speakers: int | None):
        import sherpa_onnx as so
        config = so.OfflineSpeakerDiarizationConfig(
            segmentation=so.OfflineSpeakerSegmentationModelConfig(
                pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(self._seg)),
                num_threads=self.threads),
            embedding=so.SpeakerEmbeddingExtractorConfig(model=str(self._emb), num_threads=self.threads),
            clustering=so.FastClusteringConfig(num_clusters=num_speakers if num_speakers else -1,
                                               threshold=self.threshold),
            min_duration_on=self.min_on, min_duration_off=self.min_off)
        if not config.validate():
            raise RuntimeError("sherpa 分离配置无效：模型文件缺失")
        return so.OfflineSpeakerDiarization(config)

    def diarize(self, wav16k: Path, *, num_speakers=None, min_speakers=None, max_speakers=None) -> list[SpeakerTurn]:
        # sherpa 只支持"指定人数"或"阈值自动"；min/max 仅用于事后校验并提示
        sd = self._build(num_speakers)
        audio, sr = sf.read(str(wav16k), dtype="float32", always_2d=True)
        audio = audio[:, 0]
        if sr != sd.sample_rate:
            raise RuntimeError(f"分离器要求 {sd.sample_rate} Hz，输入为 {sr} Hz（请先经 audio.to_wav16k）")
        result = sd.process(audio).sort_by_start_time()
        turns = _renumber([SpeakerTurn(int(r.speaker), float(r.start), float(r.end)) for r in result])
        n = len({t.speaker for t in turns})
        if (min_speakers and n < min_speakers) or (max_speakers and n > max_speakers):
            print(f"⚠️ 自动分离得到 {n} 人，超出提示区间 [{min_speakers}, {max_speakers}]；可用 --speakers 指定人数或调 threshold")
        return turns
