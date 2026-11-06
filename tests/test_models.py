import io, tarfile
from pathlib import Path
from models import ensure_sherpa_models

def _tar_with_model(dst: Path):
    with tarfile.open(dst, "w:bz2") as t:
        data = b"onnx"; info = tarfile.TarInfo("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"); info.size = len(data)
        t.addfile(info, io.BytesIO(data))

def test_ensure_downloads_once(tmp_path):
    seg = tmp_path / "seg.tar.bz2"; _tar_with_model(seg)
    emb = tmp_path / "emb.onnx"; emb.write_bytes(b"emb")
    cache = tmp_path / "cache"
    p1 = ensure_sherpa_models(seg.as_uri(), emb.as_uri(), cache)
    assert p1[0].name == "model.onnx" and p1[0].read_bytes() == b"onnx" and p1[1].read_bytes() == b"emb"
    seg.unlink(); emb.unlink()                       # 源消失也不再下载
    assert ensure_sherpa_models(seg.as_uri(), emb.as_uri(), cache) == p1
