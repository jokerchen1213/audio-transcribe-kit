"""_renumber 纯逻辑测试：不联网、不加载模型。"""
from backends.base import SpeakerTurn
from backends.sherpa_diar import _renumber

def _turns(ids):
    return [SpeakerTurn(s, float(i), float(i) + 0.5) for i, s in enumerate(ids)]

def test_renumber_gapped_ids_become_contiguous():
    out = _renumber(_turns([0, 1, 2, 4, 7, 8, 9]))
    assert [t.speaker for t in out] == [0, 1, 2, 3, 4, 5, 6]
    assert [(t.start, t.end) for t in out] == [(float(i), float(i) + 0.5) for i in range(7)]

def test_renumber_reappearing_speaker_keeps_number():
    assert [t.speaker for t in _renumber(_turns([4, 7, 4, 9]))] == [0, 1, 0, 2]

def test_renumber_empty():
    assert _renumber([]) == []
