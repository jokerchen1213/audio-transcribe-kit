"""Qwen3-ASR-1.7B（8-bit）on MLX，经 mlx-qwen3-asr。词级时间戳来自其原生 MLX 强制对齐器。"""
from __future__ import annotations
import logging, unicodedata
from pathlib import Path
from backends.base import ASRResult, Segment, Word

def _is_kept(ch: str) -> bool:
    """与对齐器 ForcedAlignTextProcessor.is_kept_char 同口径：字母 / 数字 / 撇号才进 token。"""
    return ch == "'" or unicodedata.category(ch)[0] in "LN"

def _reattach_punct(text: str, words: list[Word]) -> list[Word]:
    """把对齐器剥掉的标点从整段 text 回填到词流。对齐器 token = text 的"保留字符"（字母 / 数字 / 撇号）
    按序拼接（非 CJK 串先剥非保留字符再整体成一个 token，如 '3.5%'→'35'），故用指针顺序走 text：逐词消费其
    每个字符，途中跳过的非保留字符按位置附回——词内的（'3.5'）留在该词，词间 / 末尾的附到前一个词，首词之前
    的前置到首词；空白丢弃（join_words 会重插英文空格）；时间戳不变。保留字符对不上即整体放弃、原样返回
    words 并 warning，绝不抛错。"""
    if not words:
        return []
    log = logging.getLogger(__name__)
    n = len(text)
    pos = 0
    lead = ""
    texts: list[str] = []
    for i, w in enumerate(words):
        buf = ""
        for ch in w.text:
            while pos < n and not _is_kept(text[pos]):
                c = text[pos]
                if not c.isspace():
                    if buf:
                        buf += c            # 词内标点（3.5 / 10,000 / state-of-the-art）
                    elif i == 0:
                        lead += c           # 首词之前
                    else:
                        texts[i - 1] += c   # 词间标点归前一个词
                pos += 1
            if pos >= n or text[pos] != ch:
                log.warning("标点回填放弃：第 %d 个词 %r 的字符 %r 与文本位置 %d 处 %r 不匹配", i, w.text, ch, pos,
                            text[pos] if pos < n else "<EOF>")
                return words
            buf += ch
            pos += 1
        texts.append(buf)
    tail = "".join(c for c in text[pos:] if not c.isspace())
    if any(_is_kept(c) for c in tail):
        log.warning("标点回填放弃：词流耗尽后文本仍有未对齐的保留字符 %r", tail[:20])
        return words
    texts[-1] += tail
    texts[0] = lead + texts[0]
    return [Word(t, w.start, w.end) for t, w in zip(texts, words)]

DOWNLOAD_HINT = ("模型下载失败。免账号退路：① export HF_ENDPOINT=https://hf-mirror.com 后重试；"
                 "② pip install modelscope；modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir <目录1> 拉原权重，"
                 "git clone https://github.com/moona3k/mlx-qwen3-asr（scripts/convert.py 不随 pip 包分发）后用其 "
                 "scripts/convert.py --quantize 8 --encoder-bits 8 转换（参数以上游 README 为准）；"
                 "另 modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir <目录2>（对齐器为官方原始权重，无需转换）；"
                 "把 backends.toml [asr.qwen3-mlx] 的 model 指向 <目录1>、aligner 指向 <目录2>。")

class Qwen3MLX:
    name = "qwen3-mlx"

    def __init__(self, cfg: dict):
        self.model_id = cfg.get("model", "moona3k/mlx-qwen3-asr-1.7b-8bit")
        self.aligner = cfg.get("aligner", "Qwen/Qwen3-ForcedAligner-0.6B")
        self.language = cfg.get("language", "Chinese")
        self._session = None

    def _load(self):
        if self._session is None:
            from mlx_qwen3_asr import Session
            try:
                self._session = Session(model=self.model_id)
            except Exception as e:  # 网络 / 权重缺失
                raise RuntimeError(f"{DOWNLOAD_HINT}\n原始错误：{e}") from e
        return self._session

    def transcribe(self, wav16k: Path, *, context: str | None = None) -> ASRResult:
        sess = self._load()
        kw = dict(language=self.language, return_timestamps=True, return_chunks=True,
                  forced_aligner=self.aligner)
        if context:
            kw["context"] = context
        try:
            r = sess.transcribe(str(wav16k), **kw)
        except Exception as e:
            if "download" in str(e).lower() or "connect" in str(e).lower():
                raise RuntimeError(f"{DOWNLOAD_HINT}\n原始错误：{e}") from e
            raise
        # r.segments 是对齐器输出的词级条目（中文按单字、标点已剥离）；r.text 是带标点的整段文本
        words = [Word(s["text"], float(s["start"]), float(s["end"])) for s in (r.segments or [])]
        if words:
            words = _reattach_punct(r.text, words)   # 词级条目无标点，从整段文本回填
            segs = [Segment(r.text, words[0].start, words[-1].end, words)]
        else:  # 对齐器不可用时退化为整段（无词级时间戳）；TranscriptionResult 无 duration，取音频实际时长
            import soundfile as sf
            segs = [Segment(r.text, 0.0, float(sf.info(str(wav16k)).duration), [])]
        return ASRResult(segs, getattr(r, "language", None), bool(getattr(r, "truncated", False)),
                         getattr(r, "finish_reason", None), f"qwen3-mlx / {self.model_id}")
