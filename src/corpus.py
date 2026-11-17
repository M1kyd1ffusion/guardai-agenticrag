"""
语料读取 + 切片（chunking）。
SQ1 的第一个旋钮在这里：strategy / sizes / overlap_tokens 全部来自 config.yaml。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

from .common import estimate_tokens, list_files


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    order: int
    text: str
    n_tokens: int
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def load_documents(corpus_dir: str | Path) -> List[Dict[str, str]]:
    """读取 txt/md 教材。返回 [{doc_id, text}]"""
    docs = []
    for fp in list_files(corpus_dir):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"\r\n", "\n", text)
        docs.append({"doc_id": fp.stem, "text": text})
    return docs


# ------------------------------------------------------------------ 切分
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _fixed_tokens_chunks(text: str, size: int, overlap: int, cpt: float) -> List[str]:
    """按近似 token 数滑窗切分（字符数/chars_per_token 换算）。"""
    chars = max(32, int(size * cpt))
    step = max(1, chars - int(overlap * cpt))
    out, i = [], 0
    while i < len(text):
        piece = text[i: i + chars]
        if piece.strip():
            out.append(piece)
        i += step
    return out


def _sentence_chunks(text: str, size: int, cpt: float) -> List[str]:
    """句子感知切分：在不超过 size 的前提下尽量装满完整句子（#30：性价比最高）。
    保证：(a) 不切断句子；(b) 每段长度≈size，便于与其它策略在同一预算下比较。"""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    budget = size * cpt   # token 上限换算成字符上限
    out, cur = [], ""
    for s in sents:
        cand = (cur + " " + s).strip()
        if (len(cand) > budget) and cur:
            out.append(cur)
            # 允许超长单句独占一段
            cur = s
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def build_chunks(cfg: Dict[str, Any], docs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    ch_cfg = cfg["chunking"]
    emb_cfg = cfg["embedding"]
    cpt = cfg["retrieval"]["budget"]["estimate_chars_per_token"]
    strategy = ch_cfg["strategy"]
    sizes = ch_cfg["sizes"]
    overlap = int(ch_cfg.get("overlap_tokens", 0))
    min_tok = int(ch_cfg.get("min_chunk_tokens", 30))

    chunks: List[Dict[str, Any]] = []
    for size in sizes:
        for doc in docs:
            pieces = (
                _sentence_chunks(doc["text"], size, cpt)
                if strategy == "sentence"
                else _fixed_tokens_chunks(doc["text"], size, overlap, cpt)
            )
            for i, p in enumerate(pieces):
                n_tok = estimate_tokens(p, cpt)
                if n_tok < min_tok:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc['doc_id']}::{size}::{i}",
                        doc_id=doc["doc_id"],
                        order=i,
                        text=p,
                        n_tokens=n_tok,
                        meta={"chunk_size": size, "strategy": strategy},
                    ).to_dict()
                )
    return chunks
