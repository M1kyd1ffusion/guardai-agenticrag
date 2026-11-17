"""
检索：dense / bm25 / hybrid(RRF) + 可选重排 + **fill-to-budget**（DD-06）。

关键区别（为什么不用 top-k）：
  不同切片大小下固定取 k 段，送入生成端的 token 量会差好几倍，比较不公平。
  fill-to-budget 按排序依次填充直到 token 预算用尽 → 所有配置在"相同上下文成本"下比。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .common import Timer, estimate_tokens, log
from .index import tokenize


def rrf_fuse(rank_lists: List[List[int]], k: int = 60, top_n: int = 100) -> List[int]:
    """Reciprocal Rank Fusion：只看名次不看分数，免做分数归一。"""
    scores: Dict[int, float] = {}
    for lst in rank_lists:
        for rank, idx in enumerate(lst):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])][:top_n]


def retrieve(store, embedder, query: str, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """返回 (segments, debug)。segments 已按预算裁剪。"""
    r_cfg = cfg["retrieval"]
    mode = r_cfg["mode"]
    cand_n = int(r_cfg.get("rerank", {}).get("candidates_in", 50))

    # ---- 1. 召回候选 ----
    debug: Dict[str, Any] = {"mode": mode}
    if mode in ("dense", "hybrid"):
        qv = embedder.encode_query(query)
        sims, dense_idx = store.dense.search(qv, cand_n)
        debug["dense_top1_sim"] = float(sims[0]) if len(sims) else None
    if mode in ("bm25", "hybrid"):
        bscores = store.bm25.scores(tokenize(query))
        bm_idx = np.argsort(-bscores)[:cand_n].tolist()

    if mode == "dense":
        cand = [int(i) for i in dense_idx]
    elif mode == "bm25":
        cand = [int(i) for i in bm_idx]
    else:
        cand = rrf_fuse([list(map(int, dense_idx)), list(map(int, bm_idx))],
                        k=int(r_cfg.get("rrf_k", 60)), top_n=cand_n)

    # ---- 2. 可选重排（cross-encoder） ----
    if r_cfg.get("rerank", {}).get("enabled", False):
        cand = _rerank(store, query, cand, r_cfg["rerank"], cfg)

    # ---- 3. fill-to-budget 组装上下文（DD-06） ----
    budget_cfg = r_cfg["budget"]
    segs, used = [], 0
    max_tok = int(budget_cfg["max_context_tokens"])
    max_segs = int(budget_cfg["max_segments"])
    cpt = float(budget_cfg.get("estimate_chars_per_token", 4.0))

    for i in cand:
        if len(segs) >= max_segs:
            break
        ch = store.chunks[i]
        t = int(ch.get("n_tokens") or estimate_tokens(ch["text"], cpt))
        if budget_cfg.get("enabled", True) and used + t > max_tok:
            # 预算满了：若一段都没装下，强制装入第一段（否则会出现空上下文）
            if not segs:
                segs.append(ch); used += t
            continue
        segs.append(ch); used += t

    debug["n_candidates"] = len(cand)
    debug["n_segments_used"] = len(segs)
    debug["context_tokens_used"] = used
    debug["budget"] = max_tok
    return segs, debug


def _rerank(store, query: str, cand: List[int], rcfg: Dict[str, Any], cfg: Dict[str, Any]) -> List[int]:
    """cross-encoder 重排。同样用 transformers 原生加载，理由同 Embedder（避开镜像站缺文件）。"""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        name = rcfg["model"]
        device = cfg["embedding"].get("device", "cpu")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        pairs = [(query, store.chunks[i]["text"][:2000]) for i in cand]
        scores = []
        with Timer() as t:
            with torch.no_grad():
                for i in range(0, len(pairs), 8):
                    enc = tok(pairs[i: i + 8], padding=True, truncation=True,
                              max_length=512, return_tensors="pt")
                    enc = {k: v.to(device) for k, v in enc.items()}
                    s = model(**enc).logits.squeeze(-1)
                    scores.extend(s.cpu().numpy().tolist())
        order = np.argsort(-np.asarray(scores))[: int(rcfg.get("keep_top", 20))]
        log(f"重排完成 {len(pairs)} 段，{t.ms}ms")
        return [cand[j] for j in order]
    except Exception as e:
        log(f"重排失败（{e}），跳过该步")
        return cand
