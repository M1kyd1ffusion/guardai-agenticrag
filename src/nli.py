# -*- coding: utf-8 -*-
"""
本地 NLI 判据模块（判官 B 的引擎）

职责边界
--------
- 只做"这句话能不能被这段证据支持"的判断，**不判对错**（那是判官 A 的活）；
- 只输出标签，**不做拦截决策**（拦截由调用方决定，且实测是净负）；
- 模型只在第一次调用时加载（约 2s），之后常驻。

为什么单独成模块
----------------
评测脚本（stage13_faithfulness.py）和产品界面（app.py）要用**同一套**切句/切块/阈值逻辑，
否则"离线测出来的数"和"线上看到的行为"不是一回事。

模型：MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli（184M，CPU fp32）
⚠️ CPU 上必须 fp32：fp16 前向 0.90 s/对，fp32 0.097 s/对（慢 9 倍，2026-11-11 实测）。
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "DeBERTa-v3-base-mnli"

_SENT = re.compile(r"(?<=[.!?])\s+")
# 元话语：不是"事实主张"，不该拿去做忠实性判定（但也不能当依据，单列统计）
_META = re.compile(
    r"^(note to instructor|note for the instructor|note:|note\b|flag:|i should flag|\(note)", re.I)

# 缩写保护：不保护的话 "T. gondii" / "Dr. Smith" / "U.S." 会被切成碎片，
# 碎片既无法判支持、又会污染"无依据"计数（2026-11-11 抽查发现）。
_ABBR = re.compile(
    r"\b(T|Dr|Mr|Mrs|Ms|Prof|St|Jr|Sr|vs|etc|al|e\.g|i\.e|cf|Fig|fig|No|no|pp|p|Vol|Ch|Sec)\.", re.I)
_DOT = "\u0001"
# 词面守卫用：数字（含区间/千分位）与专名
_NUM = re.compile(r"\d[\d,.\-–—/]*")
_CAP = re.compile(r"\b[A-Z][A-Za-z\-]{3,}\b")

_lock = threading.Lock()
_state: Dict[str, Any] = {"tok": None, "model": None, "id2label": None}


# ------------------------------------------------------------------ 文本切分
def split_sentences(text: str, min_len: int = 25) -> List[str]:
    """切句。缩写里的点先换成占位符，切完再还原。

    min_len=25（不是 15）：太短的片段多半是切分失败产生的碎片（"gondii, so the minimum age…"），
    留着只会污染"无依据"统计。
    """
    t = _ABBR.sub(lambda m: m.group(1) + _DOT, str(text or ""))
    sents = [s.strip() for s in _SENT.split(t) if s.strip()]
    out = []
    for s in sents:
        s = s.replace(_DOT, ".")
        if len(s) >= min_len:
            out.append(s)
    return out


def split_premises(context: str, max_chars: int = 1400) -> List[str]:
    """证据切块：按空行切段，再合并到 ~1400 字符（≈350 token）一块。

    ⚠️ 不要按单行切：教材表格每行一个单元格，按行切会把 1800 token 切成 30+ 块，
    配对数量爆炸（单题 30 秒）。按段合并后通常 4–6 块。
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", str(context or "")) if b.strip()]
    if not blocks:
        return []
    out: List[str] = []
    buf = ""
    for b in blocks:
        if len(b) > max_chars:
            if buf:
                out.append(buf); buf = ""
            for i in range(0, len(b), max_chars):
                out.append(b[i:i + max_chars])
            continue
        if len(buf) + len(b) + 2 <= max_chars:
            buf = f"{buf}\n{b}".strip()
        else:
            out.append(buf); buf = b
    if buf:
        out.append(buf)
    return out


# ------------------------------------------------------------------ 模型
def available() -> bool:
    return (MODEL_DIR / "config.json").exists() and (
        (MODEL_DIR / "model.safetensors").exists() or (MODEL_DIR / "pytorch_model.bin").exists())


def _load():
    with _lock:
        if _state["model"] is not None:
            return _state
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_DIR), local_files_only=True, torch_dtype=torch.float32)
        model.eval()
        _state.update(tok=tok, model=model,
                      id2label={i: l.lower() for i, l in model.config.id2label.items()},
                      torch=torch)
        return _state


def _norm_num(s: str) -> str:
    """数字归一：统一各种短横线、去掉千分位逗号。
    不归一的话 "1,750,000" 和 "1750000" 对不上，词面守卫形同虚设（实测误报主因之一）。
    """
    s = (s or "").replace("–", "-").replace("—", "-").replace("−", "-")
    return s.replace(",", "")


def _key_tokens(s: str) -> set:
    """句中可核对的关键 token：数字与专名。用于"词面守卫"。"""
    nums = {_norm_num(x) for x in _NUM.findall(_norm_num(s))}
    return nums | set(_CAP.findall(s or ""))


def check(answer: str, context: str, tau: float = 0.5) -> Optional[Dict[str, Any]]:
    """逐句核对：返回句级标签 + 汇总。模型不可用时返回 None（调用方应降级而不是报错）。

    三态而不是两态（2026-11-11 加的，有实测依据）
    ------------------------------------------------
    NLI 对**表格类证据**系统性失灵：教材表格行是 "1977–1978 | Asian flu | N2N2 | …"，
    而答案是 "the Asian flu pandemic occurred in 1977–1978"。把整行喂给 DeBERTa，
    entailment 也只有 0.13 —— 模型读不懂表格这种前提形式，不是阈值问题。
    另有一类是证据分块截断：关键行落在切块边界外。

    直接把这些判成"无依据"会大量误报（实测抽 8 条，多数看着是有依据的）。
    所以加一道**词面守卫**：句中的数字/专名有 ≥50% 出现在证据里 → 判 "uncertain"（存疑），
    既不冤枉、也不放过真正瞎编的句子（那种句子连数字都对不上）。
    """
    if not available():
        return None
    sents = split_sentences(answer)
    prem = split_premises(context)
    if not sents or not prem:
        return {"n_sent": 0, "n_unsupported": 0, "n_uncertain": 0, "sentences": []}

    st = _load()
    tok, model = st["tok"], st["model"]
    prem_keys = set()
    for p in prem:
        prem_keys |= _key_tokens(p)
    pairs = [(p, s) for s in sents for p in prem]
    with st["torch"].no_grad():
        enc = tok([p for p, _ in pairs], [s for _, s in pairs],
                  truncation=True, max_length=512, padding=True, return_tensors="pt")
        probs = st["torch"].softmax(model(**enc).logits, dim=-1).tolist()

    detail: List[Dict[str, Any]] = []
    k = 0
    for s in sents:
        best: Dict[str, float] = {}
        best_prem = ""
        for p in prem:
            row = {st["id2label"][i]: probs[k][i] for i in range(len(probs[k]))}
            k += 1
            if not best or row.get("entailment", 0) > best.get("entailment", 0):
                best, best_prem = row, p
        ent = best.get("entailment", 0.0)
        if ent >= tau:
            status = "supported"
        else:
            keys = _key_tokens(s)
            if not keys:
                # 句中既没有数字也没有专名 → 词面无法核对。
                # 这种情况必须判"存疑"，不能判"无依据"：
                # 判不了 ≠ 判错（2026-11-11 端到端走查踩过：正确答案被标成"证据不足"）。
                status = "uncertain"
            else:
                hit = len(keys & prem_keys) / len(keys)
                # 只有一个关键 token 时要求完全命中（如句中只有一个 "60%"），否则按比例。
                ok = (len(keys) >= 2 and hit >= 0.5) or (len(keys) == 1 and hit == 1.0)
                status = "uncertain" if ok else "unsupported"
        detail.append({
            "sentence": s,
            "meta": bool(_META.match(s)),
            "p_entail": round(ent, 4),
            "p_contra": round(best.get("contradiction", 0.0), 4),
            "status": status,
            "supported": status == "supported",
        })
    facts = [d for d in detail if not d["meta"]]
    return {
        "n_sent": len(facts),
        "n_unsupported": sum(1 for d in facts if d["status"] == "unsupported"),
        "n_uncertain": sum(1 for d in facts if d["status"] == "uncertain"),
        "n_meta": len(detail) - len(facts),
        "sentences": detail,
    }
