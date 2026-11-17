"""
分诊—会诊路由（Triage–Consultation Router）

对应 PRD §4.1 的仿医院设计：

    前台预检 ──→ 判定这道题该去哪个「科室」（学科 / 教材源）
                 ├─ 单一科室明确     → 单源直答（只在这本书里检索）
                 ├─ 两个科室打架     → 会诊（各科室各取证据，并列呈现，不替学生选）
                 └─ 哪个科室都不像   → 兜底（承认超纲，不给幻觉答案）

三条设计原则（为什么这样判，不那样判）：

  1. **教育不能"主治汇总"。** 医院会诊完了由主治医生下结论；教育不行——
     汇总两个教材的口径等于替学生选了一个"正确答案"，而教学场景下
     "两本书说法不同"本身就是该让学生看见的教学内容。所以会诊态只并列、不合并。
     （对应 PRD PR-1）

  2. **两个独立信号打架 = 题目有歧义，不是"选分高的那个"。**
     向量相似度（语义）与 BM25（词面）是两种失效模式不同的信号：
     语义会把"富兰克林自传"和"富兰克林探险队"混在一起，
     词面会漏掉同义改写。它们指向不同的书时，正确动作是**并列**而非二选一。

  3. **路由错是静默失败，代价不对称。**
     普通搜索错一次用户能自己发现（结果不相关）；教育答疑错一次，
     学生会把另一本书的说法当成课本内容记下来，且**无从察觉**。
     因此阈值取保守侧：宁可多开一次会诊（多花 token），不可猜错一次。

用法（在 pipeline 目录下）：
    ./.venv/Scripts/python.exe -m src.router ask "what did McClintock find in 1859?"
    ./.venv/Scripts/python.exe -m src.router eval --limit 300        # 量分诊准确率
    ./.venv/Scripts/python.exe -m src.router calibrate               # 网格搜索阈值

本模块**不调用任何生成 API**，只做检索与判定 —— 跑多少次都不花钱。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .common import Timer, estimate_tokens, load_config, log, read_jsonl, resolve_path
from .index import HybridStore, tokenize

# ------------------------------------------------------------------ 别名表
# 关键词直通：题目里出现书名特征词 → 高置信硬路由，绕过打分。
# 这不是作弊，是真实产品做法（用户问"《白鲸》里……"时没有理由去算相似度）。
# 但必须能关掉（--no-alias），因为别名表本身是我们手写的，
# 关掉之后的准确率才是"路由算法"的真实水平。两者都报，才诚实。
SOURCE_ALIASES: Dict[str, List[str]] = {
    "moby_dick": ["moby dick", "moby-dick", "melville", "white whale", "pequod", "ahab"],
    "little_women": ["little women", "louisa may alcott", "alcott", "march sisters", "jo march"],
    "north_pole": ["north pole", "peary", "henson", "polar", "arctic"],
    "philippines": ["philippine", "philippines", "manila", "filipino"],
    "benjamin_franklin": ["franklin", "poor richard", "autobiography of benjamin"],
    "economic_peace": ["economic peace", "economic consequences of the peace", "keynes", "versailles"],
    "frederick_douglass": ["douglass", "douglass", "slavery", "abolition"],
    "common_sense": ["common sense", "thomas paine", "paine"],
    "Microbiology-WEB": ["microbiology", "bacteri", "prokaryot", "eukaryot", "virus", "microb"],
}


@dataclass
class RouterParams:
    """阈值。全部默认保守侧（宁可多会诊，不可猜错）。

    ⚠️ 2026-11-09 实测校准（不要凭感觉改）：
        域内 300 题的最佳余弦  min 0.577 / p5 0.616 / p50 0.724
        8 条域外问题的最佳余弦 0.501 ~ 0.666
        → **两个区间重叠**，单一余弦阈值不可能同时做到「不误拒」和「不漏拒」。
        所以 tau_abs 只用于「确定没有」这一类高精度判定（宁漏不误），
        模糊地带交给 `scope_suspect` 标记 + D3 的 LLM 二次判定。
        词覆盖同理：「Why does my React component re-render」的覆盖率是 1.00
        （react/component/re-render/state/change 这些词在语料里都出现过），
        所以词覆盖只能当辅助信号，不能当判据。

    ⚠️ 2026-11-10 探针集再校准（100 条超纲 ground truth 首测）：
        旧阈值 (0.60, 0.55) 超纲拒答率仅 **8%**——92% 超纲题被当域内题送去生成。
        网格搜索（域内 300 误拒 ≤1% 硬约束）帕累托点：
        **tau_abs=0.64, tau_cov_low=0.85 → 域内 0/300 误拒，超纲拒答 33%**。
        规则判定天花板即 33%（科普向超纲题余弦普遍 0.6+，双信号同时垮不掉）；
        主防线是 suspect 标记 → LLM 复核：(0.65, 0.85) 覆盖 82% 超纲、域内 13% 复核成本。
        残留：~18% 静默直通 + alias 直通先于超纲判定（同作者/陷阱类 28 条全部绕过），
        属 alias 设计问题，见 build_probe_set.py eval 输出。
    """
    tau_abs: float = 0.64      # 余弦低于此值 **且** 词覆盖低于下值 → 判「确定没有」
    tau_cov_low: float = 0.85  # 词覆盖低于此值才算"极低"（与 tau_abs 同时满足）
    tau_cov_suspect: float = 0.85   # 词覆盖低于此值 → 打「需复核」标记
    tau_suspect: float = 0.65  # 余弦低于此值 → 打「需复核」标记（域内 p5=0.616）
    tau_gap: float = 0.045     # 次选源与首选源的余弦差 ≤ 此值才算「并驾齐驱」
    rho: float = 0.15          # 次选源在 top_k 里占比 ≥ 此值 → 视为有实质存在感
    top_k: int = 20            # 计算覆盖度用的混合检索窗口
    max_rank: int = 8          # 次选源的最好名次须 ≤ 此值才算"真的相关"
    use_alias: bool = True     # 是否启用书名/作者关键词直通


# 查询侧停用词（只用于算词覆盖；语料侧不需要）
_STOP = set("""a an the is are was were be been being do does did have has had will would shall
should can could may might must of in on at to for from by with without about into over under
between and or but not no nor so than then there here this that these those it its they them
their he she him her his we our you your i my me what which who whom whose when where why how
all any both each few more most other some such only own same too very s t don now""".split())

# 元话语停用词（2026-11-10 探针集发现）：学生讨论教材常用 narrator/speaker/author 等
# 讨论措辞，这些词**语料里本来就没有**（自传和小说用第一人称，不自称 narrator），
# 但它们不指示知识范围。不算停用词的话，"What did the narrator..." 类问题的词覆盖
# 会被措辞拖垮 → 域内题被误判超纲（探针实测 2/50 误拒，amb-015/023）。
_META_STOP = set("""narrator speaker author writer essay textbook excerpt passage
according tells describes mentions writes says""".split())


@dataclass
class RouteDecision:
    mode: str                                  # single | consult | none
    primary: Optional[str] = None              # "history/north_pole"
    secondary: List[str] = field(default_factory=list)
    strength: float = 0.0                      # 首选源最佳余弦
    confidence: float = 0.0                    # 0-1，给 UI 画进度条
    reason: str = ""
    scope_suspect: bool = False                # 证据偏弱 → 需 LLM 二次判定是否超纲
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ 源映射
class SourceMap:
    """chunk 下标 ←→ (subject, source) 的双向映射，以及按源切分的下标列表。"""

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.by_source: Dict[str, List[int]] = {}
        self.subject_of: Dict[str, str] = {}
        for i, c in enumerate(chunks):
            meta = c.get("meta") or {}
            src = meta.get("source") or c.get("doc_id")
            subj = meta.get("subject") or "unknown"
            self.by_source.setdefault(src, []).append(i)
            self.subject_of[src] = subj

    @property
    def sources(self) -> List[str]:
        return list(self.by_source)

    def key(self, src: str) -> str:
        return f"{self.subject_of.get(src, 'unknown')}/{src}"

    def sources_of_subject(self, subj: str) -> List[str]:
        return [s for s, x in self.subject_of.items() if x == subj]


# ------------------------------------------------------------------ 路由
class ExpertRouter:
    def __init__(self, store: HybridStore, embedder, cfg: Dict[str, Any],
                 params: Optional[RouterParams] = None):
        self.store = store
        self.embedder = embedder
        self.cfg = cfg
        self.p = params or RouterParams()
        self.smap = SourceMap(store.chunks)
        self._n = len(store.chunks)
        self._df: Optional[Dict[str, int]] = None   # 语料词表，惰性构建（约 2-3s）

    # ---------------- 词表（超纲检测用） ----------------
    def _vocab(self) -> Dict[str, int]:
        if self._df is None:
            from collections import Counter
            df: Counter = Counter()
            for c in self.store.chunks:
                df.update(set(tokenize(c["text"])))
            self._df = dict(df)
        return self._df

    def lexical_coverage(self, query: str) -> Tuple[float, List[str]]:
        """查询的实词里，有多少在语料词表中出现过。

        为什么需要它：bge 这类双塔模型对**完全无关**的问题也会给出 0.5+ 的余弦
        （实测「矩阵特征值」0.531、「React 组件重渲染」0.562），
        单靠余弦判不了超纲。词覆盖是另一种失效模式的信号，两者互补。
        """
        df = self._vocab()
        toks = [t for t in tokenize(query) if t not in _STOP and t not in _META_STOP]
        if not toks:
            return 1.0, []
        oov = [t for t in toks if df.get(t, 0) == 0]
        return 1.0 - len(oov) / len(toks), oov

    # ---------------- 基础打分 ----------------
    def _dense_sims(self, query: str) -> np.ndarray:
        qv = self.embedder.encode_query(query)
        return self.store.dense.vectors @ qv.reshape(-1)

    def _bm25_scores(self, query: str) -> np.ndarray:
        return self.store.bm25.scores(tokenize(query))

    @staticmethod
    def _top_in(sims: np.ndarray, idxs: List[int], k: int) -> List[int]:
        """在给定下标子集里取 top-k，返回全局下标（按分数降序）。"""
        if not idxs:
            return []
        arr = sims[idxs]
        kk = min(k, len(arr))
        part = np.argpartition(-arr, kk - 1)[:kk]
        part = part[np.argsort(-arr[part])]
        return [idxs[i] for i in part]

    def _fuse(self, dense_rank: List[int], bm25_rank: List[int], top_n: int) -> List[int]:
        """RRF 融合（与 retrieve.py 同一套 k，保持口径一致）。"""
        k = int(self.cfg["retrieval"].get("rrf_k", 60))
        score: Dict[int, float] = {}
        for lst in (dense_rank, bm25_rank):
            for r, i in enumerate(lst):
                score[i] = score.get(i, 0.0) + 1.0 / (k + r + 1)
        return [i for i, _ in sorted(score.items(), key=lambda x: -x[1])][:top_n]

    # ---------------- 三个信号 ----------------
    def _signal_strength(self, dsims: np.ndarray) -> Dict[str, float]:
        """每个源的最佳余弦（bge 已归一 → 余弦，跨源可比）。"""
        return {s: float(dsims[idx].max()) for s, idx in self.smap.by_source.items()}

    def _signal_coverage(self, dsims: np.ndarray, bsc: np.ndarray, query: str
                         ) -> Tuple[Dict[str, int], Dict[str, int], List[int]]:
        """全局混合检索 top_k 里，各源各占几席 / 各自最好名次。"""
        cand_n = int(self.cfg["retrieval"].get("rerank", {}).get("candidates_in", 50))
        dense_rank = self._top_in(dsims, list(range(self._n)), cand_n)
        bm25_rank = self._top_in(bsc, list(range(self._n)), cand_n)
        fused = self._fuse(dense_rank, bm25_rank, self.p.top_k)

        count: Dict[str, int] = {s: 0 for s in self.smap.sources}
        best_rank: Dict[str, int] = {s: 10 ** 6 for s in self.smap.sources}
        for r, gi in enumerate(fused):
            src = self.store.chunks[gi].get("meta", {}).get("source") or \
                  self.store.chunks[gi].get("doc_id")
            count[src] = count.get(src, 0) + 1
            best_rank[src] = min(best_rank.get(src, 10 ** 6), r + 1)
        return count, best_rank, fused

    def _signal_alias(self, query: str) -> List[str]:
        q = " " + query.lower() + " "
        hits = []
        for src, words in SOURCE_ALIASES.items():
            for w in words:
                if w in q:
                    hits.append(src)
                    break
        return hits

    # ---------------- 判定 ----------------
    def triage(self, query: str) -> RouteDecision:
        with Timer() as t:
            dsims = self._dense_sims(query)
            bsc = self._bm25_scores(query)
            strength = self._signal_strength(dsims)
            count, best_rank, fused = self._signal_coverage(dsims, bsc, query)
        order = sorted(strength.items(), key=lambda x: -x[1])
        top_src, top_val = order[0]
        second_src, second_val = (order[1] if len(order) > 1 else (None, 0.0))

        cov_lex, oov = self.lexical_coverage(query)

        cov_order = sorted(count.items(), key=lambda x: (-x[1], best_rank[x[0]]))
        cov_main = cov_order[0][0] if cov_order else top_src
        cov_second = cov_order[1] if len(cov_order) > 1 else (None, 0)

        # 「证据偏弱」——不是判定，只是打个标记，交给 D3 的 LLM 复核
        suspect = (cov_lex < self.p.tau_cov_suspect) or (top_val < self.p.tau_suspect)

        detail = {
            "latency_ms": t.ms,
            "strength": {k: round(v, 4) for k, v in strength.items()},
            "coverage": count,
            "best_rank": {k: (None if v >= 10 ** 6 else v) for k, v in best_rank.items()},
            "signals_agree": cov_main == top_src,
            "lexical_coverage": round(cov_lex, 3),
            "oov": oov[:8],
            "suspect": suspect,
        }

        # ---- 规则 0：书名/作者直通（高置信，绕过打分） ----
        if self.p.use_alias:
            aliases = [a for a in self._signal_alias(query) if a in self.smap.by_source]
            # 只在唯一命中时直通；命中多个书名说明题目本身在比较两本书 → 走会诊
            if len(aliases) == 1:
                a = aliases[0]
                return RouteDecision(
                    mode="single", primary=self.smap.key(a),
                    strength=strength.get(a, 0.0), confidence=0.97,
                    reason=f"题干命中书名/作者特征词，直通「{self.smap.key(a)}」",
                    detail={**detail, "rule": "alias"},
                )
            if len(aliases) > 1:
                return RouteDecision(
                    mode="consult", primary=self.smap.key(aliases[0]),
                    secondary=[self.smap.key(x) for x in aliases[1:]],
                    strength=top_val, confidence=0.85,
                    reason="题干同时提到多个书目，直接进会诊",
                    detail={**detail, "rule": "alias_multi"},
                )

        # ---- 规则 1：库里确定没有（高精度、低召回：宁可漏判，不可误拒） ----
        # 两个信号都垮掉才算"确定没有"。单独一个垮掉只打 suspect 标记，
        # 因为实测两边区间重叠（见 RouterParams 的注释），单信号判会两头错。
        if top_val < self.p.tau_abs and cov_lex < self.p.tau_cov_low:
            return RouteDecision(
                mode="none", primary=None, strength=top_val,
                confidence=1.0 - min(1.0, top_val / max(self.p.tau_abs, 1e-6)),
                scope_suspect=False,
                reason=f"余弦 {top_val:.3f} < {self.p.tau_abs} 且词覆盖 "
                       f"{cov_lex:.2f} < {self.p.tau_cov_low}"
                       f"（未登录词：{', '.join(oov[:4]) or '无'}）→ 判定库里没有这个知识",
                detail={**detail, "rule": "below_abs"},
            )

        # ---- 规则 2：两个信号指向不同的书，且两边分数咬得很紧 → 有歧义 → 会诊 ----
        # 为什么加"咬得很紧"这个附加条件：语义和词面不一致，也可能是
        # 词面被一个人名带偏（"Franklin" 既是本杰明·富兰克林也是北极探险队）。
        # 若语义信号领先很多，那更可能是词面噪声，不该因此白开一次会诊。
        if cov_main != top_src and count.get(top_src, 0) > 0 \
                and strength.get(cov_main, 0.0) >= top_val - self.p.tau_gap * 2:
            sec = top_src if cov_main != top_src else None
            return RouteDecision(
                mode="consult", primary=self.smap.key(cov_main),
                secondary=[self.smap.key(top_src)] if sec else [],
                strength=top_val, confidence=0.55, scope_suspect=suspect,
                reason=f"语义信号指向「{self.smap.key(top_src)}」，"
                       f"词面信号指向「{self.smap.key(cov_main)}」，两者不一致 → 并列而非猜",
                detail={**detail, "rule": "signal_conflict"},
            )

        # ---- 规则 3：次选源既占席位、分数又够近 → 会诊 ----
        if cov_second[0] and cov_second[1] >= max(2, int(self.p.rho * self.p.top_k)) \
                and best_rank.get(cov_second[0], 10 ** 6) <= self.p.max_rank \
                and (top_val - strength.get(cov_second[0], 0.0)) <= self.p.tau_gap:
            sec = cov_second[0]
            gap = top_val - strength.get(sec, 0.0)
            return RouteDecision(
                mode="consult", primary=self.smap.key(top_src),
                secondary=[self.smap.key(sec)],
                strength=top_val, confidence=0.6, scope_suspect=suspect,
                reason=f"「{self.smap.key(sec)}」在 top{self.p.top_k} 里占 "
                       f"{cov_second[1]} 席（最好第 {best_rank[sec]} 名），"
                       f"且与首选仅差 {gap:.3f} → 跨源会诊",
                detail={**detail, "rule": "close_second", "gap": round(gap, 4)},
            )

        # ---- 默认：单源直答 ----
        margin = top_val - (second_val or 0.0)
        return RouteDecision(
            mode="single", primary=self.smap.key(top_src),
            strength=top_val, confidence=min(1.0, 0.5 + margin * 2),
            scope_suspect=suspect,
            reason=f"「{self.smap.key(top_src)}」最强（{top_val:.3f}），"
                   f"领先次选 {margin:.3f}，top{self.p.top_k} 里占 {count.get(top_src,0)} 席",
            detail={**detail, "rule": "default", "margin": round(margin, 4)},
        )

    # ---------------- 单源内的证据组装 ----------------
    def evidence(self, query: str, sources: List[str], budget_tokens: int
                 ) -> Dict[str, List[Dict[str, Any]]]:
        """会诊：每个源**各自**装满预算，互不挤占。

        为什么不共享一个预算池：共享池会让强源把弱源挤空，
        结果"会诊"变成"强源独答 + 弱源空手"，等于没会诊。
        """
        dsims = self._dense_sims(query)
        bsc = self._bm25_scores(query)
        cpt = float(self.cfg["retrieval"]["budget"]["estimate_chars_per_token"])
        out: Dict[str, List[Dict[str, Any]]] = {}
        for src in sources:
            idxs = self.smap.by_source.get(src, [])
            if not idxs:
                out[src] = []
                continue
            dr = self._top_in(dsims, idxs, 30)
            br = self._top_in(bsc, idxs, 30)
            fused = self._fuse(dr, br, 20)
            segs, used = [], 0
            for gi in fused:
                ch = self.store.chunks[gi]
                nt = int(ch.get("n_tokens") or estimate_tokens(ch["text"], cpt))
                if used + nt > budget_tokens and segs:
                    break
                segs.append({**ch, "_score": float(dsims[gi])})
                used += nt
            out[src] = segs
        return out


# ------------------------------------------------------------------ 装配
def load_router(cfg: Optional[Dict[str, Any]] = None,
                params: Optional[RouterParams] = None,
                index_dir: Optional[Path] = None) -> ExpertRouter:
    from .index import Embedder
    cfg = cfg or load_config()
    idx = index_dir or (resolve_path(cfg, "index_dir").parent / "index_expert")
    store = HybridStore.load(idx, backend=cfg["retrieval"]["backend"])
    if store is None:
        raise FileNotFoundError(
            f"找不到多源索引：{idx}\n先跑：python build_expert_pool.py build --index")
    e = cfg["embedding"]
    embedder = Embedder(
        model_name=e["model"], device=e.get("device", "cpu"),
        batch_size=int(e.get("batch_size", 32)),
        normalize=bool(e.get("normalize", True)),
        query_instruction=e.get("query_instruction", ""),
        local_files_only=bool(e.get("local_files_only", True)),
    )
    return ExpertRouter(store, embedder, cfg, params)


# ------------------------------------------------------------------ 评测
def eval_triage(router: ExpertRouter, questions: List[Dict[str, Any]],
                limit: Optional[int] = None, sample: str = "stratified",
                seed: int = 42) -> Dict[str, Any]:
    """用题集自带的 source 字段当 ground truth，量 Triage Accuracy。

    注意口径（写报告时必须写清）：
      - 只统计 mode=single 的题算「路由准确率」，因为会诊/兜底本来就没指定单一源；
      - 另报「会诊触发率」与「应会诊却单答」的漏检率；
      - 这里的 ground truth 是「这道题取材自哪本书」，不等于「人类认为该查哪本书」，
        是**构造性**标注，比人工标注便宜但更窄——报告里要说。
    """
    if limit and limit < len(questions):
        if sample == "head":
            questions = questions[:limit]
        else:
            import random
            rng = random.Random(seed)
            by_subj: Dict[str, List[Dict[str, Any]]] = {}
            for q in questions:
                by_subj.setdefault(q["subject"], []).append(q)
            picked: List[Dict[str, Any]] = []
            for subj, rows in by_subj.items():
                k = max(1, round(limit * len(rows) / len(questions)))
                picked.extend(rng.sample(rows, min(k, len(rows))))
            questions = picked[:limit]

    rows, hits, n_single, n_consult, n_none, n_suspect = [], 0, 0, 0, 0, 0
    per_subj: Dict[str, List[int]] = {}
    with Timer() as t:
        for q in questions:
            d = router.triage(q["question"])
            gold = f"{q['subject']}/{q.get('source', '')}"
            rows.append({
                "qid": q["qid"], "gold": gold, "pred": d.primary,
                "mode": d.mode, "secondary": d.secondary,
                "reason": d.reason, "rule": d.detail.get("rule"),
                "confidence": round(d.confidence, 4),
                "strength": round(d.strength, 4),
                "scope_suspect": d.scope_suspect,
                "lexical_coverage": d.detail.get("lexical_coverage"),
            })
            n_suspect += int(d.scope_suspect)
            if d.mode == "single":
                n_single += 1
                ok = int(d.primary == gold)
                hits += ok
                per_subj.setdefault(q["subject"], []).append(ok)
            elif d.mode == "consult":
                n_consult += 1
                # 会诊里如果 gold 在候选集内，算「救回来了」
                per_subj.setdefault(q["subject"], []).append(
                    int(gold == d.primary or gold in d.secondary))
            else:
                n_none += 1

    n = len(rows)
    acc_all = hits / n_single if n_single else 0.0
    acc_incl = (hits + sum(1 for r in rows if r["mode"] == "consult"
                           and (r["gold"] == r["pred"] or r["gold"] in r["secondary"]))) / n
    res = {
        "n": n,
        "seconds": round(t.ms / 1000, 1),
        "single": n_single, "consult": n_consult, "none": n_none,
        "consult_rate": round(n_consult / n, 4),
        "none_rate": round(n_none / n, 4),
        "suspect_rate": round(n_suspect / n, 4),   # 需 LLM 二次判定的比例（成本项）
        "triage_accuracy_single_only": round(acc_all, 4),
        "top1_accuracy_incl_consult": round(acc_incl, 4),
        "per_subject_single_acc": {k: round(sum(v) / len(v), 4) for k, v in per_subj.items()},
        "params": asdict(router.p),
    }
    return {"summary": res, "rows": rows}


# ------------------------------------------------------------------ CLI
def _cmd_ask(router: ExpertRouter, q: str) -> None:
    d = router.triage(q)
    print()
    log(f"问题：{q}")
    log(f"分诊：{d.mode.upper()}  置信 {d.confidence:.2f}")
    log(f"主源：{d.primary or '（无）'}")
    if d.secondary:
        log(f"并列：{', '.join(d.secondary)}")
    log(f"理由：{d.reason}")
    if d.mode != "none" and d.primary:
        srcs = [d.primary.split("/", 1)[1]] + [s.split("/", 1)[1] for s in d.secondary]
        ev = router.evidence(q, srcs, int(router.cfg["retrieval"]["budget"]["max_context_tokens"]))
        for s, segs in ev.items():
            print()
            log(f"── {s} 取到 {len(segs)} 段 ──")
            for seg in segs[:2]:
                print(f"   [{seg['_score']:.3f}] {seg['text'][:160].replace(chr(10), ' ')}…")


def main():
    ap = argparse.ArgumentParser(description="分诊—会诊路由")
    ap.add_argument("cmd", choices=["ask", "eval", "calibrate"])
    ap.add_argument("text", nargs="?", default="", help="ask 时的问题")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--sample", default="stratified", choices=["head", "stratified"])
    ap.add_argument("--no-alias", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg = load_config()
    params = RouterParams(use_alias=not a.no_alias)
    router = load_router(cfg, params)

    if a.cmd == "ask":
        if not a.text:
            log("用法：ask \"你的问题\""); sys.exit(1)
        _cmd_ask(router, a.text); return

    qdir = resolve_path(cfg, "questions")
    qs = list(read_jsonl(qdir / "pool_all.jsonl"))
    log(f"题库 {len(qs)} 题，取 {a.limit} 题评测（分层抽样, seed=42）")

    if a.cmd == "eval":
        res = eval_triage(router, qs, a.limit, a.sample)
        s = res["summary"]
        print()
        log(f"单源直答 {s['single']} / 会诊 {s['consult']} / 兜底 {s['none']}"
            f"（会诊触发率 {s['consult_rate']:.1%}）")
        log(f"★ Triage Accuracy（仅单源态）: {s['triage_accuracy_single_only']:.1%}")
        log(f"  Top-1 命中（含会诊候选）  : {s['top1_accuracy_incl_consult']:.1%}")
        log(f"  分学科（仅单源态）        : {s['per_subject_single_acc']}")
        log(f"  需 LLM 二次判定比例       : {s['suspect_rate']:.1%}（成本项）")
        log(f"  用时 {s['seconds']}s")
        if a.out:
            Path(a.out).write_text(json.dumps(res["summary"], ensure_ascii=False, indent=2),
                                   encoding="utf-8")
            Path(a.out.replace(".json", ".rows.jsonl")).write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in res["rows"]),
                encoding="utf-8")
            log(f"已写 {a.out}（+ 同名 .rows.jsonl）")
        return

    # ---- calibrate：网格搜索（在留出集上验证，避免自欺） ----
    import random
    rng = random.Random(42)
    qs2 = qs[:]
    rng.shuffle(qs2)
    cut = int(len(qs2) * 0.7)
    dev, hold = qs2[:cut][: a.limit], qs2[cut:][: max(50, a.limit // 2)]
    log(f"网格搜索：开发集 {len(dev)} 题 / 留出集 {len(hold)} 题")

    best: Tuple[float, RouterParams, Dict[str, Any]] = (-1, params, {})
    # 只搜「会诊判定」的四个旋钮。兜底阈值不参与搜索 ——
    # 题库里没有超纲题，搜它等于拿域内数据拟合一个域外边界，必然过拟合。
    for gap in (0.02, 0.03, 0.045, 0.06, 0.08):
        for rho in (0.10, 0.15, 0.20, 0.30):
            for mr in (5, 8, 12, 20):
                router.p = RouterParams(tau_gap=gap, rho=rho, top_k=20,
                                        max_rank=mr, use_alias=not a.no_alias)
                r = eval_triage(router, dev)["summary"]
                # 目标函数：单源态准确率为主；会诊率过高要罚（会诊要花 token），
                # 二次判定率过高也要罚（那也是钱）。
                score = (r["triage_accuracy_single_only"]
                         - 0.10 * max(0.0, r["consult_rate"] - 0.35)
                         - 0.05 * r["suspect_rate"])
                if score > best[0]:
                    best = (score, RouterParams(tau_gap=gap, rho=rho, top_k=20,
                                                max_rank=mr, use_alias=not a.no_alias), r)
    score, p, r = best
    log(f"开发集最优：Triage Accuracy {score:.1%}  params={asdict(p)}")
    router.p = p
    h = eval_triage(router, hold)["summary"]
    log(f"★ 留出集（防自欺）：Triage Accuracy {h['triage_accuracy_single_only']:.1%}"
        f"，会诊率 {h['consult_rate']:.1%}，兜底率 {h['none_rate']:.1%}")
    if a.out:
        Path(a.out).write_text(json.dumps({"dev": r, "holdout": h, "params": asdict(p)},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"已写 {a.out}")


if __name__ == "__main__":
    main()
