"""
一键冒烟测试（Smoke Test）：
用极小规模（默认 5 题；--n 可调）把「教材 → 索引 → 检索 → 双路径生成 → 落盘」整条链跑一遍，
目的是**尽早暴露环境与 API 问题**，不是拿结论。

用法：
    python run_smoke.py --demo          # 内置示例教材，不需要任何数据，专测环境与 API 通不通
    python run_smoke.py                 # 用真实语料（data/corpus/original）
    python run_smoke.py --n 50          # 正式 50 题 smoke test

跑完看：data/results/smoke_<时间戳>.jsonl   ← 每条含 pk_answer / ck_answer / usage，可直接翻
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common import (Cache, Timer, append_jsonl, index_source_ok, load_config, log,
                        resolve_path, write_index_source, write_jsonl)
from src.corpus import build_chunks, load_documents
from src.index import HybridStore
from src.generate import ModelClient, run_dual_path
from src.retrieve import retrieve
from src.evaluate import is_refusal_by_keywords
from src.index import Embedder

DEMO_TEXT = """Cell Walls of Prokaryotes.
The cytoplasm of prokaryotic cells has a high concentration of dissolved solutes.
Most bacteria have a cell wall that protects the cell and maintains its shape.
Peptidoglycan is a polymer composed of sugars and amino acids.
Gram-positive bacteria have a thick peptidoglycan layer and stain purple during the
Gram stain procedure. Gram-negative bacteria have a thin peptidoglycan layer and an outer
membrane containing lipopolysaccharide; they stain pink during the Gram stain procedure.
Mycoplasma species lack a cell wall entirely and are therefore resistant to many
antibiotics that target cell wall synthesis. Penicillin inhibits the cross-linking of
peptidoglycan chains, weakening the wall and causing cells to lyse.

Bacterial Staining.
The Gram stain was developed by Hans Christian Gram in 1884. The procedure uses
crystal violet as the primary stain, iodine as a mordant, and safranin as the counterstain.
Acid-fast staining is used to identify members of the genus Mycobacterium, which have
waxy mycolic acids in their cell walls."""

DEMO_QUESTIONS = [
    {"qid": "demo-1", "question": "Why do Gram-positive bacteria stain purple?",
     "reference": "Because they have a thick peptidoglycan layer."},
    {"qid": "demo-2", "question": "Which genus is identified using acid-fast staining?",
     "reference": "Mycobacterium."},
    {"qid": "demo-3", "question": "Who developed the Gram stain and in what year?",
     "reference": "Hans Christian Gram, in 1884."},
    {"qid": "demo-4", "question": "How does penicillin affect bacterial cells?",
     "reference": "It inhibits peptidoglycan cross-linking, weakening the wall and causing lysis."},
    {"qid": "demo-5", "question": "Why are Mycoplasma species resistant to certain antibiotics?",
     "reference": "Because they lack a cell wall entirely."},
]


class MockCompletions:
    """假模型：不联网、不花钱，只把收到的提示词回显成一句假答案。"""

    def create(self, **kw):
        msgs = kw.get("messages", [{}])
        prompt = msgs[-1].get("content", "")
        n = int(kw.get("n", 1))
        ans = "[MOCK] " + " ".join(prompt.split())[:60]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=ans)) for _ in range(n)],
            usage=SimpleNamespace(prompt_tokens=max(1, len(prompt) // 4), completion_tokens=20),
        )


class MockClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=MockCompletions())


def ensure_demo(cfg) -> Path:
    raw = resolve_path(cfg, "corpus_raw")
    fp = raw / "demo_prokaryotes.txt"
    fp.write_text(DEMO_TEXT, encoding="utf-8")
    qdir = resolve_path(cfg, "questions")
    write_jsonl(qdir / "demo.jsonl", DEMO_QUESTIONS)
    log(f"已生成示例语料 {fp} 与示例题 {qdir/'demo.jsonl'}")
    return fp


def sample_questions(rows, n, mode="stratified", seed=42, min_per=5):
    """
    抽样。为什么不能直接取前 N 题：
      EduScopeQA 的题目是按 type 连续排列的（science 前 50 题全是 very_specific），
      取前 N 题会让 sectional / thematic 两个粒度完全缺席 —— 而 SQ1 的核心自变量
      就是问题粒度，缺了粒度这一层，跑出来的 50 题对 SQ1 毫无信息量。
    stratified：先给每个粒度保底 min_per 题，剩余名额按各粒度占比分配（可复现）。
    """
    import random

    if mode == "head":
        return rows[:n]
    rng = random.Random(seed)
    if mode == "random":
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        picked = sorted(idx[:n])
        return [rows[i] for i in picked]

    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r.get("granularity") or "unknown", []).append(i)
    keys = sorted(by, key=lambda k: -len(by[k]))
    pool = {k: (rng.sample(by[k], len(by[k]))) for k in keys}   # 每层内部打乱，可复现
    alloc = {k: 0 for k in keys}
    left = min(n, len(rows))
    if left >= len(keys):                                       # 第零轮：先保证每层都有
        for k in keys:
            alloc[k] = 1
            left -= 1
    for k in keys:                                             # 第一轮：保底
        q = min(min_per - alloc[k], len(pool[k]) - alloc[k], left)
        if q > 0:
            alloc[k] += q
            left -= q
    total = sum(len(pool[k]) for k in keys) or 1
    while left > 0:                                            # 第二轮：按占比补
        moved = False
        for k in keys:
            if left <= 0:
                break
            want = min(len(pool[k]), int(round(n * len(pool[k]) / total)))
            while alloc[k] < want and left > 0:
                alloc[k] += 1
                left -= 1
                moved = True
        if not moved:
            break
    if left > 0:                                               # 第三轮：还有余额就均摊
        for k in keys:
            while left > 0 and alloc[k] < len(pool[k]):
                alloc[k] += 1
                left -= 1

    picked = []
    for k in keys:
        picked.extend(pool[k][:alloc[k]])
    picked.sort()
    log("分层抽样：" + " / ".join(f"{k}={alloc[k]}(池{len(pool[k])})" for k in keys))
    return [rows[i] for i in picked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--demo", action="store_true", help="用内置示例，不依赖真实数据")
    ap.add_argument("--n", type=int, default=None, help="跑几题（默认读 config.smoke.n_questions）")
    ap.add_argument("--strategy", default="S1", choices=["S1", "S2", "S3", "S4", "S5"],
                    help="提示策略。S5=DD-09 干预臂：照教材答 + 发现材料异常须举手（与 S1 只差这段指令）")
    ap.add_argument("--provider", default=None, help="覆盖 config.generation.active")
    ap.add_argument("--mock", action="store_true",
                    help="用假模型跑：不联网、不花钱、不需要 key，专测链路本身")
    ap.add_argument("--sample", default="stratified", choices=["stratified", "head", "random"],
                    help="抽样方式：stratified=按问题粒度分层（默认，SQ1 必须）；head=取前 N 题；random=随机")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，保证抽样可复现")
    ap.add_argument("--min-per-stratum", type=int, default=5, help="每个粒度保底题数")
    ap.add_argument("--budget", type=int, default=None,
                    help="覆盖 config.retrieval.budget.max_context_tokens（SQ1：做『预算 × 粒度』交互实验时用）")
    ap.add_argument("--max-segments", type=int, default=None, help="覆盖 budget.max_segments")
    ap.add_argument("--outfile", default=None,
                    help="指定结果文件名（批量实验编排时用；默认 smoke_<时间戳>.jsonl）")
    ap.add_argument("--append", action="store_true",
                    help="续写到已有文件末尾。默认会覆盖（旧文件自动备份），避免两批记录混在一起")
    ap.add_argument("--no-cache", action="store_true",
                    help="禁用磁盘缓存，强制真实调用。用来验证『隔了几天，模型版本有没有偷偷变』")
    ap.add_argument("--gran", default=None,
                    help="只跑指定粒度，逗号分隔（如 sectional,thematic）——补厚宽粒度样本用")
    ap.add_argument("--all-gran", action="store_true",
                    help="配合 --gran：不做抽样，把该粒度的题全部跑完")
    # ---- SQ2 专用：跑篡改语料 + 篡改题集 ----
    ap.add_argument("--corpus", default=None,
                    help="语料目录（默认 data/corpus/original；SQ2 传 data/corpus/altered）")
    ap.add_argument("--questions", default=None,
                    help="题目文件（默认 normal.jsonl；SQ2 传 altered.jsonl）")
    ap.add_argument("--index", default=None,
                    help="索引目录（默认 data/index；SQ2 传 data/index_altered）"
                         "⚠️ 篡改语料必须用独立索引目录，否则会把原著索引覆盖掉")
    a = ap.parse_args()

    cfg = load_config(a.config)

    # SQ1「预算 × 粒度」交互：预算档位在这里覆盖，
    # 写进 meta.json 的 retrieval.budget 也随之变化，便于事后核对每批结果用的是哪档。
    if a.budget:
        cfg["retrieval"]["budget"]["enabled"] = True
        cfg["retrieval"]["budget"]["max_context_tokens"] = a.budget
        log(f"★ 预算覆盖为 {a.budget} token（默认 {load_config(a.config)['retrieval']['budget']['max_context_tokens']}）")
    if a.max_segments:
        cfg["retrieval"]["budget"]["max_segments"] = a.max_segments

    use_demo = a.demo

    # ---- 1. 语料 ----
    if use_demo:
        ensure_demo(cfg)
        corpus_dir = resolve_path(cfg, "corpus_raw")
    elif a.corpus:
        corpus_dir = Path(a.corpus).resolve()
        if not corpus_dir.exists() or not any(corpus_dir.iterdir()):
            log(f"语料目录为空：{corpus_dir}")
            sys.exit(1)
    else:
        corpus_dir = resolve_path(cfg, "corpus_original")
        if not any(corpus_dir.iterdir()):
            log("=" * 60)
            log("data/corpus/original 是空的。先用 --demo 跑通环境，再放真实教材进来。")
            log("=" * 60)
            sys.exit(1)

    docs = load_documents(corpus_dir)
    log(f"读到 {len(docs)} 篇文档")

    chunks = build_chunks(cfg, docs)
    if not chunks:
        log("切片为空，检查语料是否有内容")
        sys.exit(1)
    log(f"切片完成：{len(chunks)} 段（strategy={cfg['chunking']['strategy']}, size={cfg['chunking']['sizes']}）")

    # ---- 2. 索引（有缓存就复用，不重复付算力）----
    idx_dir = Path(a.index).resolve() if a.index else resolve_path(cfg, "index_dir")
    # ⚠️ demo 必须用**独立**索引目录。2026-10-26 事故：--demo 只切出 1 段，
    #    与真实索引（1294 段）段数不符 → 触发重建 → 把教材索引整个覆盖成示例那段，
    #    之后 50 题真实实验全在 1 段索引上跑，结果全废。
    if use_demo and not a.index:
        idx_dir = idx_dir.parent / "index_demo"
    store = HybridStore.load(idx_dir, cfg["retrieval"]["backend"])
    # ⚠️ 段数相同 ≠ 索引就是这份语料建的。原著与篡改语料的段数只差几条，
    #    只比段数会静默复用错的那份，故再校一道来源指纹（data/index/source.json）。
    src_ok = index_source_ok(idx_dir, corpus_dir)
    if store is not None and not src_ok:
        log(f"⚠️  索引来源与语料不符（{idx_dir} ← 别处语料），将重建")
        store = None
    if store is None or len(store.chunks) != len(chunks):
        global _EMB
        emb = Embedder(cfg["embedding"]["model"], cfg["embedding"]["device"],
                       cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"],
                       cfg["embedding"].get("query_instruction", ""),
                       local_files_only=cfg["embedding"].get("local_files_only", True))
        _EMB = emb                      # 复用同一个模型实例，避免重复加载权重
        with Timer() as t:
            store = HybridStore.build(cfg, chunks, emb)
        store.save(idx_dir)
        write_index_source(idx_dir, corpus_dir)
        log(f"  索引来源已登记 → {idx_dir}/source.json")

    # ---- 3. 题目 ----
    qdir = resolve_path(cfg, "questions")
    if a.questions:
        qpath = Path(a.questions)
        qfile = qpath.resolve() if qpath.is_absolute() or qpath.exists() else qdir / a.questions
    else:
        qfile = qdir / ("demo.jsonl" if use_demo else "normal.jsonl")
    if not qfile.exists():
        if not use_demo:
            log(f"找不到题目文件 {qfile}；可先用 --demo 跑通")
            sys.exit(1)
        ensure_demo(cfg)
    from src.common import read_jsonl
    questions = list(read_jsonl(qfile))
    if a.gran and not use_demo:
        want = {x.strip() for x in a.gran.split(",") if x.strip()}
        before = len(questions)
        questions = [q for q in questions if q.get("granularity") in want]
        log(f"粒度过滤 {want}：{before} → {len(questions)} 题")
    n = a.n or int(cfg["smoke"]["n_questions"])
    if a.all_gran:
        n = len(questions)          # 该粒度全跑，不再抽样
    questions = sample_questions(questions, n, a.sample, a.seed, a.min_per_stratum)
    log(f"题目 {len(questions)} 道，来自 {qfile.name}")

    # ---- 4. 模型 ----
    cache_enabled = cfg["generation"].get("cache_enabled", True) and not a.no_cache
    if a.no_cache:
        log("★ 已禁用缓存：本次每条都会真实调用 API（用于验证模型版本是否漂移）")
    cache = Cache(resolve_path(cfg, "cache"), cache_enabled)
    providers = [a.provider] if a.provider else cfg["generation"]["active"]
    if a.mock:
        log("MOCK 模式：不会调用任何真实 API，也不需要 key")
        # ⚠️ 关键：mock 结果必须禁用缓存——否则假答案会落进 data/cache，
        # 之后跑真实 API 时会命中同一缓存键，返回的还是假答案。
        mock_cache = Cache(resolve_path(cfg, "cache"), enabled=False)
        clients = [ModelClient(cfg, p, mock_cache, client=MockClient()) for p in providers]
    else:
        clients = [ModelClient(cfg, p, cache) for p in providers]

    # ---- 5. 主循环 ----
    outdir = resolve_path(cfg, "results")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outfile = Path(a.outfile) if a.outfile else outdir / f"smoke_{stamp}.jsonl"
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    # ⚠️ 2026-10-27 事故：写文件用的是 append，重跑同名文件会把新记录接在旧记录后面。
    #    结果 10 道题的文件里有 20 条，统计时同一题被算两次，且旧记录缺新字段 →
    #    看起来像"检索命中率只有 8/20"。默认改为覆盖（旧的先备份，不丢数据）；
    #    真要续跑请显式加 --append。
    if outfile.exists() and not a.append:
        bak = outfile.with_name(outfile.name + f".bak_{stamp}")
        outfile.rename(bak)
        log(f"⚠️  同名文件已存在，旧记录备份为 {bak.name}，本次从头写")

    rows = []
    for i, q in enumerate(questions, 1):
        for cl in clients:
            try:
                segs, dbg = retrieve(store, _embedder_if_needed(cfg, store), q["question"], cfg)
                rec = run_dual_path(cl, q["question"], segs, strategy=a.strategy)
                # ---- 2026-10-27 新增：把检索到的上下文落盘 ----
                # 为什么必须存：模型答"教材里没这条"时，无法区分是
                #   ① 教材被检索到了但它不听话 → PK-leakage（我们要测的）
                #   ② 教材压根没被检索到（检索失败）→ 与模型行为无关，混进统计会污染结论
                # 只有留下原文，事后才能人工核对、才能算"上下文命中率"。
                ctx_text = "\n".join(str(s.get("text", "")) for s in segs)
                ref = str(q.get("reference") or "")
                rec.update({
                    "qid": q["qid"], "question": q["question"], "reference": q.get("reference"),
                    "is_refusal": is_refusal_by_keywords(rec.get("ck_answer", ""), cfg),
                    "retrieval": dbg,
                    "context_text": ctx_text[:4000],
                    "context_has_ref": _norm_text(ref) in _norm_text(ctx_text) if len(_norm_text(ref)) >= 3 else None,
                    "question_conflicts_ck": q.get("question_conflicts_ck"),
                })
                rows.append(rec)
                log(f"[{i}/{len(questions)}] {cl.name} | 用 {dbg['n_segments_used']} 段 / "
                    f"{dbg['context_tokens_used']} token | CK: {rec.get('ck_answer','')[:60]}...")
                append_jsonl(outfile, rec)
            except Exception as e:
                log(f"[{i}] {cl.name} 失败：{e}")

    from src.evaluate import compute_conflict_metrics
    log("=" * 60)
    log(f"完成 {len(rows)} 条记录 → {outfile}")

    # ---- 6. 运行快照（可复现性的凭据，也是论文附录材料）----
    import json as _json
    from collections import Counter as _Counter
    meta = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "result_file": outfile.name,
        "mock": bool(a.mock),
        "demo": bool(use_demo),
        "strategy": a.strategy,
        "providers": {
            p: {"model": cfg["generation"]["providers"][p]["model"],
                "base_url": cfg["generation"]["providers"][p]["base_url"],
                "temperature": cfg["generation"].get("temperature")}
            for p in providers
        },
        "judge": cfg["evaluation"]["judge"],
        "sampling": {"mode": a.sample, "seed": a.seed, "min_per_stratum": a.min_per_stratum,
                     "n_requested": n, "n_done": len(rows),
                     "granularity": dict(_Counter(q.get("granularity", "unknown") for q in questions)),
                     "qids": [q["qid"] for q in questions]},
        "corpus": {"dir": str(corpus_dir), "docs": len(docs), "chunks": len(chunks)},
        "question_file": str(qfile),
        "index_dir": str(idx_dir),
        "chunking": cfg["chunking"],
        "embedding": cfg["embedding"]["model"],
        "retrieval": {"mode": cfg["retrieval"]["mode"],
                      "rrf_k": cfg["retrieval"]["rrf_k"],
                      "rerank": cfg["retrieval"]["rerank"]["enabled"],
                      "budget": cfg["retrieval"]["budget"]},
    }
    metafile = outfile.with_suffix(".meta.json")
    metafile.write_text(_json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"运行快照 → {metafile.name}")

    if not use_demo:
        log("指标预览（需先补齐四格标签，此处仅作链路验证）：")
        log(str(compute_conflict_metrics([r for r in rows if r.get("cell")])))
    log("=" * 60)
    log("下一步：打开上面的 jsonl，翻几条看模型实际回答，把体感问题记下来告诉我。")


_EMB = None


def _norm_text(s: str) -> str:
    """只留字母数字并小写，用于判断"教材值是否真的出现在检索到的上下文里"。
    不这么做的话，en dash / 半角连字符 / 空格差异（1918–1919 vs 1918-1919）
    会造成大量漏判，把"检索命中"误记成"没命中"。"""
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def _embedder_if_needed(cfg, store):
    """仅首次需要 Embedder 编码 query（很小，不必缓存）。"""
    global _EMB
    if _EMB is None:
        _EMB = Embedder(cfg["embedding"]["model"], cfg["embedding"]["device"],
                        cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"],
                        cfg["embedding"].get("query_instruction", ""),
                        local_files_only=cfg["embedding"].get("local_files_only", True))
    return _EMB


if __name__ == "__main__":
    main()
