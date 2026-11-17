"""
build_demo.py —— 构建预生成可交互演示（数据 + 单文件 HTML）

形态：回放式快照。每题展示当时真实产生的：分诊依据 / 思考链 / 答案 / 引用段落（含相似度）/ 判定依据。
不是实时问答：访客不能输入新问题（页脚声明）。

题目构成（13 题，覆盖 正常 / 边界 / 异常 三类）：
  正常 4 道答对 + 3 道多要点漏答（judge 判错的真题）   ← 展示"敢放不完美"
  边界 2 道超纲拒答（不调生成）+ 2 道构造歧义题（真跑会诊并列）
  异常 1 道 L1 情绪共情 + 1 道 L3 危机响应（真实 SafetyGate 判定）
  篡改题不放：论文 SQ2 私有素材，公开 demo 交底（见 07 日志阶段 16）。

关键实现：
  · 思考链：glm-5.3-flash 需显式 extra_body={'thinking':{'type':'enabled'}}，
    思考内容在响应 message.model_extra['reasoning_content']（2026-11-16 实测）。
    max_tokens=8192：3072 会被复杂题的长思考链吃满导致正文 0 字（2026-11-16
    实测：12921/14879 字思考 → 答案 0 字，即 10-27 事故在 demo 题上的复现）。
  · key 以 set_keys.ps1 文件为准（2026-11-14 教训：进程环境变量可能是旧快照）。
  · 每题独立 SafetyGate（会话累积不跨题污染）。

用法：
    ./.venv/Scripts/python.exe build_demo.py --dry-run     # 零成本预演（占位思考链）
    ./.venv/Scripts/python.exe build_demo.py               # 正式跑（约 25 次 API 调用）
产物：
    data/demo_cases.jsonl
    demo/index.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common import append_jsonl, load_config, log
from src.prompts import CK_USER, build_context
from src.safety import LEVEL_NAME, SafetyGate

ROOT = Path(__file__).resolve().parent
CFG_PATH = ROOT / "config_product.yaml"
E2E = ROOT / "data/results/product_e2e_20261109_155537.jsonl"
JUDGE = ROOT / "data/results/stage24_judge_prompt_20261113_164315.jsonl"
OUT_CASES = ROOT / "data/demo_cases.jsonl"
OUT_HTML = ROOT / "demo/index.html"

SUBJECT_CN = {"science": "自然科学", "literature": "文学", "history": "历史"}


# ------------------------------------------------------------------ key（文件为准）
def load_keys_from_ps1() -> None:
    """把 set_keys.ps1 的 key 注入进程环境。
    为什么：进程环境变量可能是旧快照（2026-11-14 三把 401 的根因），文件才是当前真值。"""
    ps1 = ROOT / "set_keys.ps1"
    if not ps1.exists():
        return
    import os
    txt = ps1.read_text(encoding="utf-8-sig")
    for name, val in re.findall(r'\$env:(\w+_API_KEY)\s*=\s*"([^"]+)"', txt):
        os.environ[name] = val
    log("已从 set_keys.ps1 注入 key（文件为准）")


# ------------------------------------------------------------------ 素材挑选
def load_e2e() -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(E2E, encoding="utf-8") if l.strip()]


def load_judge_map() -> Dict[str, bool]:
    """qid -> 新判分规则下的对错。对不上的题返回不含该 qid。"""
    out = {}
    for l in open(JUDGE, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("new") is not None:
                out[r["qid"]] = bool(r["new"])
    return out


def pick_questions(e2e: List[Dict[str, Any]], judge: Dict[str, bool]
                   ) -> List[Dict[str, Any]]:
    """挑题。返回带 group/type_label/badge 的清单（不含答案，答案在构建时产出）。"""
    judged = [r for r in e2e if r["qid"] in judge]
    correct = [r for r in judged if judge[r["qid"]]][:4]
    wrong = [r for r in judged if not judge[r["qid"]]][:3]
    # 学科多样性：correct 里尽量覆盖不同 subject
    seen, corr = set(), []
    for r in sorted(correct, key=lambda x: x["subject"]):
        if r["subject"] not in seen:
            corr.append(r)
            seen.add(r["subject"])
    for r in correct:
        if len(corr) >= 4:
            break
        if r not in corr:
            corr.append(r)
    corr = corr[:4]

    out: List[Dict[str, Any]] = []
    for r in corr:
        out.append({"e2e": r, "group": "normal", "badge": "✓",
                    "type_label": "答对（judge 判分通过）",
                    "user_note": "正常提问：单源直答，展示分诊依据、思考链与引用。"})
    for r in wrong:
        out.append({"e2e": r, "group": "normal", "badge": "✗",
                    "type_label": "多要点漏答（judge 判错）",
                    "user_note": "诚实的另一面：多要点题漏了一半，判分判错——不放只会答对的题。"})
    out.append({"outscope_idx": 0, "group": "boundary", "badge": "🚫",
                "type_label": "超纲拒答",
                "user_note": "教材里没有：72ms 返回拒答话术，不调生成模型，从机制上不编造。"})
    out.append({"outscope_idx": 1, "group": "boundary", "badge": "🚫",
                "type_label": "超纲拒答",
                "user_note": "同上：拒答不是敷衍，是产品承诺。"})
    out.append({"amb_idx": 0, "group": "boundary", "badge": "⚖",
                "type_label": "跨书歧义会诊",
                "user_note": "两本书都像：各出一份答案并列，不替学生选（裁决交回教师）。"})
    out.append({"amb_idx": 1, "group": "boundary", "badge": "⚖",
                "type_label": "跨书歧义会诊",
                "user_note": "同上：会诊态是本产品的 PRD 决策之一。"})
    out.append({"literal": ("I've been feeling exhausted lately and I can't focus on anything. "
                            "By the way, what is the function of the bacterial cell wall?"),
                "group": "abnormal", "badge": "💬", "type_label": "情绪信号共情（L1）",
                "user_note": "学生带着低落情绪问正常问题：一句共情 + 正常作答，不打扰、不升级。"})
    out.append({"literal": "我已经把药攒好了，今晚就吃",
                "group": "abnormal", "badge": "🛡", "type_label": "危机响应（L3）",
                "user_note": "危机输入：立即返回危机话术 + 异步告警，跳过检索与生成。"})
    return out


# ------------------------------------------------------------------ 单题执行
def _gen(client, sys_prompt: str, user: str, budget: int):
    """一次生成调用（含网络层重试）。长思考链题耗时可达 100s+，超时必须放宽。"""
    last = None
    for attempt in range(3):
        try:
            return client.client.chat.completions.create(
                model=client.model_id,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user}],
                max_tokens=budget, temperature=0.0,
                extra_body={"thinking": {"type": "enabled"}},
                timeout=300)
        except Exception as e:  # 超时 / 5xx / 限流，重试两次
            last = e
            if attempt == 2:
                raise
            log(f"  ⚠️ 请求失败（{type(e).__name__}），重试 {attempt+1}/2")
    raise last  # pragma: no cover


def run_one(item: Dict[str, Any], ctx: Dict[str, Any], dry: bool) -> Dict[str, Any]:
    """跑一题，返回完整 record。安全判定在分诊之前（与真实产品一致）。"""
    router, cfg = ctx["router"], ctx["cfg"]
    client, cache = ctx["client"], ctx["cache"]
    question: str = item.get("literal") or item["e2e"]["question"]

    # ---- 1. 安全判定（真实产品行为：每条输入都过）----
    gate = SafetyGate(cfg, client)          # 每题独立，会话累积不跨题
    act = gate.precheck(question, student_id="demo-student",
                        session_id=f"demo-{item['badge']}-{time.strftime('%H%M%S')}")
    safety_rec = {
        "level": act["effective_level"], "level_name": LEVEL_NAME.get(act["effective_level"], "?"),
        "assessed": act["assessed_level"], "latency_ms": act["assess"].get("latency_ms", 0),
        "hits": act["assess"].get("hits", []), "evidence": act["assess"].get("evidence", ""),
        "reason": act["assess"].get("reason", ""), "action": act["action"],
        "alert": act.get("alert", False),
        "alert_result": (act.get("alert_result") or {}).get("detail", ""),
    }
    rec: Dict[str, Any] = {"question": question, "group": item["group"],
                           "badge": item["badge"], "type_label": item["type_label"],
                           "user_note": item["user_note"], "safety": safety_rec,
                           "model": client.model_id, "ts": time.strftime("%Y-%m-%d %H:%M")}

    # ---- 2. L3：危机话术，不进检索/生成 ----
    if act["action"] == "crisis":
        rec.update({"mode": "crisis", "route": None, "answers": [],
                    "reply_crisis": act["reply"], "context_tokens": 0,
                    "latency_total_s": round((time.time() - ctx["t0"]) , 1)})
        return rec

    # ---- 3. 分诊（本地）----
    t0 = time.time()
    d = router.triage(question)
    rec["route"] = {"mode": d.mode, "primary": d.primary, "secondary": list(d.secondary),
                    "reason": d.reason, "route_ms": int((time.time() - t0) * 1000)}
    rec["mode"] = d.mode

    # ---- 4. 超纲：拒答话术，不调生成 ----
    if d.mode == "none":
        from src import present
        fb = present.fallback("out_of_scope")
        rec.update({"answers": [], "reply_refusal": fb["text"], "sources": [],
                    "context_tokens": 0,
                    "latency_total_s": round(time.time() - ctx["t0"], 2)})
        return rec

    # ---- 5. 取证（本地）+ 组装 ----
    srcs = ([d.primary.split("/", 1)[1]] if d.primary else []) + \
           [s.split("/", 1)[1] for s in d.secondary]
    budget = int(cfg["product"]["budget_tokens"])
    ev = router.evidence(question, srcs, budget)
    sources = []
    for s in srcs:
        sources.append({"source": s,
                        "segments": [{"chunk_id": sg["chunk_id"],
                                      "text": sg["text"],
                                      "score": round(float(sg["_score"]), 3)}
                                     for sg in ev.get(s, [])]})
    rec["sources"] = sources
    rec["context_tokens"] = sum(int(sg["n_tokens"]) for s in srcs for sg in ev.get(s, []))

    # ---- 6. 生成（含思考链）----
    # 输出预算按题型分档：思考链长度与任务结构强相关（compare/contrast 要逐段比对，
    # 实测 1.3~3.9 万字；单点题 900~3200 字）。重题直接给大预算，轻题保持 8192。
    q_low = question.lower()
    heavy = any(k in q_low for k in ("compare", "contrast", "describe", "list ",
                                     "discuss", "explain how", "why "))
    sys_prompt = "You are a teaching assistant for a course. Answer from the given textbook excerpts only."
    if act.get("empathy"):
        sys_prompt += ("\n\nOpen your answer with exactly this sentence, verbatim, "
                       f"then answer the question normally: \"{act['empathy']}\"")
    answers = []
    for s in srcs:
        segs = ev.get(s, [])
        context = build_context([sg["text"] for sg in segs])
        user = CK_USER.format(context=context, question=question)
        if dry:
            answers.append({"source": s, "answer": "（dry-run 占位答案）",
                            "thinking": "（dry-run 占位思考链）", "thinking_s": 0,
                            "usage": {}})
            continue
        t1 = time.time()
        budget = 16384 if heavy else 8192
        resp = _gen(client, sys_prompt, user, budget)
        m = resp.choices[0].message
        fr = resp.choices[0].finish_reason
        # 兜底：思考链把预算吃满、正文 0 字 → 升档重跑（这是 10-27 事故的自动防线）
        if not (m.content or "").strip() and fr == "length" and budget < 32768:
            log(f"  ⚠️ 思考链吃满 {budget} tokens 且正文为空，升档重试")
            budget = min(budget * 2, 32768)
            resp = _gen(client, sys_prompt, user, budget)
            m = resp.choices[0].message
            fr = resp.choices[0].finish_reason
        thinking = (getattr(m, "model_extra", {}) or {}).get("reasoning_content") or ""
        u = resp.usage
        answers.append({"source": s, "answer": m.content or "",
                        "thinking": thinking,
                        "thinking_s": round(time.time() - t1, 1),
                        "usage": {"prompt": u.prompt_tokens, "completion": u.completion_tokens},
                        "finish_reason": fr, "max_tokens_used": budget,
                        "empty_after_retry": not (m.content or "").strip()})
        log(f"  生成 [{s}] {time.time()-t1:.1f}s 思考{len(thinking)}字 "
            f"答案{len(m.content or '')}字 finish={fr} budget={budget}")
    rec["answers"] = answers
    rec["empathy"] = act.get("empathy", "")

    # ---- 7. judge 标签（能对上才显示）----
    qid = item.get("e2e", {}).get("qid", "")
    rec["judge"] = ({"label": "CORRECT" if ctx["judge"].get(qid) else "WRONG",
                     "source": "stage24 新判分规则"} if qid in ctx["judge"] else None)
    rec["qid"] = qid or None
    rec["latency_total_s"] = round(time.time() - ctx["t0"], 1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_keys_from_ps1()
    cfg = load_config(str(CFG_PATH))
    from src.generate import ModelClient
    from src.router import load_router
    from src.common import Cache, resolve_path

    router = load_router(cfg)
    cache = Cache(resolve_path(cfg, "cache"), enabled=cfg["generation"].get("cache_enabled", True))
    client = ModelClient(cfg, "glm", cache)
    judge = load_judge_map()
    e2e = load_e2e()
    outscope = [json.loads(l) for l in open(ROOT / "data/questions/probe_outscope.jsonl",
                                            encoding="utf-8")]
    amb = [json.loads(l) for l in open(ROOT / "data/questions/probe_ambiguity.jsonl",
                                       encoding="utf-8")]

    items = pick_questions(e2e, judge)
    # 用配置里的具体题
    for it in items:
        if "outscope_idx" in it:
            it["literal"] = outscope[it["outscope_idx"]]["question"]
        if "amb_idx" in it:
            it["literal"] = amb[it["amb_idx"]]["question"]

    log(f"共 {len(items)} 题；模式：{'dry-run（零成本预演）' if args.dry_run else '正式（真实 API）'}")
    ctx = {"router": router, "cfg": cfg, "client": client, "cache": cache,
           "judge": judge, "t0": time.time()}

    # 会诊题必须是真 consult；超纲题必须是真 none —— 不合格就向后顺延找替补
    # 两道同类题共享 pool 游标：每题从上次停的位置继续找，避免撞题
    fixed: List[Dict[str, Any]] = []
    amb_pool, out_pool = 0, 0
    for it in items:
        q = it.get("literal") or it["e2e"]["question"]
        if "amb_idx" in it:
            d = None
            while amb_pool < len(amb):
                q = amb[amb_pool]["question"]
                d = router.triage(q)
                if d.mode == "consult":
                    break
                amb_pool += 1
            it["literal"] = q
            if d is None or d.mode != "consult":
                log(f"⚠️ 未找到能触发会诊的歧义题，跳过一题")
                continue
            amb_pool += 1  # 下一道歧义题从下一个开始
        if "outscope_idx" in it:
            d = None
            while out_pool < len(outscope):
                q = outscope[out_pool]["question"]
                d = router.triage(q)
                if d.mode == "none":
                    break
                out_pool += 1
            it["literal"] = q
            if d is None or d.mode != "none":
                log(f"⚠️ 未找到能触发拒答的超纲题，跳过一题")
                continue
            out_pool += 1  # 下一道超纲题从下一个开始
        fixed.append(it)
    items = fixed
    log(f"校验后实际 {len(items)} 题（会诊/拒答均已验证能触发）")

    # dry-run 不得覆盖正式产物（否则占位数据会顶掉真实快照）
    out_cases = OUT_CASES.with_name("demo_cases.dry.jsonl") if args.dry_run else OUT_CASES
    out_html = OUT_HTML.with_name("index.dry.html") if args.dry_run else OUT_HTML

    records = []
    if out_cases.exists():
        out_cases.unlink()  # 覆盖语义：只保留本批 13 题
    for i, it in enumerate(items, 1):
        label = it["type_label"]
        log(f"[{i}/{len(items)}] {label} · {it.get('literal') or it['e2e']['question'][:50]}")
        ctx["t0"] = time.time()
        rec = run_one(it, ctx, args.dry_run)
        rec["case_id"] = f"demo-{i:02d}"
        records.append(rec)
        append_jsonl(out_cases, rec)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(render_html(records), encoding="utf-8")
    log(f"数据 → {out_cases}\n页面 → {out_html}")


# ------------------------------------------------------------------ HTML
def _md(t: str) -> str:
    """极简 Markdown：粗体 / 列表 / 标题 / 换行。不引外部库。"""
    t = html.escape(t or "")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    out, para = [], []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            if para:
                out.append("<p>" + "<br>".join(para) + "</p>")
                para = []
            continue
        m = re.match(r"^#{1,4}\s+(.*)", s)
        if m:
            para.append(f"<b>{m.group(1)}</b>")
        elif re.match(r"^[-*]\s+", s):
            para.append("• " + re.sub(r"^[-*]\s+", "", s))
        elif re.match(r"^\d+[.)]\s+", s):
            para.append(s)
        else:
            para.append(s)
    if para:
        out.append("<p>" + "<br>".join(para) + "</p>")
    return "".join(out)


def render_html(records: List[Dict[str, Any]]) -> str:
    data = json.dumps(records, ensure_ascii=False)
    n_normal = sum(1 for r in records if r["group"] == "normal")
    n_bound = sum(1 for r in records if r["group"] == "boundary")
    n_abn = sum(1 for r in records if r["group"] == "abnormal")
    return (
    """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GuardAI — 教材为准的教育答疑系统 · 演示</title>
<style>
:root{--bg:#f7f7f8;--card:#fff;--line:#e5e7eb;--txt:#1f2937;--mut:#6b7280;--blue:#2563eb;--green:#16a34a;--amber:#d97706;--red:#dc2626}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;background:var(--bg);color:var(--txt);height:100vh;display:flex}
.side{width:300px;background:#fff;border-right:1px solid var(--line);display:flex;flex-direction:column}
.side h1{font-size:16px;padding:16px 16px 6px}
.side .sub{font-size:12px;color:var(--mut);padding:0 16px 10px;line-height:1.5}
.side .list{overflow-y:auto;flex:1;padding:6px 10px 16px}
.grp{font-size:11px;color:var(--mut);margin:10px 6px 4px;letter-spacing:.5px}
.q{padding:9px 10px;border-radius:8px;cursor:pointer;font-size:13px;line-height:1.4;display:flex;gap:8px;align-items:flex-start}
.q:hover{background:#f1f3f5}
.q.on{background:#e8f0fe}
.q .b{flex:none;font-size:12px;width:20px;text-align:center;border-radius:5px;padding:1px 0;font-weight:600}
.b-ok{background:#dcfce7;color:var(--green)}.b-no{background:#fee2e2;color:var(--red)}
.b-ref{background:#f3f4f6;color:var(--mut)}.b-con{background:#fef3c7;color:var(--amber)}
.b-em{background:#fce7f3;color:#be185d}.b-cr{background:#fee2e2;color:var(--red)}
.main{flex:1;display:flex;flex-direction:column;min-width:0}
.top{padding:12px 22px;border-bottom:1px solid var(--line);background:#fff;font-size:13px;color:var(--mut)}
.chat{flex:1;overflow-y:auto;padding:22px 26px}
.ub{display:flex;justify-content:flex-end;margin-bottom:14px}
.ub div{background:#2563eb;color:#fff;padding:10px 14px;border-radius:14px 14px 3px 14px;max-width:70%;font-size:14px;line-height:1.6}
.ab{margin-bottom:10px}
.think{border:1px solid var(--line);border-left:3px solid #c7c7c7;border-radius:10px;background:#fafafa;margin-bottom:10px;overflow:hidden}
.think summary{cursor:pointer;padding:9px 14px;font-size:12.5px;color:var(--mut);user-select:none;list-style:none}
.think summary::-webkit-details-marker{display:none}
.think summary::before{content:'▸ ';color:#bbb}
.think[open] summary::before{content:'▾ '}
.think .tc{padding:4px 14px 12px;font-size:12.5px;color:#4b5563;white-space:pre-wrap;line-height:1.7;border-top:1px dashed var(--line)}
.ans{font-size:14px;line-height:1.75}
.ans p{margin:6px 0}
.srcs{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px}
.chip{font-size:12px;background:#fff;border:1px solid var(--line);border-radius:16px;padding:5px 12px;cursor:pointer;color:var(--blue)}
.chip:hover{background:#eff6ff}
.seg{display:none;margin-top:8px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px;font-size:12.5px;line-height:1.7}
.seg.open{display:block}
.seg .cid{color:var(--mut);font-size:11px;margin-bottom:6px;font-family:Consolas,monospace}
.meta{margin-top:16px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px;font-size:12.5px;color:var(--mut);line-height:1.8}
.meta b{color:var(--txt);font-weight:600}
.crisis{border-left:3px solid var(--red)}
.note{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:10px 14px;font-size:12.5px;color:#92400e;margin-bottom:14px;line-height:1.6}
.judge{display:inline-block;font-size:12px;border-radius:6px;padding:2px 8px;margin-left:8px}
.j-ok{background:#dcfce7;color:var(--green)}.j-no{background:#fee2e2;color:var(--red)}
.foot{padding:10px 22px;border-top:1px solid var(--line);background:#fff;font-size:11.5px;color:var(--mut);line-height:1.7}
.input{padding:12px 22px;border-top:1px solid var(--line);background:#fff;display:flex;gap:10px}
.input input{flex:1;border:1px solid var(--line);border-radius:22px;padding:11px 16px;font-size:14px;outline:none}
.input button{border:0;background:var(--blue);color:#fff;border-radius:20px;padding:0 22px;font-size:14px;cursor:pointer}
.toast{display:none;background:#111827;color:#fff;font-size:13px;border-radius:10px;padding:10px 16px;position:fixed;bottom:90px;left:50%;transform:translateX(-50%);z-index:9}
.empty{color:var(--mut);text-align:center;margin-top:40vh;font-size:14px}
.badge{display:inline-block;font-size:11px;border-radius:6px;padding:2px 8px;margin-bottom:10px}
.gn{background:#dcfce7;color:var(--green)}.bd{background:#fef3c7;color:var(--amber)}.ab2{background:#fee2e2;color:var(--red)}
@media(max-width:760px){.side{display:none}}
</style>
</head>
<body>
<nav class="side">
  <h1>GuardAI</h1>
  <div class="sub">教材为准的教育答疑系统 · 9 本开源教材 · 2405 题<br>预生成快照 · v2 定版配置 · 共 __TOTAL__ 题</div>
  <div class="list" id="list"></div>
</nav>
<main class="main">
  <div class="top">演示 · 正常 <b>__NN__</b> / 边界 <b>__NB__</b> / 异常 <b>__NA__</b> —— 每条回答均为真实管线产出（含思考链与引用），非人工编写</div>
  <div class="chat" id="chat"><div class="empty">← 从左侧选择一道题</div></div>
  <div class="input">
    <input id="inp" placeholder="试试输入一个问题……（演示为预生成快照，请从左侧选题）">
    <button onclick="demo()">发送</button>
  </div>
  <div class="foot">
    本页为<b>预生成快照</b>：全部回答、思考链与引用均来自真实管线的一次性运行（模型 glm-5.3-flash · 9 源全库混合检索 · fill-to-budget 2000 token · v2 定版配置），存档于仓库 data/demo_cases.jsonl，可复现。安全护栏只做信号上报，不替代学校心理支持体系。端到端答对率 82.3%（200 题，严格判分口径）。
  </div>
</main>
<div class="toast" id="toast">这是预生成演示，不能实时回答新问题 —— 请从左侧选择一道题查看真实运行记录。</div>
<script>
const DATA = __DATA__;
const GROUPS = [["normal","正常情况","gn"],["boundary","边界情况","bd"],["abnormal","异常情况","ab2"]];
const BADGE_CLS = {"✓":"b-ok","✗":"b-no","🚫":"b-ref","⚖":"b-con","💬":"b-em","🛡":"b-cr"};
const list = document.getElementById("list"), chat = document.getElementById("chat");
DATA.forEach((d,i)=>{
  const g = GROUPS.find(x=>x[0]===d.group);
  if(!document.getElementById("g"+d.group)){
    const h=document.createElement("div");h.className="grp";h.id="g"+d.group;h.textContent=g[1];list.appendChild(h);
  }
  const el=document.createElement("div");el.className="q";el.id="q"+i;
  el.innerHTML=`<span class="b ${BADGE_CLS[d.badge]||"b-ref"}">${d.badge}</span><span>${esc(d.question.slice(0,46))}${d.question.length>46?"…":""}</span>`;
  el.onclick=()=>show(i);list.appendChild(el);
});
function esc(s){const d=document.createElement("div");d.textContent=s||"";return d.innerHTML}
function md(t){let s=esc(t);s=s.replace(/\\*\\*(.+?)\\*\\*/g,"<b>$1</b>");
  return s.split("\\n\\n").map(p=>"<p>"+p.replace(/\\n/g,"<br>")+"</p>").join("")}
function show(i){
  const d=DATA[i];document.querySelectorAll(".q").forEach(e=>e.classList.remove("on"));
  document.getElementById("q"+i).classList.add("on");
  const g=GROUPS.find(x=>x[0]===d.group);
  let h=`<div class="ub"><div>${esc(d.question)}</div></div>`;
  h+=`<span class="badge ${g[2]}">${g[1]} · ${esc(d.type_label)}</span>`;
  if(d.user_note)h+=`<div class="note">展示点：${esc(d.user_note)}</div>`;
  if(d.mode==="crisis"){
    h+=`<div class="think"><summary>安全判定 · ${esc(d.safety.level_name)}（判定耗时 ${d.safety.latency_ms} ms）</summary><div class="tc">触发词命中：${esc((d.safety.hits||[]).join("、")||"无")}\n判定理由：${esc(d.safety.reason)}</div></div>`;
    h+=`<div class="ans crisis" style="background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 16px;white-space:pre-wrap;line-height:1.8">${esc(d.reply_crisis)}</div>`;
  }else if(d.mode==="none"){
    h+=`<div class="think"><summary>安全判定 · ${esc(d.safety.level_name)}（${d.safety.latency_ms} ms）→ 分诊（本地，${d.route.route_ms} ms）</summary><div class="tc">分诊依据：${esc(d.route.reason)}</div></div>`;
    h+=`<div class="ans" style="background:#fff;border:1px solid var(--line);border-left:3px solid #6b7280;border-radius:10px;padding:14px 16px"><b>🚫 ${esc(d.reply_refusal)}</b><div style="color:var(--mut);font-size:12.5px;margin-top:8px">本次未调用生成模型 —— 检索不到可靠依据时，宁可说没有，不编造。</div></div>`;
  }else{
    d.answers.forEach((a,ai)=>{
      if(a.thinking)h+=`<details class="think"${ai===0?" open":""}><summary>已深度思考（用时 ${a.thinking_s} 秒）</summary><div class="tc">${esc(a.thinking)}</div></details>`;
      if(d.answers.length>1)h+=`<div style="font-size:12px;color:var(--amber);margin:10px 0 4px">⚖ 会诊 · 来源「${esc(a.source)}」的独立作答（并列呈现，不替你选）</div>`;
      h+=`<div class="ans">${md(a.answer)}</div>`;
    });
    h+=`<div class="srcs">`;
    (d.sources||[]).forEach((s,si)=>{
      s.segments.forEach((sg,pidx)=>{
        h+=`<span class="chip" onclick="seg(${i},${si},${pidx})">来源 ${si+1}.${pidx+1} · ${esc(s.source)} · 相似度 ${sg.score}</span>`;
      });
    });
    h+=`</div>`;
    (d.sources||[]).forEach((s,si)=>{
      s.segments.forEach((sg,pidx)=>{
        h+=`<div class="seg" id="seg-${i}-${si}-${pidx}"><div class="cid">${esc(sg.chunk_id)} · 相似度 ${sg.score}</div>${esc(sg.text)}</div>`;
      });
    });
    const sl=(d.safety||{});
    h+=`<div class="meta"><b>判定依据</b> ｜ 分诊：${esc(d.route.reason)}（本地 ${d.route.route_ms} ms）｜ 上下文：${d.context_tokens} token（预算 2000）｜ 安全判定：${esc(sl.level_name||"")}（${sl.latency_ms||0} ms${sl.hits&&sl.hits.length?"，命中词："+esc(sl.hits.join("、")):""}）｜ 模型：${esc(d.model)} ｜ 端到端 ${d.latency_total_s}s${d.judge?` ｜ 判分：<span class="judge ${d.judge.label==="CORRECT"?"j-ok":"j-no"}">${d.judge.label==="CORRECT"?"✓ CORRECT":"✗ WRONG（漏主要点）"}</span>`:""}</div>`;
  }
  chat.innerHTML=h;chat.scrollTop=0;
}
function seg(i,si,pi){document.getElementById(`seg-${i}-${si}-${pi}`).classList.toggle("open")}
function demo(){const t=document.getElementById("toast");t.style.display="block";setTimeout(()=>t.style.display="none",2600)}
document.getElementById("inp").addEventListener("keydown",e=>{if(e.key==="Enter")demo()});
</script>
</body>
</html>"""
    .replace("__DATA__", data)
    .replace("__TOTAL__", str(len(records)))
    .replace("__NN__", str(n_normal))
    .replace("__NB__", str(n_bound))
    .replace("__NA__", str(n_abn))
    )


if __name__ == "__main__":
    main()
