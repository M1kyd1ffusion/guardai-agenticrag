"""
GuardAI · 教材为准的教育答疑系统 —— 交互界面（Gradio）

设计要点（为什么这样排版）：
  1. **先给分诊，再给答案**：分诊 200ms 就出来，答案 3-8 秒。
     用户第一眼看到"这题我去哪本书查"，等待感立刻降低。
  2. **答案逐字流式输出**：首字 1 秒内出现，观感接近实时。
  3. **会诊态并排、不合并**：两本书各给一份答案，谁也不替学生选（PRD PR-1）。
  4. **证据可展开**：每个答案都能点开看它引用了哪几段原文，这是"可核查"。
  5. **置信度与耗时同步展示**：把工程数据摊在界面上，面试演示不用另做 PPT。

启动（在仓库根目录下）：
    # 先把三把 key 放进环境变量（Windows: $env:ZHIPU_API_KEY="..."）
    python app.py
    浏览器打开 http://127.0.0.1:7860
    # 首次运行前先建索引：python scripts/rebuild_index.py

不调判题：判题是离线评测用的，用户不该等它。
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import gradio as gr

from src.common import Cache, cache_key, load_config, resolve_path
from src.prompts import CK_USER, S1_SYSTEM, build_context
from src import present
from src import nli as nli_mod
from src.safety import LEVEL_NAME, SafetyGate

CFG_PATH = Path(__file__).resolve().parent / "config_product.yaml"
SUBJECT_CN = {"science": "自然科学", "literature": "文学", "history": "历史"}
MODE_CN = {
    "single": ("单源直答", "#2563eb", "在这本书里找到了明确依据"),
    "consult": ("多源会诊", "#d97706", "两本书都像，各取证据并列呈现，不替你选"),
    "none": ("超出教材范围", "#6b7280", "接进来的教材里没有可靠依据，不猜"),
}

_STATE: Dict[str, Any] = {}


# ------------------------------------------------------------------ 装配
def boot() -> Dict[str, Any]:
    if _STATE:
        return _STATE
    from src.generate import ModelClient
    from src.router import load_router

    cfg = load_config(str(CFG_PATH))
    _STATE["cfg"] = cfg
    _STATE["router"] = load_router(cfg)
    cache = Cache(resolve_path(cfg, "cache"), enabled=cfg["generation"].get("cache_enabled", True))
    _STATE["cache"] = cache
    _STATE["gen"] = ModelClient(cfg, "glm", cache)
    # 安全护栏：与生成端共用 client（glm-flash 便宜），但逻辑完全是独立一层。
    # 放在 boot 里只建一次 —— SafetyGate 内部有会话累积状态（SessionRisk）。
    _STATE["gate"] = SafetyGate(cfg, _STATE["gen"])
    return _STATE


# ------------------------------------------------------------------ 流式生成（带缓存）
def _params_for(client, system: str, user: str) -> Dict[str, Any]:
    """与 ModelClient.chat 完全同构的参数，保证缓存键一致（UI 命中的正是评测跑过的结果）。"""
    g = client.cfg["generation"]
    temp = client.pconf.get("temperature", g["temperature"])
    max_tok = client._max_tokens_override or g.get("max_output_tokens", 512)
    messages = ([{"role": "system", "content": system}] if system.strip() else [])
    messages = messages + [{"role": "user", "content": user}]
    p = {"model": client.model_id, "temperature": temp, "n": 1,
         "max_tokens": max_tok, "messages": messages}
    if client.pconf.get("reasoning_effort"):
        p["reasoning_effort"] = client.pconf["reasoning_effort"]
    return p


def stream_answer(client, cache: Cache, system: str, user: str) -> Iterator[str]:
    params = _params_for(client, system, user)
    key = cache_key(client.name, client.model_id, params)
    hit = cache.get(key)
    if hit is not None:
        yield hit["texts"][0]
        return
    text = ""
    try:
        stream = client.client.chat.completions.create(**{**params, "stream": True})
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None) or ""
            if piece:
                text += piece
                yield text
        cache.put(key, {"texts": [text], "usage": {}, "cached": False})
    except Exception as e:  # noqa: BLE001
        yield f"（生成失败：{e}）"


# ------------------------------------------------------------------ 卡片渲染
_CSS = """
<style>
.wrap { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
.card { border:1px solid #e5e7eb; border-left:4px solid #2563eb; border-radius:10px;
        padding:14px 16px; margin:10px 0; background:#fff; }
.card h4 { margin:0 0 8px 0; font-size:15px; }
.tag { display:inline-block; font-size:12px; padding:2px 8px; border-radius:999px;
       background:#f3f4f6; color:#374151; margin-right:6px; }
.ans { font-size:14.5px; line-height:1.75; color:#111827; white-space:pre-wrap; }
.muted { color:#6b7280; font-size:12.5px; }
.ev { background:#fafafa; border:1px solid #eee; border-radius:8px; padding:10px 12px;
      margin:8px 0; font-size:13px; line-height:1.7; color:#374151; }
.mono { font-family: ui-monospace, Consolas, monospace; font-size:12px; color:#6b7280; }
.warn { background:#fffbeb; border:1px solid #fde68a; color:#92400e;
        border-radius:8px; padding:10px 12px; font-size:13px; margin:8px 0; }
/* 呈现层（v1）：置信徽章 / 句级标注 / 兜底话术 */
.conf { display:inline-block; font-size:12px; padding:2px 9px; border-radius:999px;
        margin-left:6px; font-weight:600; }
.unsup { border-bottom:2px dotted #dc2626; cursor:help; }
.fb { background:#f9fafb; border:1px dashed #d1d5db; color:#374151; border-radius:8px;
      padding:10px 12px; font-size:13px; line-height:1.7; margin:8px 0; white-space:pre-wrap; }
.fb b { color:#111827; }
.notice { background:#eff6ff; border:1px solid #bfdbfe; color:#1e3a8a; border-radius:8px;
          padding:10px 12px; font-size:13px; line-height:1.7; }
</style>
"""


def _esc(t: str) -> str:
    return html.escape(t or "")


def render_route(d) -> str:
    name, color, sub = MODE_CN.get(d.mode, ("未知", "#6b7280", ""))
    prim = d.primary or "—"
    sec = "、".join(d.secondary) if d.secondary else ""
    books = prim + (f" ＋ {sec}" if sec else "")
    conf = int(max(0.0, min(1.0, d.confidence)) * 100)
    warn = ""
    if d.scope_suspect:
        warn = ('<div class="warn">⚠️ 这条证据偏弱（可能超出教材范围），'
                '建议对照课程指定教材核对后再采用。</div>')
    return f"""
<div class="wrap"><div class="card" style="border-left-color:{color}">
  <h4>🩺 分诊：<span style="color:{color}">{name}</span>
      <span class="tag">置信 {conf}%</span>
      <span class="tag">{'已锁定学科' if d.mode=='single' else sub}</span></h4>
  <div class="muted">查的教材：<b>{_esc(books)}</b></div>
  <div class="muted" style="margin-top:6px">{_esc(d.reason)}</div>
  {warn}
</div></div>"""


def _annotate_html(text: str, nli: Dict[str, Any] | None) -> str:
    """把"证据不支持"的句子标成红色虚线（只标注，不删不改）。

    ⚠️ 必须先转义再替换：直接对原文替换，答案里的 & < > 被 escape 后会改字符，
    句子就匹配不上了（标注静默失效，比报错更难发现）。
    """
    out = _esc(text)
    for s in (nli or {}).get("sentences", []):
        # 只标"真正无依据"的句子；"存疑"（词面对得上但 NLI 不认）不标，
        # 否则表格类证据会把大量正确答案标红（判官 B 的已知失效模式）。
        if s.get("meta") or (s.get("status") or "").lower() not in ("unsupported",):
            continue
        sent = _esc((s.get("sentence") or "").strip())
        if len(sent) < 15 or sent not in out:
            continue
        tip = f"教材未覆盖（支持度 {s.get('p_entail', 0):.2f}）"
        out = out.replace(sent, f'<span class="unsup" title="{tip}">{sent}</span>', 1)
    return out


def render_answers(items: List[Dict[str, Any]], done: bool) -> str:
    """items = [{book, text, segs, conf, nli, fb}]"""
    if not items:
        return ""
    multi = len(items) > 1
    out = ['<div class="wrap">']
    if multi:
        fb = present.fallback("ambiguous")
        out.append(f'<div class="fb"><b>{fb["title"]}</b>：{_esc(fb["text"])}</div>')
    for it in items:
        color = "#d97706" if multi else "#2563eb"
        badge = present.badge(it["conf"]) if it.get("conf") else ""
        out.append(f'''<div class="card" style="border-left-color:{color}">
  <h4>📖 {_esc(it["book"])}{badge}</h4>''')
        if it.get("fb"):
            out.append(f'<div class="fb"><b>{it["fb"]["title"]}</b>：{_esc(it["fb"]["text"])}</div>')
        body = _annotate_html(it.get("text") or "", it.get("nli"))
        out.append(f'<div class="ans">{body or "<span class=\'muted\'>生成中…</span>"}</div>')
        if it.get("conf"):
            out.append(f'<div class="muted" style="margin-top:6px">{_esc(it["conf"]["hint"])}'
                       f'　·　依据：{_esc("；".join(it["conf"]["reasons"])[:120])}</div>')
        segs = it.get("segs") or []
        if segs:
            rows = "".join(
                f'<div class="ev"><span class="mono">第 {s.get("order","?")} 段 · '
                f'相似度 {s.get("_score",0):.2f}</span><br>{_esc(s["text"][:420])}…</div>'
                for s in segs[:3])
            out.append(f'<details {"open" if not done else ""}><summary class="muted">'
                       f'查看引用原文（{len(segs)} 段）</summary>{rows}</details>')
        out.append("</div>")
    out.append("</div>")
    return "".join(out)


# ------------------------------------------------------------------ 主流程
def _safety_card(level: int, text: str, alert: Dict[str, Any] | None, ms: int) -> str:
    """危机卡片。配色按等级走：L3 红、L1/L2 琥珀。
    只承诺真的做到的事：没配 webhook 就不说"已通知老师"。"""
    color = "#dc2626" if level >= 3 else "#d97706"
    icon = "🛡️" if level >= 3 else "💬"
    title = "现在最重要的不是这道题" if level >= 3 else "先停一下"
    detail = ""
    if alert:
        if alert.get("sent"):
            detail = "已把这次对话同步给你的任课老师/辅导员。"
        else:
            detail = "（演示环境：未配置告警通道，实际部署会同步给老师）"
    body = "<br>".join(_esc(x) for x in text.split("\n"))
    return (f'<div class="wrap"><div class="card" style="border-left-color:{color}">'
            f'<h4>{icon} {title}</h4><div class="fb">{body}</div>'
            f'<div class="muted" style="margin-top:8px">{_esc(detail)}</div></div></div>')


def ask(question: str, budget: int, use_nli: bool = True,
        use_safety: bool = True) -> Iterator[Tuple[str, str, str]]:
    question = (question or "").strip()
    if not question:
        yield "<div class='muted'>请输入一个问题。</div>", "", ""
        return
    S = boot()
    router, cfg = S["router"], S["cfg"]
    cache, gen = S["cache"], S["gen"]

    t0 = time.time()

    # ---- 安全护栏：在分诊之前。L3 直接返回危机话术，不进检索、不调生成 ----
    act: Dict[str, Any] | None = None
    gate = S.get("gate")
    if use_safety and gate is not None:
        yield ("<div class='muted'>🛡️ 正在检查这条消息…</div>", "", "")
        try:
            act = gate.precheck(question, student_id="demo-student",
                               session_id=time.strftime("%Y%m%d%H%M%S"))
        except Exception as e:                      # noqa: BLE001
            # 护栏自己出错不能挡住答疑：降级为"不拦"，并把事实写进 meta。
            yield (f"<div class='muted'>护栏不可用（{type(e).__name__}），本次按普通问答处理</div>",
                   "", "")
            act = None
        if act and act.get("action") == "crisis":
            ms = int((time.time() - t0) * 1000)
            a = act.get("assess", {})
            yield (
                '<div class="wrap"><div class="card" style="border-left-color:#dc2626">'
                '<h4>🛡️ 已触发安全响应</h4>'
                f'<div class="fb">判定等级 <b>{LEVEL_NAME.get(act["effective_level"], "?")}</b>'
                f'（本轮 {LEVEL_NAME.get(a.get("level", 0), "?")}）<br>'
                f'触发原句：{_esc(a.get("evidence") or "（模型未给出原句）")}</div></div></div>',
                _safety_card(act["effective_level"], act["reply"],
                             act.get("alert_result"), ms),
                f'<div class="muted mono">安全检查 {ms} ms ｜ 已跳过检索与生成'
                f'（不消耗生成成本）｜ 判定模型 {gen.model_id}</div>',
            )
            return

    yield ("<div class='muted'>🩺 分诊中…（本地判定，通常 0.2 秒）</div>", "", "")
    d = router.triage(question)
    route_html = render_route(d)
    yield route_html, "<div class='muted'>📚 正在取证…</div>", ""

    budget_tokens = int(budget) * 100   # 滑杆单位是「×100 token」

    # ---- 失败态 A：库里没有 ----
    if d.mode == "none":
        fb = present.fallback("out_of_scope")
        yield route_html, (
            f'<div class="wrap"><div class="card" style="border-left-color:#6b7280">'
            f'<h4>🚫 {fb["title"]}</h4>'
            f'<div class="fb">{_esc(fb["text"])}</div></div></div>'), \
            f'<div class="muted">分诊耗时 {(time.time()-t0)*1000:.0f} ms · 未调用生成模型（省一次调用）</div>'
        return

    srcs = ([d.primary.split("/", 1)[1]] if d.primary else []) + \
           [s.split("/", 1)[1] for s in d.secondary]
    ev = router.evidence(question, srcs, budget_tokens)
    t_route = time.time() - t0

    items: List[Dict[str, Any]] = []
    for src in srcs:
        segs = ev.get(src, [])
        label = f"{SUBJECT_CN.get(S['router'].smap.subject_of.get(src,''), '')} · {src}"
        items.append({"book": label, "text": "", "segs": segs,
                      "conf": None, "nli": None, "fb": None})
    yield route_html, render_answers(items, done=False), ""

    # 逐源生成，逐字流式
    # L1/L2 的共情不是另插一段固定文案，而是让生成端在同一个 prompt 里带出来——
    # 固定文案接在教材答案前面读起来像机器人贴标签。
    sys_prompt = S1_SYSTEM
    if act and act.get("empathy"):
        sys_prompt = (S1_SYSTEM + "\n\nOpen your answer with exactly this sentence, "
                      f"verbatim, then answer the question normally: \"{act['empathy']}\"")
    for i, src in enumerate(srcs):
        segs = ev.get(src, [])
        prompt = CK_USER.format(context=build_context([s["text"] for s in segs]),
                               question=question)
        acc = ""
        for acc in stream_answer(gen, cache, sys_prompt, prompt):
            items[i]["text"] = acc
            yield route_html, render_answers(items, done=False), ""
        items[i]["text"] = acc

    # ---- 呈现层：句级核对 + 置信分级（答案已显示，这步不阻塞首字） ----
    ctx_tokens = sum(int(s.get("n_tokens", 0)) for segs in ev.values() for s in segs)
    nli_note = ""
    if use_nli and len(srcs) == 1:
        segs = ev.get(srcs[0], [])
        ctx = build_context([s["text"] for s in segs])
        res = nli_mod.check(items[0]["text"], ctx) if nli_mod.available() else None
        items[0]["nli"] = res
        if res:
            nli_note = (f' ｜ 逐句核对 {res["n_sent"]} 句：'
                        f'{res["n_unsupported"]} 句无依据'
                        + (f'、{res["n_uncertain"]} 句存疑' if res.get("n_uncertain") else ""))
    p = present.plan(d.mode, bool(d.scope_suspect), ctx_tokens, budget_tokens,
                     items[0].get("nli"))
    items[0]["conf"] = p["confidence"]
    items[0]["fb"] = p["fallback"]

    total = time.time() - t0
    safe_note = ""
    if act:
        a = act.get("assess", {})
        safe_note = (f' ｜ 安全判定 {LEVEL_NAME.get(act.get("effective_level", 0), "?")}'
                     f'（{a.get("latency_ms", 0)} ms）')
        if act.get("alert"):
            safe_note += " · 已留痕" + ("并告警" if (act.get("alert_result") or {}).get("sent")
                                       else "（未配告警通道）")
    meta = (f'<div class="muted mono">分诊 {t_route*1000:.0f} ms ｜ 总耗时 {total:.1f} s ｜ '
            f'上下文 {ctx_tokens} token ｜ 教材 {len(srcs)} 本 ｜ 模型 {gen.model_id}'
            f'{safe_note}{nli_note}</div>')
    yield route_html, render_answers(items, done=True), meta


# ------------------------------------------------------------------ 界面
EXAMPLES = [
    "What specific items did McClintock find among the Eskimos in 1859?",
    "What is the function of the bacterial cell wall?",
    "What advice did Mrs. March give to each of her daughters before she left?",
    "What reward does Captain Ahab promise to the first man who sights the White Whale?",
    "How do you compute the eigenvalues of a symmetric matrix?",
]


def build_ui() -> gr.Blocks:
    cfg = load_config(str(CFG_PATH))
    budget = int(cfg["product"]["budget_tokens"])
    # Gradio 6：css / theme 挪到 launch()，放在 Blocks() 里只报警告不生效
    with gr.Blocks(title="GuardAI · 教材为准的教育答疑系统") as demo:
        gr.HTML("""
<div class="wrap">
  <h2 style="margin:0 0 4px 0">GuardAI · 教材为准的教育答疑系统</h2>
  <div class="muted">先分诊，再检索，答案必须出自指定教材；两本书都像时并列呈现，不替你选；
  找不到依据就说找不到，不猜。已接入 <b>9 个源 / 3 个学科</b>（自然科学、文学、历史）。</div>
  <div class="notice" style="margin-top:10px">""" + present.first_use_notice(9) + """</div>
</div>""")
        with gr.Row():
            with gr.Column(scale=4):
                q = gr.Textbox(label="你的问题", placeholder="用英文提问（教材是英文的），例如：What is the function of the bacterial cell wall?",
                               lines=2)
            with gr.Column(scale=1):
                b = gr.Slider(8, 40, value=budget // 100, step=2,
                              label="上下文预算（×100 token）")
                use_nli = gr.Checkbox(value=True, label="逐句核对依据（本地 NLI，实验性：约 2–4 秒，标注有误报）")
                use_safety = gr.Checkbox(
                    value=True,
                    label="学生安全护栏（先判风险：L3 走危机响应、跳过检索；L1/L2 只加一句共情）")
                btn = gr.Button("提问", variant="primary")
        route = gr.HTML()
        answers = gr.HTML()
        meta = gr.HTML()
        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[q], label="示例（含一道超纲题，看兜底行为）")

        btn.click(ask, [q, b, use_nli, use_safety], [route, answers, meta], show_progress="hidden")
        q.submit(ask, [q, b, use_nli, use_safety], [route, answers, meta], show_progress="hidden")
        gr.HTML('<div class="muted" style="margin-top:12px">'
                '分诊与检索在本地运行（bge + BM25，无网络）；生成调用国产大模型；'
                '判题不进界面（离线评测用）；安全护栏只做信号上报，不替代学校心理支持体系。</div>')
    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True,
               css=_CSS,
               theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"))
