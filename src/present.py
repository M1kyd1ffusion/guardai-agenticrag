# -*- coding: utf-8 -*-
"""
呈现层（v1 收口件之一）：把"不确定性"变成用户看得见的东西

为什么需要这一层
----------------
文献（RGB / 信任校准）与本文实测都指向同一件事：用户对 RAG 的过度信任，
主要不是"答案错"，而是"答案看起来很确定、错得没有痕迹"。
生成端已经榨干（prompt 层被证伪四次），检索端也改不动信任——
唯一还没做、且零 API 成本的抓手就是**呈现**。

三个组件（对应 06 §15 的三段流程第 3 段）
------------------------------------------
1. 置信分级（高/中/低）—— 把一堆内部信号压成一个用户能懂的等级；
2. 三类兜底话术 —— 三种失败态各有一句人话，禁止一律"抱歉我不知道"；
3. 首用声明 —— 第一次就把能力边界说清（只回答这 9 本书）。

设计纪律
--------
- **零 API 成本**：全部是规则 + 模板，不调用任何模型；
- **可解释**：每一级置信都能说出"为什么"，不允许黑箱打分；
- **不改答案**：只加标注/话术，不截断、不拒答（判官 B 的结论：拦截是净负）。

用法
    from src.present import grade, fallback, badge, first_use_notice, annotate
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# 1. 三类兜底话术（失败态各一句人话）
# --------------------------------------------------------------------------
FALLBACK = {
    # A. 库里没有（分诊 none）
    "out_of_scope": (
        "这 9 本教材里没有找到能回答这个问题的内容。\n"
        "常见原因：问题超出教材范围（其他书里的情节、时事、生活常识，或教材没覆盖的主题）。\n"
        "建议：换成教材里的概念来问，或先确认这个知识点是否在课程指定教材里。"
    ),
    # B. 有但证据不足（suspect / 证据薄 / 判官 B 大面积不支持）
    "weak_evidence": (
        "我找到了一些相关段落，但证据不足以给出确定答案。\n"
        "下面附上原文片段，建议你对照原文自己判断；如果是作业或考试范围，请以老师给的材料为准。"
    ),
    # C. 问题含糊 / 多源说法不一致（会诊态）
    "ambiguous": (
        "这个问题可能有两种理解，或两本教材的说法不完全一致，所以下面并列呈现。\n"
        "系统不替你裁决：如果课程指定了某本教材，请以那本为准。"
    ),
}

# 话术的短标题（卡片头，避免用户只看正文）
FALLBACK_TITLE = {
    "out_of_scope": "教材里没有",
    "weak_evidence": "证据不足",
    "ambiguous": "两种可能",
}


def fallback(kind: str) -> Dict[str, str]:
    kind = kind if kind in FALLBACK else "weak_evidence"
    return {"kind": kind, "title": FALLBACK_TITLE[kind], "text": FALLBACK[kind]}


# --------------------------------------------------------------------------
# 2. 置信分级（规则透明，可解释）
# --------------------------------------------------------------------------
LEVEL_CN = {"high": "高", "mid": "中", "low": "低"}
LEVEL_COLOR = {"high": "#059669", "mid": "#d97706", "low": "#dc2626"}
LEVEL_HINT = {
    "high": "教材明确支持，可直接采用",
    "mid": "证据基本对得上，建议对照原文再确认",
    "low": "证据薄弱，仅供参考，不要直接引用",
}


def grade(mode: str,
          suspect: bool,
          ctx_tokens: int,
          budget_tokens: int,
          n_sent: Optional[int] = None,
          n_unsupported: Optional[int] = None,
          n_uncertain: int = 0,
          consult: bool = False) -> Dict[str, Any]:
    """把内部信号压成 高/中/低 三级。

    规则（写死在这里，改之前先想清楚为什么）：
      低：判官 B 不支持句占比 > 50%，或证据量 < 40% 预算
      中：suspect 标记 / 不支持句占比 20%–50% / 证据量 < 80% 预算 / 会诊态（并列不裁决）
      高：单源 + 不 suspect + 证据足 + 不支持句占比 ≤ 20%

    为什么会诊态最高只能给"中"：两本书并列表述时，系统本身没有裁决，
    "高置信"会误导用户以为系统已经选定了口径。
    """
    reasons: List[str] = []
    level = "high"

    def to(lv: str, why: str) -> None:
        nonlocal level
        order = {"low": 0, "mid": 1, "high": 2}
        if order[lv] < order[level]:
            level = lv
        reasons.append(why)

    if n_sent and n_unsupported is not None:
        # 两条规则分开用（不要合成一个"加权无依据"）：
        #   ① 无依据句（既不被 NLI 支持、词面也对不上）——这是**硬信号**，比例到 30% 就压到"低"；
        #   ② 存疑句（NLI 不支持但数字/专名在证据里，或句中无可核对词）——这是**软信号**，
        #      只有过半句子都存疑时才压到"中"。
        # 为什么分开：软信号占多数时（如短答案全被转述）不能把正确答案打成"低"，
        # 否则用户会认为系统坏了（端到端走查踩过）。
        r_uns = n_unsupported / max(1, n_sent)
        r_unc = n_uncertain / max(1, n_sent)
        if r_uns >= 0.3:
            to("low", f"判官B：{n_unsupported}/{n_sent} 句无依据")
        elif r_uns > 0:
            to("mid", f"判官B：{n_unsupported}/{n_sent} 句无依据")
        elif r_unc >= 0.5:
            to("mid", f"判官B：{n_uncertain}/{n_sent} 句未能核实")
        else:
            reasons.append(f"判官B：{n_sent - n_unsupported - n_uncertain}/{n_sent} 句有依据"
                           + (f"、{n_uncertain} 句存疑" if n_uncertain else ""))

    if budget_tokens > 0:
        fill = ctx_tokens / budget_tokens
        if fill < 0.4:
            to("low", f"证据量仅 {fill:.0%} 预算")
        elif fill < 0.8:
            to("mid", f"证据量 {fill:.0%} 预算")

    if suspect:
        to("mid", "检索信号偏弱（疑似超出教材范围）")
    if consult or mode == "consult":
        to("mid", "两本教材并列，系统未裁决")

    return {
        "level": level,
        "cn": LEVEL_CN[level],
        "color": LEVEL_COLOR[level],
        "hint": LEVEL_HINT[level],
        "reasons": reasons,
    }


def badge(g: Dict[str, Any]) -> str:
    """置信徽章（HTML）。"""
    return (f'<span class="conf" style="background:{g["color"]}1a;color:{g["color"]};'
            f'border:1px solid {g["color"]}55">置信：{g["cn"]}</span>')


# --------------------------------------------------------------------------
# 3. 首用声明（能力边界先讲清）
# --------------------------------------------------------------------------
def first_use_notice(n_books: int = 9) -> str:
    return (f"本助手只依据指定的 {n_books} 本教材回答（微生物 / 文学 / 历史），"
            "不回答教材之外的常识、时事与其他书籍内容；"
            "回答下方会标注置信度与原文依据，证据不足时会直接说明。")


# --------------------------------------------------------------------------
# 4. 句级标注（判官 B 的结果只标注、不拦截）
# --------------------------------------------------------------------------
def annotate(answer: str, sentences: List[Dict[str, Any]]) -> str:
    """把"证据不支持"的句子包一层虚线，鼠标悬停显示原因。

    实现注意：按句子原文做字符串替换，替换失败（模型输出被截断/端口改了标点）就原样返回，
    绝不因为标注失败而丢内容。
    """
    if not answer:
        return ""
    out = answer
    for s in sentences or []:
        if s.get("meta") or s.get("supported"):
            continue
        sent = (s.get("sentence") or "").strip()
        if len(sent) < 15 or sent not in out:
            continue
        tip = f'教材未覆盖（支持度 {s.get("p_entail", 0):.2f}）'
        out = out.replace(
            sent,
            f'<span class="unsup" title="{tip}">{sent}</span>', 1)
    return out


# --------------------------------------------------------------------------
# 5. 组装：一次问询的完整"呈现决策"
# --------------------------------------------------------------------------
def plan(mode: str,
         suspect: bool,
         ctx_tokens: int,
         budget_tokens: int,
         nli: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """给 app 层用的统一入口：返回该显示哪类话术、哪级置信。"""
    if mode == "none":
        return {"action": "refuse", "fallback": fallback("out_of_scope"),
                "confidence": None}
    n_unc = (nli or {}).get("n_uncertain", 0)
    if mode == "consult":
        g = grade(mode, suspect, ctx_tokens, budget_tokens,
                  (nli or {}).get("n_sent"), (nli or {}).get("n_unsupported"),
                  n_unc, consult=True)
        return {"action": "answer", "fallback": fallback("ambiguous"), "confidence": g}

    g = grade(mode, suspect, ctx_tokens, budget_tokens,
              (nli or {}).get("n_sent"), (nli or {}).get("n_unsupported"), n_unc)
    fb = fallback("weak_evidence") if g["level"] == "low" or suspect else None
    return {"action": "answer", "fallback": fb, "confidence": g}
