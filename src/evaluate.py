"""
评估：四格标签 + 归因 + 指标。

四格（源自 #38 SABER）：
    pk_correct / ck_correct → C11 / C10 / C01 / C00
    ⚠️ 规范倒置：本研究的 ground truth 是【教材】，不是现实事实。

    ⚠️⚠️ 记号冲突（2026-10-26 修正，务必读懂再引用 SABER 的格子号）：
    SABER 原文里真值 = 现实事实，篡改上下文后「记忆对、上下文错」= C10，
    且它期望模型坚持记忆 → 在 SABER 里 C10 是【好】的。
    本研究真值 = 教材（含篡改版教材），于是同一个行为被算成：
        pk_correct（裸考 vs 教材篡改值）= False，ck_correct（跟教材）= True
        → 落进 C01，不是 C10。
    结论：
        - 篡改题上「跟教材」= CK-adherence → 格子 C01（本研究视为好）
        - 篡改题上「跟记忆」= PK-leakage  → 格子 C00（本研究视为失败）
        - 因此不要把本研究的 C10 与 SABER 的 C10 混用；论文里必须显式声明这一重命名。
    另注：C01 在本研究中恒为「好」（自然题=检索救回，篡改题=跟教材），语义自洽。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from .prompts import (ATTRIBUTION_JUDGE, CORRECTNESS_JUDGE, FLAG_JUDGE,
                      REFUSAL_JUDGE)


# ------------------------------------------------------------------ 标签
def label_cell(pk_correct: bool, ck_correct: bool) -> str:
    a, b = int(bool(pk_correct)), int(bool(ck_correct))
    return {(1, 1): "C11", (1, 0): "C10", (0, 1): "C01", (0, 0): "C00"}[(a, b)]


def is_refusal_by_keywords(text: str, cfg: Dict[str, Any]) -> bool:
    ev = cfg["evaluation"]
    low = text.lower()
    kws = ev.get("refusal_keywords_en", []) + ev.get("refusal_keywords_zh", [])
    return any(k.lower() in low for k in kws)


def _label_of(text: str) -> str:
    """取模型回的第一个词并大写。judge 被约束为只输出一个标签，但保险起见只认开头。"""
    t = text.strip().strip(".!\"'`").upper()
    return t.split()[0] if t else ""


def _must_label(text: str, allowed: tuple, what: str) -> str:
    """裁判回复必须落在允许集合内，否则抛错。

    ⚠️ 2026-10-27 事故（代价：22 题误判）：
       glm-5.3 旗舰默认开思考模式，max_tokens 全烧在推理链上 → content 返回空字符串。
       原代码 `_label_of("") == "CORRECT"` 得 False，等于**把"裁判没说话"当成"裁判说答错"**。
       结果 22 道答案与教材值一字不差的题被判 WRONG，C00 虚高、CK-adherence 被低估 24pp。
    原则：判分失败必须显式失败（该条不计入统计），绝不能默认算错——
      默认算错会让指标朝"看起来更严格"的方向系统性偏移，而且完全不可见。
    """
    lab = _label_of(text)
    if lab not in allowed:
        raise ValueError(f"{what}：裁判回复无法解析 {text[:60]!r}"
                         f"（空回复通常是思考模式吃满 max_tokens；检查 thinking_control）")
    return lab


def judge_refusal(client, answer: str) -> bool:
    r = client.chat("", REFUSAL_JUDGE.format(answer=answer[:1500]))
    return _must_label(r["texts"][0], ("REFUSAL", "ANSWER"), "拒答判定") == "REFUSAL"


def attribute(client, answer: str, ck_value: str, pk_value: str) -> str:
    """判定答案跟了教材（TEXTBOOK）、跟了记忆（PRIOR）、还是别的（NEITHER）。"""
    r = client.chat("", ATTRIBUTION_JUDGE.format(answer=answer[:1500],
                                                 ck_value=ck_value, pk_value=pk_value))
    lab = _must_label(r["texts"][0], ("TEXTBOOK", "PRIOR", "NEITHER"), "归因判定")
    return {"TEXTBOOK": "CK", "PRIOR": "PK", "NEITHER": "NEITHER"}[lab]


def _norm_value(s: str) -> str:
    """归一化：只留字母数字并小写。用于「答案里到底有没有出现参考答案」的包含判定。
    这样 1977–1978 / 1977-1978 / 1977–1978 with spaces 都能对上。"""
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


# 答案里出现这些短语，说明它在"指出矛盾 / 拒答 / 否认"，不是在陈述答案。
# 典型：教材被篡改成 1933–1934，模型答"摘录里没有 1968 年这次，表格写的是 1933–1934 配 H15N2，与题目不符"
# —— 它虽然含 H15N2，但语义是拒绝作答。规则若只做包含判定会把它误判成"答对"。
_NEGATION_MARKERS = (
    "does not", "do not", "doesn't", "don't", "cannot", "can't", "unable",
    "no information", "not mentioned", "not stated", "not specified",
    "no evidence", "no data", "not identify", "does not match", "inconsistent",
)


def _contains_value(answer: str, reference: str) -> Optional[bool]:
    """规则优先判定。返回 True=确定命中、None=判不了交给 LLM。

    ⚠️ 为什么要规则优先（2026-10-29 事故，另一类误判）：
       LLM 裁判自己也有一份世界知识。面对「教材值被篡改成违背现实」的题，
       它会按**现实对错**判，而不是按**与参考答案一致**判 —— 于是
       "Richard Altmann wrote Cellular Pathology"（与教材一致）被判 WRONG，
       因为裁判知道那是 Virchow 的。
       这是规范倒置在**裁判层面**的复现：真值定义改了，裁判的立场没跟着改。
    对短答案（年份/数值/人名）做字符串包含判定没有这个问题：
    它只问"答案里有没有这个值"，完全不关心这个值在现实里对不对。
    """
    ref = _norm_value(reference)
    ans = _norm_value(answer)
    if len(ref) < 3 or not ans:
        return None                      # 太短的值（如 "2"）易误匹配，交给 LLM
    if ref not in ans:
        return None                      # 不含 ≠ 判错（可能是改写表达），仍交 LLM
    low = str(answer).lower()
    if any(m in low for m in _NEGATION_MARKERS):
        return None                      # 含否定语气 → 可能在指出矛盾，交给 LLM 判
    return True


def is_correct(client, answer: str, reference: str) -> bool:
    # 第一道：规则判定（零成本、不受裁判世界知识影响）
    hit = _contains_value(answer, reference)
    if hit is True:
        return True
    # 第二道：LLM 裁判，提示词已明确要求它不要用现实知识
    r = client.chat("", CORRECTNESS_JUDGE.format(answer=answer[:1500], reference=reference))
    # ⚠️ 千万别写 `"CORRECT" in text.upper()` —— "INCORRECT" 里就含 "CORRECT"，
    #    会把判错的样本当成对的。只认开头那个标签词。
    return _must_label(r["texts"][0], ("CORRECT", "WRONG"), "正确性判定") == "CORRECT"


# ---------------------------------------------------------------- 是否提示异常
# ⚠️ 注意这里的短路方向与 is_correct **相反**，别照抄：
#   is_correct 是「规则命中确定正例 → 短路」（因为 LLM 会拿现实知识否定教材值）；
#   judge_flag  是「规则命中确定负例 → 短路」（因为 LLM 会把正常转折词当成举报）。
#   两个方向的失败模式不同，不能共用一套逻辑。
#
#   强标记：S5 指令里指定的格式开头，出现即说明它确实举手了 → 直接判 FLAGGED。
_FLAG_STRONG = (
    "note to instructor", "note for the instructor", "flag:", "flagging:",
    "i should flag", "i'd like to flag", "suspected error", "possible error",
    "this appears to be an error", "appears to be a typo", "should be verified",
    "instructor should verify", "worth verifying",
)
#   弱线索：答案里只要有任何"转折 / 质疑 / 矛盾"的措辞，就交 LLM 判，不短路。
#   只有**完全平铺直叙**的答案才会被规则直接判 SILENT —— 那种答案不可能举报了。
#   库做得宽是故意的：短路省一次 API，误判省不掉一次重跑。
_FLAG_WEAK = (
    "however", "but ", "although", "though", "note:", "note ", "note that",
    "conflict", "contradict", "inconsistent", "discrepan", "in reality",
    "in fact", "actually", "appears", "seems", "unusual", "surprising",
    "error", "typo", "mistake", "incorrect", "wrong", "verify", "caution",
    "warning", "suspicious", "doubt", "should be", "rather than", "instead of",
    "not the", "differs from", "differ from", "unlike", "on the other hand",
)


def _flag_by_rule(answer: str) -> Optional[bool]:
    """返回 True=确定 FLAGGED、False=确定 SILENT、None=判不了交给 LLM。"""
    low = str(answer or "").lower()
    if not low.strip():
        return False                                  # 没说话自然没举手
    if any(m in low for m in _FLAG_STRONG):
        return True
    if not any(m in low for m in _FLAG_WEAK):
        return False
    return None


def judge_flag(client, answer: str) -> bool:
    """模型有没有「举手」——指出教材材料本身有问题。

    这是 DD-09 信任校准的操作化指标：只判"答没答对"看不出模型是盲从还是警觉，
    必须再问一句"它有没有说材料有问题"。盲从（材料错了还照答不吭声）= 过度信任，
    与"不听教材"同属失校准，但方向相反，混在一个指标里会互相抵消、两边都看不出来。
    """
    hit = _flag_by_rule(answer)
    if hit is not None:
        return hit
    r = client.chat("", FLAG_JUDGE.format(answer=answer[:1500]))
    return _must_label(r["texts"][0], ("FLAGGED", "SILENT"), "提示异常判定") == "FLAGGED"


# ------------------------------------------------------------------ 指标
def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_conflict_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """DD-04 v2 的核心指标。

    records 每条需含：cell(期望四格) / is_refusal / attribution(CK|PK|NEITHER)

    指标：
      CK_adherence   = C10 格里答案遵循教材的比例            （本研究期望高）
      PK_leakage     = C10 格里答案回退到记忆的比例          （本研究期望低）
      OAcc           = C00 格里正确拒答的比例                （Zhou）
      OR             = C11 格里错误拒答（过度拒答）的比例     （Zhou）
      abstention_f1  = 拒答决策相对 C00 金标准的 F1          （#38）
      coverage / risk_on_answered / score                    （选择性预测 +1/-1/0）
    """
    def sub(cell: str) -> List[Dict[str, Any]]:
        return [r for r in records if r.get("cell") == cell]

    out: Dict[str, Any] = {"n_total": len(records)}
    for c in ("C11", "C10", "C01", "C00"):
        out[f"n_{c}"] = len(sub(c))

    # --- C10：篡改题归因 ---
    c10 = sub("C10")
    ans10 = [r for r in c10 if not r.get("is_refusal")]
    out["CK_adherence"] = round(_safe_div(sum(1 for r in ans10 if r.get("attribution") == "CK"), len(ans10)), 4)
    out["PK_leakage"]   = round(_safe_div(sum(1 for r in ans10 if r.get("attribution") == "PK"), len(ans10)), 4)
    out["unneeded_abstention_on_C10"] = round(_safe_div(sum(1 for r in c10 if r.get("is_refusal")), len(c10)), 4)

    # --- C00：超纲题 → 应当拒答 ---
    c00 = sub("C00")
    tp = sum(1 for r in c00 if r.get("is_refusal"))
    out["OAcc"] = round(_safe_div(tp, len(c00)), 4)
    gold_refuse = set(id(r) for r in c00)
    pred_refuse = set(id(r) for r in records if r.get("is_refusal"))
    inter = len(gold_refuse & pred_refuse)
    prec = _safe_div(inter, len(pred_refuse))
    rec = _safe_div(inter, len(gold_refuse))
    out["abstention_f1"] = round(_safe_div(2 * prec * rec, prec + rec), 4)
    out["abstention_precision"] = round(prec, 4)
    out["abstention_recall"] = round(rec, 4)

    # --- C11：教材和记忆都对 → 不应拒答 ---
    c11 = sub("C11")
    out["OR"] = round(_safe_div(sum(1 for r in c11 if r.get("is_refusal")), len(c11)), 4)

    # --- 选择性预测：+1 / -1 / 0 ---
    answered = [r for r in records if not r.get("is_refusal")]
    wrong = sum(1 for r in answered if r.get("attribution") != "CK")
    out["coverage"] = round(_safe_div(len(answered), len(records)), 4)
    out["risk_on_answered"] = round(_safe_div(wrong, len(answered)), 4)
    out["score"] = round(_safe_div((len(answered) - wrong) - wrong, len(records)), 4)

    # --- 成本侧（服务 SQ3）---
    pt = [r.get("usage", {}).get("prompt_tokens") or 0 for r in records]
    ct = [r.get("usage", {}).get("completion_tokens") or 0 for r in records]
    out["avg_prompt_tokens"] = round(sum(pt) / len(records), 1) if records else 0
    out["avg_completion_tokens"] = round(sum(ct) / len(records), 1) if records else 0
    return out


# ---------------------------------------------------------------- 信任校准
def trust_calibration_table(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """DD-09 的核心交叉表：**有没有举手 × 听了谁**。

    为什么单看 CK-adherence 或单看 flag_rate 都判不出校准：
      「沉默地照着错材料答」和「举手并照着答」在 CK-adherence 上是同一个数，
      但前者是过度信任（over-trust）、后者才是校准（calibrated）。
      两个指标各自都看不到这件事，只有交叉才看得到。

    四格（只在**材料确实异常**的题上有意义，正常题上分别对应误报）：

      flagged & CK    校准 calibrated          —— 照教材答，同时指出材料可疑      ← 应然
      silent & CK     过度信任 over-trust      —— 照着可疑材料答，一声不吭        ← 教学事故
      flagged & PK    不合作 non-cooperative   —— 指出问题但拒不给教材口径的答案
      silent & PK     不信任 under-trust       —— 直接抛开教材按记忆答            ← 本研究原以为会很多

    ⚠️ 「应然」依赖规范选择：本研究把教材定为课程内的真值，故 flagged & CK 才是理想。
       若换成「以现实事实为真值」的规范（SABER 那一套），应然格会变成 flagged & PK。
       论文里必须显式声明这个选择，并做敏感性分析（两种规范下各算一遍）。

    ⚠️ 缺字段的记录必须排除，不能算作沉默（2026-10-29 自查发现）：
       `bool(r.get("flagged"))` 对没判过的记录（None）返回 False，
       等于把"裁判没说话"当成"模型没举手"——这与 10-27 那次
       "把空回复当成判错"是同一种错误，只是方向相反，同样会系统性偏移指标。
       故此处只统计 flagged 明确为 True/False 的记录，并单独报出缺失条数。
    """
    recs = [r for r in records if isinstance(r.get("flagged"), bool)]
    n_missing = len(records) - len(recs)

    def flagged(r) -> bool:
        return r["flagged"] is True

    def is_ck(r) -> bool:
        return r.get("attribution") == "CK"

    def is_pk(r) -> bool:
        return r.get("attribution") == "PK"

    def is_off(r) -> bool:
        return r.get("attribution") == "NEITHER"

    def cnt(pred) -> int:
        return sum(1 for r in recs if pred(r))

    def both(a, b):
        def _p(r):
            return a(r) and b(r)
        return _p

    def _q(a, b):
        return {"n": cnt(both(a, b)),
                "pct": round(_safe_div(cnt(both(a, b)), n), 4)}

    n = len(recs)
    if not n:
        return {"n": 0}
    not_f = lambda r: not flagged(r)

    return {
        "n": n,
        "n_missing": n_missing,          # 未判出 flagged 的条数：非 0 就必须在论文里报出来
        "flag_rate": round(_safe_div(cnt(flagged), n), 4),
        "calibrated":  _q(flagged, is_ck),   # 举手 + 跟教材   ← 本研究应然
        "over_trust":  _q(not_f, is_ck),     # 沉默 + 跟教材
        "non_coop":    _q(flagged, is_pk),   # 举手 + 跟记忆
        "under_trust": _q(not_f, is_pk),     # 沉默 + 跟记忆
        "off_topic":   _q(lambda r: True, is_off),   # 拒答/跑题，与举手与否无关
        # 有条件提示率：只在「真的照教材答了」的题里，有多大比例顺带举了手。
        # 这个数比裸 flag_rate 更能说明问题——分母排除了那些压根没答的题。
        "flag_rate_given_ck": round(
            _safe_div(cnt(both(flagged, is_ck)), cnt(is_ck)), 4),
    }
