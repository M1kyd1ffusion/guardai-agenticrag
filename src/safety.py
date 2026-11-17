"""
安全护栏：输入侧风险分级 + 会话累积 + 危机分流 + 告警 + 输出侧复核。

为什么单独成模块（而不是塞进 router 或 generate）：
  风险与意图**正交**。学生可以一边问"细菌细胞壁的成分"，一边说"反正我考不过了"。
  塞进意图分类器 = 逼模型二选一，两个都做不好。所以它是独立的一层判定，
  在意图分诊之前执行，只回答一个问题：**这个人现在安全吗。**

五条设计原则（每条都对应一个具体的失败模式）：
  1. 词表只作"提级信号"，绝不作为放行闸门。隐喻永远识别不全，"我不想活了"
     可能一句关键词都不出现；反过来"考完想死"是正常口语。所以关键词只当提示。
  2. 解析失败按 L1 处理，不按 L0。模型返回格式错乱，恰恰是最该保守的时刻。
  3. 只有 L3 升级人工；L1/L2 只留痕。误报的代价不对称，但"宁可多报"也不成立——
     一句"这题好难我要死了"就推给辅导员，学生会立刻觉得被监视，信任当场碎掉。
  4. 告警永不阻塞学生：超时 3 秒、异常吞掉、始终落盘。告警服务挂了不能让学生收不到回复。
  5. 输出侧复核失败按放行处理（fail-open）。它只是额外一道，不是唯一一道；
     输入侧已经处理了"人怎么了"，这里再因为服务抖动卡掉正常答案就是负收益。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .common import Timer, append_jsonl, log
from .prompts import (OUTPUT_SAFETY_SYSTEM, OUTPUT_SAFETY_USER_TMPL,
                      RISK_HINT_TMPL, RISK_SYSTEM, RISK_USER_TMPL)

# ------------------------------------------------------------------ 等级
LEVEL_NONE, LEVEL_DISTRESS, LEVEL_IDEATION, LEVEL_IMMINENT = 0, 1, 2, 3
LEVEL_NAME = {0: "L0 无信号", 1: "L1 低落", 2: "L2 意念", 3: "L3 紧急"}

# ------------------------------------------------------------------ 词表快筛
# 只做"提级提示"，不决定分级。命中 ≠ 风险，未命中 ≠ 安全。
KW_SELF_HARM = [
    "自杀", "自残", "割腕", "跳楼", "跳河", "上吊", "结束生命", "了结自己",
    "不想活", "不想活了", "活不下去", "活着没意思", "死了算了", "想死", "去死",
    "安眠药", "遗书", "告别", "最后一次", "解脱了",
    "suicide", "kill myself", "end my life", "want to die", "better off dead",
    "self-harm", "self harm", "cut myself", "overdose", "no reason to live",
]
KW_VIOLENCE = [
    "杀了他", "杀了她", "杀了他们", "报复", "同归于尽", "让他们付出代价",
    "弄死", "捅死", "炸了",
    "kill him", "kill her", "kill them", "hurt them", "revenge", "make them pay",
]
KW_DESPAIR = [
    "撑不下去", "撑不住了", "没人在意", "没人关心", "毫无意义", "没有意义",
    "看不到希望", "熬不下去", "我完了", "对不起所有人", "拖累",
    "can't go on", "no one cares", "hopeless", "worthless", "give up",
]
_ALL_KW = [(k, "self_harm") for k in KW_SELF_HARM] \
    + [(k, "violence") for k in KW_VIOLENCE] \
    + [(k, "distress") for k in KW_DESPAIR]


def quick_screen(text: str) -> List[str]:
    """返回命中的关键词（去重、保序）。空列表不代表安全，只是没有字面信号。"""
    t = (text or "").lower()
    hits, seen = [], set()
    for kw, _ in _ALL_KW:
        if kw.lower() in t and kw not in seen:
            hits.append(kw)
            seen.add(kw)
    return hits


# ------------------------------------------------------------------ 解析
_LEVEL_RE = re.compile(r'"?level"?\s*[:：]\s*([0-3])')
_CAT_RE = re.compile(r'"?category"?\s*[:：]\s*"?([a-z_]+)"?')


def parse_risk(raw: str) -> Dict[str, Any]:
    """容错解析。三级兜底：标准 JSON → 正则抠字段 → 全失败按 L1。

    第三级为什么是 L1 而不是 L0：模型吐不出格式，往往意味着输入本身让它"卡住"了。
    这时候放行（L0）等于在最不确定的时刻最激进。
    """
    raw = (raw or "").strip()
    out: Dict[str, Any] = {"level": LEVEL_DISTRESS, "category": "distress",
                           "evidence": "", "reason": "", "parse_ok": False}
    if not raw:
        out["reason"] = "模型返回空内容，按 L1 保守处理"
        return out

    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            lv = int(d.get("level", LEVEL_DISTRESS))
            out.update({
                "level": lv if lv in (0, 1, 2, 3) else LEVEL_DISTRESS,
                "category": str(d.get("category") or "distress"),
                "evidence": str(d.get("evidence") or "")[:300],
                "reason": str(d.get("reason") or "")[:200],
                "parse_ok": True,
            })
            return out
        except Exception:                                   # noqa: BLE001
            pass

    # 二级兜底：字段级正则（模型偶尔输出半截 JSON 或带前后缀说明）
    ml, mc = _LEVEL_RE.search(raw), _CAT_RE.search(raw)
    if ml:
        out.update({"level": int(ml.group(1)),
                    "category": mc.group(1) if mc else "unknown",
                    "parse_ok": False,
                    "reason": "JSON 不完整，按正则取 level"})
        return out
    out["reason"] = "无法解析模型输出，按 L1 保守处理"
    return out


# ------------------------------------------------------------------ 判定器
class RiskAssessor:
    """风险分级。默认用生成端同一个 client（glm-5.3-flash，便宜）；可单独注入。"""

    def __init__(self, cfg: Dict[str, Any], client, provider: Optional[str] = None):
        self.cfg = cfg
        self.client = client
        self.provider = provider or getattr(client, "name", "?")
        s = cfg.get("safety", {}) or {}
        self.enabled = bool(s.get("enabled", True))
        self.max_history = int(s.get("max_history_turns", 3))

    def assess(self, query: str, history: Optional[List[str]] = None) -> Dict[str, Any]:
        """返回 {level, category, evidence, reason, hits, latency_ms, raw, parse_ok}。"""
        if not self.enabled:
            return {"level": LEVEL_NONE, "category": "none", "evidence": "",
                    "reason": "护栏未启用", "hits": [], "latency_ms": 0,
                    "raw": "", "parse_ok": True}

        hits = quick_screen(query)
        # 多轮累积：把最近几轮学生原话拼进同一条待判文本。
        # 必须做——学生会分三轮才说完（"最近很累"→"没人在意"→"我已经想好了"），
        # 单轮判定必然漏，而"多轮补"几乎零成本。
        ctx_lines = []
        if history:
            ctx_lines = [h for h in history[-self.max_history:] if h.strip()]
        merged = "\n".join(ctx_lines + [query]) if ctx_lines else query

        hint = RISK_HINT_TMPL.format(hits="、".join(hits)) if hits else ""
        user = RISK_USER_TMPL.format(query=merged, hint=hint)

        with Timer() as t:
            try:
                resp = self.client.chat(RISK_SYSTEM, user, temperature=0.0, max_tokens=400)
                raw = (resp.get("texts") or [""])[0]
                usage = resp.get("usage", {})
            except Exception as e:                          # noqa: BLE001
                # 判定服务挂了不能把整个答疑也挂掉：按 L1 处理并标记 degraded。
                log(f"⚠️  风险判定调用失败（{type(e).__name__}: {str(e)[:80]}），按 L1 降级")
                return {"level": LEVEL_DISTRESS, "category": "distress", "evidence": "",
                        "reason": f"判定失败降级：{type(e).__name__}", "hits": hits,
                        "latency_ms": 0, "raw": "", "parse_ok": False, "degraded": True}

        d = parse_risk(raw)
        d.update({"hits": hits, "latency_ms": t.ms, "raw": raw[:800],
                  "usage": usage, "provider": self.provider})
        return d


# ------------------------------------------------------------------ 会话累积
class SessionRisk:
    """同一会话内逐轮累积。规则：最近 window 轮里有 escalate_after 轮 ≥L2 → 升到 L3。

    为什么要它：真实的危机表达很少一次说完。单轮 L2 只留痕，连续 L2 才是模式。
    """

    def __init__(self, window: int = 3, escalate_after: int = 2):
        self.window = window
        self.escalate_after = escalate_after
        self.levels: Deque[int] = deque(maxlen=window)
        self.turns: List[str] = []

    def update(self, level: int, query: str = "") -> int:
        self.levels.append(int(level))
        if query:
            self.turns.append(query)
        if len(self.levels) >= self.escalate_after \
                and sum(1 for x in self.levels if x >= LEVEL_IDEATION) >= self.escalate_after:
            return LEVEL_IMMINENT
        return int(level)

    def history(self) -> List[str]:
        return list(self.turns)


# ------------------------------------------------------------------ 分流
DEFAULT_CRISIS_REPLY = (
    "我很在意你刚才说的话，也想认真回应你。\n\n"
    "现在最重要的不是这道题。如果你有伤害自己的念头，请立刻联系能马上到你身边的人：\n"
    "· 心理援助热线：{hotline}\n"
    "· 学校心理咨询中心：{counsellor}\n"
    "· 紧急情况请直接拨打 110 或 120\n\n"
    "如果你愿意，也可以先告诉我你现在在哪里、身边有没有人。"
    "我不做心理治疗，但我不会把你一个人留在这里。"
)

DEFAULT_EMPATHY = {
    LEVEL_DISTRESS: "听起来这段时间不太好过。",
    LEVEL_IDEATION: "你刚才说的话我认真看到了。",
}


def decide(level: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把等级翻译成"产品动作"。L1/L2 不打扰、不升级；只有 L3 走危机响应。"""
    s = cfg.get("safety", {}) or {}
    if level >= LEVEL_IMMINENT:
        tpl = s.get("crisis_reply") or DEFAULT_CRISIS_REPLY
        try:
            text = tpl.format(hotline=s.get("hotline", "（请在此填写本地心理援助热线）"),
                              counsellor=s.get("counsellor", "（请在此填写校内咨询中心电话）"))
        except (KeyError, IndexError, ValueError):
            # 配置里的话术带了别的占位符就原样返回 —— 宁可少填一格，也不能 500。
            text = tpl
        return {"action": "crisis", "reply": text, "empathy": "",
                "skip_rag": True, "alert": True}
    if level >= LEVEL_DISTRESS:
        return {"action": "answer_with_empathy", "reply": "",
                "empathy": (s.get("empathy") or DEFAULT_EMPATHY).get(
                    level, "听起来这段时间不太好过。"),
                "skip_rag": False, "alert": level >= LEVEL_IDEATION}
    return {"action": "answer", "reply": "", "empathy": "",
            "skip_rag": False, "alert": False}


# ------------------------------------------------------------------ 告警
class AlertSink:
    """告警出口。两件事永远做：①落盘审计 ②有 webhook 就发。

    为什么不用 Dify 的「人工介入」节点那种"暂停等审批"：学生会卡在等待里。
    这里要的是发出去就完事（fire-and-forget），学生侧的回复不等任何人。
    """

    def __init__(self, cfg: Dict[str, Any], path: Optional[Path] = None):
        s = cfg.get("safety", {}) or {}
        self.webhook = os.environ.get("SAFETY_ALERT_WEBHOOK") or s.get("alert_webhook") or ""
        self.timeout = float(s.get("alert_timeout_sec", 3))
        self.path = Path(path) if path else Path("data/results/safety_alerts.jsonl")

    def send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
        try:
            append_jsonl(self.path, rec)
        except Exception as e:                              # noqa: BLE001
            log(f"⚠️  告警落盘失败：{e}")

        if not self.webhook:
            return {"sent": False, "detail": "未配置 webhook（已落盘）"}
        try:
            import urllib.request
            body = json.dumps({"msgtype": "text",
                               "text": {"content": self._format(payload)}},
                              ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                self.webhook, data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return {"sent": True, "detail": f"HTTP {r.status}"}
        except Exception as e:                              # noqa: BLE001
            # 告警失败绝不向上抛：学生不能因为告警服务挂了而收不到回复。
            log(f"⚠️  告警发送失败（{type(e).__name__}: {str(e)[:80]}），已落盘")
            return {"sent": False, "detail": f"{type(e).__name__}: {str(e)[:80]}"}

    @staticmethod
    def _format(p: Dict[str, Any]) -> str:
        return (f"【学生风险告警】{LEVEL_NAME.get(p.get('level', 0), '?')}\n"
                f"会话：{p.get('session_id', '-')}\n"
                f"学生：{p.get('student_id', '-')}\n"
                f"触发原句：{p.get('evidence') or '（模型未给出原句）'}\n"
                f"学生输入：{(p.get('query') or '')[:200]}\n"
                f"时间：{p.get('ts', time.strftime('%Y-%m-%d %H:%M:%S'))}")


# ------------------------------------------------------------------ 输出侧复核
def check_output(client, answer: str, context: str,
                 cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """只查"这句话本身有没有害"，不判对错（线上没有参考答案）。

    失败按放行处理（fail-open）：它只是额外一道，不能因为审核抖动卡掉正常答案。
    """
    if not (cfg or {}).get("safety", {}).get("output_check", True):
        return {"safe": True, "reason": "未启用", "fixed_text": ""}
    try:
        resp = client.chat(OUTPUT_SAFETY_SYSTEM,
                           OUTPUT_SAFETY_USER_TMPL.format(
                               context=(context or "")[:4000], answer=answer),
                           temperature=0.0, max_tokens=600)
        raw = (resp.get("texts") or [""])[0]
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return {"safe": True, "reason": "复核输出无法解析，放行", "fixed_text": ""}
        d = json.loads(m.group(0))
        return {"safe": bool(d.get("safe", True)),
                "reason": str(d.get("reason") or "")[:200],
                "fixed_text": str(d.get("fixed_text") or "")}
    except Exception as e:                                  # noqa: BLE001
        return {"safe": True, "reason": f"复核失败放行：{type(e).__name__}",
                "fixed_text": ""}


# ------------------------------------------------------------------ 门面
class SafetyGate:
    """一次调用走完：判定 → 累积 → 分流 →（必要时）告警。"""

    def __init__(self, cfg: Dict[str, Any], client, alert_path: Optional[Path] = None):
        self.cfg = cfg
        self.assessor = RiskAssessor(cfg, client)
        s = cfg.get("safety", {}) or {}
        self.session = SessionRisk(
            window=int(s.get("session_window", 3)),
            escalate_after=int(s.get("session_escalate_after", 2)))
        self.sink = AlertSink(cfg, alert_path)

    def precheck(self, query: str, student_id: str = "-",
                 session_id: str = "-") -> Dict[str, Any]:
        a = self.assessor.assess(query, history=self.session.history())
        effective = self.session.update(a["level"], query)
        act = decide(effective, self.cfg)
        act.update({"assessed_level": a["level"], "effective_level": effective,
                    "assess": a})
        if act.get("alert"):
            act["alert_result"] = self.sink.send({
                "level": effective, "student_id": student_id,
                "session_id": session_id, "evidence": a.get("evidence", ""),
                "reason": a.get("reason", ""), "query": query,
                "category": a.get("category", "")})
        return act
