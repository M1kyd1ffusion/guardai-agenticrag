"""
生成端：三家国产模型统一调用 + **PK/CK 双路径** + S1/S2/S4 策略。

DD-04 v2 的关键：
  每个问题都跑两条路径并**完整持久化**：
    PK-only        —— 不给上下文，看模型凭记忆答什么（这条同时也是 09 文件 Step2 的"裸考"）
    CK-conditioned —— 给上下文，看模型读完教材答什么
  两条路径各自的答案 + 是否正确，构成四格标签 C11/C10/C01/C00。
  ⚠️ 如果不落盘，事后补算归因要把所有 API 重跑一遍。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .common import Cache, Timer, cache_key, log
from .prompts import CK_USER, PK_ONLY_USER, SYSTEMS, build_context


def _looks_like_bad_param(e: Exception) -> bool:
    """判断异常是否像"厂商不认识我传的参数"（400 / invalid / unknown field）。
    宁可误判一次：去掉 extra_body 只是回到厂商默认行为，不会让整批崩掉。"""
    s = str(e).lower()
    return any(k in s for k in ("400", "invalid", "unknown", "unrecognized", "extra", "thinking"))


class ModelClient:
    """三家都走 OpenAI 兼容接口，只有 base_url / 模型名 / key 不同。"""

    def __init__(self, cfg: Dict[str, Any], provider: str, cache: Cache, client=None,
                 model: Optional[str] = None, max_tokens: Optional[int] = None):
        self.cfg = cfg
        # 裁判端需要比生成端更大的输出上限：glm-5.3 强制思考，思考链本身就吃掉几百 token，
        # 上限太小 → content 空（2026-10-27 的 22 题误判）。故允许调用方单独指定。
        self._max_tokens_override = max_tokens
        self.name = provider
        self.pconf = cfg["generation"]["providers"][provider]
        self.cache = cache
        # model 覆盖：judge 需要用「旗舰档」而生成端用「轻量档」，同一家会有两个模型名。
        # 2026-10-26 修：此前 ModelClient 只认 providers.*.model，导致
        # evaluation.judge.model 这个配置项一直被忽略，judge 实际用了生成端的轻量模型。
        self._model_override = model
        if client is None:
            from openai import OpenAI
            api_key = os.environ.get(self.pconf["env_key"], "")
            if not api_key:
                raise RuntimeError(
                    f"环境变量 {self.pconf['env_key']} 未设置。\n"
                    f"  Windows PowerShell 里执行： $env:{self.pconf['env_key']}=\"你的key\"\n"
                    f"  （重启终端后失效；永久生效见 README §2）"
                )
            self.client = OpenAI(api_key=api_key, base_url=self.pconf["base_url"],
                                 timeout=cfg["generation"].get("timeout_sec", 120))
        else:
            self.client = client

    @property
    def model_id(self) -> str:
        return self._model_override or self.pconf["model"]

    def chat(self, system: str, user: str, temperature: Optional[float] = None,
             n: int = 1, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        g = self.cfg["generation"]
        # 温度优先级：调用方 > 厂商配置 > 全局默认。
        # 为什么要有厂商级温度：kimi-k3 只接受 temperature=1（2026-11-09 实测 400
        # "invalid temperature: only 1 is allowed for this model"），强制思考的模型
        # 大多不开放温度调节。判分要温度 0 的惯例在它这里行不通，只能按厂商放。
        temp = temperature if temperature is not None \
            else self.pconf.get("temperature", g["temperature"])
        max_tok = max_tokens or self._max_tokens_override or g.get("max_output_tokens", 512)

        params = {
            "model": self.model_id, "temperature": temp, "n": n, "max_tokens": max_tok,
        }
        # kimi-k3 不接受空 system 消息（2026-11-09 实测 400：role 'system' must not be
        # empty）；glm/deepseek 则无所谓。空 system 就干脆不发这条消息，对三家都安全。
        messages = ( [{"role": "system", "content": system}] if system.strip() else [] )
        messages.append({"role": "user", "content": user})
        params["messages"] = messages
        # 关掉思考模式：新一带 DeepSeek 默认开启，输出会全落在 reasoning_content 上，
        # 把 max_tokens 烧光后 content 变成空字符串 —— 样本直接损坏。
        # 只有厂商声明支持该参数时才传，避免智谱 / Kimi 收到不认识的参数报错。
        if self.pconf.get("thinking_control") and not g.get("thinking", False):
            params["extra_body"] = {"thinking": {"type": "disabled"}}
        # 智谱 GLM-5.x 系列【始终】思考，关不掉，只能调强度（low / high / max，默认 max）。
        # 默认 max 的思考量经常上千 token，会把输出上限吃满 → content 返回空串。
        # 官方推荐用 reasoning_effort 控制，这是顶层参数，不是 extra_body。
        if self.pconf.get("reasoning_effort"):
            params["reasoning_effort"] = self.pconf["reasoning_effort"]
        key = cache_key(self.name, self.model_id, params)
        hit = self.cache.get(key)
        if hit is not None:
            return {**hit, "cached": True}

        last_err = None
        for attempt in range(g.get("max_retries", 3)):
            try:
                with Timer() as t:
                    resp = self.client.chat.completions.create(**params)
                out = {
                    "texts": [c.message.content or "" for c in resp.choices],
                    "usage": {
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                        "latency_ms": t.ms,
                    },
                    "cached": False,
                }
                # 空内容 + 用满输出上限 = 思考模式把 token 烧在推理链上的典型症状。
                # 2026-10-27 因这个现象丢过 22 题的判分，故就地报警，别等统计时才发现。
                if (not out["texts"][0].strip()
                        and out["usage"].get("completion_tokens") == max_tok):
                    log(f"⚠️  {self.name}/{self.model_id} 返回空内容，且 completion_tokens 用满 {max_tok}"
                        f" —— 疑似思考模式吃满，检查该 provider 的 thinking_control")
                self.cache.put(key, out)
                return out
            except Exception as e:                       # 指数退避
                last_err = e
                # 401/鉴权失败重试没有意义（key 错了等多久都一样），
                # 而且一批里每题都白白烧 3 次重试 × 退避 7 秒。直接抛。
                s401 = str(e)
                if "401" in s401 or "authentication" in s401.lower() \
                        or "身份验证" in s401 or "invalid api key" in s401.lower():
                    raise RuntimeError(
                        f"{self.name} 鉴权失败（{s401[:120]}）。"
                        f"环境变量 {self.pconf['env_key']} 里的 key 无效或已轮换 —— "
                        f"换新终端/重启进程后再试，不要反复重试。") from e
                # 有的厂商不认 thinking 参数（或字段名不同）直接 400。
                # 反复重试同一个 400 没意义，去掉该参数再试——宁可开思考也别整批崩掉。
                if (params.get("extra_body") or params.get("reasoning_effort")) \
                        and _looks_like_bad_param(e):
                    params.pop("extra_body", None)
                    params.pop("reasoning_effort", None)
                    log(f"{self.name} 不接受思考控制参数（{str(e)[:80]}），已去掉后重试")
                    continue
                wait = 2 ** attempt
                log(f"调用失败（第 {attempt+1} 次）：{e} → {wait}s 后重试")
                import time as _t
                _t.sleep(wait)
        raise RuntimeError(f"{self.name} 调用失败：{last_err}")


# ------------------------------------------------------------------ 双路径生成
def run_dual_path(client: ModelClient, question: str, segs: List[Dict[str, Any]],
                  strategy: str = "S1") -> Dict[str, Any]:
    """返回一个 record：含 pk / ck 两条路径的输出与用量。"""
    g_cfg = client.cfg["generation"]
    system = SYSTEMS.get(strategy, SYSTEMS["S1"])
    context = build_context([s["text"] for s in segs])

    rec: Dict[str, Any] = {
        "provider": client.name, "model": client.model_id, "strategy": strategy,
        "n_segments": len(segs),
    }

    # ---- PK-only（裸考 / 09 Step2 同一次调用）----
    if g_cfg.get("dual_path", {}).get("enabled", True):
        pk = client.chat("", PK_ONLY_USER.format(question=question))
        rec["pk_answer"] = pk["texts"][0]
        rec["pk_usage"] = pk["usage"]

    # ---- CK-conditioned（正常 RAG）----
    if strategy == "S4":
        n = int(g_cfg.get("s4_samples", 3))
        ck = client.chat(system, CK_USER.format(context=context, question=question),
                         temperature=max(0.7, 0.7), n=n)
        rec["ck_answer"] = ck["texts"][0]
        rec["ck_samples"] = ck["texts"]
        rec["ck_usage"] = ck["usage"]
        rec["ck_agreement"] = _agreement(ck["texts"])
    else:
        ck = client.chat(system, CK_USER.format(context=context, question=question))
        rec["ck_answer"] = ck["texts"][0]
        rec["ck_usage"] = ck["usage"]

    rec["context_tokens"] = sum(int(s.get("n_tokens", 0)) for s in segs)
    return rec


def _agreement(texts: List[str]) -> float:
    """黑盒版的"信念估计"代理：多次采样的一致性比例（对应 #38 的 multi-trace）。
    完全一致=1.0，全都不同=1/n。"""
    if not texts:
        return 0.0
    norm = [t.strip().lower() for t in texts]
    counts: Dict[str, int] = {}
    for t in norm:
        counts[t] = counts.get(t, 0) + 1
    return max(counts.values()) / len(norm)
