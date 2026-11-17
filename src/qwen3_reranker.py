"""
Qwen3-Reranker 的本地包装器（绕开 sentence-transformers，严格按官方模型卡实现）。

官方代码的三个关键点（README transformers 段，2026-11-11 逐码点核对）：
  1) 只截断 middle（Instruct/Query/Document），prefix/suffix 永远完整保留；
  2) padding_side='left'，取 logits[:, -1]；等价实现：右侧手动 padding +
     按 attention_mask 取每个样本真实最后一个 token 的 logits；
  3) suffix 以"空的 think 块"结尾（开标签+空行+闭标签+空行），让模型直接输出 yes/no。

首次实现踩过的坑（已修，勿回退）：
  - truncation=True 会把 suffix 截掉 → 模型看不到作答指令 → 分数全是噪声，
    实测"无关段得分反而高于相关段"；
  - 右 padding 下取 logits[:, -1] 取到 pad 位置 → 分数随批次内容漂移。

用法与 sentence_transformers.CrossEncoder.predict 兼容。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on '
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
# 官方 suffix：结尾是"空 think 块"。用 chr() 拼接，避免该标签在工具传输中被改写。
_T_OPEN = chr(60) + "think" + chr(62)
_T_CLOSE = chr(60) + "/think" + chr(62)
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n' + _T_OPEN + '\n\n' + _T_CLOSE + '\n\n'
DEFAULT_INSTRUCT = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class Qwen3Reranker:
    def __init__(self, path: str, max_length: int = 512, device: str = "cpu",
                 instruction: str = DEFAULT_INSTRUCT):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, local_files_only=True,
        ).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.instruction = instruction
        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.no_id = self.tok.convert_tokens_to_ids("no")
        self.prefix_ids = self.tok.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(SUFFIX, add_special_tokens=False)
        self.mid_cap = max_length - len(self.prefix_ids) - len(self.suffix_ids)
        assert self.mid_cap > 64, f"max_length={max_length} 太小，middle 只剩 {self.mid_cap}"

    def _middles(self, pairs: Sequence[Tuple[str, str]]) -> List[str]:
        return [
            f"<Instruct>: {self.instruction}\n<Query>: {q}\n<Document>: {d[:2000]}"
            for q, d in pairs
        ]

    @torch.no_grad()
    def predict(self, pairs: Sequence[Tuple[str, str]], batch_size: int = 8,
                show_progress_bar: bool = False) -> np.ndarray:
        mids = self._middles(pairs)
        out: List[float] = []
        pad_id = self.tok.pad_token_id or 0
        for i in range(0, len(mids), batch_size):
            part = mids[i: i + batch_size]
            enc = self.tok(part, add_special_tokens=False)["input_ids"]
            seqs = [self.prefix_ids + ids[: self.mid_cap] + self.suffix_ids
                    for ids in enc]
            maxlen = max(len(s) for s in seqs)
            input_ids = torch.tensor(
                [s + [pad_id] * (maxlen - len(s)) for s in seqs], device=self.device)
            attn = torch.tensor(
                [[1] * len(s) + [0] * (maxlen - len(s)) for s in seqs],
                device=self.device)
            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
            last = attn.sum(dim=1) - 1                 # 每个样本真实末 token 的位置
            rows = logits[torch.arange(len(seqs)), last]
            yn = rows[:, [self.no_id, self.yes_id]].float()
            probs = torch.softmax(yn, dim=-1)[:, 1]    # P(yes)
            out.extend(probs.cpu().tolist())
        return np.array(out)
