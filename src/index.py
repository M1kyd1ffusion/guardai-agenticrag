"""
索引：稠密（bge 嵌入）+ 稀疏（BM25）。
默认用 numpy 暴力余弦检索 —— 对本项目的语料规模（数千到数万段）完全够用且零依赖；
语料变大后把 config.retrieval.backend 改成 faiss 即可（需 pip install faiss-cpu）。
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .common import Timer, log


# ------------------------------------------------------------------ 嵌入
class Embedder:
    """直接用 transformers 加载 bge，不走 sentence-transformers。

    为什么要这样：sentence-transformers 6.x 加载 bge 系列时会要求仓库里存在
    `config_sentence_transformers.json`，而国内镜像站（hf-mirror）常常没有同步这个文件，
    结果就是权重下载完了却报 FileNotFoundError。bge 本体就是一个 BERT，
    用 AutoModel + mean pooling 加载完全等价，且少一层依赖、少一个坑。
    """

    def __init__(self, model_name: str, device: str = "cpu",
                 batch_size: int = 32, normalize: bool = True,
                 query_instruction: str = "", max_length: int = 512,
                 local_files_only: bool = True):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device
        self.batch = batch_size
        self.normalize = normalize
        self.qi = query_instruction
        self.max_len = max_length
        # ⚠️ 2026-10-27 事故：from_pretrained 默认会先联网查"模型有没有新版本"，
        #    国内访问 huggingface.co 超时（WinError 10060），明明模型已在本地却加载失败，
        #    报错长得像"缺模型"，实际只是卡在校验这一步。
        #    本项目模型已在 ~/.cache/huggingface/hub 缓存齐全 → 默认只读本地。
        #    真要首次下载新模型时，把这个参数设为 False（或临时设 HF_HUB_OFFLINE=0）。
        self.local_only = local_files_only
        self.tok = self._load(AutoTokenizer, "tokenizer", model_name)
        self.model = self._load(AutoModel, "model", model_name)
        self.model.to(device)
        self.model.eval()

    def _load(self, cls, what: str, model_name: str):
        """先走本地缓存；本地真没有才回退联网（并明确告诉你正在下载）。"""
        try:
            return cls.from_pretrained(model_name, local_files_only=self.local_only)
        except Exception as e:  # noqa: BLE001
            if not self.local_only:
                raise
            msg = str(e).splitlines()[0] if str(e) else e.__class__.__name__
            log(f"本地缓存里没找到 {model_name} 的{what}（{msg}），改为联网加载……")
            log("  如果这行之后卡住或超时，说明本机访问 huggingface.co 不通：")
            log("  请先设 HF_ENDPOINT=https://hf-mirror.com 再重跑，或用别的机器下好缓存拷过来。")
            return cls.from_pretrained(model_name, local_files_only=False)

    @staticmethod
    def _mean_pooling(last_hidden, attention_mask, torch):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode(self, texts: List[str]) -> np.ndarray:
        import torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch):
                chunk = texts[i: i + self.batch]
                enc = self.tok(chunk, padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc)
                emb = self._mean_pooling(out.last_hidden_state, enc["attention_mask"], torch)
                if self.normalize:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                outs.append(emb.cpu().numpy())
        return np.vstack(outs) if outs else np.zeros((0, 1), dtype=np.float32)

    def encode_docs(self, texts: List[str]) -> np.ndarray:
        with Timer() as t:
            vecs = self._encode(texts)
        log(f"嵌入完成：{len(texts)} 段，用时 {t.ms/1000:.1f}s")
        return vecs

    def encode_query(self, q: str) -> np.ndarray:
        text = (self.qi + q) if self.qi else q
        return self._encode([text])[0]


# ------------------------------------------------------------------ BM25
class BM25Index:
    def __init__(self, tokenized: List[List[str]]):
        from rank_bm25 import BM25Okapi
        self.core = BM25Okapi(tokenized)
        self.size = len(tokenized)

    def scores(self, q_tokens: List[str]) -> np.ndarray:
        return np.asarray(self.core.get_scores(q_tokens), dtype=np.float32)


def tokenize(text: str) -> List[str]:
    return [t for t in "".join(
        c.lower() if c.isalnum() else " " for c in text
    ).split() if t]


# ------------------------------------------------------------------ 稠密检索
class DenseIndex:
    def __init__(self, vectors: np.ndarray, backend: str = "numpy"):
        self.vectors = vectors.astype(np.float32)
        self.backend = backend
        self.faiss = None
        if backend == "faiss":
            try:
                import faiss  # type: ignore
                dim = self.vectors.shape[1]
                self.faiss = faiss.IndexFlatIP(dim)
                self.faiss.add(self.vectors)
                log(f"FAISS 索引已建：{self.vectors.shape}")
            except Exception as e:
                log(f"FAISS 不可用（{e}），自动回退 numpy 后端")
                self.backend = "numpy"

    def search(self, qvec: np.ndarray, top_n: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.faiss is not None:
            sims, idx = self.faiss.search(qvec.reshape(1, -1), top_n)
            return sims[0], idx[0]
        sims = self.vectors @ qvec.reshape(-1)
        idx = np.argpartition(-sims, min(top_n, len(sims) - 1))[:top_n]
        idx = idx[np.argsort(-sims[idx])]
        return sims[idx], idx


# ------------------------------------------------------------------ 索引包
class HybridStore:
    """把 chunks / dense / bm25 三件套打包存到磁盘，避免每次重算嵌入。"""

    def __init__(self, chunks: List[Dict[str, Any]], dense: DenseIndex, bm25: BM25Index):
        self.chunks = chunks
        self.dense = dense
        self.bm25 = bm25

    @staticmethod
    def build(cfg: Dict[str, Any], chunks: List[Dict[str, Any]], embedder: Embedder) -> "HybridStore":
        texts = [c["text"] for c in chunks]
        vecs = embedder.encode_docs(texts)
        backend = cfg["retrieval"]["backend"]
        tok = [tokenize(t) for t in texts]
        return HybridStore(chunks, DenseIndex(vecs, backend), BM25Index(tok))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        np.save(p / "vectors.npy", self.dense.vectors)
        with open(p / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25.core, f)
        log(f"索引已保存 → {p}")

    @classmethod
    def load(cls, path: str | Path, backend: str = "numpy") -> Optional["HybridStore"]:
        p = Path(path)
        if not (p / "chunks.pkl").exists():
            return None
        with open(p / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        vectors = np.load(p / "vectors.npy")
        from rank_bm25 import BM25Okapi
        with open(p / "bm25.pkl", "rb") as f:
            core = pickle.load(f)
        store = cls(chunks, DenseIndex(vectors, backend), BM25Index(["dummy"]))
        store.bm25.core = core
        log(f"索引已加载 ← {p}（{len(chunks)} 段）")
        return store
