#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebuild_index.py —— 重建检索索引（语料或切片参数变了才需要跑）

什么时候要跑：
  ① 换了教材 / 语料内容变了
  ② 改了 config.yaml 的 chunking.sizes 或 strategy（SQ1 的切片档位）
  ③ 索引文件被意外覆盖（比如误跑 --demo 把教材索引盖成示例）

成本：纯本地计算，不花 API 钱。26 章 / 1294 段约 1–2 分钟（CPU）。
产出：data/index/{chunks.pkl, bm25.pkl, vectors.npy}

用法：
    .\\.venv\\Scripts\\python.exe rebuild_index.py
    .\\.venv\\Scripts\\python.exe rebuild_index.py --corpus data/corpus/altered   # 给篡改语料建索引
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 仓库根目录（src/ 在上一层）

from src.common import (Timer, load_config, log, resolve_path,  # noqa: E402
                        write_index_source)
from src.corpus import build_chunks, load_documents  # noqa: E402
from src.index import Embedder, HybridStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None,
                    help="语料目录（默认取 config.paths.corpus_original；给篡改语料就传 corpus_altered）")
    ap.add_argument("--out", default=None, help="索引输出目录（默认 data/index_expert，即产品加载的目录）")
    a = ap.parse_args()

    cfg = load_config()
    if a.corpus:
        cdir = Path(a.corpus).resolve()
    else:
        cdir = resolve_path(cfg, "corpus_original")

    if not cdir.exists() or not any(cdir.iterdir()):
        log(f"语料目录为空：{cdir}")
        sys.exit(1)

    docs = load_documents(cdir)
    log(f"读到 {len(docs)} 篇文档 ← {cdir}")
    chunks = build_chunks(cfg, docs)
    if not chunks:
        log("切片为空，检查语料")
        sys.exit(1)
    log(f"切片完成：{len(chunks)} 段"
        f"（strategy={cfg['chunking']['strategy']}, size={cfg['chunking']['sizes']}）")

    emb = Embedder(cfg["embedding"]["model"], cfg["embedding"]["device"],
                   cfg["embedding"]["batch_size"], cfg["embedding"]["normalize"],
                   cfg["embedding"].get("query_instruction", ""),
                   local_files_only=cfg["embedding"].get("local_files_only", True))
    with Timer() as t:
        store = HybridStore.build(cfg, chunks, emb)
    log(f"嵌入完成，用时 {t.ms/1000:.1f}s")

    # 默认写 data/index_expert：src/router.py 的 build_router() 就加载这个目录
    out = (Path(a.out).resolve() if a.out
           else resolve_path(cfg, "index_dir").parent / "index_expert")
    out.mkdir(parents=True, exist_ok=True)
    store.save(out)
    # 登记索引来源指纹：原著与篡改语料段数几乎相同，只比段数会误复用错索引
    write_index_source(out, cdir)
    log(f"★ 索引已保存 → {out}（{len(store.chunks)} 段，来源 {cdir}）")

    # 自检：读回来确认段数一致
    back = HybridStore.load(out, cfg["retrieval"]["backend"])
    if back is None or len(back.chunks) != len(chunks):
        log("⚠️ 回读校验失败，索引可能没写全")
        sys.exit(1)
    log(f"✓ 回读校验通过：{len(back.chunks)} 段")
    log(f"  首段 doc_id = {back.chunks[0].get('doc_id')}")


if __name__ == "__main__":
    main()
