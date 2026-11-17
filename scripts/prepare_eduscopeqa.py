"""
把 EduScopeQA 官方数据接进本管线。

用法（在 pipeline 目录下）：
    # 第 1 步：你先 clone 官方仓库（题目在里面）
    #   git clone https://github.com/Amay-J/EduScopeQA.git  (任意位置，比如 F:/学习/UM/文献/datasets-external/)
    # 第 2 步：在 clone 下来的目录里跑它的教材下载脚本（趁挂代理时做！）
    #   pip install requests PyPDF2
    #   python download_sources.py
    # 第 3 步：本脚本做对接
    python prepare_eduscopeqa.py --repo <EduScopeQA路径> --corpus <下载好的教材文本目录> --subjects science

产出：
    data/questions/normal.jsonl   ← 题目（含参考答案、粒度、学科、原chunk定位）
    data/corpus/original/         ← 教材文本（后续实验的索引对象）
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 仓库根目录（src/ 在上一层）
from src.common import list_files, load_config, log, resolve_path, write_jsonl

_GRAN_MAP = {  # 官方 type 值 → 本管线的粒度名（宽松匹配）
    "specific": "specific", "very_specific": "specific",
    "sectional": "sectional", "section": "sectional",
    "thematic": "thematic", "theme": "thematic",
}


def map_granularity(raw: str) -> str:
    low = (raw or "").lower()
    for k, v in _GRAN_MAP.items():
        if k in low:
            return v
    return low or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="clone 下来的 EduScopeQA 仓库路径")
    ap.add_argument("--corpus", default=None, help="教材 txt/md 所在目录（download_sources.py 的产出）")
    ap.add_argument("--subjects", default="science", help="逗号分隔：science,literature,history")
    a = ap.parse_args()

    cfg = load_config()
    repo = Path(a.repo).resolve()
    if not repo.exists():
        log(f"仓库路径不存在：{repo}"); sys.exit(1)

    # ---------- 题目 ----------
    rows = []
    for subj in [s.strip() for s in a.subjects.split(",") if s.strip()]:
        qfile = repo / subj / "questions.json"
        if not qfile.exists():
            log(f"⚠ 跳过 {subj}：找不到 {qfile}"); continue
        data = json.loads(qfile.read_text(encoding="utf-8"))
        for i, item in enumerate(data):
            rows.append({
                "qid": f"{subj}-{i:04d}",
                "question": item.get("question", "").strip(),
                "reference": item.get("answer", "").strip(),
                "granularity": map_granularity(item.get("type", "")),
                "subject": subj,
                "source_chunk": item.get("source_chunk", ""),
            })
        log(f"{subj}: {len(data)} 题")

    qdir = resolve_path(cfg, "questions")
    n = write_jsonl(qdir / "normal.jsonl", rows)
    log(f"★ 题目已写入 {qdir/'normal.jsonl'}（{n} 条）")
    from collections import Counter
    log("粒度分布：" + str(dict(Counter(r['granularity'] for r in rows))))

    # ---------- 教材 ----------
    if a.corpus:
        src = Path(a.corpus).resolve()
        if not src.exists():
            log(f"⚠ 教材目录不存在：{src}"); sys.exit(1)
        dst = resolve_path(cfg, "corpus_original")
        copied = 0
        for fp in list_files(src, (".txt", ".md")):
            # 跳过官方仓库自带的脚本/说明
            if fp.parent == repo:
                continue
            target = dst / f"{fp.stem}.txt"
            shutil.copyfile(fp, target)
            copied += 1
            log(f"教材已拷入：{fp.name} → {target.name}")
        log(f"★ 教材共 {copied} 个文件 → {dst}")
        if copied == 0:
            log("⚠ 没找到 txt/md。若 OpenStax 是 PDF 且转换失败，请手动转成 txt 后放入。")
    else:
        log("（未提供 --corpus，只转换了题目；教材稍后可重跑本脚本补充）")


if __name__ == "__main__":
    main()
