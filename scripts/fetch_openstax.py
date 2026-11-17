#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_openstax.py —— 从 OpenStax 官方 GitHub 源抓取教材正文，转成纯 txt。

为什么不用 PDF：
  EduScopeQA 自带的 download_sources.py 是直接下 OpenStax 的 PDF 再 pypdf 提取，
  但 assets.openstax.org 的直链现在返回 403；而且 PDF 提取出来的文本带页眉页脚、
  断词连字符、图表乱序，做检索语料质量差。
  OpenStax 官方把每本书的 CNXML 源文件放在 GitHub 上（CC BY-NC-SA 授权），
  结构干净（title/para/list/table 都是独立标签），转出来的文本质量更高。

用法：
    # 默认抓微生物学（EduScopeQA 的 science 学科），写到 ../EduScopeQA-main/science/input/
    python fetch_openstax.py

    # 指定输出目录 / 指定书
    python fetch_openstax.py --repo openstax/osbooks-microbiology --out "F:/.../science/input"

依赖：只用标准库（urllib）。国内直连 raw.githubusercontent.com 会被重置，
      脚本默认走 ghproxy.net 代理前缀，可用 --mirror 换。
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

# GitHub raw 的国内可用中转；留空字符串则直连 raw.githubusercontent.com
MIRRORS = {
    "ghproxy": "https://ghproxy.net/https://raw.githubusercontent.com/",
    "direct": "https://raw.githubusercontent.com/",
}


# ---------------------------------------------------------------- 网络

def http_get(url, timeout=60, retries=3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}\n  last error: {last}")


def raw_url(mirror, repo, path, ref="main"):
    base = MIRRORS[mirror]
    if mirror == "direct":
        return f"{base}{repo}/{ref}/{path}"
    # 中转前缀形如 https://ghproxy.net/https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>
    return f"{base}{repo}/{ref}/{path}"


def get_head_sha(repo):
    """拿 main 分支最新 commit sha，写进语料清单做版本快照。"""
    try:
        data = http_get(f"https://api.github.com/repos/{repo}/commits/main", timeout=30, retries=2)
        return json.loads(data).get("sha", "")[:10]
    except Exception:  # noqa: BLE001
        return "unknown"


def list_modules(repo, mirror):
    """列出仓库里所有 modules/*/index.cnxml。"""
    url = f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"
    try:
        tree = json.loads(http_get(url, timeout=90))["tree"]
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"取仓库文件树失败（api.github.com 不通）：{e}\n"
            f"兜底：用 --module-ids 手动给出章节 id，或在浏览器下载仓库 ZIP 后本地转换。"
        )
    mods = [t["path"] for t in tree if re.fullmatch(r"modules/m\d+/index\.cnxml", t["path"])]
    return sorted(mods)


# ---------------------------------------------------------------- CNXML → 文本

def local(tag):
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def clean(s):
    s = re.sub(r"[ \t\u00a0]+", " ", s or "")
    s = re.sub(r"\s*\n\s*", " ", s)
    return s.strip()


BLOCK_TITLES = {"title"}
BLOCK_PARA = {"para"}
BLOCK_ITEM = {"item"}
SKIP = {"metadata", "media"}  # metadata 是书目信息；media 里没有正文

# 教材每章末尾的习题区（CNXML 用 class 标出来）。
# 为什么要单独拆出来、不放进检索语料：
#   1) 习题区占全书 6.4%，内容是 "- endoplasmic reticulum / - Golgi apparatus / C" 这种
#      选项 + 答案字母，对回答问题纯属噪音，还白白吃掉上下文预算（直接影响成本维度）；
#   2) 有极小概率与 benchmark 题目撞车（实测 778 题里 3 题 6-gram 覆盖率 >60%），
#      留在索引里会让 CK 路径被质疑"抄教材课后题"= 文档级污染。
# 拆出来后单列一个文件，用于泄漏审计，不进索引。
EXERCISE_CLASSES = {
    "multiple-choice", "fill-in-the-blank", "short-answer", "critical-thinking",
    "matching", "true-false", "review-questions", "exercises",
}


def walk(el, out, ex_out):
    """按文档顺序把 CNXML 拆成 (kind, text)；习题区进 ex_out，其余进 out。"""
    for child in el:
        ln = local(child.tag)
        cls = set((child.get("class") or "").split())
        if ln == "section" and cls & EXERCISE_CLASSES:
            walk(child, ex_out, ex_out)
            continue
        if ln in SKIP:
            continue
        if ln in BLOCK_TITLES:
            t = clean("".join(child.itertext()))
            if t:
                out.append(("h", t))
        elif ln in BLOCK_PARA:
            t = clean("".join(child.itertext()))
            if t:
                out.append(("p", t))
        elif ln in BLOCK_ITEM:
            t = clean("".join(child.itertext()))
            if t:
                out.append(("li", t))
        elif ln == "table":
            for row in child.iter():
                if local(row.tag) != "row":
                    continue
                cells = [clean("".join(c.itertext())) for c in row if local(c.tag) == "entry"]
                cells = [c for c in cells if c]
                if cells:
                    out.append(("row", " | ".join(cells)))
        else:
            # section / note / example / exercise / figure / blockquote … 递归下去
            walk(child, out, ex_out)
    return out, ex_out


def cnxml_to_text(xml_bytes):
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        raise RuntimeError(f"XML 解析失败: {e}")
    body, ex = walk(root, [], [])
    return body, ex


def render(blocks):
    """块序列 → 纯文本。空行分段，标题单独成行。"""
    lines = []
    for kind, text in blocks:
        if kind == "h":
            lines.append("")
            lines.append(text)
            lines.append("")
        elif kind == "p":
            lines.append(text)
            lines.append("")
        elif kind == "li":
            lines.append(f"- {text}")
        elif kind == "row":
            lines.append(text)
    txt = "\n".join(lines)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip() + "\n"


# ---------------------------------------------------------------- 合集目录（章节顺序）

def parse_collection(xml_bytes):
    """
    返回 [(chapter_title or None, [module_id, ...]), ...]，保持书里的顺序。
    合集结构是 <subcollection><md:title>章名</md:title><col:content><col:module document="m58780"/>
    """
    root = ET.fromstring(xml_bytes)
    plan = []

    def title_of(el):
        for c in el:
            if local(c.tag) == "title":
                return clean("".join(c.itertext()))
        return ""

    def collect(el, chapter):
        for c in el:
            ln = local(c.tag)
            if ln == "subcollection":
                collect(c, title_of(c) or chapter)
            elif ln == "content":
                collect(c, chapter)
            elif ln == "module":
                mid = c.get("document")
                if mid:
                    plan.append((chapter, mid))

    for c in root:
        if local(c.tag) == "content":
            collect(c, "")

    # 按章分组，保持首次出现顺序
    grouped, order = {}, []
    for chapter, mid in plan:
        if chapter not in grouped:
            grouped[chapter] = []
            order.append(chapter)
        grouped[chapter].append(mid)
    return [(ch, grouped[ch]) for ch in order]


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="openstax/osbooks-microbiology")
    ap.add_argument("--collection", default="collections/microbiology.collection.xml")
    ap.add_argument("--out", default="data/corpus/openstax")
    ap.add_argument("--txt-name", default="Microbiology-WEB.txt")
    ap.add_argument("--ex-out", default="data/corpus/openstax_exercises",
                    help="习题区输出目录（不进索引，仅供泄漏审计）")
    ap.add_argument("--cache", default=None, help="CNXML 缓存目录，默认 <out>/.cnxml")
    ap.add_argument("--mirror", default="ghproxy", choices=list(MIRRORS))
    ap.add_argument("--sleep", type=float, default=0.25, help="每次请求间隔秒数，避免被限流")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache).expanduser().resolve() if args.cache else out_dir / ".cnxml"
    cache.mkdir(parents=True, exist_ok=True)

    sha = get_head_sha(args.repo)
    print(f"[1/4] 仓库 {args.repo} @ {sha}   镜像={args.mirror}")

    print(f"[2/4] 读取合集目录 {args.collection} ...")
    coll_bytes = http_get(raw_url(args.mirror, args.repo, args.collection))
    plan = parse_collection(coll_bytes)
    n_mods = sum(len(m) for _, m in plan)
    print(f"      {len(plan)} 个单元 / {n_mods} 个模块")

    mods = list_modules(args.repo, args.mirror)
    known = {m for _, ms in plan for m in ms}
    extra = [p for p in mods if re.search(r"m(\d+)", p).group(0) not in known]
    print(f"[3/4] 仓库共 {len(mods)} 个模块，合集外另有 {len(extra)} 个（附录等）")

    def fetch_module(mid):
        p = cache / f"{mid}.cnxml"
        if p.exists() and p.stat().st_size > 0:
            return p.read_bytes()
        data = http_get(raw_url(args.mirror, args.repo, f"modules/{mid}/index.cnxml"))
        p.write_bytes(data)
        time.sleep(args.sleep)
        return data

    parts = []
    header = (
        "Source: OpenStax Microbiology — official CNXML source, "
        f"github.com/{args.repo} @ {sha}\n"
        "License: CC BY-NC-SA 4.0. Converted by pipeline/fetch_openstax.py on "
        f"{time.strftime('%Y-%m-%d')} for the EduScopeQA science retrieval corpus.\n"
    )
    parts.append(header)

    done = 0
    total = n_mods + len(extra)
    chap_no = 0
    ex_parts = ["Exercises removed from the retrieval corpus (kept for leakage audit only).\n"]
    for chapter, mids in plan:
        if chapter:
            chap_no += 1
            parts.append(f"\n\n===== Chapter {chap_no}: {chapter} =====\n")
            ex_parts.append(f"\n\n===== Chapter {chap_no}: {chapter} =====\n")
        elif chap_no == 0:
            parts.append("\n\n===== Front Matter =====\n")
        for mid in mids:
            try:
                blocks, ex_blocks = cnxml_to_text(fetch_module(mid))
            except Exception as e:  # noqa: BLE001
                print(f"      ! {mid} 失败：{e}")
                continue
            done += 1
            if done % 20 == 0:
                print(f"      进度 {done}/{total}")
            # 模块自己的 <title> 就是小节名，render 时已是首行
            parts.append(render(blocks))
            if ex_blocks:
                ex_parts.append(render(ex_blocks))

    for path in extra:
        mid = re.search(r"m(\d+)", path).group(0)
        try:
            blocks, ex_blocks = cnxml_to_text(fetch_module(mid))
            parts.append(render(blocks))
            if ex_blocks:
                ex_parts.append(render(ex_blocks))
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"      ! {mid} 失败：{e}")

    body = "\n".join(parts)
    target = out_dir / args.txt_name
    target.write_text(body, encoding="utf-8")

    ex_dir = Path(args.ex_out).expanduser().resolve() if args.ex_out else out_dir.parent / (out_dir.name + "_exercises")
    ex_dir.mkdir(parents=True, exist_ok=True)
    ex_target = ex_dir / (target.stem + ".exercises.txt")
    ex_body = "\n".join(ex_parts)
    ex_target.write_text(ex_body, encoding="utf-8")

    mb = target.stat().st_size / 1e6
    words = len(body.split())
    print(f"[4/4] 写出 {target}")
    print(f"      模块 {done}/{total}，{mb:.2f} MB，{words:,} 词，约 {int(words*1.35):,} tokens")
    print(f"      习题另行存放（不进索引，仅供泄漏审计）：{ex_target}"
          f"（{ex_target.stat().st_size/1e6:.2f} MB）")

    # 版本快照清单，直接可并入 run_manifest
    manifest = out_dir / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "repo": args.repo,
                "commit": sha,
                "collection": args.collection,
                "modules_total": total,
                "modules_ok": done,
                "output": target.name,
                "bytes": target.stat().st_size,
                "words": words,
                "exercises_file": str(ex_target),
                "exercises_bytes": ex_target.stat().st_size,
                "exercises_in_index": False,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "license": "CC BY-NC-SA 4.0",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"      版本快照 {manifest}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
