"""
通用工具：配置加载、路径、缓存、token 估算、日志。
用户不需要改这个文件。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import yaml

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- config
def load_config(path: str | None = None) -> Dict[str, Any]:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """把 config.paths 里的相对路径解析成绝对路径，并保证目录存在。"""
    rel = cfg["paths"][key]
    p = (ROOT / rel) if not os.path.isabs(rel) else Path(rel)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- io
def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- tokens
def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """粗略估算 token 数，用于 fill-to-budget 的预算控制。
    若有 tiktoken 会用更准的估算；没有也不影响运行。"""
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / chars_per_token))


# ---------------------------------------------------------------- cache
def cache_key(*parts: Any) -> str:
    raw = "||".join(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Cache:
    """磁盘 KV 缓存，用于 API 调用去重。
    同一 (provider, model, prompt, params) 只花一次钱；误改配置会生成新键，安全。"""

    def __init__(self, dirpath: str | Path, enabled: bool = True):
        self.dir = Path(dirpath)
        self.enabled = enabled
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        fp = self.dir / f"{key}.json"
        if fp.exists():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        fp = self.dir / f"{key}.json"
        tmp = fp.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        tmp.replace(fp)


# ---------------------------------------------------------------- misc
class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.ms = int((time.time() - self.t0) * 1000)
        return False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def list_files(d: str | Path, exts: Iterable[str] = (".txt", ".md")) -> List[Path]:
    p = Path(d)
    if not p.exists():
        return []
    out = []
    for ext in exts:
        out.extend(sorted(p.rglob(f"*{ext}")))
    return out


# ------------------------------------------------------------ 索引来源指纹
# 为什么要这个：索引按【段数】判断是否复用，而原著语料和篡改语料的段数几乎一样
# （都是 1294 段）。只比段数的话，跑篡改集时会静默复用原著索引 ——
# 整批 SQ2 数据全废且极难察觉。故用语料指纹把索引和它的来源语料绑死。
def corpus_fingerprint(corpus_dir: str | Path) -> Dict[str, Any]:
    p = Path(corpus_dir)
    files = sorted(p.glob("*.txt")) or sorted(p.glob("*.md"))
    return {
        "corpus": str(p.resolve()),
        "n_files": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "names": [f.name for f in files][:5],
    }


def index_source_ok(idx_dir: Path, corpus_dir: str | Path) -> bool:
    """索引目录里的 source.json 与当前语料指纹一致才允许复用。"""
    fp = Path(idx_dir) / "source.json"
    if not fp.exists():
        return False
    try:
        old = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return old.get("bytes") == corpus_fingerprint(corpus_dir).get("bytes") \
        and old.get("corpus") == str(Path(corpus_dir).resolve())


def write_index_source(idx_dir: Path, corpus_dir: str | Path) -> None:
    fp = Path(idx_dir) / "source.json"
    fp.write_text(json.dumps(corpus_fingerprint(corpus_dir), ensure_ascii=False, indent=2),
                  encoding="utf-8")
