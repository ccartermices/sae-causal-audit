"""D0 取数（直连 HTTP 版）：三个公开 EEG 数据源 -> data/raw/

为什么不用 snapshot_download
---------------------------
HF 的 snapshot 会在目标目录里建 `.cache/huggingface/download/` 并在每个文件上
反复创建/清理临时文件与 .lock；本机对批量删除有拦截，109 被试（3052 个文件）
的下载在跑到第 9 个文件时被中断。直连 `resolve/main/<path>` 逐文件拉取没有这个
问题，而且天然可断点续传（已存在且大小一致就跳过）。

三个源（2026-11-17 实测）：
  - physionet.org / bnci-horizon-2020.eu / github / kaggle：不可达
  - hf-mirror.com：可达（EEGMMIDB 与 BCI IV-2a 的镜像都在上面）
  - bci.med.tsinghua.edu.cn：可达（SSVEP Benchmark）

用法
----
    python scripts/d00_fetch.py --source all
    python scripts/d00_fetch.py --source mmidb --subjects S001,S002
    python scripts/d00_fetch.py --source ssvep --n-subjects 10
    python scripts/d00_fetch.py --source 2a --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
LOG = REPORTS / "d00_fetch.log"

HF = "https://hf-mirror.com/datasets"
TSINGHUA = "https://bci.med.tsinghua.edu.cn/upload/yijun"

REPOS = {
    "mmidb": "raei/Schalk-2009-EEGMotorMovementImagery",
    "2a": "raei/Brunner-2008-BCICompetition2008GrazDataSetA",
}

_lock = threading.Lock()
mbytes = 0


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def list_tree(repo: str) -> list[dict]:
    """递归列出 repo 内所有文件（含大小）。HF 的 tree API 每页 1000 条。"""
    out, url = [], f"https://hf-mirror.com/api/datasets/{repo}/tree/main?recursive=true"
    while url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        for it in r.json():
            if it["type"] == "file":
                out.append({"path": it["path"], "size": it.get("size", 0)})
        link = r.headers.get("Link", "")
        url = None
        if 'rel="next"' in link:
            url = link.split("<")[1].split(">")[0]
    return out


def get_one(url: str, dest: Path, expect: int, retries: int = 6) -> str:
    """下载单文件；已存在且大小一致则跳过。返回状态字符串。

    并发过高会触发 hf-mirror 的 429 限流（实测 24 并发在 521/1526 处被拦），
    因此 429 单独退避：优先尊重 Retry-After，否则 10→20→40 s 指数退避。
    """
    global mbytes
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if not expect or dest.stat().st_size == expect:
            return "skip"
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 0)) or 10 * (2 ** (attempt - 1))
                    time.sleep(min(wait, 120))
                    continue
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                n = 0
                with tmp.open("wb") as f:
                    for blk in r.iter_content(1 << 20):
                        f.write(blk)
                        n += len(blk)
                tmp.replace(dest)
            with _lock:
                mbytes += n
            return "ok"
        except Exception as ex:  # noqa: BLE001
            if attempt == retries:
                return f"FAIL {type(ex).__name__}: {ex}"
            time.sleep(min(2.0 * attempt, 20))
    return "FAIL rate-limited"


def fetch_hf(source: str, patterns: list[str], dest_root: Path, workers: int) -> list[Path]:
    repo = REPOS[source]
    files = list_tree(repo)
    picked = [f for f in files if any(f["path"].startswith(p) or f["path"] == p for p in patterns)]
    log(f"{source}: {len(picked)}/{len(files)} files selected from {repo}")
    todo = [(f, dest_root / f["path"], f"{HF}/{repo}/resolve/main/{f['path']}") for f in picked]
    done: list[Path] = []
    fails = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(get_one, u, d, s): d for f, d, u in todo for s in [f["size"]]}
        for i, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            st = fut.result()
            if st.startswith("FAIL"):
                fails.append((p, st))
            else:
                done.append(p)
            if i % 200 == 0 or i == len(futs):
                dt = time.time() - t0
                log(f"{source}: {i}/{len(futs)} done, {mbytes/1e6:.0f} MB, "
                    f"{mbytes/1e6/max(dt,1):.1f} MB/s, fails={len(fails)}")
    for p, st in fails[:10]:
        log(f"{source}: FAIL {p.relative_to(RAW)} :: {st}")
    return done


def fetch_tsinghua(names: list[str], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for name in names:
        p = dest / name
        st = get_one(f"{TSINGHUA}/{name}", p, 0)
        log(f"ssvep: {name} -> {st} ({p.stat().st_size/1e6:.1f} MB)" if p.exists() else f"ssvep: {name} -> {st}")
        if p.exists():
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all", choices=["all", "mmidb", "2a", "ssvep"])
    ap.add_argument("--subjects", default="")
    ap.add_argument("--n-subjects", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    mpath = RAW / "MANIFEST_raw.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    t0 = time.time()
    new: list[Path] = []

    if a.source in ("all", "mmidb"):
        subs = [s.strip() for s in a.subjects.split(",") if s.strip()]
        new += fetch_hf("mmidb", [f"{s}/" for s in subs] if subs else ["S0", "S1"], RAW / "eegmmidb", a.workers)

    if a.source in ("all", "2a"):
        new += fetch_hf("2a", [""], RAW / "bci2a", a.workers)

    if a.source in ("all", "ssvep"):
        names = [f"S{i}.mat.7z" for i in range(1, a.n_subjects + 1)]
        names += ["Freq_Phase.mat", "64-channels.loc", "Sub_info.txt", "Readme.txt"]
        new += fetch_tsinghua(names, RAW / "ssvep")

    log(f"manifest: hashing {len(new)} new files ...")
    for p in new:
        rel = p.relative_to(RAW).as_posix()
        manifest[rel] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    tot = sum(v["bytes"] for v in manifest.values())
    log(f"DONE files={len(manifest)} total={tot/1e9:.2f} GB elapsed={(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
