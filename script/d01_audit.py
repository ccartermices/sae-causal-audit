"""D0/P0 审计：把「数据到底长什么样」变成可复算的事实文件。

产出（全部由脚本生成，人工不得编辑）：
  data/registry_facts.json     各源的结构性事实（被试数/通道/采样率/试次数）
  data/audit_annotations.csv   逐 run 的事件计数与时长（EEGMMIDB）
  data/audit_2a_runs.csv       逐 session 的类别计数（BCI IV-2a）

为什么单独一层
--------------
登记表只吃 `data/` 下的文件。若让登记表直接读 1526 个 EDF，构建会慢到不可用；
所以先把原始数据审计成两个小文件，登记表再读小文件（C7 毒化小文件即可证明绑定）。
原始字节的完整性由 `check_registry.py` 的 C10 用 MANIFEST_raw.json 的 sha256 守。

用法
----
    python scripts/d01_audit.py                 # 全部
    python scripts/d01_audit.py --only facts
    python scripts/d01_audit.py --only annotations
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (REPORTS / "d01_audit.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------- mmidb
def mmidb_runs() -> list[dict]:
    import mne

    rows = []
    edfs = sorted((RAW / "eegmmidb").glob("S*/*.edf"))
    for i, p in enumerate(edfs, 1):
        try:
            raw = mne.io.read_raw_edf(p, preload=False, verbose="ERROR")
            from collections import Counter

            c = Counter(raw.annotations.description)
            dur = raw.n_times / raw.info["sfreq"]
            rows.append(dict(
                subject=p.parent.name, run=p.stem.split("R")[-1],
                file=p.name, n_chans=len(raw.ch_names),
                sfreq=raw.info["sfreq"], duration_s=round(dur, 2),
                n_T0=c.get("T0", 0), n_T1=c.get("T1", 0), n_T2=c.get("T2", 0),
                n_other=len(raw.annotations) - c.get("T0", 0) - c.get("T1", 0) - c.get("T2", 0),
            ))
        except Exception as ex:  # noqa: BLE001
            rows.append(dict(subject=p.parent.name, run=p.stem.split("R")[-1], file=p.name,
                             n_chans=-1, sfreq=-1, duration_s=-1, n_T0=-1, n_T1=-1, n_T2=-1,
                             n_other=-1, error=f"{type(ex).__name__}: {ex}"))
        if i % 150 == 0:
            log(f"mmidb annotations: {i}/{len(edfs)}")
    return rows


# ----------------------------------------------------------------------- 2a
def bci2a_runs() -> list[dict]:
    import numpy as np
    import scipy.io as sio

    rows = []
    for p in sorted((RAW / "bci2a").glob("*.mat")):
        m = sio.loadmat(p, struct_as_record=False, squeeze_me=True)
        for ri, run in enumerate(np.atleast_1d(m["data"])):
            # 这个镜像里 y 是标签数组（1..4），不是 BNCI Horizon 那种 T1/T2 结构体。
            # T 文件的头几个 run 是**无标签校准**（y 为空），必须单独计数，不能当 0 试次。
            y = np.atleast_1d(np.asarray(run.y, dtype=float)).ravel()
            y = y[np.isfinite(y)].astype(int)
            counts = {str(int(k)): int((y == k).sum()) for k in np.unique(y)} if y.size else {}
            arts = (np.atleast_1d(np.asarray(run.artifacts, dtype=float)).ravel()
                    if hasattr(run, "artifacts") else np.array([]))
            x = run.X
            rows.append(dict(
                file=p.name, run=ri, n_samples=int(x.shape[0]), n_chans=int(x.shape[1]),
                sfreq=float(run.fs), classes="/".join(map(str, np.atleast_1d(run.classes))),
                labeled=int(y.size > 0), n_trials=int(y.size),
                counts=json.dumps(counts, ensure_ascii=False),
                # artifacts 是逐试次的 0/1 标记：这里要的是「被标伪迹的试次数」，
                # 不是数组长度（长度恒等于试次数，曾导致伪迹率被算成 100%）
                n_artifacts=int(np.nansum(arts)) if arts.size else 0,
            ))
    return rows


# ------------------------------------------------------------------- ssvep
def ssvep_facts() -> dict:
    import py7zr
    import scipy.io as sio
    import tempfile

    p = RAW / "ssvep" / "S1.mat.7z"
    if not p.exists():
        return {}
    with tempfile.TemporaryDirectory() as td:
        with py7zr.SevenZipFile(p, "r") as z:
            names = z.getnames()
            z.extract(path=td, targets=names)
        m = sio.loadmat(str(Path(td) / names[0]), struct_as_record=False, squeeze_me=True)
    d = m["data"]
    return dict(shape=list(d.shape), n_chans=int(d.shape[0]), n_samples=int(d.shape[1]),
                n_targets=int(d.shape[2]), n_blocks=int(d.shape[3]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "facts", "annotations"])
    a = ap.parse_args()

    facts_p = DATA / "registry_facts.json"
    facts = json.loads(facts_p.read_text(encoding="utf-8")) if facts_p.exists() else {}
    facts["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if a.only in ("all", "annotations"):
        rows = mmidb_runs()
        if rows:
            with (DATA / "audit_annotations.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            log(f"wrote audit_annotations.csv rows={len(rows)}")
            facts["mmidb_run_table"] = dict(
                n_rows=len(rows),
                n_edf=sum(1 for r in rows if r["n_chans"] > 0),
                n_error=sum(1 for r in rows if r["n_chans"] < 0),
                subjects=sorted({r["subject"] for r in rows}),
            )
        rows2 = bci2a_runs()
        if rows2:
            with (DATA / "audit_2a_runs.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows2[0].keys()))
                w.writeheader()
                w.writerows(rows2)
            log(f"wrote audit_2a_runs.csv rows={len(rows2)}")

    if a.only in ("all", "facts"):
        # ---- mmidb
        subs = sorted({p.parent.name for p in (RAW / "eegmmidb").glob("S*/*.edf")})
        edfs = list((RAW / "eegmmidb").glob("S*/*.edf"))
        n_ch, sf = 0, 0.0
        if edfs:
            import mne
            r = mne.io.read_raw_edf(edfs[0], preload=False, verbose="ERROR")
            n_ch, sf = len(r.ch_names), float(r.info["sfreq"])
        facts["mmidb"] = dict(n_subjects=len(subs), n_edf=len(edfs), n_chans=n_ch, sfreq=sf)

        # ---- 2a
        mats = sorted((RAW / "bci2a").glob("*.mat"))
        n_sub = len({p.stem[:3] for p in mats})
        sr = facts.get("_2a_runs_tmp", {})
        n_ch = sf2 = 0
        n_runs = 0
        if mats:
            import scipy.io as sio
            import numpy as np
            m = sio.loadmat(mats[0], struct_as_record=False, squeeze_me=True)
            runs = np.atleast_1d(m["data"])
            n_runs = len(runs)
            n_ch = int(runs[0].X.shape[1])
            sf2 = float(runs[0].fs)
        facts["bci2a"] = dict(n_files=len(mats), n_subjects=n_sub,
                             n_chans_total=n_ch, n_chans_eog=3,
                             n_chans_eeg=max(n_ch - 3, 0),
                             sfreq=sf2, n_runs_per_session=n_runs)

        # 从审计表汇总「有标签 run 数 / 试次数」。T 文件开头是无标签校准 run，
        # 所以「总 run 数」不等于「可用 run 数」——这两个数必须分开记，
        # 否则 288 试次这个论文里要引的数字会被校准段稀释。
        audit_2a = DATA / "audit_2a_runs.csv"
        if audit_2a.exists():
            import csv as _csv

            with audit_2a.open(encoding="utf-8") as f:
                rr = list(_csv.DictReader(f))
            per_file: dict[str, dict] = {}
            for r in rr:
                d = per_file.setdefault(r["file"], dict(runs=0, labeled=0, trials=0))
                d["runs"] += 1
                if r["labeled"] == "1":
                    d["labeled"] += 1
                    d["trials"] += int(r["n_trials"])
            lab = sorted({v["labeled"] for v in per_file.values()})
            tri = sorted({v["trials"] for v in per_file.values()})
            facts["bci2a"].update(
                n_labeled_runs_per_session=lab[-1] if len(lab) == 1 else -1,
                n_trials_per_session=tri[-1] if len(tri) == 1 else -1,
                n_labeled_runs_values=lab,
                n_trials_values=tri,
                files_with_fewer_than_9_runs=sorted(
                    k for k, v in per_file.items() if v["runs"] != 9),
            )

        # ---- ssvep
        sf3 = ssvep_facts()
        n7 = len(list((RAW / "ssvep").glob("S*.mat.7z")))
        if sf3:
            sf3.update(n_subjects_files=n7, sfreq=250.0)
            facts["ssvep"] = sf3

        facts_p.write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
        log(f"wrote registry_facts.json keys={list(facts)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
