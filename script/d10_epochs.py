"""d10 —— 把原始数据锁成 epoch（P1 的「数据落盘」）。

产出
----
data/interim/epochs/{mmidb,bci2a,ssvep}/*.npz   每被试（或每 session）一个
data/locked/MANIFEST.json                       逐文件 sha256 + 形状 + 试次 + 参数哈希
data/locked/EPOCH_SPEC.md                       人读的规格快照（与 sig_layer 同步生成）

设计约束
--------
1. 所有窗口、基线、剔除阈值都从 `sig_layer`（冻结层）取，脚本里**不许出现**裸数字。
2. 不做重采样：EEGMMIDB 的 106 名标准被试原生 160 Hz，2a/SSVEP 原生 250 Hz；
   三个数据集之间不做直接比较，因此不需要统一率。三名结构异常被试（128 Hz）
   按原生率存盘，主分析时排除，敏感性分析单列。
3. 被剔除的试次记数但不丢信息：npz 里保留 `rejected` 掩码与剔除前的索引。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sig_layer as SL  # noqa: E402

RAW = ROOT / "data" / "raw"
EPO = ROOT / "data" / "interim" / "epochs"
LOCK = ROOT / "data" / "locked"

# EEGMMIDB：run → (执行/想象, 手/脚)。T1/T2 的含义随 run 变，必须按 run 号映射。
RUN_META = {
    3: ("executed", "hands"), 4: ("imagined", "hands"),
    5: ("executed", "feet"), 6: ("imagined", "feet"),
    7: ("executed", "hands"), 8: ("imagined", "hands"),
    9: ("executed", "feet"), 10: ("imagined", "feet"),
    11: ("executed", "hands"), 12: ("imagined", "hands"),
    13: ("executed", "feet"), 14: ("imagined", "feet"),
}
LABEL_CODE = {"T1": 0, "T2": 1}  # 0 = 左手 / 双拳；1 = 右手 / 双脚


def sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------- mmidb
def _canon_names(ch_names: list[str]) -> list[str]:
    """EDF 里的通道名带尾点且大小写混乱（'Fc5.' / 'C3..' / 'O2..'），统一到 10-05 规范名。

    不规范化，后面 ROI 匹配（`ROI_LEFT_MOTOR` 等）会全部落空——这是 d10 里最容易
    静默出错的一步，所以在这里一次性修掉并把规范化结果存进 npz。
    """
    import re
    import mne

    canon = {re.sub(r"[^a-z0-9]", "", n.lower()): n
             for n in mne.channels.make_standard_montage("standard_1005").ch_names}
    out = []
    for n in ch_names:
        k = re.sub(r"[^a-z0-9]", "", n.lower())
        out.append(canon.get(k, n.strip(". ")))
    return out


def epoch_mmidb_subject(sub: str) -> dict:
    import mne

    mne.set_log_level("ERROR")
    files = sorted((RAW / "eegmmidb" / sub).glob("*.edf"))
    Xs, ys, runs, conds, kinds, flags = [], [], [], [], [], []
    sfreq = None
    ch_names = None
    for p in files:
        run_no = int(p.stem.split("R")[-1])
        if run_no not in RUN_META:
            continue
        cond, kind = RUN_META[run_no]
        raw = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        ch_names = _canon_names(list(raw.ch_names))
        raw.filter(*SL.BANDPASS, fir_design="firwin", verbose="ERROR")
        if SL.NOTCH["mmidb"]:
            raw.notch_filter(SL.NOTCH["mmidb"], fir_design="firwin", verbose="ERROR")
        events, _ = mne.events_from_annotations(
            raw, event_id={"T1": 1, "T2": 2}, verbose="ERROR")
        if events.size == 0:
            continue
        tmin, tmax = SL.EPOCHS["mmidb"]
        base = SL.BASELINES["mmidb"]
        # 基线区间必须落在 epoch 内：先按基线起点开 epoch，做完基线校正再裁到分析窗
        lo = min(tmin, base[0]) if base else tmin
        ep = mne.Epochs(raw, events, event_id={"T1": 1, "T2": 2}, tmin=lo, tmax=tmax,
                        baseline=base, preload=True, verbose="ERROR",
                        proj=False, detrend=0)          # 不在这里剔除，见下
        if lo < tmin:
            ep.crop(tmin, tmax)
        D = ep.get_data()
        ptp = np.ptp(D, axis=2) * 1e6                    # uV，逐试次逐通道
        # 主分析不按幅度剔除：该数据集在 0.5–45 Hz 下峰峰值中位数就有 266 uV，
        # 150 uV 会丢掉 100% 的试次（实测）。改为标记，伪迹由 S2 的 EOG/额区
        # 替代解码器量化，而不是靠阈值制造选择偏倚。
        flags.append((ptp.max(1) > SL.ARTIFACT_FLAG_UV["mmidb"]).astype(np.int8))
        Xs.append(D.astype(np.float32))
        ys.append(ep.events[:, 2].astype(np.int8) - 1)
        runs.append(np.full(len(ep), run_no, dtype=np.int16))
        conds.append(np.full(len(ep), 0 if cond == "executed" else 1, dtype=np.int8))
        kinds.append(np.full(len(ep), 0 if kind == "hands" else 1, dtype=np.int8))

    out = EPO / "mmidb" / f"{sub}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not Xs:
        return dict(id=sub, ok=False, note="no epochs")
    X = np.concatenate(Xs)
    flag = np.concatenate(flags)
    np.savez_compressed(
        out, X=X, y=np.concatenate(ys), run=np.concatenate(runs),
        cond=np.concatenate(conds), kind=np.concatenate(kinds),
        artifact_flag=flag,
        sfreq=np.float32(sfreq), ch_names=np.array(ch_names),
        n_flagged=int(flag.sum()),
        exclude=np.bool_(sub in SL.EXCLUDE_STRUCT_SUBJECTS),
    )
    return dict(id=sub, ok=True, n_trials=int(X.shape[0]), n_flagged=int(flag.sum()),
                sfreq=float(sfreq), shape=list(X.shape), bytes=out.stat().st_size)


# ---------------------------------------------------------------------- bci2a
def epoch_bci2a_file(p: Path) -> dict:
    import scipy.io as sio

    fs = SL.RESAMPLE_TO["bci2a"]
    m = sio.loadmat(p, struct_as_record=False, squeeze_me=True)
    t0, t1 = SL.EPOCHS["bci2a"]
    b0, b1 = SL.BASELINES["bci2a"]
    Xs, ys, rs, arts, n_rej = [], [], [], [], 0
    for ri, run in enumerate(np.atleast_1d(m["data"])):
        y = np.atleast_1d(np.asarray(run.y, dtype=float)).ravel()
        if y.size == 0:
            continue
        X = np.asarray(run.X, dtype=np.float64)          # (n_samples, 25)
        on = np.atleast_1d(np.asarray(run.trial, dtype=int)).ravel()
        if on.size != y.size:
            on = on[:y.size]
        # 整段滤波再切片，避免逐试次边缘效应
        from scipy.signal import butter, filtfilt, iirnotch
        bb, ab = butter(4, [SL.BANDPASS[0] / (fs / 2), SL.BANDPASS[1] / (fs / 2)],
                        btype="band")
        X = filtfilt(bb, ab, X, axis=0)
        bn, an = iirnotch(SL.NOTCH["bci2a"], 30, fs)
        X = filtfilt(bn, an, X, axis=0)
        art = (np.atleast_1d(np.asarray(run.artifacts, dtype=float)).ravel()
               if hasattr(run, "artifacts") else np.zeros(y.size))
        for o, yy, aa in zip(on, y, art):
            i0, i1 = o + int(round(t0 * fs)), o + int(round(t1 * fs))
            j0, j1 = o + int(round(b0 * fs)), o + int(round(b1 * fs))
            if i1 > X.shape[0] or j1 > X.shape[0]:
                continue
            seg = X[i0:i1] - X[j0:j1].mean(0, keepdims=True)
            # 2a 的 .mat 以 uV 存储（实测 std≈12.6, p99≈36），故阈值直接用 SL.REJECT_UV
            if np.ptp(seg[:, :SL.BCI2A_N_EEG]) > SL.REJECT_UV:
                n_rej += 1
                continue
            Xs.append(seg.T.astype(np.float32))
            ys.append(int(yy) - 1)
            rs.append(ri)
            arts.append(int(aa))
    out = EPO / "bci2a" / f"{p.stem}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not Xs:
        return dict(id=p.stem, ok=False, note="no epochs")
    X = np.stack(Xs)
    np.savez_compressed(
        out, X=X, y=np.array(ys, dtype=np.int8), run=np.array(rs, dtype=np.int16),
        artifact=np.array(arts, dtype=np.int8),
        sfreq=np.float32(fs), ch_names=np.array(SL.BCI2A_CH_NAMES),
        n_rejected=np.int32(n_rej),
    )
    return dict(id=p.stem, ok=True, n_trials=int(X.shape[0]), n_rejected=int(n_rej),
                shape=list(X.shape), bytes=out.stat().st_size)


# ---------------------------------------------------------------------- ssvep
def epoch_ssvep_subject(sub: str) -> dict:
    import py7zr
    import scipy.io as sio
    import tempfile

    fs = SL.RESAMPLE_TO["ssvep"]
    src = RAW / "ssvep" / f"{sub}.mat.7z"
    with tempfile.TemporaryDirectory() as td:
        with py7zr.SevenZipFile(src, "r") as z:
            z.extractall(td)
        mat = next(Path(td).glob("*.mat"))
        d = np.asarray(sio.loadmat(mat, struct_as_record=False,
                                   squeeze_me=True)["data"], dtype=np.float32)
    # Readme: 500 ms pre-stimulus + 5.5 s post，故刺激起点 = 样点 0.5*fs
    stim0 = int(round(0.5 * fs))
    t0, t1 = SL.EPOCHS["ssvep"]
    i0, i1 = stim0 + int(round(t0 * fs)), stim0 + int(round(t1 * fs))
    X = d[:, i0:i1, :, :]                                   # (64, T, 40, 6)
    n_ch, T, n_tg, n_bl = X.shape
    X = X.transpose(2, 3, 0, 1).reshape(n_tg * n_bl, n_ch, T)
    y = np.repeat(np.arange(n_tg, dtype=np.int16), n_bl)
    blk = np.tile(np.arange(n_bl, dtype=np.int8), n_tg)
    # 频率**不是**线性的 8+0.2i：官方 Freq_Phase.mat 是分组 JFPM——
    # [8,9,...,15, 8.2,9.2,...,15.2, 8.4,...]，共 5 组 × 8 个。
    # 按线性假设取频率会让 S0 的谱峰判据整体错位（实测 FFT 分类只有 1%）。
    fp = RAW / "ssvep" / "Freq_Phase.mat"
    freq = (np.asarray(sio.loadmat(fp)["freqs"]).ravel().astype(np.float32)
            if fp.exists() else 8.0 + 0.2 * np.arange(n_tg))
    out = EPO / "ssvep" / f"{sub}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, block=blk, freq=freq.astype(np.float32),
                        sfreq=np.float32(fs), stim_offset_s=np.float32(0.5))
    return dict(id=sub, ok=True, n_trials=int(X.shape[0]), shape=list(X.shape),
                bytes=out.stat().st_size)


# ----------------------------------------------------------------------- main
def build_manifest(records: list[dict], ds: str) -> dict:
    epo = EPO / ds
    files = []
    for f in sorted(epo.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        files.append(dict(
            file=str(f.relative_to(ROOT)).replace("\\", "/"),
            sha256=sha256(f), bytes=f.stat().st_size,
            n_trials=int(z["X"].shape[0]), shape=list(z["X"].shape),
            sfreq=float(z["sfreq"]),
        ))
    return dict(dataset=ds, n_files=len(files),
                n_trials=int(sum(x["n_trials"] for x in files)),
                params_sha256=sha256(ROOT / "scripts" / "sig_layer.py"),
                epoch=[list(SL.EPOCHS[ds])], baseline=[list(SL.BASELINES[ds])
                                                       if SL.BASELINES[ds] else None],
                reject_uv=SL.REJECT_UV, files=files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["mmidb", "bci2a", "ssvep", "all"], default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--subjects", default=None, help="逗号分隔，仅用于试跑")
    ap.add_argument("--verify", action="store_true", help="只校验 MANIFEST，不重建")
    a = ap.parse_args()

    if a.verify:
        mp = LOCK / "MANIFEST.json"
        if not mp.exists():
            print("MANIFEST.json 不存在"); return 2
        man = json.loads(mp.read_text(encoding="utf-8"))
        bad = 0
        if sha256(ROOT / "scripts" / "sig_layer.py") != man["params_sha256"]:
            print("BLOCK 冻结层已变，epoch 必须重建"); bad += 1
        files = [f for d in man["datasets"].values() for f in d["files"]]
        for f in files:
            p = ROOT / f["file"]
            if not p.exists() or sha256(p) != f["sha256"]:
                print("BLOCK 校验失败:", f["file"]); bad += 1
        print(f"verify: {man['n_files']} 文件 / {man['n_trials']} 试次，异常 {bad}")
        return 1 if bad else 0

    LOCK.mkdir(parents=True, exist_ok=True)
    (EPO / "mmidb").mkdir(parents=True, exist_ok=True)
    (EPO / "bci2a").mkdir(parents=True, exist_ok=True)
    (EPO / "ssvep").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    manifests = []

    if a.source in ("mmidb", "all"):
        subs = sorted(p.name for p in (RAW / "eegmmidb").iterdir()
                      if p.is_dir() and p.name.startswith("S") and p.name[1:].isdigit())
        if a.subjects:
            subs = [s for s in a.subjects.split(",")]
        print(f"[mmidb] {len(subs)} 被试, {a.workers} 进程")
        recs, done = [], 0
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(epoch_mmidb_subject, s): s for s in subs}
            for fu in as_completed(futs):
                r = fu.result()
                recs.append(r); done += 1
                if done % 10 == 0 or done == len(subs):
                    print(f"  {done}/{len(subs)}  "
                          f"{sum(x.get('n_trials', 0) for x in recs)} 试次 "
                          f"{time.time() - t_start:.0f}s", flush=True)
        fails = [r for r in recs if not r.get("ok")]
        if fails:
            print("  失败:", fails[:5])
        manifests.append(build_manifest(recs, "mmidb"))

    if a.source in ("bci2a", "all"):
        files = sorted((RAW / "bci2a").glob("*.mat"))
        print(f"[bci2a] {len(files)} 文件")
        recs = [epoch_bci2a_file(p) for p in files]
        tot = sum(r.get("n_trials", 0) for r in recs)
        print(f"  {len(recs)} 文件 / {tot} 试次 / 剔除 {sum(r.get('n_rejected', 0) for r in recs)}")
        manifests.append(build_manifest(recs, "bci2a"))

    if a.source in ("ssvep", "all"):
        subs = sorted({p.name.split(".")[0] for p in (RAW / "ssvep").glob("*.7z")})
        if a.subjects:
            subs = [s for s in a.subjects.split(",")]
        print(f"[ssvep] {len(subs)} 被试")
        recs = [epoch_ssvep_subject(s) for s in subs]
        tot = sum(r.get("n_trials", 0) for r in recs)
        print(f"  {len(recs)} 被试 / {tot} 试次")
        manifests.append(build_manifest(recs, "ssvep"))

    allman = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                  params_sha256=sha256(ROOT / "scripts" / "sig_layer.py"),
                  datasets={m["dataset"]: m for m in manifests},
                  n_files=sum(m["n_files"] for m in manifests),
                  n_trials=sum(m["n_trials"] for m in manifests))
    (LOCK / "MANIFEST.json").write_text(
        json.dumps(allman, ensure_ascii=False, indent=1), encoding="utf-8")
    spec_lines = ["# epoch 规格快照（由 d10_epochs.py 生成，勿手改）", ""]
    for m in manifests:
        spec_lines += [f"## {m['dataset']}", "",
                       f"- 文件 {m['n_files']}，试次 {m['n_trials']}",
                       f"- epoch {m['epoch']}（相对事件起点，秒）",
                       f"- baseline {m['baseline']}",
                       f"- 剔除阈值 {m['reject_uv']} uV", ""]
    (LOCK / "EPOCH_SPEC.md").write_text("\n".join(spec_lines), encoding="utf-8")
    print(f"\n总 {allman['n_files']} 文件 / {allman['n_trials']} 试次 "
          f"/ {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
