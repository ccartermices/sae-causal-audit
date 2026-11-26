"""d31 — T1 正式解码：BCI2a 四类运动想象，T 段训 / E 段测。

为什么要这一步
--------------
到 T1 之前，所有「SAE 有没有用」的读数都建在**注入的已知源**上（T0-C/E/G、T1 边界）。
那测的是仪器能力。真正的科学问题要换成**真实标签**：解码器能不能从真实 EEG 里
把四类运动想象分开，以及 SAE 的消融是否改变这个能力。若解码器本身接不上标签，
后面所有消融读数都无法与噪声区分——所以先立这个基线。

协议（`03_判据与闸门` §5，冻结）
------------------------------
* 训练只用 `*T.npz`（runs 3–7）；测试只用 `*E.npz`（runs 3–7 的另一场）。
* 超参（训练轮数）只在 T 段内用 **5 折分层 CV** 选；E 段只在最后前向一次。
* 每导 z 化，均值/方差**只由 T 段估**，再作用到 E 段。
* 剔除：epoch 锁已按 150 µV 峰峰值做过（`artifact` 标记），这里只用 `artifact==0`。

跑什么
------
1. **主结果**：22 导，9 被试 × 3 种子，平衡准确率 + Cohen kappa。
2. **N4**：只用 3 导 EOG 训练同一架构。期望「显著高于机会、远低于全通道」；
   若接近全通道，说明 T1 的解码能力主要由眼电驱动，整体结论需重写。
3. **N1**：被试内置换标签后重训（3 次置换）。期望回到机会水平 0.25。

产物：`probe/t1_decoder.json`、`reports/T1_decoder.md`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import det_lock  # noqa: E402
import sig_layer as SL  # noqa: E402

EPO = ROOT / "data" / "interim" / "epochs" / "bci2a"
PROBE = ROOT / "probe"
REPORTS = ROOT / "reports"


# ------------------------------------------------------------------ 数据读取
def load_session(sub: str, sess: str):
    """读一场（T 或 E）；只保留 artifact==0 的试次。"""
    z = np.load(EPO / f"{sub}{sess}.npz", allow_pickle=True)
    keep = z["artifact"] == 0
    return dict(X=z["X"][keep].astype(np.float32), y=z["y"][keep].astype(np.int64),
                run=z["run"][keep], ch=[str(c) for c in z["ch_names"]],
                n_drop=int((~keep).sum()), sfreq=float(z["sfreq"]))


def zscore(Xtr: np.ndarray, Xte: np.ndarray):
    """逐导 z 化；统计量只来自训练段（防止 E 段信息泄漏）。"""
    mu = Xtr.mean((0, 2), keepdims=True)
    sd = Xtr.std((0, 2), keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _flat(X, idx):
    """(n, C, T) → (n·T, |idx|)，时间为主序。"""
    return X[:, idx, :].transpose(0, 2, 1).reshape(-1, len(idx))


def _unflat(Y, n, T, C):
    """(n·T, C) → (n, C, T)。"""
    return Y.reshape(n, T, C).transpose(0, 2, 1)


def regress_out(Xtr, Xte, full_eog_idx, keep_idx):
    """把 EEG 逐导对 3 导 EOG 做线性回归并减掉预测部分（Gratton 型回归矫正）。

    回归系数**只在训练段**上估（训练试次的时间点全部拼接），再原样作用到 E 段。
    返回 (Xtr_clean, Xte_clean, r2_per_chan)：r2 是 EOG 能解释的方差占比，
    用来判断「EEG 里有多少只是眼电的线性像」。
    """
    ntr, _, T = Xtr.shape
    nte = Xte.shape[0]
    Etr, Ete = _flat(Xtr, full_eog_idx), _flat(Xte, full_eog_idx)
    Atr = np.c_[Etr, np.ones(len(Etr))]
    Ate = np.c_[Ete, np.ones(len(Ete))]
    Ytr, Yte = _flat(Xtr, keep_idx), _flat(Xte, keep_idx)
    coef, *_ = np.linalg.lstsq(Atr, Ytr, rcond=None)
    r2 = 1.0 - ((Ytr - Atr @ coef) ** 2).sum(0) / np.maximum((Ytr ** 2).sum(0), 1e-20)
    return (_unflat(Ytr - Atr @ coef, ntr, T, len(keep_idx)),
            _unflat(Yte - Ate @ coef, nte, T, len(keep_idx)),
            r2)


# ------------------------------------------------------------------ 模型与训练
def build_model(n_chans: int, n_times: int, seed: int):
    from braindecode.models import ShallowFBCSPNet
    return ShallowFBCSPNet(n_chans=n_chans, n_outputs=SL.T1_N_CLASSES,
                           n_times=n_times, drop_prob=0.5)


def train_once(Xtr, ytr, Xva, yva, epochs: int, seed: int,
               snapshots: tuple[int, ...] = (), bs: int = 32, lr: float = 1e-3):
    """训练一个解码器；返回 (model, {epoch: 验证平衡准确率})。"""
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    model = build_model(Xtr.shape[1], Xtr.shape[2], seed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.long, device=dev)
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)

    snaps = {}
    n = len(Xt)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i:i + bs].to(dev)
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
        if ep in snapshots:
            snaps[ep] = balanced_acc(model, Xva, yva)
    return model, snaps


def predict(model, X, bs: int = 128) -> np.ndarray:
    import torch
    dev = next(model.parameters()).device
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs], dtype=torch.float32, device=dev)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def balanced_acc(model, X, y) -> float:
    yh = predict(model, X)
    return float(np.mean([(yh[y == c] == c).mean() for c in range(SL.T1_N_CLASSES)
                          if (y == c).any()]))


def cohen_kappa(y_true, y_pred) -> float:
    k = SL.T1_N_CLASSES
    cm = np.zeros((k, k))
    np.add.at(cm, (y_true, y_pred), 1)
    po = np.trace(cm) / cm.sum()
    pe = float((cm.sum(0) * cm.sum(1)).sum() / cm.sum() ** 2)
    return float((po - pe) / max(1 - pe, 1e-12))


# ------------------------------------------------------------------ 单被试流程
def run_subject(sub: str, chans: str = "eeg", seed: int = 0, permute: bool = False,
                cv_folds: int = SL.T1_CV_FOLDS, verbose: bool = True) -> dict:
    from sklearn.model_selection import StratifiedKFold

    tr = load_session(sub, "T")
    te = load_session(sub, "E")
    eog_idx = [i for i, c in enumerate(tr["ch"]) if c in SL.T1_EOG_CH]
    eeg_idx = [i for i, c in enumerate(tr["ch"]) if c not in SL.T1_EOG_CH]
    r2_eog = None
    if chans == "eog":
        sel, Xt_raw, Xe_raw = eog_idx, tr["X"][:, eog_idx], te["X"][:, eog_idx]
    elif chans == "eeg_reg":
        # 先把 EEG 里能被 EOG 线性解释的部分减掉（系数只用 T 段估）
        Xt_raw, Xe_raw, r2_eog = regress_out(tr["X"], te["X"], eog_idx, eeg_idx)
        sel = eeg_idx
    else:
        sel, Xt_raw, Xe_raw = eeg_idx, tr["X"][:, eeg_idx], te["X"][:, eeg_idx]
    Xtr_all, Xte_all = zscore(Xt_raw, Xe_raw)
    ytr_all, yte = tr["y"], te["y"]

    rng = np.random.RandomState(1000 + seed)
    y_fit = rng.permutation(ytr_all) if permute else ytr_all

    # --- 阶段 A：T 段内 5 折，选训练轮数（只看 T，不看 E）
    grid = SL.T1_EPOCH_GRID
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_snap = {g: [] for g in grid}
    for f, (ia, ib) in enumerate(skf.split(Xtr_all, y_fit)):
        _, snaps = train_once(Xtr_all[ia], y_fit[ia], Xtr_all[ib], y_fit[ib],
                              epochs=max(grid), seed=seed * 100 + f, snapshots=grid)
        for g in grid:
            fold_snap[g].append(snaps[g])
    cv_mean = {g: float(np.mean(v)) for g, v in fold_snap.items()}
    best_ep = max(grid, key=lambda g: cv_mean[g])

    # --- 阶段 B：全 T 段重训，E 段只前向一次
    model, _ = train_once(Xtr_all, y_fit, Xtr_all[:1], y_fit[:1],
                          epochs=best_ep, seed=seed)
    yh = predict(model, Xte_all)
    bacc = float(np.mean([(yh[yte == c] == c).mean()
                          for c in range(SL.T1_N_CLASSES)]))
    kap = cohen_kappa(yte, yh)

    rec = dict(subject=sub, chans=chans, seed=seed, permute=bool(permute),
               n_train=int(len(Xtr_all)), n_test=int(len(Xte_all)),
               n_drop_T=tr["n_drop"], n_drop_E=te["n_drop"],
               cv_mean={str(k): round(v, 4) for k, v in cv_mean.items()},
               best_epoch=int(best_ep), bal_acc=round(bacc, 4), kappa=round(kap, 4))
    if r2_eog is not None:
        rec["eog_r2_median"] = round(float(np.median(r2_eog)), 4)
        rec["eog_r2_max"] = round(float(np.max(r2_eog)), 4)
    if verbose:
        print(f"  {sub} {chans:3s} seed{seed}{' perm' if permute else ''} | "
              f"CV {cv_mean} → {best_ep} ep | E: bacc {bacc:.3f} κ {kap:+.3f}",
              flush=True)
    return rec


# ------------------------------------------------------------------------ 主
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str, default="A01,A02,A03,A04,A05,A06,A07,A08,A09")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--chans", type=str, default="eeg",
                    choices=("eeg", "eog", "eeg_reg"))
    ap.add_argument("--with-eog", action="store_true",
                    help="额外跑 3 导 EOG（N4）与 EOG 回归矫正（eeg_reg）两条对照")
    ap.add_argument("--permute-reps", type=int, default=3,
                    help="N1 标签置换重复次数；0 表示不跑")
    ap.add_argument("--tag", type=str, default="main")
    a = ap.parse_args()

    info = det_lock.lock(0)
    print(f"[det] {info}", flush=True)

    subs = [s.strip() for s in a.subjects.split(",") if s.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    t0 = time.time()
    out = dict(protocol=dict(cv_folds=SL.T1_CV_FOLDS, epoch_grid=list(SL.T1_EPOCH_GRID),
                             chance=1.0 / SL.T1_N_CLASSES, det=info,
                             split="train=T(E never seen), test=E"),
               runs=[], n1=[])

    for sub in subs:
        for i, sd in enumerate(seeds):
            out["runs"].append(run_subject(sub, chans=a.chans, seed=sd))
            # N4 与 EOG 回归矫正（只在主条件下附带跑，避免重复）
            if a.chans == "eeg" and a.with_eog:
                if i == 0:
                    out["runs"].append(run_subject(sub, chans="eog", seed=0))
                out["runs"].append(run_subject(sub, chans="eeg_reg", seed=sd))

    # N1：置换标签（每被试每个置换一次）
    for rep in range(a.permute_reps):
        for sub in subs:
            out["n1"].append(run_subject(sub, chans=a.chans, seed=rep, permute=True,
                                         verbose=False))

    out["elapsed_s"] = round(time.time() - t0, 1)
    PROBE.mkdir(exist_ok=True)
    (PROBE / f"t1_decoder_{a.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(out, a.tag)
    print(f"[done] {out['elapsed_s']} s", flush=True)
    return 0


def write_report(out: dict, tag: str) -> None:
    def agg(rows):
        v = np.array([r["bal_acc"] for r in rows])
        k = np.array([r["kappa"] for r in rows])
        return v.mean(), v.std(), k.mean()

    lines = ["# T1 解码基线（BCI2a 四类运动想象，T→E）", "",
             f"自动生成，勿手改。确定性锁：`{out['protocol']['det']}`。",
             f"切分：{out['protocol']['split']}；机会水平 {out['protocol']['chance']:.2f}。", ""]

    real = [r for r in out["runs"] if r["chans"] == "eeg" and not r["permute"]]
    eog = [r for r in out["runs"] if r["chans"] == "eog"]
    reg = [r for r in out["runs"] if r["chans"] == "eeg_reg"]
    if real:
        m, s, k = agg(real)
        lines += ["## 主结果（22 导 EEG）", "",
                  "| 被试 | 种子 | 训练轮数 | E 段平衡准确率 | kappa |", "|---|---|---|---|---|"]
        for r in real:
            lines.append(f"| {r['subject']} | {r['seed']} | {r['best_epoch']} | "
                         f"{r['bal_acc']:.3f} | {r['kappa']:+.3f} |")
        lines += ["", f"**合并：平衡准确率 {m:.3f} ± {s:.3f}，kappa {k:+.3f}**"
                      f"（机会 0.250）", ""]
    if eog:
        m, s, k = agg(eog)
        lines += ["## N4 对照（仅 3 导 EOG）", "",
                  f"平衡准确率 {m:.3f} ± {s:.3f}，kappa {k:+.3f}。"
                  f"全通道 kappa {agg(real)[2]:+.3f}。", "",
                  "| 被试 | EEG κ | EOG κ | EOG 可解释方差（中位 R²） |", "|---|---|---|---|"]
        for sub in sorted({r["subject"] for r in eog}):
            ke = np.mean([r["kappa"] for r in real if r["subject"] == sub])
            ko = np.mean([r["kappa"] for r in eog if r["subject"] == sub])
            rr = [r.get("eog_r2_median") for r in reg if r["subject"] == sub]
            lines.append(f"| {sub} | {ke:+.3f} | {ko:+.3f} | "
                         f"{(rr[0] if rr and rr[0] is not None else float('nan')):.3f} |")
        lines.append("")
    if reg:
        m, s, k = agg(reg)
        lines += ["## EOG 回归矫正后（22 导 EEG）", "",
                  f"平衡准确率 {m:.3f} ± {s:.3f}，kappa {k:+.3f}"
                  f"（矫正前 {agg(real)[2]:+.3f}）。", ""]
    if out["n1"]:
        m, s, k = agg(out["n1"])
        lines += ["## N1 对照（被试内置换标签）", "",
                  f"平衡准确率 {m:.3f} ± {s:.3f}，kappa {k:+.3f}"
                  f"（期望回到 {out['protocol']['chance']:.2f}）。", ""]
    (REPORTS / f"T1_decoder_{tag}.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
