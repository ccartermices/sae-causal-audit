"""P0 标签语义侧检：T1/T2 是否携带「左手 vs 右手」的方向信息。

为什么注解计数不够
------------------
`d01_audit.py` 只能证明每个 run 的注解条数/时长/通道数正常，证明不了
"R03 里的 T1 真的是左手"。若 T1/T2 被整体互换，计数完全看不出来。

两个可判定的内部证据（都只用 R03/R07/R11 —— 唯一提供左/右手方向的 run）
--------------------------------------------------------------------------
A. 侧化方向：对侧抑制 ⇒ 左手(T1) 时右半球 mu 更强抑制。
   侧化量 l = log(左簇功率) − log(右簇功率)；标签正确则 mean(l|T1) > mean(l|T2)。
   **只用单通道 C3/C4 会被噪声淹没**（首版 7/11、显著 2/11），故改为簇平均 +
   CAR + 峰峰值剔除。

B. 可分性：若标签真带方向信息，被试内 CSP+LDA 的交叉验证准确率应显著高于 0.5。
   这一条对"方向"不敏感（互换标签也能分），但能证明标签不是噪声。

两者合起来才够：B 说明"有信号"，A 说明"方向对"。

用法 / 产出
-----------
    python scripts/d03_label_sanity.py     →  reports/P0_label_sanity.md
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "eegmmidb"
REPORTS = ROOT / "reports"

DISPUTED = ["S038", "S082", "S088", "S089", "S100", "S104"]
CONTROLS = ["S001", "S002", "S003", "S004", "S005"]
RUNS = [3, 7, 11]
BAND = (8.0, 13.0)
TMIN, TMAX = 0.5, 3.5
REJECT_UV = 150.0

LEFT = ("c3", "fc3", "cp3")
RIGHT = ("c4", "fc4", "cp4")


def _pick(ch_names: list[str], stems: tuple[str, ...]) -> list[str]:
    out = []
    for s in stems:
        for c in ch_names:
            if c.rstrip(".").lower() == s:
                out.append(c)
                break
    return out


def one_subject(sub: str) -> dict | None:
    import mne
    from scipy import stats
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from mne.decoding import CSP

    Xs, ys = [], []
    lrs, labs = [], []
    sfreq = None
    for r in RUNS:
        p = RAW / sub / f"{sub}R{r:02d}.edf"
        if not p.exists():
            continue
        raw = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
        raw.set_eeg_reference("average", verbose="ERROR")
        raw.filter(*BAND, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        ev, eid = mne.events_from_annotations(raw, verbose="ERROR")
        keep = {k: eid[k] for k in ("T1", "T2") if k in eid}
        if not keep:
            continue
        ep = mne.Epochs(raw, ev, keep, tmin=TMIN, tmax=TMAX, baseline=None,
                        preload=True, reject=dict(eeg=REJECT_UV * 1e-6),
                        verbose="ERROR")
        lch = _pick(ep.ch_names, LEFT)
        rch = _pick(ep.ch_names, RIGHT)
        if len(lch) < 2 or len(rch) < 2:
            continue
        for lab in keep:
            sel = ep[lab]
            if len(sel) == 0:
                continue
            d = sel.get_data() ** 2
            pl = d[:, [ep.ch_names.index(c) for c in lch], :].mean(axis=(1, 2))
            pr = d[:, [ep.ch_names.index(c) for c in rch], :].mean(axis=(1, 2))
            lrs.append(np.log(pl) - np.log(pr))
            labs.append(np.full(len(pl), 0 if lab == "T1" else 1))
            Xs.append(sel.get_data())
            ys.append(np.full(len(sel), 0 if lab == "T1" else 1))

    if not lrs:
        return None
    lr = np.concatenate(lrs)
    y = np.concatenate(labs)
    a, b = lr[y == 0], lr[y == 1]
    if a.size < 5 or b.size < 5:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)

    X = np.concatenate(Xs)
    yy = np.concatenate(ys)
    acc = float("nan")
    if min(np.bincount(yy)) >= 4:
        clf = make_pipeline(
            CSP(n_components=4, reg=None, log=True, norm_trace=False),
            LinearDiscriminantAnalysis())
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        scores = cross_val_score(clf, X, yy, cv=cv, n_jobs=1)
        acc = float(np.mean(scores))

    return dict(sub=sub, sfreq=sfreq, n=int(y.size), n_t1=int(a.size), n_t2=int(b.size),
                mean_t1=float(a.mean()), mean_t2=float(b.mean()),
                diff=float(a.mean() - b.mean()), t=float(t), p=float(p),
                direction_ok=bool(a.mean() > b.mean()), acc=acc)


def main() -> int:
    t0 = time.time()
    rows = []
    for s in CONTROLS + DISPUTED:
        r = one_subject(s)
        if r is None:
            print(f"  {s}: 数据不足，跳过")
            continue
        rows.append(r)
        print(f"  {s}: n={r['n']:3d} diff={r['diff']:+.3f} t={r['t']:+.2f} p={r['p']:.3g} "
              f"方向{'✓' if r['direction_ok'] else '✗'} acc={r['acc']:.3f}")

    def grp(s):
        return "对照" if s in CONTROLS else "争议"

    ok_c = sum(r["direction_ok"] for r in rows if r["sub"] in CONTROLS)
    n_c = sum(1 for r in rows if r["sub"] in CONTROLS)
    ok_d = sum(r["direction_ok"] for r in rows if r["sub"] in DISPUTED)
    n_d = sum(1 for r in rows if r["sub"] in DISPUTED)
    sig = sum(r["p"] < 0.05 and r["direction_ok"] for r in rows)
    accs = [r["acc"] for r in rows if np.isfinite(r["acc"])]
    mean_acc = float(np.mean(accs)) if accs else float("nan")
    n_above = sum(a > 0.6 for a in accs)
    # 检验「准确率是否高于机会」：对被试做单样本单侧 t 检验（比把连续量切成 >0.6 更有力）
    from scipy import stats as st
    if len(accs) >= 3:
        tt = st.ttest_1samp(accs, 0.5, alternative="greater")
        acc_t, acc_p = float(tt.statistic), float(tt.pvalue)
        acc_sd = float(np.std(accs, ddof=1))
    else:
        acc_t = acc_p = acc_sd = float("nan")

    L = []
    A = L.append
    A("# P0 标签语义侧检（左手 vs 右手）")
    A("")
    A("> 由 `scripts/d03_label_sanity.py` 生成，只用 R03/R07/R11（唯一区分左/右手的 run）。")
    A("> 预处理：CAR + 8–13 Hz + 峰峰值 150 µV 剔除；窗 0.5–3.5 s。")
    A("")
    A("## A 侧化方向（对侧抑制）")
    A("")
    A("l = log(左簇 C3/FC3/CP3) − log(右簇 C4/FC4/CP4)。标签正确时 mean(l|T1) > mean(l|T2)。")
    A("")
    A("| 被试 | 组 | sfreq | n | mean l (T1) | (T2) | 差 | t | p | 方向 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['sub']} | {grp(r['sub'])} | {r['sfreq']:.0f} | {r['n']} | "
          f"{r['mean_t1']:+.3f} | {r['mean_t2']:+.3f} | {r['diff']:+.3f} | "
          f"{r['t']:+.2f} | {r['p']:.3g} | {'✓' if r['direction_ok'] else '✗'} |")
    A("")
    A("## B 可分性（CSP + LDA，4 折被试内交叉验证）")
    A("")
    A("| 被试 | 组 | 准确率 |")
    A("|---|---|---|")
    for r in rows:
        A(f"| {r['sub']} | {grp(r['sub'])} | {r['acc']:.3f} |")
    A("")
    A("## 结论")
    A("")
    A(f"- 方向正确：对照 **{ok_c}/{n_c}**，争议 **{ok_d}/{n_d}**，全样本显著（p<0.05 且方向对）**{sig}/{len(rows)}**。")
    A(f"- 平均交叉验证准确率 **{mean_acc:.3f}**（机会 0.5）；准确率 >0.6 的被试 **{n_above}/{len(accs)}**"
      + f"；单样本单侧 t 检验（H0: 0.5）t={acc_t:.2f}, p={acc_p:.2g}, SD={acc_sd:.3f}。")
    if ok_c == n_c and ok_d == n_d and n_above >= max(2, len(accs) // 2):
        A("- **判定：标签方向与可分性均成立**，可继续使用；逐试次噪声上界交给 P4 的标签洗牌零对照。")
    elif np.isfinite(acc_p) and acc_p < 0.05:
        A("- **判定：可分性成立但方向不稳**——标签含真实信息，但「T1=左手」这一步在本样本上")
        A("  不能靠 C3/C4 侧化单独确认（样本量与个体差异所致）。处理方式：正文只主张「标签可学」，")
        A("  不主张「方向已核验」；若审稿追问，用 P1 的 CSP 模式可视化补证。")
    else:
        A("- **判定：本样本不足以判定**。CSP+LDA 未显著高于机会水平，说明这条侧检在这个被试量下")
        A("  没有检验力，**不能据此否定标签**。结论按「未核验」写，并在 P4 用标签洗牌给出噪声上界。")
        A("- 下一步（P1）：用 109 人全样本做同一检验，得到人群水平的可分性与方向一致率，再决定正文口径。")
    A("")
    A(f"_耗时 {time.time() - t0:.0f} s。_")

    (REPORTS / "P0_label_sanity.md").write_text("\n".join(L), encoding="utf-8")

    # 摘要落到 data/，供 registry 取用（正文数字只能来自产物）
    import json
    summary = dict(
        n_subjects=len(rows),
        n_direction_ok=sum(r["direction_ok"] for r in rows),
        n_direction_ok_controls=ok_c,
        n_controls=n_c,
        n_direction_ok_disputed=ok_d,
        n_disputed=n_d,
        n_sig=sum(1 for r in rows if r["p"] < 0.05 and r["direction_ok"]),
        acc_mean=round(mean_acc, 4) if np.isfinite(mean_acc) else None,
        acc_t=round(acc_t, 4) if np.isfinite(acc_t) else None,
        acc_p=round(acc_p, 6) if np.isfinite(acc_p) else None,
        acc_sd=round(acc_sd, 4) if np.isfinite(acc_sd) else None,
        n_acc_above_60=n_above,
        n_acc=len(accs),
        subjects=[dict(sub=r["sub"], diff=round(r["diff"], 4), p=round(r["p"], 5),
                       direction_ok=r["direction_ok"],
                       acc=None if not np.isfinite(r["acc"]) else round(r["acc"], 4))
                  for r in rows],
    )
    (ROOT / "data" / "label_sanity.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"wrote reports/P0_label_sanity.md | 方向 对照{ok_c}/{n_c} 争议{ok_d}/{n_d} | "
          f"acc 均值 {mean_acc:.3f}, >0.6 的 {n_above}/{len(accs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
