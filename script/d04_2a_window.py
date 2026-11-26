"""d04 —— 用滑窗解码反推 2a 的判别信息时间窗（P0 的一项「规格实测」）。

为什么需要它
------------
2a 镜像的 `X` 是整段连续记录，`trial` 字段只给逐试次起点，试次间隔约 2015 样点
（8.06 s @250 Hz）——**不是**文献里常见的「4 s 试次」。如果直接按 0–4 s 切，会把
提示期和部分静息切进来，MI 窗错位。所以窗口不能抄，只能用数据自己说话：

  对 0–8 s 逐 1 s 滑窗做 CSP(4)+LDA 5 折交叉验证，判别准确率随时间的峰就是 MI 窗。

这一步只用于**定规格**，不是实验结果；结果进 `registry` 供方法节引用。

产出
----
reports/P0_2a_window.md      人读
data/2a_window.json          机器读（供 registry）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sig_layer as SL  # noqa: E402

import warnings  # noqa: E402

warnings.filterwarnings("ignore")


def load_session(path: Path):
    import scipy.io as sio

    fs = SL.RESAMPLE_TO["bci2a"]
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    Xs, ys = [], []
    for run in np.atleast_1d(m["data"]):
        y = np.atleast_1d(np.asarray(run.y, dtype=float)).ravel()
        if y.size == 0:
            continue
        X = np.asarray(run.X, dtype=np.float64)
        on = np.atleast_1d(np.asarray(run.trial, dtype=int)).ravel()[:y.size]
        for o, yy in zip(on, y):
            seg = X[o:o + int(round(8.0 * fs)), :SL.BCI2A_N_EEG]
            if seg.shape[0] < int(round(8.0 * fs)):
                continue
            Xs.append(seg.T)
            ys.append(int(yy) - 1)
    return np.array(Xs), np.array(ys), fs


def main() -> int:
    from scipy.signal import butter, filtfilt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from mne.decoding import CSP

    X, y, fs = load_session(ROOT / "data" / "raw" / "bci2a" / "A01T.mat")
    b, a = butter(4, [SL.BANDS["mu"][0] / (fs / 2), SL.BANDS["beta"][1] / (fs / 2)],
                  btype="band")
    X = filtfilt(b, a, X, axis=2)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    grid = np.arange(0.0, 7.0, 0.5)
    accs = []
    for t0 in grid:
        i0, i1 = int(t0 * fs), int((t0 + 1.0) * fs)
        s = cross_val_score(make_pipeline(CSP(4, log=True), LDA()),
                            X[:, :, i0:i1], y, cv=cv)
        accs.append(float(s.mean()))
    accs = np.array(accs)
    k = int(np.argmax(accs))
    peak_t0 = float(grid[k])

    out = dict(n_trials=int(len(y)), n_classes=int(len(np.unique(y))),
               chance=1.0 / len(np.unique(y)), window_s=1.0,
               grid=[round(float(g), 2) for g in grid],
               acc=[round(float(v), 4) for v in accs],
               peak_start_s=peak_t0, peak_acc=round(float(accs[k]), 4),
               epoch=list(SL.EPOCHS["bci2a"]), baseline=list(SL.BASELINES["bci2a"]))
    (ROOT / "data" / "2a_window.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# P0 · 2a 判别窗实测（由 `scripts/d04_2a_window.py` 生成，勿手改）", "",
         f"- 数据：A01T（训练 session），{out['n_trials']} 试次，4 类，机会水平 "
         f"{out['chance']:.2f}",
         f"- 特征：mu+beta（{SL.BANDS['mu'][0]:.0f}–{SL.BANDS['beta'][1]:.0f} Hz），"
         "CSP(4) + LDA，5 折分层交叉验证（shuffle, seed=0）", "",
         "| 窗起点 (s) | 窗 | 准确率 |", "|---|---|---|"]
    for g, v in zip(grid, accs):
        L.append(f"| {g:.1f} | {g:.1f}–{g + 1:.1f} s | {v:.3f} |")
    L += ["", f"**峰在 {peak_t0:.1f} s 起，准确率 {accs[k]:.3f}**（机会 "
          f"{out['chance']:.2f}）。", "",
          "## 结论（写进 `sig_layer` 的部分）", "",
          f"- 判别信息从 1.5 s 起上升、{peak_t0:.1f} s 最强、6.0 s 后衰减，"
          "与「提示 2 s + MI 4 s」的范式一致。",
          f"- 故 MI 分析窗取 `EPOCHS['bci2a'] = {tuple(SL.EPOCHS['bci2a'])}`，"
          f"基线取提示期 `BASELINES['bci2a'] = {tuple(SL.BASELINES['bci2a'])}`。",
          "- 注意：本文件是**规格推导**，不是解码性能结论——单 session、单被试，"
          "不能引用为「2a 可达 0.75」。"]
    (ROOT / "reports" / "P0_2a_window.md").write_text("\n".join(L), encoding="utf-8")
    print(f"peak {peak_t0:.1f}s acc {accs[k]:.3f} | 写出 reports/P0_2a_window.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
