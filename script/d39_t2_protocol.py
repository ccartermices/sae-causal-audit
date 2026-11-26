"""d39 — T2 是否值得扩 run：三条切分协议的解码器强度比较。

背景
----
D38 的 T2 只用了 6 个 hands run 里的 4 训 2 测（60/30），每被试训练试次只有 60，
解码器明显弱（9 人 0.552、106 人 0.552，κ +0.104）。审稿人会问：扩到更多 run
能不能把它救回来？能，就该换协议；不能，就该把 `t2_bacc` 明确降级为仪器线。

三条协议
--------
| 代号 | 类别 | 训练/测试 | 训练试次 | 说明 |
|---|---|---|---|---|
| P0 | 2 类左/右拳 | runs 3,4,7,8 → 11,12 | 60 / 30 | 现行（D38） |
| P1 | 2 类左/右拳 | 6 个 hands run 留一 run（6 折） | 75 / 15 | 用满 90 试次，**无偏** |
| P2 | 4 类左/右/双拳/双脚 | runs 3–10 → 11–14 | 120 / 60 | 训练试次翻倍，类别数对齐 T1 |

P2 的可行性来自 `d10_epochs.RUN_META`：run 3/7/11 与 4/8/12 是左右拳，
5/9 与 6/10 是双拳/双脚，12 个 run 恰好构成 3 个完整 block（3–6、7–10、11–14）。

判据
----
比 **Cohen κ** 而非 bacc：四类的机会水平是 0.25，直接比 bacc 会误判。
若三条协议的 κ 在噪声内不可区分 → 扩 run 无收益，维持 P0 并把 `t2_bacc` 降级。

产物：`probe/t2_proto_sweep.json`
"""


import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import det_lock  # noqa: E402

det_lock.lock(0)

import d38_t2_mmidb as D  # noqa: E402

EPO = D.EPO

BLOCKS = [(3, 4, 5, 6), (7, 8, 9, 10), (11, 12, 13, 14)]
HANDS = (3, 4, 7, 8, 11, 12)


def load(sub, runs, n_classes):
    z = np.load(EPO / f"{sub}.npz", allow_pickle=True)
    m = np.isin(z["run"], runs)
    y = z["y"][m].astype(np.int64)
    if n_classes == 4:
        y = z["kind"][m].astype(np.int64) * 2 + y
    return dict(X=z["X"][m].astype(np.float32), y=y, run=z["run"][m])


def kappa(y, pred, n_cls):
    cm = np.zeros((n_cls, n_cls))
    np.add.at(cm, (y, pred), 1)
    po = np.trace(cm) / cm.sum()
    pe = float((cm.sum(0) * cm.sum(1)).sum() / cm.sum() ** 2)
    return float((po - pe) / max(1 - pe, 1e-12))


def bacc(logits, y, n_cls):
    pr = logits.argmax(1)
    return float(np.mean([(pr[y == c] == c).mean() for c in range(n_cls) if (y == c).any()]))


def one(Xtr, ytr, Xte, yte, n_cls, epochs, seed):
    D.T2_CLASSES = n_cls                       # build() 按模块常量定输出维度
    mdl = D.train_once(Xtr, ytr, epochs=epochs, seed=seed, bs=16, lr=3e-4, wd=1e-3)
    lg = D.D32._logits(mdl, Xte)
    return bacc(lg, yte, n_cls), kappa(yte, lg.argmax(1), n_cls)


def main(subs):
    rows = []
    for s in subs:
        d = load(s, HANDS, 2)
        tr = np.isin(d["run"], (3, 4, 7, 8)); te = np.isin(d["run"], (11, 12))
        Xtr, Xte = D.zscore(d["X"][tr], d["X"][te])
        p0 = one(Xtr, d["y"][tr], Xte, d["y"][te], 2, 60, 0)
        print(f"  {s} P0 done {p0}", flush=True)

        accs, kaps = [], []
        for r in HANDS:
            tr2 = d["run"] != r; te2 = d["run"] == r
            A, B = D.zscore(d["X"][tr2], d["X"][te2])
            a, k = one(A, d["y"][tr2], B, d["y"][te2], 2, 60, 0)
            accs.append(a); kaps.append(k)
        p1 = (float(np.mean(accs)), float(np.mean(kaps)))

        d4 = load(s, sum(BLOCKS, ()), 4)
        tr4 = np.isin(d4["run"], BLOCKS[0] + BLOCKS[1]); te4 = np.isin(d4["run"], BLOCKS[2])
        A, B = D.zscore(d4["X"][tr4], d4["X"][te4])
        p2s = {}
        for ep in (40, 60, 100):
            p2s[ep] = one(A, d4["y"][tr4], B, d4["y"][te4], 4, ep, 0)
        rows.append((s, p0, p1, p2s))
        print(f"{s}  P0 {p0[0]:.3f}/κ{p0[1]:+.3f} | P1 {p1[0]:.3f}/κ{p1[1]:+.3f} | "
              + " ".join(f"P2@{ep} {v[0]:.3f}/κ{v[1]:+.3f}" for ep, v in p2s.items()), flush=True)

    print("\n--- 均值 ---")
    print("P0 bacc %.3f κ %+.3f" % (np.mean([r[1][0] for r in rows]), np.mean([r[1][1] for r in rows])))
    print("P1 bacc %.3f κ %+.3f" % (np.mean([r[2][0] for r in rows]), np.mean([r[2][1] for r in rows])))
    for ep in (40, 60, 100):
        print("P2@%d bacc %.3f κ %+.3f" % (ep,
              np.mean([r[3][ep][0] for r in rows]), np.mean([r[3][ep][1] for r in rows])))
    return rows


if __name__ == "__main__":
    import json
    subs = sys.argv[1].split(",") if len(sys.argv) > 1 else [f"S{i:03d}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)]
    rows = main(subs)
    out = {"protocols": {
        "P0_current_2class_60_30": [list(r[1]) for r in rows],
        "P1_loso6_2class_75_15": [list(r[2]) for r in rows],
        "P2_4class_120_60_ep40": [list(r[3][40]) for r in rows],
        "P2_4class_120_60_ep60": [list(r[3][60]) for r in rows],
        "P2_4class_120_60_ep100": [list(r[3][100]) for r in rows]},
        "subjects": [r[0] for r in rows],
        "note": "每格 = (平衡准确率, Cohen kappa)。P1 为 6 折均值，是无偏估计（90 试次各测一次）；"
                "P0 的测试集只有 30 试次，单次抽样方差约 ±0.09。"}
    pathlib.Path("probe/t2_proto_sweep.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved probe/t2_proto_sweep.json  n=", len(rows))
