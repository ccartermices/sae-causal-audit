"""d34 — S2 伪迹拆分：top-k 因果特征的效果，能否被「只含伪迹」的解码器复现？

问题
----
D30 已证明：3 导 EOG 单独就能拿到全通道 κ 的 85%，且与被试间 κ 解离。
于是每个「因果特征」都要回答：它消融后把试次推向的那个新类别，
是不是**眼电通路本来就想推的方向**？若是，这个特征的消融效应就是伪迹中介的。

操作化（本节是 S2 的可执行版本）
--------------------------------
对特征 $i$，记消融后预测改变的试次集合 $F_i$，以及被推向的新类别 $c_i(t)$（$t\\in F_i$）。

* $\\pi_{\\rm raw}(i)=\\Pr[\\hat y_{\\rm sub}(t)=c_i(t)\\mid t\\in F_i]$，
  其中 $\\hat y_{\\rm sub}$ 是**替代解码器**（只用 3 导 EOG，或只用 6 导额区）的预测。
* $\\pi_{\\rm base}(i)=\\Pr[\\hat y_{\\rm sub}(t)=c_i(t)\\mid t\\notin F_i]$（同被试、同一批测试试次）。
* **超额一致率** $\\pi_{\\rm exc}=\\pi_{\\rm raw}-\\pi_{\\rm base}$。

$\\pi_{\\rm exc}$ 高 ⇒ 消融的方向与替代解码器独立偏好的方向一致 ⇒ 该特征的效应可由该通路复现。
判据（`03` §1 的 S2）：top-k 特征中 $\\pi_{\\rm exc}\\ge0.50$ 的比例即为**伪迹中介占比** $\\pi_{\\rm art}$。
本脚本只报数，不设通过线（S2 是必报项）。

产物：`probe/t1_s2_{tag}.json`、`reports/T1_S2_{tag}.md`
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
sys.path.insert(0, str(ROOT))

import det_lock  # noqa: E402
import sig_layer as SL  # noqa: E402
from sae.token_sae import TopKSAE  # noqa: E402

import d31_t1_decoder as D31  # noqa: E402
import d32_t1_sae_ablate as D32  # noqa: E402

PROBE = ROOT / "probe"
REPORTS = ROOT / "reports"

FRONTAL = ("Fz", "FC3", "FC1", "FCz", "FC2", "FC4")


def train_cv(Xtr, ytr, seed: int) -> tuple:
    """T 段内 5 折 CV 选轮数，再用全 T 段重训（与 d31 同协议）。"""
    from sklearn.model_selection import StratifiedKFold
    grid = SL.T1_EPOCH_GRID
    skf = StratifiedKFold(n_splits=SL.T1_CV_FOLDS, shuffle=True, random_state=seed)
    snaps = {g: [] for g in grid}
    for f, (ia, ib) in enumerate(skf.split(Xtr, ytr)):
        _, sn = D31.train_once(Xtr[ia], ytr[ia], Xtr[ib], ytr[ib],
                               epochs=max(grid), seed=seed * 100 + f, snapshots=grid)
        for g in grid:
            snaps[g].append(sn[g])
    best = max(grid, key=lambda g: np.mean(snaps[g]))
    model, _ = D31.train_once(Xtr, ytr, Xtr[:1], ytr[:1], epochs=best, seed=seed)
    return model, best


def run_subject(sub: str, seed: int, topk: int, src: str, n_null: int = 20,
                verbose=True) -> dict:
    tr = D31.load_session(sub, "T")
    te = D31.load_session(sub, "E")
    ch = tr["ch"]
    eeg = [i for i, c in enumerate(ch) if c not in SL.T1_EOG_CH]
    eog = [i for i, c in enumerate(ch) if c in SL.T1_EOG_CH]
    fro = [i for i, c in enumerate(ch) if c in FRONTAL]
    ytr, yte = tr["y"], te["y"]

    prev = json.loads((PROBE / f"t1_sae_ablate_{src}.json").read_text(encoding="utf-8"))
    rec = [r for r in prev["runs"] if r["subject"] == sub][0]

    # --- 主解码器与 SAE（确定性 → 与 d32 完全一致）
    Xtr, Xte = D31.zscore(tr["X"][:, eeg], te["X"][:, eeg])
    model, ep = train_cv(Xtr, ytr, seed)
    assert ep == rec["epochs"], f"轮数不一致 {ep} vs {rec['epochs']}（确定性被破坏？）"
    logit0 = D32._logits(model, Xte)
    Htr, Hte = D32.feature_map(model, Xtr), D32.feature_map(model, Xte)
    F = Htr.shape[1]
    # 步数必须与 d32 一致，否则翻转试次不是由产生 eff 的那个 SAE 引起的
    sae = TopKSAE(F, prev["d_dict"], prev["k"], seed=seed)
    sae.fit(D32.tokens_of(Htr), steps=int(prev.get("steps", 3000)), bs=1024, lr=2e-3)

    # --- 替代解码器：仅 EOG / 仅额区
    subs = {}
    for name, idx in (("eog", eog), ("frontal", fro)):
        A, B = D31.zscore(tr["X"][:, idx], te["X"][:, idx])
        mdl, _ = train_cv(A, ytr, seed)
        subs[name] = D32._logits(mdl, B).argmax(1)

    # --- top-k 因果特征
    eff = np.array(rec["eff"])
    ok = ~np.isnan(eff)
    top = np.where(ok)[0][np.argsort(-eff[ok])[:topk]]
    abl = D32.Ablater(model, sae, Hte, prev["m_abl"])
    y0 = logit0.argmax(1)

    rows = []
    pooled = {nm: ([], []) for nm in subs}
    for i in top:
        y1 = abl.logits(int(i)).argmax(1)
        fl = np.flatnonzero(y1 != y0)
        if len(fl) == 0:
            continue
        target = y1[fl]
        row = dict(atom=int(i), eff=float(eff[i]), n_flip=int(len(fl)),
                   flip_rate=float(len(fl) / len(y0)))
        for name, pred in subs.items():
            row[f"pi_raw_{name}"] = float((pred[fl] == target).mean())
            row[f"pi_base_{name}"] = float((pred == y1).mean())
            pooled[name][0].append(fl)
            pooled[name][1].append(target)
        rows.append(row)

    # --- 合并样本 + 类别置换零分布
    # 单原子只翻转几个试次，逐原子算一致率没有统计意义；把 top-k 的翻转试次并起来，
    # 用「把这些试次的目标类别随机重排」当零分布——它保留类别边际、破坏与试次的对应。
    stat = {}
    for name in subs:
        T = np.concatenate(pooled[name][0]); C = np.concatenate(pooled[name][1])
        pred = subs[name]
        raw = float((pred[T] == C).mean())
        r2 = np.random.RandomState(seed + 7)
        nulls = np.array([(pred[T] == r2.permutation(C)).mean() for _ in range(500)])
        stat[name] = dict(raw=raw, null_mean=float(nulls.mean()),
                          null_p95=float(np.percentile(nulls, 95)),
                          p=float((np.sum(nulls >= raw) + 1) / (len(nulls) + 1)),
                          n_flip=int(len(T)))

    out = dict(subject=sub, seed=seed, k=int(prev["k"]), n_top=int(len(rows)),
               stat=stat, rows=rows,
               main_acc=float((y0 == yte).mean()),
               sub_acc={nm: float((p == yte).mean()) for nm, p in subs.items()})
    if verbose:
        print(f"  {sub} | 主 {out['main_acc']:.3f} | "
              + " ".join(f"{nm} {out['sub_acc'][nm]:.3f} "
                         f"(π {stat[nm]['raw']:.2f}, 零 {stat[nm]['null_mean']:.2f}, "
                         f"p={stat[nm]['p']:.3f})" for nm in subs), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str, default="A01,A03,A07")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--from-ablate", type=str, default="k8")
    ap.add_argument("--tag", type=str, default="main")
    a = ap.parse_args()

    info = det_lock.lock(0)
    print(f"[det] {info}", flush=True)
    subs = [s.strip() for s in a.subjects.split(",") if s.strip()]
    t0 = time.time()
    out = dict(topk=a.topk, src=a.from_ablate, det=info, runs=[])
    for sub in subs:
        out["runs"].append(run_subject(sub, a.seed, a.topk, a.from_ablate, a.n_null))
    out["elapsed_s"] = round(time.time() - t0, 1)
    (PROBE / f"t1_s2_{a.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    summarize(out, a.tag)
    print(f"[done] {out['elapsed_s']} s", flush=True)
    return 0


def summarize(out: dict, tag: str) -> None:
    lines = ["# T1 S2 伪迹拆分：top-k 因果特征的效应能否被伪迹通路复现", "",
             f"自动生成。取因果效应最强的 {out['topk']} 个特征（k={out['runs'][0]['k']}）。",
             "",
             "把 top-k 特征消融后翻转的试次并起来，看**替代解码器**（仅 3 导 EOG / 仅 6 导额区）",
             "是否也把它们推向同一个类别。零分布 = 把这些试次的目标类别随机重排（500 次），",
             "保留类别边际、破坏与试次的对应。", "",
             "| 被试 | 主解码 acc | 替代 acc | 翻转试次 | 一致率 π | 零分布均值 | 零 p95 | p |",
             "|---|---|---|---|---|---|---|---|"]
    for r in out["runs"]:
        for nm, lab in (("eog", "仅 EOG"), ("frontal", "仅额区")):
            x = r["stat"][nm]
            lines.append(f"| {r['subject']} | {r['main_acc']:.3f} | "
                         f"{r['sub_acc'][nm]:.3f}（{lab}） | {x['n_flip']} | "
                         f"{x['raw']:.3f} | {x['null_mean']:.3f} | {x['null_p95']:.3f} | "
                         f"{x['p']:.3f} |")
    lines += ["", "## 逐特征（各被试 top-10）", ""]
    for r in out["runs"]:
        rows = sorted(r["rows"], key=lambda x: -x["eff"])[:10]
        lines += [f"### {r['subject']}", "",
                  "| 原子 | 效应 | 翻转试次 | π(EOG 指向一致) | π(额区) |",
                  "|---|---|---|---|---|"]
        for x in rows:
            lines.append(f"| {x['atom']} | {x['eff']:+.4f} | {x['n_flip']} | "
                         f"{x['pi_raw_eog']:.2f} | {x['pi_raw_frontal']:.2f} |")
        lines.append("")
    (REPORTS / f"T1_S2_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
