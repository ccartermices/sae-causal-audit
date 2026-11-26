#!/usr/bin/env python
"""d28 — G 的可行性探针：per-trial 分解能否定位？

为什么先做这个
--------------
E/F 的根因是「字典全局共享」：原子要么匹配全数据的空间分布，要么什么都不匹配。
G 的思路是把字典拆到试次级。但那是个大工程（per-trial 字典 / 超网络），
所以先用**最便宜、无参数**的对照把上限摸出来：

    每个试次的 90 个 token（6 帧 × 15 频点 × 64 导）单独做 SVD，
    取前 K 个空间模式，看注入试次里有没有一个模式落在 C3。

- 若 per-trial SVD（理论上限）都定位不到 ⇒ G 不必投入
- 若能 ⇒ 给出「需要多少 token / 什么 K」的定量依据，再做可学习的 per-trial 字典

判据
----
对每个试次算 best = max_k |cos(v_k, g)|（g 为注入位置的真值方向），
比较注入试次与干净试次：
  - 若注入试次的 best 分布整体右移（AUC ≥0.75），说明 per-trial 分解能区分
  - 并报「注入试次里 best > 0.8 的比例」

输出
----
  probe/g_pertrial_{tag}.json
  reports/T0_G_pertrial_{tag}.md
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
sys.path.insert(0, str(ROOT / "sae"))

import d22_s0_synth as D22      # noqa: E402
import d24_e_designs as D24     # noqa: E402
import d25_e_l1_verify as D25   # noqa: E402

SIGMA = 0.020
F0 = 10.0


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U / (n1*n2)，即 P(pos > neg)。"""
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 处理并列
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for i in np.where(cnt > 1)[0]:
        m = inv == i
        ranks[m] = ranks[m].mean()
    r1 = ranks[:len(pos)].sum()
    return float((r1 - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trial", type=int, default=1200)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--k", type=int, default=4, help="每试次保留的空间模式数")
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    tag = a.tag or "z"
    out = dict(n_trial=a.n_trial, snr_db=float(a.snr), K=a.k, sigma=SIGMA, runs=[])

    for seed in [int(v) for v in a.seeds.split(",") if v.strip()]:
        X0, ch_names, fs = D22.load_trials(a.n_trial, seed)
        n = len(X0)
        is_inj = np.zeros(n, bool)
        is_inj[np.random.default_rng(seed).permutation(n)[: n // 2]] = True
        P, d = D24.load_geometry(ch_names)
        w = np.exp(-(d ** 2) / (2 * SIGMA ** 2)); w /= w.max()
        sel = np.where(w > 1e-4)[0]
        cond = D25.build_condition(X0, is_inj, fs, w, sel, int(np.argmax(w)),
                                   F0, a.snr, seed, 100_000)
        tok = cond["T"]                              # (n*Tf*F, C)
        m = cond["m"]
        Tf, F, C = m["Tf"], m["F"], cond["C"]
        g = cond["g"]
        print(f"seed {seed} | {n} 试次（注入 {is_inj.sum()}）| 每试次 {Tf*F} token "
              f"× {C} 维 | K={a.k}", flush=True)

        # per-trial SVD：每个试次的 token 矩阵 (Tf*F, C)，按**同一试次**切
        toks = tok.reshape(n, Tf * F, C)
        best = np.zeros(n)
        best_k = np.zeros(n, int)
        var_top = np.zeros(n)
        t0 = time.time()
        for i in range(n):
            X = toks[i] - toks[i].mean(0)            # 去该试次的均值（跨 token）
            # 每个 token 已 z 化，但逐试次中心化能去掉该试次的共模
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            V = Vt[:a.k]                             # (K, C) 空间模式
            Vn = V - V.mean(1, keepdims=True)
            Vn /= np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-12
            cs = np.abs(Vn @ g)
            best[i] = cs.max()
            best_k[i] = int(cs.argmax())
            var_top[i] = float((S[:a.k] ** 2).sum() / max((S ** 2).sum(), 1e-12))

        a_ = auc(best[is_inj], best[~is_inj])
        frac_hi = float(np.mean(best[is_inj] > 0.8))
        frac_lo = float(np.mean(best[~is_inj] > 0.8))
        print(f"  注入试次 best: 中位 {np.median(best[is_inj]):.2f} "
              f"（>0.8 占 {frac_hi:.0%}）| 干净试次: 中位 {np.median(best[~is_inj]):.2f} "
              f"（>0.8 占 {frac_lo:.0%}）", flush=True)
        print(f"  AUC {a_:.3f} | 前 K 方差占比中位 {np.median(var_top):.2f} "
              f"| K 的分布 {np.bincount(best_k[is_inj], minlength=a.k)} "
              f"[{time.time()-t0:.0f}s]", flush=True)

        out["runs"].append(dict(seed=seed, auc=float(a_), K=a.k,
                                inj_median=float(np.median(best[is_inj])),
                                cln_median=float(np.median(best[~is_inj])),
                                inj_frac_hi=frac_hi, cln_frac_hi=frac_lo,
                                var_top_median=float(np.median(var_top)),
                                best_inj=[float(v) for v in best[is_inj]],
                                best_cln=[float(v) for v in best[~is_inj]]))

    (ROOT / "probe").mkdir(exist_ok=True)
    jp = ROOT / "probe" / f"g_pertrial_{tag}.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        "# T0-G 可行性探针：per-trial 分解能否定位",
        "",
        f"每个试次的 {out['runs'][0]['K'] and '90'} 个 token 单独做 SVD（无参数、理论上限），"
        f"取前 K={a.k} 个空间模式，看注入试次里有没有一个落在注入位置。",
        f"σ={SIGMA*1000:.0f} mm（N_eff≈3.1），SNR {a.snr:+.0f} dB。",
        "",
        "| 种子 | 注入 best 中位 | 干净 best 中位 | 注入 >0.8 占比 | 干净 >0.8 占比 | AUC | 前 K 方差占比 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in out["runs"]:
        lines.append(f"| {r['seed']} | {r['inj_median']:.2f} | {r['cln_median']:.2f} | "
                     f"{r['inj_frac_hi']:.0%} | {r['cln_frac_hi']:.0%} | {r['auc']:.3f} | "
                     f"{r['var_top_median']:.2f} |")
    lines += ["", "AUC = P(注入试次的 best > 干净试次的 best)；0.5 = 无区分，≥0.75 才算能区分。"]
    rp = ROOT / "reports" / f"T0_G_pertrial_{tag}.md"
    rp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写出 {jp.name} / {rp.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
