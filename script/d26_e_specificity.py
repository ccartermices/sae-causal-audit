#!/usr/bin/env python
"""d26 — 覆盖率的**位置特异性**：一个必须加的对照。

d25 的教训
----------
PR 约束（τ=2）把覆盖率从 0.41 推到 0.83，四个注入中心给出**完全相同**的数字。
查下来：注入位置 C3 的覆盖率 0.804，而**全脑 64 个位置的中位覆盖率是 0.840**
（C3 排第 48/64）。也就是说，PR 约束把字典变成了「铺满全脑的双导原子库」，
覆盖率再高也不含「学到了注入源」的信息。

所以覆盖率必须配一条位置特异性：

    spec = cov(目标位置) − median_{c∈全脑} cov(c)

只有 spec > SPEC_MIN 才说明字典对该位置**有偏好**，而不只是「恰好有个原子在那儿」。
本脚本对若干设计同时报绝对覆盖率与特异性。

输出
----
  probe/e_specificity_{tag}.json
  reports/T0_E_specificity_{tag}.md
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

SPEC_MIN = 0.15      # 目标位置覆盖率须比全脑中位高这么多
SIGMA = 0.020
F0 = 10.0
CENTER = "C3"


def all_positions_cov(Dm: np.ndarray, P: np.ndarray, sd_f: np.ndarray,
                      sigma: float) -> np.ndarray:
    """以每个导为中心生成真值方向，返回该位置的字典覆盖率。"""
    C = len(P)
    out = []
    for c in range(C):
        dc = np.linalg.norm(P - P[c], axis=1)
        wc = np.exp(-(dc ** 2) / (2 * sigma ** 2))
        wc /= wc.max()
        s = np.where(wc > 1e-4)[0]
        g = np.zeros(C)
        g[s] = wc[s] / sd_f[s]
        g -= g.mean()
        g /= np.linalg.norm(g) + 1e-12
        out.append(float(np.abs(Dm @ g).max()))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trial", type=int, default=1200)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--sae-dict", type=int, default=1024)
    ap.add_argument("--sae-k", type=int, default=16)
    ap.add_argument("--sae-steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-tokens", type=int, default=100_000)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)

    seeds = [int(v) for v in a.seeds.split(",") if v.strip()]
    X0, ch_names, fs = D22.load_trials(a.n_trial, 0)
    n = len(X0)
    is_inj = np.zeros(n, bool)
    is_inj[np.random.default_rng(0).permutation(n)[: n // 2]] = True
    P, d = D24.load_geometry(ch_names)
    w = np.exp(-(d ** 2) / (2 * SIGMA ** 2)); w /= w.max()
    sel = np.where(w > 1e-4)[0]
    cond = D25.build_condition(X0, is_inj, fs, w, sel, int(np.argmax(w)),
                               F0, a.snr, 0, a.max_tokens)
    ei, ej = D24.knn_edges(P, k=4)
    groups = D24.group_masks(ch_names)
    sd_f = cond["m"]["sd"][cond["fi"]]

    designs = [
        ("base", "base", 0.0, 0.0),
        ("l1_1.0", "l1", 1.0, 0.0),
        ("l1_0.3", "l1", 0.3, 0.0),
        ("pr_tau2", "pr", 3.0, 2.0),
        ("pr_tau3", "pr", 3.0, 3.0),
        ("pr_tau6", "pr", 3.0, 6.0),
    ]

    out = dict(n_trial=int(n), sigma=SIGMA, snr_db=float(a.snr), spec_min=SPEC_MIN,
               sae_dict=a.sae_dict, sae_k=a.sae_k, sae_steps=a.sae_steps, lr=a.lr,
               results=[], seed=0)
    print(f"{'设计':10s} {'目标cov':>8s} {'全脑中位':>8s} {'全脑p90':>8s} "
          f"{'C3 排名':>8s} {'特异性':>8s} {'R²':>7s} 判定")
    for name, reg, lam, tau in designs:
        covs, meds, p90s, ranks, specs, r2s = [], [], [], [], [], []
        for s in seeds:
            t0 = time.time()
            sae = D24.RegSAE(cond["C"], a.sae_dict, a.sae_k, seed=s, ei=ei, ej=ej,
                             groups=groups)
            sae.fit(cond["T"][cond["tr"]], steps=a.sae_steps, lr=a.lr, reg=reg,
                    lam=lam, tau=tau)
            dirs = sae.atom_dirs().astype(np.float64)
            Dm = dirs - dirs.mean(1, keepdims=True)
            Dm /= np.linalg.norm(Dm, axis=1, keepdims=True) + 1e-12
            tgt = float(np.abs(Dm @ cond["g"]).max())
            allc = all_positions_cov(Dm, P, sd_f, SIGMA)
            rank = int((allc > tgt).sum()) + 1
            idx, _ = sae.encode_sparse(cond["T"][cond["te"]])
            Xhat = sae.reconstruct(cond["T"][cond["te"]])
            te = cond["T"][cond["te"]]
            r2 = 1 - float(((Xhat - te) ** 2).sum()) / max(
                float(((te - te.mean(0)) ** 2).sum()), 1e-12)
            covs.append(tgt); meds.append(float(np.median(allc)))
            p90s.append(float(np.percentile(allc, 90))); ranks.append(rank)
            specs.append(tgt - float(np.median(allc))); r2s.append(float(r2))
            out["results"].append(dict(design=name, reg=reg, lam=lam, tau=tau, seed=s,
                                       cov_target=tgt, cov_median=float(np.median(allc)),
                                       cov_p90=float(np.percentile(allc, 90)),
                                       rank=rank, spec=tgt - float(np.median(allc)),
                                       r2=float(r2), dt=time.time() - t0))
            print(f"  {name:10s} seed={s} | {tgt:.2f} | 中位 {np.median(allc):.2f} "
                  f"| 排 {rank:2d}/64 | spec {tgt-np.median(allc):+.2f} | R² {r2:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        print(f"{name:10s} {np.mean(covs):8.2f} {np.mean(meds):8.2f} {np.mean(p90s):8.2f} "
              f"{np.median(ranks):8.0f} {np.mean(specs):+8.2f} {np.mean(r2s):7.3f} "
              f"{'PASS' if np.mean(specs) > SPEC_MIN else '-'}", flush=True)

    tag = a.tag or "z"
    (ROOT / "probe").mkdir(exist_ok=True)
    jp = ROOT / "probe" / f"e_specificity_{tag}.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # 汇总表
    lines = [
        "# T0-E：覆盖率的**位置特异性**（必须加的对照）",
        "",
        "只报绝对覆盖率会被「原子变窄」骗：PR 约束（τ=2）把覆盖率刷到 0.83，"
        "但注入位置 C3 在全脑 64 个位置里排 **48/64**，中位覆盖率 0.84 比它还高 —— "
        "字典只是铺满了全脑，什么都没学到。",
        "",
        f"故加一条：`spec = cov(目标) − median(cov(全脑 64 位置))`，"
        f"要求 spec > {SPEC_MIN}。",
        "",
        f"σ={SIGMA*1000:.0f} mm（N_eff≈3.1），SNR {a.snr:+.0f} dB，"
        f"字典 {a.sae_dict} × k={a.sae_k}，{a.sae_steps} 步，lr {a.lr:g}，"
        f"{len(seeds)} 个种子均值。",
        "",
        "| 设计 | 目标 cov | 全脑中位 | 全脑 p90 | C3 排名 | **特异性** | R² | 判定 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, reg, lam, tau in designs:
        rs = [r for r in out["results"] if r["design"] == name]
        lines.append(
            f"| {name} | {np.mean([r['cov_target'] for r in rs]):.2f} | "
            f"{np.mean([r['cov_median'] for r in rs]):.2f} | "
            f"{np.mean([r['cov_p90'] for r in rs]):.2f} | "
            f"{np.median([r['rank'] for r in rs]):.0f}/64 | "
            f"**{np.mean([r['spec'] for r in rs]):+.2f}** | "
            f"{np.mean([r['r2'] for r in rs]):.3f} | "
            f"{'**PASS**' if np.mean([r['spec'] for r in rs]) > SPEC_MIN else '—'} |")
    rp = ROOT / "reports" / f"T0_E_specificity_{tag}.md"
    rp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写出 {jp.name} / {rp.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
