#!/usr/bin/env python
"""d25 — E 的验证：方向 L1 的稳定性、泛化与机制。

d24 上 λ=1.0 把覆盖率从 0.41 推到 0.94，但 λ=3 又掉回 0.54 —— 非单调，必须查清是
「真的学到了注入源」还是「碰巧有个单导原子落在 C3」。

本脚本回答四个问题：
  1. **跨种子**：同一 λ 换 5 个种子，覆盖率是稳定在 0.9 还是跳？
  2. **跨中心**：C3 / Oz / Pz / Fz 四个注入位置是否都能到 0.8？
  3. **跨尺度**：N_eff = 1.5 / 3 / 8 / 20 时，L1 能把边界推到多低？
  4. **机制**：覆盖率最高的原子，方向真的落在注入导上吗（argmax + 前 6 分量）？

输出
----
  probe/e_l1_{tag}.json
  reports/T0_E_l1_{tag}.md
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

F0 = {"C3": 10.0, "Oz": 11.0, "Pz": 12.0, "Fz": 9.0}
DEFAULT_LAM = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
DEFAULT_SIGMA = (0.012, 0.020, 0.030, 0.045)   # N_eff ≈ 1.5 / 3.1 / 8.8 / 20.4


def pr(v):
    v = np.abs(v) ** 2
    return float(v.sum() ** 2 / np.maximum((v ** 2).sum(), 1e-12))


def build_condition(X0, is_inj, fs, w, sel, main_ch, f0, snr, seed, max_tokens):
    """注入 + token + 真值方向，返回一整套训练/评估素材。"""
    n = len(X0)
    env = D22.envelope(X0.shape[-1], fs, D22.INJ_WIN_S[0], D22.INJ_WIN_S[1], D22.INJ_RAMP_S)
    tt = np.arange(X0.shape[-1]) / fs
    ph = np.random.default_rng(seed + 1000).uniform(0, 2 * np.pi, n)
    p_bg = float(np.median(D22.band_power(X0[is_inj][:, main_ch, :], fs, f0)))
    amp = float(np.sqrt(2.0 * p_bg * 10 ** (snr / 10) / max(float((env ** 2).mean()), 1e-12)))
    X = X0.copy()
    X[:, sel, :] += (amp * w[sel])[None, :, None] * env[None, None, :] * np.sin(
        2 * np.pi * f0 * tt[None, None, :] + ph[:, None, None])
    X[~is_inj] = X0[~is_inj]

    tok, m = D22.build_tokens(X, fs, ref_trials=np.where(~is_inj)[0])
    C, F = m["C"], m["F"]
    T_ = ((tok.reshape(-1, F, C) - m["mu"]) / m["sd"]).reshape(-1, C).astype(np.float32)
    trial_of = m["trial_of"]
    fi = int(np.argmin(np.abs(m["fb"] - f0)))
    tr = np.where(np.isin(trial_of, np.where(~is_inj)[0]))[0]
    te = np.where(np.isin(trial_of, np.where(is_inj)[0]))[0]
    rng = np.random.default_rng(seed)
    if len(tr) > max_tokens:
        tr = rng.choice(tr, max_tokens, replace=False)
    if len(te) > 20000:
        te = rng.choice(te, 20000, replace=False)
    g = np.zeros(C)
    g[sel] = w[sel] / m["sd"][fi][sel]
    g -= g.mean()
    g /= np.linalg.norm(g) + 1e-12
    return dict(T=T_, tr=tr, te=te, g=g, C=C, n=n, trial_of=trial_of,
                is_inj=is_inj, tok=tok, m=m, fi=fi)


def run_one(cond, lam, seed, sae_dict, sae_k, steps, ei, ej, groups, lr=2e-3,
            reg="l1", tau=6.0):
    sae = D24.RegSAE(cond["C"], sae_dict, sae_k, seed=seed, ei=ei, ej=ej, groups=groups)
    sae.fit(cond["T"][cond["tr"]], steps=steps, lr=lr,
            reg=(reg if lam > 0 else "base"), lam=lam, tau=tau)
    dirs = sae.atom_dirs().astype(np.float64)
    Dm = dirs - dirs.mean(1, keepdims=True)
    Dm /= np.linalg.norm(Dm, axis=1, keepdims=True) + 1e-12
    cos_all = np.abs(Dm @ cond["g"])
    best = int(np.argmax(cos_all))
    idx, _ = sae.encode_sparse(cond["T"][cond["te"]])
    Xhat = sae.reconstruct(cond["T"][cond["te"]])
    te = cond["T"][cond["te"]]
    r2 = 1 - float(((Xhat - te) ** 2).sum()) / max(float(((te - te.mean(0)) ** 2).sum()), 1e-12)
    return dict(cov=float(cos_all[best]), best_atom=best,
                best_dir=[float(v) for v in dirs[best]],
                pr_median=float(np.median([pr(dirs[i]) for i in range(len(dirs))])),
                alive=int((np.bincount(idx.ravel(), minlength=len(dirs)) > 0).sum()),
                r2=float(r2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trial", type=int, default=1200)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--sae-dict", type=int, default=1024)
    ap.add_argument("--sae-k", type=int, default=16)
    ap.add_argument("--sae-steps", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=100_000)
    ap.add_argument("--lam", type=str, default=",".join(str(v) for v in DEFAULT_LAM))
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--centers", type=str, default="C3,Oz,Pz,Fz")
    ap.add_argument("--sigma", type=str, default="")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--reg", type=str, default="l1")
    ap.add_argument("--tau", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)

    lams = [float(v) for v in a.lam.split(",") if v.strip()]
    seeds = [int(v) for v in a.seeds.split(",") if v.strip()]
    centers = [c.strip() for c in a.centers.split(",") if c.strip()]
    sigmas = [float(v) for v in a.sigma.split(",") if v.strip()] or [DEFAULT_SIGMA[1]]

    X0, ch_names, fs = D22.load_trials(a.n_trial, a.seed)
    n = len(X0)
    is_inj = np.zeros(n, bool)
    is_inj[np.random.default_rng(a.seed).permutation(n)[: n // 2]] = True
    P, d = D24.load_geometry(ch_names)
    ei, ej = D24.knn_edges(P, k=4)
    groups = D24.group_masks(ch_names)

    out = dict(n_trial=int(n), snr_db=float(a.snr), sae_dict=a.sae_dict, sae_k=a.sae_k,
               sae_steps=a.sae_steps, lam_grid=lams, seeds=seeds, runs=[])

    # ---- ① 跨种子 + ② 跨中心（固定 σ）
    print(f"== 跨种子 / 跨中心（σ={sigmas[0]*1000:.0f} mm）==", flush=True)
    for center in centers:
        f0 = F0.get(center, 10.0)
        w = np.exp(-(d ** 2) / (2 * sigmas[0] ** 2)); w /= w.max()
        sel = np.where(w > 1e-4)[0]; main_ch = int(np.argmax(w))
        ne = float((w).sum() ** 2 / (w ** 2).sum())
        cond = build_condition(X0, is_inj, fs, w, sel, main_ch, f0, a.snr, 0, a.max_tokens)
        print(f"\n  中心 {center} @ {f0} Hz（N_eff {ne:.1f}，{len(sel)} 导）", flush=True)
        rows = []
        for lam in lams:
            covs, prs, r2s = [], [], []
            for s in seeds:
                t0 = time.time()
                r = run_one(cond, lam, s, a.sae_dict, a.sae_k, a.sae_steps, ei, ej,
                            groups, lr=a.lr, reg=a.reg, tau=a.tau)
                covs.append(r["cov"]); prs.append(r["pr_median"]); r2s.append(r["r2"])
                rows.append(dict(kind="seed", center=center, f0=f0, n_eff=ne, lam=lam,
                                 seed=s, **r))
                print(f"    λ={lam:<4g} seed={s} | 覆盖率 {r['cov']:.2f} | PR {r['pr_median']:5.1f} "
                      f"| R² {r['r2']:.3f} | 最佳原子 {r['best_atom']:4d} "
                      f"| argmax {ch_names[int(np.argmax(np.abs(r['best_dir'])))]} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
            print(f"    → λ={lam:<4g} 覆盖率 {np.mean(covs):.2f} ± {np.std(covs):.2f} "
                  f"（min {np.min(covs):.2f}, max {np.max(covs):.2f}）| "
                  f"≥0.8 的种子 {sum(c >= 0.8 for c in covs)}/{len(covs)}", flush=True)
        out["runs"].append(dict(center=center, f0=f0, n_eff=ne, sigma=sigmas[0], rows=rows))

    # ---- ③ 跨尺度
    if len(sigmas) > 1:
        print("\n== 跨尺度（λ 取 d24 的最优点，5 个种子）==", flush=True)
        best_lam = 1.0
        for sigma in sigmas:
            w = np.exp(-(d ** 2) / (2 * sigma ** 2)); w /= w.max()
            sel = np.where(w > 1e-4)[0]; main_ch = int(np.argmax(w))
            ne = float(w.sum() ** 2 / (w ** 2).sum())
            cond = build_condition(X0, is_inj, fs, w, sel, main_ch, 10.0, a.snr, 0, a.max_tokens)
            covs = []
            for s in seeds:
                r = run_one(cond, best_lam, s, a.sae_dict, a.sae_k, a.sae_steps, ei, ej,
                            groups, lr=a.lr, reg=a.reg, tau=a.tau)
                covs.append(r["cov"])
            print(f"  σ={sigma*1000:5.1f} mm | N_eff {ne:5.1f} | 覆盖率 {np.mean(covs):.3f} "
                  f"± {np.std(covs):.3f}（{np.round(covs,2)}）", flush=True)
            out.setdefault("scale", []).append(dict(sigma=float(sigma), n_eff=ne,
                                                    lam=best_lam,
                                                    cov_mean=float(np.mean(covs)),
                                                    cov_std=float(np.std(covs)),
                                                    covs=[float(c) for c in covs]))

    tag = a.tag or "z"
    (ROOT / "probe").mkdir(exist_ok=True)
    jp = ROOT / "probe" / f"e_l1_{tag}.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {jp.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
