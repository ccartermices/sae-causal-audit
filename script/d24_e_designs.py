#!/usr/bin/env python
"""d24 — E：用 C 的合成注入平台筛「能让原子局部化」的 SAE 设计。

背景（D22）
-----------
标准 TopK SAE 的原子只能表示 ≥8 等效导联的模式，且与 SNR 无关。
本脚本把「仪器设计」当成自变量：固定数据、固定注入真值（N_eff≈3），
只换字典的结构约束，看哪个能把覆盖率推到 0.8。

候选设计
--------
  base   无正则（对照，复现 D22 的结论）
  l1     decoder 方向 L1：把原子推向「少数导」
  tv     导联图上总变差：惩罚相邻导之间的跳变（空间连续）
  l1tv   L1 + TV：局部**且**连续（单导 δ 不被允许）
  group  按 8 个脑区的组 lasso：原子只在一两个脑区有分量，组内可分

验收（两条都要）
----------------
  1. 覆盖率 ≥ COV_TARGET（0.80）且 > 随机基线 p95
  2. 重建 R² ≥ 0.90 × base 的 R² —— 否则只是「靠毁掉重建换局部化」

输出
----
  probe/e_designs_{tag}.json
  reports/T0_E_designs_{tag}.md
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

import token_sae as TS          # noqa: E402
import d22_s0_synth as D22      # noqa: E402

COV_TARGET = 0.80
R2_RATIO_MIN = 0.90
# 目标尺度：σ=0.020 m 时 N_eff ≈ 3.1（D22 的分辨率曲线实测）
TARGET_SIGMA = 0.020
CENTER = "C3"
F0 = 10.0


# ---------------------------------------------------------------- 导联图与分组
def load_geometry(ch_names: list[str]):
    import mne

    mont = mne.channels.make_standard_montage("standard_1005")
    pos = mont.get_positions()["ch_pos"]
    P = np.array([pos[c] for c in ch_names], dtype=float)
    d = np.linalg.norm(P - P[ch_names.index(CENTER)], axis=1)
    return P, d


def knn_edges(P: np.ndarray, k: int = 4):
    from scipy.spatial.distance import cdist

    D = cdist(P, P)
    np.fill_diagonal(D, np.inf)
    ei, ej = [], []
    for i in range(len(P)):
        for j in np.argsort(D[i])[:k]:
            ei.append(i); ej.append(j)
    return np.array(ei), np.array(ej)


BRAIN_GROUPS = {
    "LF": ("Fp1", "AF7", "AF3", "F7", "F5", "F3", "F1", "FC5", "FC3", "FC1"),
    "RF": ("Fp2", "AF4", "AF8", "F2", "F4", "F6", "F8", "FC2", "FC4", "FC6"),
    "LC": ("C5", "C3", "C1", "CP5", "CP3", "CP1", "T7", "TP7"),
    "RC": ("C2", "C4", "C6", "CP2", "CP4", "CP6", "T8", "TP8"),
    "LP": ("P7", "P5", "P3", "P1", "PO7", "PO3"),
    "RP": ("P2", "P4", "P6", "P8", "PO4", "PO8"),
    "LO": ("O1",),
    "RO": ("O2",),
    "MID": ("Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz", "Iz",
            "FT7", "FT8", "T9", "T10"),
}


def group_masks(ch_names: list[str]):
    out = []
    for name, chs in BRAIN_GROUPS.items():
        idx = np.array([ch_names.index(c) for c in chs if c in ch_names])
        if len(idx):
            out.append((name, idx))
    return out


# ---------------------------------------------------------------- 带正则的 SAE
class RegSAE:
    """TopK SAE + 可选的结构正则（作用在 decoder 方向上）。"""

    def __init__(self, d_in, d_dict, k, seed=0, ei=None, ej=None, groups=None,
                 init_local=None):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.k, self.d_in, self.d_dict = k, d_in, d_dict
        self.enc = nn.Linear(d_in, d_dict)
        self.dec = nn.Linear(d_dict, d_in, bias=False)
        nn.init.kaiming_uniform_(self.enc.weight)
        with torch.no_grad():
            self.dec.weight.copy_(self.enc.weight.t())
            if init_local is not None:
                # 把前若干个原子直接设成已知的高斯包，检验「初始化能否保持」
                W = self.dec.weight.data
                for i, v in enumerate(init_local):
                    if i >= d_dict:
                        break
                    u = torch.tensor(v, dtype=torch.float32)
                    u = u / (u.norm() + 1e-8) * 0.3
                    W[:, i] = u
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enc.to(self.device); self.dec.to(self.device)
        self.ei = None if ei is None else torch.tensor(ei, device=self.device)
        self.ej = None if ej is None else torch.tensor(ej, device=self.device)
        self.groups = groups

    def fit(self, X, steps=2000, bs=512, lr=2e-3, reg="base", lam=0.0, tau=6.0):
        import torch

        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=lr)
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = len(Xt)
        for _ in range(steps):
            xb = Xt[torch.randint(0, n, (min(bs, n),), device=self.device)]
            a = torch.relu(self.enc(xb))
            top = torch.topk(a, self.k, dim=1)
            a_sp = torch.zeros_like(a).scatter(1, top.indices, top.values)
            loss = ((self.dec(a_sp) - xb) ** 2).mean()
            W = self.dec.weight                       # (C, D_dict)
            if reg in ("l1", "l1tv") and lam > 0:
                loss = loss + lam * W.abs().mean()
            if reg in ("tv", "l1tv") and lam > 0:
                loss = loss + lam * (W[self.ei] - W[self.ej]).abs().mean()
            if reg == "pr" and lam > 0:
                # 参与率约束：PR_i = (Σd²)²/Σd⁴ 是该原子的等效导联数（对整体尺度不变）。
                # 与 L1 不同，它**不惩罚重复** —— 多个同位置、不同宽度的原子可以共存。
                d2 = W ** 2
                pri = d2.sum(0) ** 2 / (d2 ** 2).sum(0).clamp(min=1e-12)
                loss = loss + lam * torch.relu(pri - tau).mean()
            if reg == "group" and lam > 0:
                g = sum(W[idx].norm(dim=0).mean() for _, idx in self.groups) / len(self.groups)
                loss = loss + lam * g
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def encode_sparse(self, X, batch=4096):
        import torch

        idxs, vals = [], []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=self.device)
                a = torch.relu(self.enc(xb))
                top = torch.topk(a, self.k, dim=1)
                idxs.append(top.indices.cpu().numpy())
                vals.append(top.values.cpu().numpy())
        return np.concatenate(idxs), np.concatenate(vals)

    def reconstruct(self, X, batch=4096):
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=self.device)
                a = torch.relu(self.enc(xb))
                top = torch.topk(a, self.k, dim=1)
                a_sp = torch.zeros_like(a).scatter(1, top.indices, top.values)
                out.append(self.dec(a_sp).cpu().numpy())
        return np.concatenate(out)

    def atom_dirs(self):
        return self.dec.weight.detach().cpu().numpy().T     # (D_dict, C)


def pr(v):
    v = np.abs(v) ** 2
    return float(v.sum() ** 2 / np.maximum((v ** 2).sum(), 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trial", type=int, default=1200)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--sae-dict", type=int, default=1024)
    ap.add_argument("--sae-k", type=int, default=16)
    ap.add_argument("--sae-steps", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=100_000)
    ap.add_argument("--sigma", type=float, default=TARGET_SIGMA)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)

    rng = np.random.default_rng(a.seed)
    X0, ch_names, fs = D22.load_trials(a.n_trial, a.seed)
    n = len(X0)
    is_inj = np.zeros(n, bool)
    is_inj[rng.permutation(n)[: n // 2]] = True

    P, d = load_geometry(ch_names)
    w_all = np.exp(-(d ** 2) / (2 * a.sigma ** 2))
    ne = float((w_all / w_all.max()).sum() ** 2 / ((w_all / w_all.max()) ** 2).sum())
    w = w_all / w_all.max()
    sel = np.where(w > 1e-4)[0]
    main_ch = int(np.argmax(w))
    print(f"目标尺度：σ={a.sigma*1000:.0f} mm → N_eff={ne:.1f}（{len(sel)} 导参与）", flush=True)

    env = D22.envelope(X0.shape[-1], fs, D22.INJ_WIN_S[0], D22.INJ_WIN_S[1], D22.INJ_RAMP_S)
    tt = np.arange(X0.shape[-1]) / fs
    ph = np.random.default_rng(a.seed + 1000).uniform(0, 2 * np.pi, n)
    p_bg = float(np.median(D22.band_power(X0[is_inj][:, main_ch, :], fs, F0)))
    amp = float(np.sqrt(2.0 * p_bg * 10 ** (a.snr / 10) / max(float((env ** 2).mean()), 1e-12)))

    X = X0.copy()
    X[:, sel, :] += (amp * w[sel])[None, :, None] * env[None, None, :] * np.sin(
        2 * np.pi * F0 * tt[None, None, :] + ph[:, None, None])
    X[~is_inj] = X0[~is_inj]

    tok, m = D22.build_tokens(X, fs, ref_trials=np.where(~is_inj)[0])
    C, F = m["C"], m["F"]
    mu, sd = m["mu"], m["sd"]
    T_ = ((tok.reshape(-1, F, C) - mu) / sd).reshape(-1, C).astype(np.float32)
    trial_of = m["trial_of"]
    fi = int(np.argmin(np.abs(m["fb"] - F0)))
    tr = np.where(np.isin(trial_of, np.where(~is_inj)[0]))[0]
    te = np.where(np.isin(trial_of, np.where(is_inj)[0]))[0]
    if len(tr) > a.max_tokens:
        tr = rng.choice(tr, a.max_tokens, replace=False)
    if len(te) > 20000:
        te = rng.choice(te, 20000, replace=False)

    # 真值方向
    g = np.zeros(C)
    g[sel] = w[sel] / sd[fi][sel]
    g -= g.mean()
    g /= np.linalg.norm(g) + 1e-12
    # 随机基线
    rc = []
    for _ in range(20):
        R = rng.standard_normal((a.sae_dict, C))
        R -= R.mean(1, keepdims=True)
        R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
        rc.append(float(np.abs(R @ g).max()))
    rand_p95 = float(np.percentile(rc, 95))

    ei, ej = knn_edges(P, k=4)
    groups = group_masks(ch_names)
    # 初始化候选：以注入中心为中心的 8 个不同宽度高斯包
    init_local = []
    for s in (0.008, 0.012, 0.016, 0.02, 0.026, 0.033, 0.045, 0.06):
        init_local.append(np.exp(-(d ** 2) / (2 * s ** 2)))

    designs = [
        ("base", "base", 0.0, None),
        ("l1_0.3", "l1", 0.3, None), ("l1_1.0", "l1", 1.0, None),
        ("l1_3.0", "l1", 3.0, None), ("l1_10", "l1", 10.0, None),
        ("tv_3.0", "tv", 3.0, None),
        ("l1tv_3.0", "l1tv", 3.0, None), ("l1tv_10", "l1tv", 10.0, None),
        ("group_1.0", "group", 1.0, None), ("group_3.0", "group", 3.0, None),
        ("group_10", "group", 10.0, None),
        ("pr_0.3", "pr", 0.3, None), ("pr_1.0", "pr", 1.0, None),
        ("pr_3.0", "pr", 3.0, None),
        ("init_local", "base", 0.0, init_local),
        ("init_local_l1_1.0", "l1", 1.0, init_local),
    ]

    out = dict(n_trial=int(n), sigma=float(a.sigma), n_eff=ne, snr_db=float(a.snr),
               sae_dict=a.sae_dict, sae_k=a.sae_k, sae_steps=a.sae_steps,
               rand_p95=rand_p95, cov_target=COV_TARGET, designs=[], seed=a.seed)
    base_r2 = None
    print(f"\n{'设计':22s} {'覆盖率':>6s} {'R²':>7s} {'活原子':>7s} {'PR中位':>6s} {'R²比':>6s} 判定")
    for name, reg, lam, init in designs:
        t0 = time.time()
        sae = RegSAE(C, a.sae_dict, a.sae_k, seed=a.seed, ei=ei, ej=ej,
                     groups=groups, init_local=init)
        sae.fit(T_[tr], steps=a.sae_steps, reg=reg, lam=lam)
        dirs = sae.atom_dirs().astype(np.float64)
        Dm = dirs - dirs.mean(1, keepdims=True)
        Dm /= np.linalg.norm(Dm, axis=1, keepdims=True) + 1e-12
        cov = float(np.abs(Dm @ g).max())

        idx, val = sae.encode_sparse(T_[te])
        Xhat = sae.reconstruct(T_[te])
        sse = float(((Xhat - T_[te]) ** 2).sum())
        sst = float(((T_[te] - T_[te].mean(0)) ** 2).sum())
        r2 = 1 - sse / max(sst, 1e-12)
        if base_r2 is None:
            base_r2 = r2
        alive = int((np.bincount(idx.ravel(), minlength=a.sae_dict) > 0).sum())
        prm = float(np.median([pr(dirs[i]) for i in range(a.sae_dict)]))
        passed = bool(cov >= COV_TARGET and cov > rand_p95 and r2 >= R2_RATIO_MIN * base_r2)
        out["designs"].append(dict(name=name, reg=reg, lam=lam,
                                   has_init=init is not None, cov=cov, r2=float(r2),
                                   alive=alive, pr_median=prm,
                                   r2_ratio=float(r2 / max(base_r2, 1e-12)),
                                   passed=passed))
        print(f"{name:22s} {cov:6.2f} {r2:7.3f} {alive:7d} {prm:6.1f} "
              f"{r2/base_r2:6.2f} {'PASS' if passed else '-'}  [{time.time()-t0:.0f}s]",
              flush=True)

    tag = a.tag or "z"
    (ROOT / "probe").mkdir(exist_ok=True)
    jp = ROOT / "probe" / f"e_designs_{tag}.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        "# T0-E：筛「能让原子局部化」的 SAE 设计",
        "",
        f"固定注入真值 $N_{{eff}}={ne:.1f}$（σ={a.sigma*1000:.0f} mm）、SNR {a.snr:+.0f} dB、"
        f"字典 {a.sae_dict} × k={a.sae_k}，只换结构约束。",
        "",
        f"- 验收：覆盖率 ≥ {COV_TARGET} 且 > 随机 p95 {rand_p95:.2f}，"
        f"**且** R² ≥ {R2_RATIO_MIN} × base R²（防止靠毁重建换局部化）",
        "",
        "| 设计 | 正则 | λ | 覆盖率 | R² | 活原子 | 通道 PR 中位 | R²/base | 判定 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in out["designs"]:
        lines.append(
            f"| {r['name']} | {r['reg']} | {r['lam']:g} | {r['cov']:.2f} | {r['r2']:.3f} | "
            f"{r['alive']} | {r['pr_median']:.1f} | {r['r2_ratio']:.2f} | "
            f"{'**PASS**' if r['passed'] else '—'} |")
    rp = ROOT / "reports" / f"T0_E_designs_{tag}.md"
    rp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写出 {jp.name} / {rp.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
