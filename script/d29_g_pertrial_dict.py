#!/usr/bin/env python
r"""d29 — G：per-trial 稀疏字典 + 试次级读出，并给出仪器可分辨的 SNR 边界。

前情
----
D23/D24（全局共享字典 + 单原子读出）的特异性恒为 0；D25（per-trial SVD）的
AUC 0.97 但精确率只有 17%。

本脚本先修掉 D25 留下的两个口径问题，再看还剩多少：

**口径问题一：参照方向用错了。**
  D25 的 g 是解析高斯包 $w/\sigma$。但 token 是 STFT **幅度**，$|Z|$ 是非线性变换：
  边缘导上注入是二阶小量，注入实际留下的方向被压窄成 $w^2$ 形状。
  实测单试次 $\cos(\cdot, g_w)$ 上限只有 0.57–0.72，且 SNR 提到 30 dB 也不涨。
  改用**经验签名**（训练试次上 注入均值 − 干净均值，不是逐试次真值，故不循环）：
  $$g_{\rm emp}=\mathrm{unit}\big(\overline{V}_{\rm inj}[f_0]-\overline{V}_{\rm cln}[f_0]\big)$$

**口径问题二：频点用真值挑的。**
  拆成两个量：`oracle`（用真值频点，测方向精度）与 `sel`（无监督按跨导局部性挑频点，端到端）。

字典三种（重建各自自洽，不共用）
  global        D_0（L1 窄原子）
  pertrial_res  D_0 + 该试次残差的 top-$k_r$ 方向（$x_h=x_h^{g}+R V^\top V$）
  pertrial_mod  $D_t=D_0+B\,\mathrm{diag}(\alpha_t)M$，$\alpha_t$ 由该试次池化特征给出

判据（03_判据 §1.3 S5-v2）
  命中 = 该试次 $\cos\ge$ PT_HIT_COS(0.80)；通过 = 注入命中 $\ge$ PT_HIT_MIN(0.60)
  且 干净误报 $\le$ PT_FAR_MAX(0.05)

输出
----
  probe/g_dict_{tag}.json
  reports/T0_G_dict_{tag}.md
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
import sig_layer as SL          # noqa: E402

SIGMA = 0.020
F0 = 10.0
# 判据数值一律从冻结层读，不在脚本里重复定义（D26）
PT_HIT_COS = SL.PT_HIT_COS
PT_HIT_MIN = SL.PT_HIT_MIN
PT_FAR_MAX = SL.PT_FAR_MAX


# ---------------------------------------------------------------- 工具
def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def unit_rows(V: np.ndarray) -> np.ndarray:
    V = np.atleast_2d(np.asarray(V, float))
    V = V - V.mean(1, keepdims=True)
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for i in np.where(cnt > 1)[0]:
        m = inv == i
        ranks[m] = ranks[m].mean()
    r1 = ranks[:len(pos)].sum()
    return float((r1 - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def r2_of(xh: np.ndarray, x: np.ndarray) -> float:
    return 1 - float(((xh - x) ** 2).sum()) / max(
        float(((x - x.mean((0, 1))) ** 2).sum()), 1e-12)


# ---------------------------------------------------------------- per-trial 调制字典
class PertrialMod:
    """D_t = D_0 + B diag(alpha_t) M；D_0 冻结，只学 B / M / W_alpha。"""

    def __init__(self, D0: np.ndarray, d_alpha: int = 16, seed: int = 0, device=None):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.D0 = torch.tensor(D0, dtype=torch.float32, device=self.device)
        C, K = self.D0.shape
        self.C, self.K = C, K
        self.B = nn.Parameter(torch.zeros(C, d_alpha, device=self.device))
        nn.init.normal_(self.B, std=0.02)
        # M 必须非零，否则 α≡0 且 ∂L/∂W_α ∝ M ≡ 0，两边同时死锁
        self.M = nn.Parameter(torch.zeros(d_alpha, K, device=self.device))
        nn.init.normal_(self.M, std=0.01)
        self.Wa = nn.Linear(C, d_alpha).to(self.device)
        # α 从 0 起步：初始 D_t ≡ D_0，训练只是「在阶段 1 的收敛解上做增量」
        nn.init.zeros_(self.Wa.weight)
        nn.init.zeros_(self.Wa.bias)

    def dicts(self, pool):
        import torch
        a = torch.tanh(self.Wa(pool))
        return self.D0[None] + torch.einsum("ca,ta,ak->tck", self.B, a, self.M)

    def fit(self, X, trial_of, n_trial, pool_np, steps=800, nt=32, ns=28, k=4,
            lr=3e-4, lam_a=0.1, lam_m=1e-2):
        import torch

        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        pool_t = torch.tensor(pool_np, dtype=torch.float32, device=self.device)
        order = np.argsort(trial_of, kind="mergesort")
        bounds = np.searchsorted(trial_of[order], np.arange(n_trial + 1))
        buckets = [order[bounds[i]:bounds[i + 1]] for i in range(n_trial)]
        buckets = [b for b in buckets if len(b) > 0]
        rng = np.random.default_rng(1234)
        opt = torch.optim.Adam([self.B, self.M] + list(self.Wa.parameters()), lr=lr)
        for it in range(steps):
            sel_t = rng.choice(len(buckets), min(nt, len(buckets)), replace=False)
            pk = [min(ns, len(buckets[t])) for t in sel_t]
            idx = np.concatenate([buckets[t][rng.choice(len(buckets[t]), p, replace=False)]
                                  for t, p in zip(sel_t, pk)])
            own = np.repeat(np.arange(len(sel_t)), pk)
            Dt = self.dicts(pool_t[sel_t])
            Dg = Dt[torch.tensor(own, device=self.device)]
            xb = Xt[torch.tensor(idx, device=self.device)]
            a = torch.relu(torch.einsum("nc,nck->nk", xb, Dg))
            top = torch.topk(a, k, dim=1)
            a_sp = torch.zeros_like(a).scatter(1, top.indices, top.values)
            xh = torch.einsum("nk,nck->nc", a_sp, Dg)
            loss = ((xh - xb) ** 2).mean()
            loss = loss + lam_a * (torch.tanh(self.Wa(pool_t[sel_t])) ** 2).mean()
            loss = loss + lam_m * (self.M ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def recon_batch(self, Xcube, pool, k=4):
        """Xcube (nt, T, C) → 重建 (nt, T, C)。"""
        import torch
        with torch.no_grad():
            Dt = self.dicts(torch.tensor(pool, dtype=torch.float32, device=self.device))
            xb = torch.tensor(Xcube, dtype=torch.float32, device=self.device)
            a = torch.relu(torch.einsum("tnc,tck->tnk", xb, Dt))
            top = torch.topk(a, k, dim=2)
            a_sp = torch.zeros_like(a).scatter(2, top.indices, top.values)
            return torch.einsum("tnk,tck->tnc", a_sp, Dt).cpu().numpy()


# ---------------------------------------------------------------- 单试次读出
def band_maps(Xt: np.ndarray, Tf: int, F: int) -> np.ndarray:
    """(Tf*F, C) → (F, C)：每个频点上对帧求平均的 z 化图谱。"""
    return Xt.reshape(Tf, F, Xt.shape[1]).mean(0)


def locality(V: np.ndarray) -> np.ndarray:
    """跨导局部性：去均值后 最大值 − 中位值。局灶源高，弥散节律低。"""
    Vd = V - V.mean(1, keepdims=True)
    return Vd.max(1) - np.median(Vd, axis=1)


def readouts(Xt, Xh, g_emp, fi, Tf, F):
    """一个试次在所有条件下的读出（|cos|，已按试次自身归一）。

    返回 dict，键为读出名。V 均为该条件的重建频点图谱。
    """
    Vraw = band_maps(Xt, Tf, F)
    Vrec = band_maps(Xh, Tf, F)
    o = {}
    # 无监督：按重建的局部性挑频点（端到端可用）
    fb = int(np.argmax(locality(Vrec)))
    o["sel"] = float(abs(unit(Vrec[fb]) @ g_emp))
    o["_fb"] = fb
    # 上界：用真值频点，只测「方向精度」
    o["oracle"] = float(abs(unit(Vrec[fi]) @ g_emp))
    # 参照：不经字典的原始图谱（说明表征里本来有多少信息）
    o["oracle_raw"] = float(abs(unit(Vraw[fi]) @ g_emp))
    o["sel_raw"] = float(abs(unit(Vraw[int(np.argmax(locality(Vraw)))]) @ g_emp))
    return o


def svd_atom(Xt: np.ndarray, g: np.ndarray, k: int) -> float:
    """D25 的口径：该试次 token 矩阵的 top-k 右奇异向量里，最像 g 的那个。

    与「试次级重建读出」的区别只在于**读出量**：这里要求单个正交基对准源，
    而重建读出允许任意原子组合。同字典、同数据，只换这一个量。
    """
    Xc = Xt - Xt.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return float(np.abs(unit_rows(Vt[:k]) @ g).max())


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trial", type=int, default=1200)
    ap.add_argument("--snr", type=str, default="null,10,20",
                    help="逗号分隔；`null` = 完全不注入（零对照，应给 AUC≈0.5）")
    ap.add_argument("--sae-dict", type=int, default=512)
    ap.add_argument("--sae-k", type=int, default=4)
    ap.add_argument("--sae-steps", type=int, default=2500)
    ap.add_argument("--lam-l1", type=float, default=1.0)
    ap.add_argument("--d-alpha", type=int, default=16)
    ap.add_argument("--k-res", type=int, default=24)
    ap.add_argument("--mod-steps", type=int, default=1200)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备 {dev}", flush=True)

    raw_snrs = [v.strip() for v in a.snr.split(",") if v.strip()]
    snrs = [None if v.lower() in ("null", "none", "off") else float(v) for v in raw_snrs]
    seeds = [int(v) for v in a.seeds.split(",") if v.strip()]
    out = dict(n_trial=a.n_trial, snr_grid=snrs, sae_dict=a.sae_dict, sae_k=a.sae_k,
               sae_steps=a.sae_steps, lam_l1=a.lam_l1, d_alpha=a.d_alpha, k_res=a.k_res,
               mod_steps=a.mod_steps, train_frac=a.train_frac, sigma=SIGMA,
               centers=F0, hit_cos=PT_HIT_COS, hit_min=PT_HIT_MIN, far_max=PT_FAR_MAX,
               runs=[])
    DSETS = ("global", "pertrial_res", "pertrial_mod")
    RDS = (("oracle", "oracle_raw"), ("sel", "sel_raw"))

    for seed in seeds:
        X0, ch_names, fs = D22.load_trials(a.n_trial, seed)
        n = len(X0)
        is_inj = np.zeros(n, bool)
        is_inj[np.random.default_rng(seed).permutation(n)[: n // 2]] = True

        P, d = D24.load_geometry(ch_names)
        w = np.exp(-(d ** 2) / (2 * SIGMA ** 2)); w /= w.max()
        sel = np.where(w > 1e-4)[0]
        main_ch = int(np.argmax(w))
        ne = float(w.sum() ** 2 / (w ** 2).sum())
        ei, ej = D24.knn_edges(P, k=4)
        groups = D24.group_masks(ch_names)

        # 训练/测试按注入与否分层
        rng = np.random.default_rng(seed + 77)
        tr = np.zeros(n, bool)
        for cls in (True, False):
            idx = np.where(is_inj == cls)[0]
            rng.shuffle(idx)
            tr[idx[: int(round(a.train_frac * len(idx)))]] = True
        te = ~tr

        env = D22.envelope(X0.shape[-1], fs, *D22.INJ_WIN_S, D22.INJ_RAMP_S)
        tt = np.arange(X0.shape[-1]) / fs
        ph = np.random.default_rng(seed + 1000).uniform(0, 2 * np.pi, n)
        p_bg = float(np.median(D22.band_power(X0[is_inj][:, main_ch, :], fs, F0)))

        print(f"\n=== seed {seed} | {n} 试次（注入 {is_inj.sum()}）| N_eff {ne:.1f} "
              f"| 训练 {tr.sum()} / 测试 {te.sum()} ===", flush=True)

        for snr in snrs:
            t0 = time.time()
            if snr is None:
                # 零对照：不做任何注入。is_inj 仍把一半试次标成「注入」，
                # 因此任何真实效应都会消失，AUC 应回落到 0.5。
                amp = 0.0
            else:
                amp = float(np.sqrt(2.0 * p_bg * 10 ** (snr / 10)
                                    / max(float((env ** 2).mean()), 1e-12)))
            X = X0.copy()
            X[:, sel, :] += (amp * w[sel])[None, :, None] * env[None, None, :] * np.sin(
                2 * np.pi * F0 * tt[None, None, :] + ph[:, None, None])
            X[~is_inj] = X0[~is_inj]

            tok, m = D22.build_tokens(X, fs, ref_trials=np.where(~is_inj)[0])
            C, F, Tf = m["C"], m["F"], m["Tf"]
            A = ((tok.reshape(-1, F, C) - m["mu"]) / m["sd"]).reshape(n, Tf, F, C)
            Tflat = A.reshape(-1, C)
            trial_of = np.repeat(np.arange(n), Tf * F)
            fi = int(np.argmin(np.abs(m["fb"] - F0)))

            # 参照方向：经验签名（只用训练试次标定）
            Vfi = A[:, :, fi, :].mean(1)                       # (n, C)
            g_emp = unit(Vfi[tr & is_inj].mean(0) - Vfi[tr & ~is_inj].mean(0))
            gw = np.zeros(C); gw[sel] = w[sel]
            g_w = unit(gw)
            g_rand = unit(np.random.default_rng(seed + 5).standard_normal(C))

            tr_tok = np.where(tr[trial_of])[0]
            if len(tr_tok) > 120_000:
                tr_tok = rng.choice(tr_tok, 120_000, replace=False)

            # 阶段 1
            sae = D24.RegSAE(C, a.sae_dict, a.sae_k, seed=seed, ei=ei, ej=ej, groups=groups)
            sae.fit(Tflat[tr_tok], steps=a.sae_steps, reg="l1", lam=a.lam_l1)
            D0 = sae.atom_dirs().astype(np.float32).T           # (C, K)

            te_trials = np.where(te)[0]
            cube = A[te_trials].reshape(len(te_trials), Tf * F, C)   # (n_te, Tf*F, C)
            # RegSAE.reconstruct 的 topk 作用在 dim=1（token 维），必须喂 2 维
            # (N, C)。喂 3 维会让 topk 在 token 轴上取，重建直接失效（R² 为负）。
            Xh_g = np.concatenate([sae.reconstruct(cube[i:i + 32].reshape(-1, C))
                                   for i in range(0, len(cube), 32)]).reshape(cube.shape)
            r2_g = r2_of(Xh_g, cube)

            # 阶段 2a：残差字典 D_t = [D_0, V_t]（V_t 为该试次对 D_0 残差的 top-k 右奇异向量）
            Xh_r = np.empty_like(Xh_g)
            D_res = []
            for i in range(len(cube)):
                R = cube[i] - Xh_g[i]
                R = R - R.mean(0)
                U, S, Vt = np.linalg.svd(R, full_matrices=False)
                V = Vt[: a.k_res]                                # (kr, C)
                Xh_r[i] = Xh_g[i] + (R @ V.T) @ V
                D_res.append(np.concatenate([D0, V.T.astype(np.float32)], axis=1))
            r2_r = r2_of(Xh_r, cube)

            # 阶段 2b：调制字典
            pool_all = A.mean((1, 2))                             # (n, C) 试次级池化
            mod = PertrialMod(D0, d_alpha=a.d_alpha, seed=seed, device=dev)
            mod.fit(Tflat, trial_of, n, pool_all, steps=a.mod_steps, k=a.sae_k)
            Xh_m = np.concatenate([mod.recon_batch(cube[i:i + 32], pool_all[te_trials[i:i + 32]],
                                                   k=a.sae_k)
                                   for i in range(0, len(cube), 32)])
            r2_m = r2_of(Xh_m, cube)

            conds = {"global": (Xh_g, r2_g), "pertrial_res": (Xh_r, r2_r),
                     "pertrial_mod": (Xh_m, r2_m)}
            # 每种字典自己的原子方向，用于「单原子」读出（2×2 表每格要有出处）
            with torch.no_grad():
                Dt_mod_te = mod.dicts(torch.tensor(pool_all[te_trials], dtype=torch.float32,
                                                   device=dev)).cpu().numpy()
            # 每个试次一份 (C,K) 字典：global 对所有试次相同（这正是 AUC=0.5 的原因）
            Dts = {"global": [D0] * len(te_trials), "pertrial_res": D_res,
                   "pertrial_mod": Dt_mod_te}
            rows = []
            for j, t in enumerate(te_trials):
                rec = dict(trial=int(t), inj=bool(is_inj[t]))
                for name, (Xh, _) in conds.items():
                    rd = readouts(cube[j], Xh[j], g_emp, fi, Tf, F)
                    for kk, vv in rd.items():
                        rec[f"{name}|{kk}"] = vv
                    rec[f"{name}|atom"] = float(
                        np.abs(unit_rows(Dts[name][j].T) @ g_emp).max())
                # D25 口径：无字典的 per-trial SVD + 单原子
                rec["svd|atom"] = svd_atom(cube[j], g_emp, a.k_res)
                # 参照对照：随机方向
                rec["rand|oracle_raw"] = float(abs(unit(band_maps(cube[j], Tf, F)[fi]) @ g_rand))
                rec["an|g_w"] = float(abs(
                    unit(band_maps(Xh_g[j], Tf, F)[fi]) @ g_w))
                rows.append(rec)

            inj_v = np.array([r["inj"] for r in rows])
            summ = {}
            keys = [k for k in rows[0] if "|" in k and not k.split("|")[1].startswith("_")]
            for k in keys:
                v = np.array([r[k] for r in rows])
                summ[k] = dict(
                    inj_med=float(np.median(v[inj_v])),
                    cln_med=float(np.median(v[~inj_v])),
                    inj_hit=float(np.mean(v[inj_v] >= PT_HIT_COS)),
                    cln_hit=float(np.mean(v[~inj_v] >= PT_HIT_COS)),
                    auc=float(auc(v[inj_v], v[~inj_v])))
            for s in summ.values():
                s["passed"] = bool(s["inj_hit"] >= PT_HIT_MIN and s["cln_hit"] <= PT_FAR_MAX)
            # 无监督选频点的准确率
            fb_acc = {}
            for name in DSETS:
                v = np.array([r[f"{name}|_fb"] for r in rows])
                fb_acc[name] = float(np.mean(v[inj_v] == fi))
            # 解析参照的对比
            an_gw = float(np.median([r["an|g_w"] for r in rows]))

            out["runs"].append(dict(seed=seed, snr=(None if snr is None else float(snr)),
                                    r2=dict(global_=float(r2_g), res=float(r2_r),
                                            mod=float(r2_m)),
                                    fb_acc=fb_acc, an_gw_median=an_gw, summary=summ))

            lab = "无注入" if snr is None else f"{snr:+.0f} dB"
            print(f"\n  SNR {lab} | R² 全局 {r2_g:.3f} / 残差 {r2_r:.3f} / "
                  f"调制 {r2_m:.3f} | 选频点准确率 "
                  + " ".join(f"{k} {v:.0%}" for k, v in fb_acc.items()), flush=True)
            print(f"  {'条件|读出':30s} {'注入中位':>7s} {'干净中位':>7s} {'注入命中':>7s} "
                  f"{'干净误报':>7s} {'AUC':>6s}  判定")
            for k in sorted(summ, key=lambda s: -summ[s]["auc"]):
                s = summ[k]
                print(f"  {k:30s} {s['inj_med']:7.2f} {s['cln_med']:7.2f} "
                      f"{s['inj_hit']:7.0%} {s['cln_hit']:7.0%} {s['auc']:6.3f}  "
                      f"{'PASS' if s['passed'] else '-'}", flush=True)
            print(f"  （解析参照 g_w 的 cos 中位 {an_gw:.2f}，经验签名口径见上）"
                  f"  [{time.time()-t0:.0f}s]", flush=True)

    tag = a.tag or "z"
    (ROOT / "probe").mkdir(exist_ok=True)
    jp = ROOT / "probe" / f"g_dict_{tag}.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- 报告
    lines = [
        "# T0-G：per-trial 稀疏字典与试次级读出",
        "",
        f"注入 $N_{{eff}}={ne:.1f}$（σ={SIGMA*1000:.0f} mm，{ch_names[main_ch]}）"
        f"@ {F0:g} Hz；{a.n_trial} 试次，注入/干净各半；"
        f"训练 {int(a.train_frac*100)}% / 测试 {int(100-a.train_frac*100)}%（分层）。",
        "",
        "参照方向 $g_{\\rm emp}$ 由**训练试次**标定（注入均值 − 干净均值），"
        "不是逐试次真值；测试试次与标定试次不相交。",
        "",
        f"命中 = $\\cos\\ge${PT_HIT_COS}；判据：注入命中 $\\ge${PT_HIT_MIN} 且 "
        f"干净误报 $\\le${PT_FAR_MAX}。",
        "",
        "## 主表（按 SNR）",
        "",
        "| SNR | 条件 \\| 读出 | 注入中位 | 干净中位 | 注入命中 | 干净误报 | AUC | 种子全过 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for snr in snrs:
        lab = "无注入" if snr is None else f"{snr:+.0f}"
        keys = [k for k in out["runs"][0]["summary"]]
        for k in sorted(keys, key=lambda s: -np.mean(
                [r["summary"][s]["auc"] for r in out["runs"] if r["snr"] == snr])):
            ss = [r["summary"][k] for r in out["runs"] if r["snr"] == snr]
            lines.append(
                f"| {lab} | `{k}` | "
                f"{np.mean([s['inj_med'] for s in ss]):.2f} | "
                f"{np.mean([s['cln_med'] for s in ss]):.2f} | "
                f"{np.mean([s['inj_hit'] for s in ss]):.0%} | "
                f"{np.mean([s['cln_hit'] for s in ss]):.0%} | "
                f"{np.mean([s['auc'] for s in ss]):.3f} | "
                f"{sum(s['passed'] for s in ss)}/{len(ss)} |")
    lines += ["", "## 重建质量 / 选频点准确率", "",
              "| 种子 | SNR | R² 全局 | R² 残差 | R² 调制 | 选频点准确率(全局) | 选频点(残差) |",
              "|---|---|---|---|---|---|---|"]
    for r in out["runs"]:
        lab = "无注入" if r["snr"] is None else f"{r['snr']:+.0f}"
        lines.append(f"| {r['seed']} | {lab} | {r['r2']['global_']:.3f} | "
                     f"{r['r2']['res']:.3f} | {r['r2']['mod']:.3f} | "
                     f"{r['fb_acc']['global']:.0%} | {r['fb_acc']['pertrial_res']:.0%} |")
    rp = ROOT / "reports" / f"T0_G_dict_{tag}.md"
    rp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写出 {jp.name} / {rp.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
