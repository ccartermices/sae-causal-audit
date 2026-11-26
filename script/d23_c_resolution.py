#!/usr/bin/env python
"""d23 — 原子的空间分辨率下限：注入源的空间跨度扫描。

C（d22）已经证明两件事：
  1. 局部（3 导）注入在 +21 dB SIR 下也取不到，且**字典覆盖率与 SNR 完全无关**
  2. 半球（27 导）注入在 0 dB 就通过，覆盖率 0.77

所以「多强的信号才够」是错的问题，正确的问题是「**多大的空间尺度才够**」。
本脚本把注入源做成高斯空间包，用等效导联数 N_eff = (Σw)²/Σw² 标定跨度，
从单导扫到全脑，测字典覆盖率怎么变。曲线的拐点就是原子的空间分辨率下限。

输出
----
  probe/c_resolution_{tag}.json
  reports/T0_C_resolution_{tag}.md
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

import sig_layer as SL          # noqa: E402
import token_sae as TS          # noqa: E402
import d22_s0_synth as D22      # noqa: E402

# 高斯包宽度（米，标准头模型半径约 0.09 m）。先验给，不看结果调。
SIGMA_GRID = (0.004, 0.008, 0.012, 0.02, 0.03, 0.045, 0.065, 0.09)
CENTERS = ("C3", "Oz")


def gauss_weights(ch_names: list[str], center: str):
    """按 10-05 坐标到 center 的距离给高斯权重，返回 (idx, w, 实际 N_eff)。"""
    import mne

    mont = mne.channels.make_standard_montage("standard_1005")
    pos = mont.get_positions()["ch_pos"]
    P = np.array([pos[c] for c in ch_names], dtype=float)
    c0 = P[ch_names.index(center)]
    d = np.linalg.norm(P - c0, axis=1)
    return d


def weights_from_sigma(d: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(d ** 2) / (2 * sigma ** 2))


def n_eff(w: np.ndarray) -> float:
    w = w / w.max()
    return float(w.sum() ** 2 / (w ** 2).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trial", type=int, default=1200)
    ap.add_argument("--snr", type=float, default=10.0,
                    help="固定 SNR（已证与结果无关，见 d22）")
    ap.add_argument("--sae-dict", type=int, default=1024)
    ap.add_argument("--sae-k", type=int, default=16)
    ap.add_argument("--sae-steps", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=100_000)
    ap.add_argument("--centers", type=str, default="C3,Oz")
    ap.add_argument("--sigma", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)

    sigmas = tuple(float(v) for v in a.sigma.split(",") if v.strip()) or SIGMA_GRID
    centers = tuple(c.strip() for c in a.centers.split(",") if c.strip())
    f0 = {"C3": 10.0, "Oz": 11.0, "Pz": 12.0, "Fz": 9.0}

    rng = np.random.default_rng(a.seed)
    X0, ch_names, fs = D22.load_trials(a.n_trial, a.seed)
    n = len(X0)
    is_inj = np.zeros(n, bool)
    is_inj[rng.permutation(n)[: n // 2]] = True
    env = D22.envelope(X0.shape[-1], fs, D22.INJ_WIN_S[0], D22.INJ_WIN_S[1], D22.INJ_RAMP_S)
    env_msq = float((env ** 2).mean())
    tt = np.arange(X0.shape[-1]) / fs
    ph = np.random.default_rng(a.seed + 1000).uniform(0, 2 * np.pi, n)
    print(f"试次 {n}（注入 {is_inj.sum()}）| 导联 {len(ch_names)} | SNR 固定 {a.snr:+.0f} dB "
          f"| σ 网格 {sigmas}", flush=True)

    out = dict(n_trial=int(n), snr_db=float(a.snr), sigma_grid=list(sigmas),
               sae_dict=a.sae_dict, sae_k=a.sae_k, sae_steps=a.sae_steps,
               token_band=list(D22.TOKEN_BAND), inj_win_s=list(D22.INJ_WIN_S),
               curves=[], seed=a.seed)

    for center in centers:
        f0c = f0.get(center, 10.0)
        d = gauss_weights(ch_names, center)
        rec = dict(center=center, f0=f0c, points=[])
        print(f"\n=== 中心 {center} @ {f0c} Hz（到最近导 {d.min():.3f} m）===", flush=True)
        for sigma in sigmas:
            t0 = time.time()
            w_all = weights_from_sigma(d, sigma)
            ne = n_eff(w_all)
            w = w_all / w_all.max()
            main_ch = int(np.argmax(w))
            X = X0.copy()
            p_bg = float(np.median(D22.band_power(X0[is_inj][:, main_ch, :], fs, f0c)))
            p_target = p_bg * 10.0 ** (a.snr / 10.0)
            amp = float(np.sqrt(2.0 * p_target / max(env_msq, 1e-12)))
            sel = np.where(w > 1e-4)[0]
            X[:, sel, :] += (amp * w[sel])[None, :, None] * env[None, None, :] * np.sin(
                2 * np.pi * f0c * tt[None, None, :] + ph[:, None, None])
            X[~is_inj] = X0[~is_inj]

            tok, m = D22.build_tokens(X, fs, ref_trials=np.where(~is_inj)[0])
            C, F = m["C"], m["F"]
            mu, sd = m["mu"], m["sd"]
            T_ = ((tok.reshape(-1, F, C) - mu) / sd).reshape(-1, C).astype(np.float32)
            trial_of, fi_of = m["trial_of"], m["fbin_of"]
            fi = int(np.argmin(np.abs(m["fb"] - f0c)))
            tr = np.where(np.isin(trial_of, np.where(~is_inj)[0]))[0]
            if len(tr) > a.max_tokens:
                tr = rng.choice(tr, a.max_tokens, replace=False)
            sae = TS.TopKSAE(d_in=C, d_dict=a.sae_dict, k=a.sae_k, seed=a.seed)
            sae.fit(T_[tr], steps=a.sae_steps)
            idx, val = sae.encode_sparse(T_)
            act = TS.atom_trial_activation(idx, val, trial_of, n, a.sae_dict)
            dirs = sae.atom_dirs().astype(np.float64)

            # 真值方向：z 空间里注入源的轮廓 ∝ w/sd（sd 是该导在 f0 的背景 std）
            g = np.zeros(C)
            g[sel] = w[sel] / sd[fi][sel]
            g -= g.mean()
            g /= np.linalg.norm(g) + 1e-12

            Dm = dirs - dirs.mean(1, keepdims=True)
            Dm /= (np.linalg.norm(Dm, axis=1, keepdims=True) + 1e-12)
            cos_all = np.abs(Dm @ g)
            cov_max = float(cos_all.max())
            cov5 = float(np.sort(cos_all)[-5:].mean())

            # 类特异度 top-5 的定位准确度
            ct = act[is_inj].mean(0) - act[~is_inj].mean(0)
            top5 = np.argsort(-ct)[:5]
            c5 = np.array([abs(Dm[i] @ g) for i in top5])
            # oracle：不经 SAE
            A4 = tok.reshape(n, m["Tf"], F, C)
            amp_f0 = A4[:, :, fi, :].mean(1)
            p_orc = amp_f0[is_inj].mean(0) - amp_f0[~is_inj].mean(0)
            pv = p_orc - p_orc.mean()
            orc_cos = float((pv / (np.linalg.norm(pv) + 1e-12)) @ g)

            # 随机字典基线：同样是 `sae_dict` 个 64 维随机方向，max|cos| 会是多少？
            # 没有这条线，覆盖率 0.37 与 0.77 都读不出意义（实测两者都接近/超过随机）。
            rc = []
            for _ in range(20):
                R = rng.standard_normal((a.sae_dict, C))
                R -= R.mean(1, keepdims=True)
                R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
                rc.append(float(np.abs(R @ g).max()))
            rand_cov = float(np.median(rc))
            rand_cov_p95 = float(np.percentile(rc, 95))

            null = []
            for _ in range(150):
                pick = rng.choice(a.sae_dict, 5, replace=False)
                null.append(max(abs(Dm[i] @ g) for i in pick))
            null95 = float(np.percentile(null, 95))

            rec["points"].append(dict(
                sigma=float(sigma), n_eff=float(ne), n_sel=int(len(sel)),
                cov_max=cov_max, cov_top5=cov5,
                rand_cov=rand_cov, rand_cov_p95=rand_cov_p95,
                top5_cos_mean=float(c5.mean()), top5_cos_max=float(c5.max()),
                null_p95=null95, oracle_cos=orc_cos,
                passed=bool(c5.mean() > null95 and cov_max > rand_cov_p95),
            ))
            print(f"  σ={sigma*1000:5.1f} mm | N_eff {ne:5.1f}（{len(sel):2d} 导）| "
                  f"覆盖率 {cov_max:.2f}（随机 {rand_cov:.2f}±）| top5 均 {c5.mean():.2f} | "
                  f"零对照 p95 {null95:.2f} | oracle {orc_cos:+.2f} | "
                  f"{'PASS' if (c5.mean() > null95 and cov_max > rand_cov_p95) else '-'} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        out["curves"].append(rec)

    tag = a.tag or "z"
    (ROOT / "probe").mkdir(exist_ok=True)
    jp = ROOT / "probe" / f"c_resolution_{tag}.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        f"# T0-C：原子的空间分辨率下限（SNR 固定 {a.snr:+.0f} dB）",
        "",
        "注入源做成以中心导为原点的高斯空间包，用等效导联数 "
        "$N_{eff}=(\\sum w)^2/\\sum w^2$ 标定跨度。信噪比固定，只改空间跨度 —— "
        "因为 d22 已证明覆盖率与 SNR 无关。",
        "",
        f"- 试次 {out['n_trial']}（一半注入），字典 {a.sae_dict} × k={a.sae_k}，{a.sae_steps} 步",
        "- 覆盖率 = 字典里最接近真值方向的原子余弦（结构性上界，与排序器无关）",
        "- 随机基线 = 同样 `sae_dict` 个 64 维随机方向的 max|cos| 中位。"
        "**覆盖率不超过随机基线，就说明字典里没有该方向的信息**（不是「弱」，是「无」）",
        "- 判定 = 类特异度 top-5 的平均余弦超过零对照 p95",
        "",
        "| 中心 | σ (mm) | N_eff | 参与导 | 覆盖率 max | 随机基线 | top-5 均余弦 | 零对照 p95 | oracle | 判定 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in out["curves"]:
        for p in rec["points"]:
            lines.append(
                f"| {rec['center']} | {p['sigma']*1000:.1f} | {p['n_eff']:.1f} | {p['n_sel']} | "
                f"{p['cov_max']:.2f} | {p['rand_cov']:.2f} | {p['top5_cos_mean']:.2f} | "
                f"{p['null_p95']:.2f} | "
                f"{p['oracle_cos']:+.2f} | {'**PASS**' if p['passed'] else '—'} |")
    rp = ROOT / "reports" / f"T0_C_resolution_{tag}.md"
    rp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写出 {jp.name} / {rp.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
