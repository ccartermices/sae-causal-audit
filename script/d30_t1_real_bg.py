#!/usr/bin/env python
r"""d30 — T1：真实背景（BCI2a，22 导）上的可分辨边界。

要回答的问题
------------
G（d29）在 mmidb 合成背景上给出的结论是：**换读出量**之后，试次内的局灶源在
+20 dB 可定位（AUC 0.999、命中 91%），+10 dB 是边界（AUC 0.98 但命中 36–44%）。
这套读出搬到真实 EEG 上还成立吗？真实 EEG 的等效可分辨 SNR 落在边界哪一侧？

为什么不能直接拿真实数据当检验
------------------------------
真实 EEG 没有「源位置」真值，观察性验证只能得到否证（S0a/S0b/S0c 已经做过）。
所以这里沿用 C 的做法：往**真实试次**里注入已知源，保留全部真实背景结构
（1/f、mu 节律、共模、肌电、伪迹、电极噪声）。真值已知，背景真实。

三种背景，正交地分离两个混淆因素
--------------------------------
| 背景 | 做了什么 | 保留 | 去掉 |
|---|---|---|---|
| `real`  | 原样 | 全部真实结构 | — |
| `phase` | 逐试次逐通道 FFT 幅度不变、相位均匀随机 | 功率谱、单通道幅度分布 | **跨通道结构**（共模、相干、诱发成分） |
| `white` | 逐通道除以自身平滑幅度谱 | 跨通道结构、瞬时非平稳性 | **谱形**（1/f、mu 峰） |

`real − phase` 给出「跨通道结构」的代价，`real − white` 给出「谱形」的代价。
这两个是 S0c「去共模后余弦 0.58 < 零对照 0.62」的直接量化。

判据（T1-bound，先验确定）
--------------------------
对每个 (背景, SNR)：以**试次级重建方向**（G 的胜出读出）读 $\lvert\cos\rvert$，
  - 命中：注入试次 $\lvert\cos\rvert \ge$ `PT_HIT_COS`
  - 误报：干净试次 $\lvert\cos\rvert \ge$ `PT_HIT_COS`
  - 通过：命中 $\ge$ `PT_HIT_MIN` 且误报 $\le$ `PT_FAR_MAX`，且 `snr >= PT_SNR_FLOOR_MIN_DB`
零对照：`snr=null` 不注入，AUC 必须回落到 0.45–0.55。

输出
----
  probe/t1_real_bg_<tag>.json
  reports/T1_real_bg_<tag>.md
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
import d22_s0_synth as D22      # noqa: E402
import d24_e_designs as D24     # noqa: E402
import d29_g_pertrial_dict as D29   # noqa: E402

EPO = ROOT / "data" / "interim" / "epochs" / "bci2a"
# bci2a 只有中央—顶—枕区，22 个 EEG 导；EOG1/2/3 无标准坐标，且不应进源定位
DROP_CH = ("EOG1", "EOG2", "EOG3")
SRC_CH = "C3"                   # 与 d29 同位置，便于跨数据集直接比较
SIGMA = 0.025                   # 22 导间距约 35 mm；25 mm 下 N_eff ≈ 2–3
F0 = 10.0


# ---------------------------------------------------------------- 数据
def load_bci2a(n_trial: int, seed: int = 0):
    """按被试顺序铺开取试次，只用 22 个 EEG 导。"""
    files = sorted(EPO.glob("*.npz"), key=lambda p: p.stem)
    per = max(1, int(np.ceil(n_trial / len(files))))
    xs = []
    for p in files:
        X = np.load(p)["X"]
        xs.append(X[:min(per, len(X))])
        if sum(len(v) for v in xs) >= n_trial:
            break
    X = np.concatenate(xs)
    z0 = np.load(files[0])
    ch_all = [str(c) for c in z0["ch_names"]]
    fs = float(z0["sfreq"])
    keep = [i for i, c in enumerate(ch_all) if c not in DROP_CH]
    X = X[:n_trial][:, keep, :].astype(np.float32)
    ch = [ch_all[i] for i in keep]
    rng = np.random.default_rng(seed + 31337)
    idx = rng.permutation(len(X))[:n_trial]
    return X[idx], ch, fs


def load_geometry(ch_names: list[str]):
    import mne

    mont = mne.channels.make_standard_montage("standard_1005")
    pos = mont.get_positions()["ch_pos"]
    P = np.array([pos[c] for c in ch_names], dtype=float)
    d = np.linalg.norm(P - P[ch_names.index(D24.CENTER)], axis=1)
    return P, d


# ---------------------------------------------------------------- 背景变体
def bg_variant(X: np.ndarray, fs: float, mode: str, seed: int = 0) -> np.ndarray:
    """real / phase / white，见模块 docstring 的表。"""
    if mode == "real":
        return X
    n, C, T = X.shape
    F = np.fft.rfft(X, axis=-1)
    mag = np.abs(F)
    if mode == "phase":
        rng = np.random.default_rng(seed + 9001)
        ph = rng.uniform(0, 2 * np.pi, mag.shape)
        # DC 与 Nyquist 必须实：相位置 0（DC 另置幅值为 0，等价去均值）
        ph[:, :, 0] = 0.0
        ph[:, :, -1] = 0.0
        mag[:, :, 0] = 0.0
        Y = np.fft.irfft(mag * np.exp(1j * ph), n=T, axis=-1)
        return Y.astype(np.float32)
    if mode == "white":
        from scipy.signal import welch

        # 逐试次白化不可用：单个 4 s 试次的 10 Hz 谱估计方差极大，重尾；
        # 而且 1/f 在 >50 Hz 的 PSD 小到 1e-11，1/sqrt(S) 会把高频噪声放大 1e6 倍，
        # 时域方差直接爆炸（实测 σ 9 µV → 1.4e4 µV），SNR 标定随之失效。
        # 改为：① 跨试次中位谱（稳定）② 只在 1–40 Hz 整形 ③ 加下限 + 余弦 taper。
        fq, P = welch(X, fs=fs, nperseg=min(T, int(round(2.0 * fs))), axis=-1)
        Pmed = np.median(P, axis=0)                     # (C, nf)
        fr = np.fft.rfftfreq(T, d=1.0 / fs)
        S = np.stack([np.interp(fr, fq, Pmed[c]) for c in range(C)])
        lo, hi = 1.0, 40.0
        m = (fr >= lo) & (fr <= hi)
        floor = 0.05 * float(np.median(S[:, (fr >= 6) & (fr <= 20)]))
        g = np.ones_like(S)
        g[:, m] = 1.0 / np.sqrt(S[:, m] + floor)
        # 余弦 taper，避免 m 边界不连续造成振铃
        for edge, inner in ((lo, lo + 1.0), (hi, hi - 1.0)):
            t = ((fr - min(edge, inner)) / abs(inner - edge)).clip(0, 1)
            w = 0.5 - 0.5 * np.cos(np.pi * t)
            g = np.where(((fr >= min(edge, inner)) & (fr <= max(edge, inner)))[None, :],
                         g * w[None, :] + (1 - w)[None, :], g)
        Y = np.fft.irfft(F * g, n=T, axis=-1)
        Y = Y - Y.mean(-1, keepdims=True)
        return Y.astype(np.float32)
    raise ValueError(mode)


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trial", type=int, default=600)
    ap.add_argument("--bgs", type=str, default="real",
                    help="逗号分隔：real,phase,white")
    ap.add_argument("--snr", type=str, default="null,0,5,10,15,20")
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--f0", type=float, default=F0, help="注入频率（Hz）")
    ap.add_argument("--sae-dict", type=int, default=512)
    ap.add_argument("--sae-k", type=int, default=4)
    ap.add_argument("--sae-steps", type=int, default=1500)
    ap.add_argument("--lam-l1", type=float, default=1.0)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--tag", type=str, default="z")
    a = ap.parse_args()

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"设备 {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

    bgs = [v.strip() for v in a.bgs.split(",") if v.strip()]
    raw = [v.strip() for v in a.snr.split(",") if v.strip()]
    snrs = [None if v.lower() in ("null", "none", "off") else float(v) for v in raw]
    seeds = [int(v) for v in a.seeds.split(",") if v.strip()]

    f0 = a.f0
    out = dict(dataset="bci2a", n_trial=a.n_trial, sigma=a.sigma, src_ch=SRC_CH,
               f0=f0, hit_cos=SL.PT_HIT_COS, hit_min=SL.PT_HIT_MIN,
               far_max=SL.PT_FAR_MAX, snr_floor=SL.PT_SNR_FLOOR_MIN_DB,
               sae=dict(d=a.sae_dict, k=a.sae_k, steps=a.sae_steps, lam=a.lam_l1),
               runs=[])

    for seed in seeds:
        X0, ch, fs = load_bci2a(a.n_trial, seed)
        n = len(X0)
        print(f"\n=== seed {seed} | {n} 试次 | {len(ch)} 导 | fs {fs:.0f} Hz "
              f"| T {X0.shape[-1]} ===", flush=True)

        P, d = load_geometry(ch)
        w = np.exp(-(d ** 2) / (2 * a.sigma ** 2)); w /= w.max()
        sel = np.where(w > 1e-4)[0]
        main_ch = ch.index(SRC_CH)
        ne = float(w.sum() ** 2 / (w ** 2).sum())
        ei, ej = D24.knn_edges(P, k=4)
        groups = D24.group_masks(ch)

        # 注入/干净各半，训练/测试按注入分层
        rng = np.random.default_rng(seed + 77)
        is_inj = np.zeros(n, bool)
        is_inj[rng.permutation(n)[: n // 2]] = True
        tr = np.zeros(n, bool)
        for cls in (True, False):
            idx = np.where(is_inj == cls)[0]
            rng.shuffle(idx)
            tr[idx[: int(round(a.train_frac * len(idx)))]] = True
        te = ~tr

        env = D22.envelope(X0.shape[-1], fs, *D22.INJ_WIN_S, D22.INJ_RAMP_S)
        tt = np.arange(X0.shape[-1]) / fs
        ph = np.random.default_rng(seed + 1000).uniform(0, 2 * np.pi, n)
        print(f"  注入导 {[ch[i] for i in sel]}（N_eff {ne:.1f}）| 训练 {tr.sum()} / "
              f"测试 {te.sum()}", flush=True)

        for bg in bgs:
            Xb = bg_variant(X0, fs, bg, seed)
            # 背景质控：逐通道相对幅度（白化后若某导被放大成主导，读数不可比）
            amp_bg = np.median(np.abs(Xb), axis=(0, 2))
            print(f"\n  --- 背景 {bg} | 逐导幅度中位 {amp_bg.min():.3g}–{amp_bg.max():.3g} "
                  f"---", flush=True)

            for snr in snrs:
                t0 = time.time()
                if snr is None:
                    amp = 0.0
                else:
                    p_bg = float(np.median(D22.band_power(Xb[is_inj][:, main_ch, :], fs, f0)))
                    amp = float(np.sqrt(2.0 * p_bg * 10 ** (snr / 10)
                                        / max(float((env ** 2).mean()), 1e-12)))
                X = Xb.copy()
                X[:, sel, :] += (amp * w[sel])[None, :, None] * env[None, None, :] * np.sin(
                    2 * np.pi * f0 * tt[None, None, :] + ph[:, None, None])
                X[~is_inj] = Xb[~is_inj]

                tok, m = D22.build_tokens(X, fs, ref_trials=np.where(~is_inj)[0])
                C, F, Tf = m["C"], m["F"], m["Tf"]
                A = ((tok.reshape(-1, F, C) - m["mu"]) / m["sd"]).reshape(n, Tf, F, C)
                Tflat = A.reshape(-1, C)
                trial_of = np.repeat(np.arange(n), Tf * F)
                fi = int(np.argmin(np.abs(m["fb"] - f0)))

                Vfi = A[:, :, fi, :].mean(1)
                g_emp = D29.unit(Vfi[tr & is_inj].mean(0) - Vfi[tr & ~is_inj].mean(0))

                tr_tok = np.where(tr[trial_of])[0]
                if len(tr_tok) > 120_000:
                    tr_tok = np.random.default_rng(seed).choice(
                        tr_tok, 120_000, replace=False)

                sae = D24.RegSAE(C, a.sae_dict, a.sae_k, seed=seed,
                                 ei=ei, ej=ej, groups=groups)
                sae.fit(Tflat[tr_tok], steps=a.sae_steps, reg="l1", lam=a.lam_l1)
                D0 = sae.atom_dirs().astype(np.float32).T          # (C, K)

                z_main = float(A[is_inj][:, :, fi, main_ch].mean()
                               - A[~is_inj][:, :, fi, main_ch].mean())
                te_trials = np.where(te)[0]
                cube = A[te_trials].reshape(len(te_trials), Tf * F, C)
                Xh = np.concatenate([sae.reconstruct(cube[i:i + 32].reshape(-1, C))
                                     for i in range(0, len(cube), 32)]).reshape(cube.shape)
                r2 = D29.r2_of(Xh, cube)
                atom_g = float(np.abs(D29.unit_rows(D0.T) @ g_emp).max())

                rows = []
                for j, t in enumerate(te_trials):
                    rec = dict(trial=int(t), inj=bool(is_inj[t]))
                    rd = D29.readouts(cube[j], Xh[j], g_emp, fi, Tf, F)
                    for kk, vv in rd.items():
                        rec[f"rec|{kk}"] = vv
                    # 单原子读出（D23/D24 口径）：对所有试次相同，AUC 必为 0.5
                    rec["rec|atom"] = atom_g
                    rows.append(rec)

                inj_v = np.array([r["inj"] for r in rows])
                summ = {}
                for rd in ("oracle", "sel", "oracle_raw", "sel_raw", "atom"):
                    v = np.array([r[f"rec|{rd}"] for r in rows])
                    s = dict(inj_med=float(np.median(v[inj_v])),
                             cln_med=float(np.median(v[~inj_v])),
                             inj_hit=float(np.mean(v[inj_v] >= SL.PT_HIT_COS)),
                             cln_hit=float(np.mean(v[~inj_v] >= SL.PT_HIT_COS)),
                             auc=float(D29.auc(v[inj_v], v[~inj_v])))
                    # D28：绝对命中线随维度漂移，改判 AUC + 分离度；
                    # 绝对命中率仍记录，只作描述量。
                    s["sep"] = s["inj_hit"] - s["cln_hit"]
                    s["passed"] = bool(s["auc"] >= SL.T1_AUC_MIN
                                       and s["sep"] >= SL.T1_SEP_MIN)
                    summ[rd] = s

                # 实得 SIR（注入源功率 / 总功率），说明 SNR 标定是否可信
                if snr is None:
                    sir = float("nan")
                else:
                    p_tot = float(np.median(D22.band_power(X[:, main_ch, :], fs, f0)))
                    sir = float(10 * np.log10(max(p_tot, 1e-30) / max(p_tot - amp ** 2 *
                                          float((env ** 2).mean()) / 2, 1e-30)))

                out["runs"].append(dict(seed=seed, bg=bg, snr=snr, r2=float(r2),
                                        sir_db=sir, z_main=z_main, n_te=int(te.sum()),
                                        summary=summ))
                print(f"  {bg:5s} SNR {('null' if snr is None else f'{snr:+.0f}'):>5s} "
                      f"| R² {r2:.3f} | " + "  ".join(
                          f"{k} {v['auc']:.3f}/{v['inj_hit']:.0%}/{v['cln_hit']:.0%}"
                          for k, v in summ.items() if k in ("oracle", "sel", "atom"))
                      + f" | {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------ 输出
    (ROOT / "probe").mkdir(exist_ok=True)
    (ROOT / "probe" / f"t1_real_bg_{a.tag}.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    lines = [f"# T1 真实背景可分辨边界（bci2a，{a.n_trial} 试次，σ={a.sigma*1000:.0f} mm，"
             f"f0={f0:g} Hz）", "",
             "自动生成，勿手改。AUC（命中率/误报率），命中线 `|cos| ≥ "
             f"{SL.PT_HIT_COS}`（仅描述量）。`z_main` 是注入试次在主道真值频点上的平均 z 增量，"
             "用作「SNR 刻度 → 实际信噪」的仪表读数。通过 = "
             f"AUC ≥ {SL.T1_AUC_MIN} 且 分离度 ≥ {SL.T1_SEP_MIN}（D28）。", "",
             "| 背景 | SNR | z_main | R² | " + " | ".join(
                 f"`{k}`" for k in ("oracle", "sel", "oracle_raw", "atom")) + " | 通过 |",
             "|---|---|---|---|" + "---|" * 4 + "---|"]
    for r in out["runs"]:
        s = r["summary"]
        lab = "null" if r["snr"] is None else f"{r['snr']:+.0f}"
        z = "—" if r.get("z_main") is None else f"{r['z_main']:+.1f}"
        lines.append(
            f"| {r['bg']} | {lab} | {z} | {r['r2']:.3f} | " + " | ".join(
                f"{s[k]['auc']:.3f} ({s[k]['inj_hit']:.0%}/{s[k]['cln_hit']:.0%})"
                for k in ("oracle", "sel", "oracle_raw", "atom"))
            + f" | {'✓' if s['oracle']['passed'] else '—'} |")
    (ROOT / "reports" / f"T1_real_bg_{a.tag}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写出 probe/t1_real_bg_{a.tag}.json 与 reports/T1_real_bg_{a.tag}.md",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
