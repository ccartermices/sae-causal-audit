#!/usr/bin/env python
"""d22 — C：合成注入，测「原子层在什么 SNR 下才成立」。

为什么需要这个实验
------------------
S0a（SSVEP）与 S0b/S0c（mmidb）都判负，但两者都是**观察性**的：我们只知道
「真实数据上原子层取不到已知事实」，不知道**为什么**，也不知道**多强的信号才够**。
一份方法学论文必须回答后者，否则「原子不可解释」只是一句否定。

做法
----
往**真实** mmidb 试次里注入一个已知的源：
  - 空间上只占 3 个相邻导（真值地形已知）
  - 频带上只在 f0 ± 1 Hz（真值频率已知）
  - 时间上只在 epoch 内 1.0–3.0 s（真值潜伏期已知）
背景保留全部真实结构（1/f、共模、肌电、伪迹），因此结论对外部有效。

同一批试次、同一注入窗、同一相位，只改**幅度**得到 SNR 曲线 —— 曲线上的每一点
只差一个标量，不是不同数据集。

判据（S0-synth）
----------------
对每个 (配置, SNR)：
  1. 用**类特异度**（注入试次均值 − 干净试次均值）选 top-5 原子（D21-1 的教训：
     平均激活选的是「响得最多」的原子，与类别无关）
  2. 地形：这 5 个原子的方向 argmax 是否落在注入的 3 导集合里（≥3/5 记过）
  3. 形状：与真值 z 轮廓的余弦是否超过**零对照**（随机 5 原子，200 次，|cos| p95）
  4. 频率：注入 token 上的激活加权频点与 f0 的偏差

同时给出 **oracle 上界**（不经 SAE，直接在 f0 频点上算注入−干净的幅度差）：
oracle 能定位的 SNR 下限就是数据本身的检测限，SAE 与它的差值是**仪器代价**。

输出
----
  probe/c_synth.json
  reports/T0_C_synth.md
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

EPO = ROOT / "data" / "interim" / "epochs" / "mmidb"

# ---------------------------------------------------------------- 冻结的仪器参数
# 先验确定，不看结果调。改了要记 DECISIONS。
TOKEN_WIN_S = 1.0                # 1 s 窗 → 1 Hz 分辨率（f0 间隔 1 Hz 可分辨）
TOKEN_HOP_S = 0.5                # 50% 重叠
TOKEN_BAND = (6.0, 20.0)         # 覆盖 theta/alpha/beta
INJ_WIN_S = (1.0, 3.0)           # epoch 坐标（epoch 从刺激后 0.5 s 起）
INJ_RAMP_S = 0.25
INJ_WEIGHTS = (1.0, 0.62, 0.62)  # 主道 1，两侧邻导 0.62

# 真值配置：每组 = (名字, 三导集合, f0)。导联集合都取相邻三导，符合体积传导的
# 空间尺度；频率取 mu/alpha/beta 三个常用节律。
CONFIGS = [
    ("C3", ["C3", "FC3", "CP3"], 10.0),
    ("Oz", ["Oz", "O1", "POz"], 11.0),
    ("Fz", ["Fz", "FCz", "Cz"], 9.0),
    ("Pz", ["Pz", "CPz", "POz"], 12.0),
]

SNR_GRID = (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0)
# 左半球导（10-05 规则：数字为奇数即左侧；中线给 0.5）。
# 用作「大尺度平滑」的对照注入：如果原子只能表示弥散模式，这一组应当明显更容易。
HEMI_MID = ("z",)
HIT_MIN = 3                      # top-5 里至少 3 个 argmax 落入注入集
N_NULL = 200
PASS_MIN_FRAC = 0.5              # 参与判定的原子里有 ≥50% 余弦过零对照


# ---------------------------------------------------------------- 数据
def load_trials(n_trial: int, seed: int = 0):
    """按被试顺序取试次：先取每人前几试次，尽量把被试铺开。

    S088/S092/S100 是 128 Hz（`SL.EXCLUDE_STRUCT_SUBJECTS`），epoch 长度 449 点，
    与本实验的 160 Hz / 561 点几何不一致，直接排除。
    """
    files = sorted(EPO.glob("*.npz"), key=lambda p: p.stem)
    files = [p for p in files if p.stem not in SL.EXCLUDE_STRUCT_SUBJECTS]
    per = max(1, int(np.ceil(n_trial / len(files))))
    xs = []
    for p in files:
        X = np.load(p)["X"]
        xs.append(X[:min(per, len(X))])
        if sum(len(v) for v in xs) >= n_trial:
            break
    X = np.concatenate(xs)[:n_trial]
    z0 = np.load(files[0])
    ch = [str(c) for c in z0["ch_names"]]
    fs = float(z0["sfreq"])
    return X.astype(np.float32), ch, fs


# ---------------------------------------------------------------- 注入
def envelope(T: int, fs: float, t0: float, t1: float, ramp: float) -> np.ndarray:
    t = np.arange(T) / fs
    e = ((t >= t0) & (t <= t1)).astype(np.float64)
    n_r = max(1, int(round(ramp * fs)))
    h = np.hanning(2 * n_r + 1)
    h /= h.sum()
    return np.convolve(e, h, mode="same")


def band_power(x: np.ndarray, fs: float, f0: float, half: float = 1.0) -> np.ndarray:
    """(n, T) → (n,) 在 f0±half 的功率（Welch，均值）。"""
    from scipy.signal import welch

    fq, P = welch(x, fs=fs, nperseg=int(round(1.5 * fs)), axis=-1)
    b = (fq >= f0 - half) & (fq <= f0 + half)
    return P[:, b].mean(1)


def inject(X: np.ndarray, ch_idx: list[int], w: np.ndarray, f0: float,
           snr_db: float, fs: float, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """返回注入后的数据与本次注入的实测量（目标 SNR、实得 SIR、相位等）。"""
    n, C, T = X.shape
    env = envelope(T, fs, INJ_WIN_S[0], INJ_WIN_S[1], INJ_RAMP_S)
    t = np.arange(T) / fs
    ph = rng.uniform(0, 2 * np.pi, n)          # 逐试次随机相位（更真实；|STFT| 不受影响）
    # 背景在该导上的 f0 频带功率（中位），用来定标幅度
    p_bg = np.median(band_power(X[:, ch_idx[0], :], fs, f0))
    p_target = p_bg * 10.0 ** (snr_db / 10.0)
    a = float(np.sqrt(2.0 * p_target / max(float((env ** 2).mean()), 1e-12)))

    Xo = X.copy()
    for c, wc in zip(ch_idx, w):
        Xo[:, c, :] += (a * wc) * env[None, :] * np.sin(
            2 * np.pi * f0 * t[None, :] + ph[:, None])
    # 实得 SIR：注入源在主道 f0 频带的功率 / 注入后总功率
    p_tot = float(np.median(band_power(Xo[:, ch_idx[0], :], fs, f0)))
    return Xo, dict(amp=float(a), p_bg=float(p_bg), p_target=float(p_target),
                    sir=float(p_target / max(p_tot - p_target, 1e-12)),
                    env_power=float((env ** 2).mean()))


# ---------------------------------------------------------------- token
def build_tokens(X: np.ndarray, fs: float, ref_trials: np.ndarray | None = None):
    """(n,C,T) → (token 矩阵, 索引, 频点表, 帧中心, 逐 (f,c) 的 mu/sd)。

    `ref_trials` 指定算标准化统计量用的试次。合成实验里必须传**干净试次**：
    否则注入源的功率会被计入 sd，z 值被自己稀释（+10 dB 时 sd 由注入主导，
    注入反而「看不见」）。真实场景的对应做法是用一段静息基线定标。
    """
    from scipy.signal import stft

    nperseg = int(round(TOKEN_WIN_S * fs))
    hop = int(round(TOKEN_HOP_S * fs))
    f, t, Z = stft(X, fs=fs, nperseg=nperseg, noverlap=nperseg - hop, axis=-1)
    sel = (f >= TOKEN_BAND[0]) & (f <= TOKEN_BAND[1])
    fb = f[sel]
    A = np.abs(Z[:, :, sel, :]).astype(np.float32)      # (n,C,F,Tf)
    del Z
    n, C, F, Tf = A.shape
    A4 = A.transpose(0, 3, 2, 1)                        # (n,Tf,F,C)
    del A
    ref = slice(None) if ref_trials is None else ref_trials
    flat_ref = A4[ref].reshape(-1, F, C)
    mu = flat_ref.mean(0)                               # (F,C)
    sd = flat_ref.std(0)
    # mmidb 的幅度单位是 **V**（~1e-5），固定 epsilon 1e-6 与信号同量级，
    # 会把 z 化变成「除以一个常数」。floor 必须相对于 sd 自身的量级。
    sd = sd + 1e-6 * max(float(np.median(sd[sd > 0])), 1e-30)
    tok = A4.reshape(n * Tf * F, C)
    trial_of = np.repeat(np.arange(n), Tf * F)
    fbin_of = np.tile(np.tile(np.arange(F), Tf), n)
    tf_of = np.tile(np.repeat(np.arange(Tf), F), n)
    # 帧中心换算到 epoch 坐标（scipy 的 t 从 nperseg/2 起，已是 epoch 起点算）
    tvec = t
    return tok, dict(n=n, C=C, F=F, Tf=Tf, fb=fb, tvec=tvec, mu=mu, sd=sd,
                     trial_of=trial_of, fbin_of=fbin_of, tf_of=tf_of)


def hemisphere_set(ch_names: list[str]):
    """返回 (导索引, 权重)。左半球 1，中线 0.5，右半球 0。"""
    import re

    idx, w = [], []
    for i, c in enumerate(ch_names):
        m = re.search(r"(\d+)$", c)
        if not m:
            idx.append(i); w.append(0.5)          # Fz/Cz/Pz 等中线
            continue
        num = int(m.group(1))
        if num % 2 == 1:
            idx.append(i); w.append(1.0)
        elif c.startswith(("Fp", "AF", "F", "FC", "C", "CP", "P", "PO", "O", "FT", "T", "TP")):
            idx.append(i); w.append(0.0)
    idx = np.array(idx); w = np.array(w)
    return idx, w


def pr(v: np.ndarray) -> float:
    v = np.abs(v) ** 2
    return float(v.sum() ** 2 / max((v ** 2).sum(), 1e-12))


# ---------------------------------------------------------------- 单条件
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trial", type=int, default=1200)
    ap.add_argument("--snr", type=str, default=",".join(str(v) for v in SNR_GRID))
    ap.add_argument("--conf", type=str, default="all", help="配置名，逗号分隔或 all")
    ap.add_argument("--repr", type=str, default="z", choices=["z", "raw"])
    ap.add_argument("--inj-mode", type=str, default="local", choices=["local", "hemi"])
    ap.add_argument("--sae-dict", type=int, default=1024)
    ap.add_argument("--sae-k", type=int, default=16)
    ap.add_argument("--sae-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=200_000)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    snrs = [float(v) for v in a.snr.split(",") if v.strip()]
    confs = CONFIGS if a.conf == "all" else [c for c in CONFIGS if c[0] in a.conf.split(",")]
    if not confs:
        raise SystemExit(f"未知配置 {a.conf}")
    if a.inj_mode == "hemi":
        confs = [(n, cs, f) for n, cs, f in confs]

    rng = np.random.default_rng(a.seed)
    X0, ch_names, fs = load_trials(a.n_trial, a.seed)
    n = len(X0)
    is_inj = np.zeros(n, bool)
    is_inj[rng.permutation(n)[: n // 2]] = True
    print(f"试次 {n}（注入 {is_inj.sum()} / 干净 {(~is_inj).sum()}）| 导联 {len(ch_names)} "
          f"| fs {fs} | 表示 {a.repr}", flush=True)

    out = dict(n_trial=int(n), n_inj=int(is_inj.sum()), fs=float(fs), repr=a.repr,
               sae_dict=a.sae_dict, sae_k=a.sae_k, sae_steps=a.sae_steps,
               token_win_s=TOKEN_WIN_S, token_hop_s=TOKEN_HOP_S,
               token_band=list(TOKEN_BAND), inj_win_s=list(INJ_WIN_S),
               inj_weights=list(INJ_WEIGHTS), configs=[], seed=a.seed)

    for cname, ch_set, f0 in confs:
        if a.inj_mode == "hemi":
            ch_idx, w = hemisphere_set(ch_names)
            ch_set = [ch_names[i] for i, ww in zip(ch_idx, w) if ww > 0.9]
            cname = f"hemi_{cname}"
        else:
            missing = [c for c in ch_set if c not in ch_names]
            if missing:
                print(f"!! 配置 {cname} 缺导联 {missing}，跳过", flush=True)
                continue
            ch_idx = np.array([ch_names.index(c) for c in ch_set])
            w = np.array(INJ_WEIGHTS, float)
        w = w / w.max()                  # 主道权重保持 1
        print(f"\n=== 配置 {cname}：{ch_set} @ {f0} Hz ===", flush=True)

        rec = dict(name=cname, ch_set=ch_set, f0=float(f0), snr=[])
        # 相位与背景功率在配置内固定，跨 SNR 一致 → 曲线上的点只差一个标量
        env = envelope(X0.shape[-1], fs, INJ_WIN_S[0], INJ_WIN_S[1], INJ_RAMP_S)
        env_msq = float((env ** 2).mean())
        tt = np.arange(X0.shape[-1]) / fs
        ph = np.random.default_rng(a.seed + 1000).uniform(0, 2 * np.pi, n)
        p_bg = float(np.median(band_power(X0[is_inj][:, ch_idx[0], :], fs, f0)))
        for snr_db in snrs:
            t_start = time.time()
            X = X0.copy()
            p_target = p_bg * 10.0 ** (snr_db / 10.0)
            amp = float(np.sqrt(2.0 * p_target / max(env_msq, 1e-12)))
            for c, wc in zip(ch_idx, w):
                X[:, c, :] += (amp * wc) * env[None, :] * np.sin(
                    2 * np.pi * f0 * tt[None, :] + ph[:, None])
            X[~is_inj] = X0[~is_inj]        # 干净试次原样

            p_tot = float(np.median(band_power(X[is_inj][:, ch_idx[0], :], fs, f0)))
            sir = p_target / max(p_tot - p_target, 1e-12)

            tok, m = build_tokens(X, fs, ref_trials=np.where(~is_inj)[0])
            C, F, Tf = m["C"], m["F"], m["Tf"]
            fb, mu, sd = m["fb"], m["mu"], m["sd"]
            if a.repr == "z":
                T_ = ((tok.reshape(-1, F, C) - mu) / sd).reshape(-1, C).astype(np.float32)
            else:
                T_ = tok
            trial_of, fbin_of = m["trial_of"], m["fbin_of"]
            tr_trial = np.where(~is_inj)[0]
            tr_mask = np.isin(trial_of, tr_trial)
            tr_idx = np.where(tr_mask)[0]
            if len(tr_idx) > a.max_tokens:
                tr_idx = rng.choice(tr_idx, a.max_tokens, replace=False)
            sae = TS.TopKSAE(d_in=C, d_dict=a.sae_dict, k=a.sae_k, seed=a.seed)
            sae.fit(T_[tr_idx], steps=a.sae_steps)
            idx, val = sae.encode_sparse(T_)
            act = TS.atom_trial_activation(idx, val, trial_of, n, a.sae_dict)
            dirs = sae.atom_dirs()                       # (D_dict, C)，行 = 原子

            # ---- 类特异度
            w_inj = act[is_inj].mean(0) - act[~is_inj].mean(0)
            top5 = np.argsort(-w_inj)[:5]
            argmax_ch = np.abs(dirs).argmax(1)           # (D_dict,)
            hit_ch = [ch_names[argmax_ch[i]] in ch_set for i in top5]

            # ---- 真值 z 轮廓（f0 最近的 bin 上，注入源的 z 幅度 ∝ w/sd）
            fi = int(np.argmin(np.abs(fb - f0)))
            g = w.astype(np.float64) / sd[fi][ch_idx]
            gvec = np.zeros(C)
            gvec[ch_idx] = g
            gvec = gvec - gvec.mean()
            gvec /= np.linalg.norm(gvec) + 1e-12

            def _cos(d):
                v = np.asarray(d, float)
                v = v - v.mean()
                return float((v / (np.linalg.norm(v) + 1e-12)) @ gvec)

            # 字典覆盖率：不问「排序对不对」，只问「字典里有没有真值方向」。
            # 若 max|cos| 本身就很低，任何排序器都不可能选对 —— 这是结构性判据。
            Dm = dirs.astype(np.float64)
            Dm = Dm - Dm.mean(1, keepdims=True)
            Dm /= (np.linalg.norm(Dm, axis=1, keepdims=True) + 1e-12)
            cos_all = np.abs(Dm @ gvec)
            cov_max = float(cos_all.max())
            cov_top5 = float(np.sort(cos_all)[-5:].mean())

            cos5 = np.array([_cos(dirs[i]) for i in top5])
            cos_abs5 = np.abs(cos5)
            cos_best = float(np.max(cos_abs5))

            # ---- 频率读数：注入 token 上的激活加权频点
            m_inj = is_inj[trial_of]
            wsum = np.zeros(a.sae_dict)
            fsum = np.zeros(a.sae_dict)
            np.add.at(wsum, idx[m_inj].ravel(), val[m_inj].ravel())
            np.add.at(fsum, idx[m_inj].ravel(),
                      (val[m_inj] * fb[fbin_of[m_inj]][:, None]).ravel())
            fpref = np.where(wsum > 1e-9, fsum / np.maximum(wsum, 1e-9), np.nan)
            f_err = float(np.nanmedian(np.abs(fpref[top5] - f0)))

            # ---- 潜伏期读数
            tf_inj = m["tf_of"][m_inj]
            tsum = np.zeros(a.sae_dict)
            np.add.at(tsum, idx[m_inj].ravel(),
                      (val[m_inj] * m["tvec"][tf_inj][:, None]).ravel())
            tpref = np.where(wsum > 1e-9, tsum / np.maximum(wsum, 1e-9), np.nan)
            t_err = float(np.nanmedian(np.abs(tpref[top5] - 0.5 * sum(INJ_WIN_S))))

            # ---- 零对照：随机 5 原子
            null_cos, null_hit = [], []
            for _ in range(N_NULL):
                pick = rng.choice(a.sae_dict, 5, replace=False)
                null_cos.append(max(abs(_cos(dirs[i])) for i in pick))
                null_hit.append(sum(ch_names[argmax_ch[i]] in ch_set for i in pick))
            null_p95 = float(np.percentile(null_cos, 95))
            null_hit_p95 = float(np.percentile(null_hit, 95))

            # ---- oracle：不经 SAE，直接在 f0 频点上算 注入−干净 的幅度差
            A4 = tok.reshape(n, Tf, F, C)
            amp_f0 = A4[:, :, fi, :].mean(1)              # (n, C) 该试次 f0 上的平均幅度
            p_orc = amp_f0[is_inj].mean(0) - amp_f0[~is_inj].mean(0)
            orc_ch = ch_names[int(np.argmax(np.abs(p_orc)))]
            pv = p_orc - p_orc.mean()
            orc_cos = float((pv / (np.linalg.norm(pv) + 1e-12)) @ gvec)

            act_inj = act[is_inj].mean(0)
            act_cln = act[~is_inj].mean(0)
            ratio = act_inj[top5] / np.maximum(act_cln[top5], 1e-6)
            n_hit = int(sum(hit_ch))
            frac_cos = float(np.mean(cos_abs5 > null_p95))
            passed = bool(n_hit >= HIT_MIN and frac_cos >= PASS_MIN_FRAC and cos_best > null_p95)
            print(f"  SNR {snr_db:+5.1f} dB (实得 SIR {10*np.log10(max(sir,1e-12)):+5.1f} dB, "
                  f"amp {amp:.3g}) | 命中 {n_hit}/5 | cos 最佳 {cos_best:+.2f} "
                  f"(零对照 p95 {null_p95:.2f}, 过线 {frac_cos:.0%}) | 覆盖率 {cov_max:.2f} | "
                  f"f 误差 {f_err:.2f} Hz "
                  f"| t 误差 {t_err:.2f} s | oracle {orc_ch} cos {orc_cos:+.2f} | "
                  f"{'PASS' if passed else '-'}  [{time.time()-t_start:.0f}s]", flush=True)

            rec["snr"].append(dict(
                snr_db=float(snr_db), sir_db=float(10 * np.log10(max(sir, 1e-12))),
                amp=amp, n_hit=n_hit, hit_ch=[bool(v) for v in hit_ch],
                top5=[int(i) for i in top5], top5_ch=[ch_names[argmax_ch[i]] for i in top5],
                cos_best=cos_best, cos5=[float(v) for v in cos5],
                dict_cov_max=cov_max, dict_cov_top5=cov_top5,
                cos_abs5=[float(v) for v in cos_abs5],
                frac_cos_pass=frac_cos, null_cos_p95=null_p95,
                null_hit_p95=null_hit_p95, freq_err=f_err, latency_err=t_err,
                oracle_ch=orc_ch, oracle_cos=orc_cos, oracle_hit=bool(orc_ch in ch_set),
                passed=passed, atom_pr_median=float(np.median([pr(dirs[i])
                                                               for i in range(a.sae_dict)])),
                top5_contrast=[float(v) for v in w_inj[top5]],
                top5_act_inj=[float(v) for v in act_inj[top5]],
                top5_act_clean=[float(v) for v in act_cln[top5]],
                top5_ratio=[float(v) for v in ratio],
            ))
        out["configs"].append(rec)

    # ---------------------------------------------------------------- 汇总
    tag = a.tag or a.repr
    (ROOT / "probe").mkdir(exist_ok=True)
    jp = ROOT / "probe" / f"c_synth_{tag}.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        f"# T0-C：合成注入的原子层成立边界（表示 = `{a.repr}`）",
        "",
        "往**真实** mmidb 试次里注入已知源（空间 3 导、频带 f0±1 Hz、时间 epoch 内 "
        f"{INJ_WIN_S[0]}–{INJ_WIN_S[1]} s），同一批数据只改幅度扫 SNR。",
        "",
        f"- 试次 {out['n_trial']}（注入 {out['n_inj']} / 干净 {out['n_trial']-out['n_inj']}），"
        f"导联 {len(ch_names)}，fs {fs:g} Hz",
        f"- token：{TOKEN_WIN_S:g} s 窗 / {TOKEN_HOP_S:g} s 步，{TOKEN_BAND[0]:g}–"
        f"{TOKEN_BAND[1]:g} Hz；字典 {a.sae_dict} × k={a.sae_k}，{a.sae_steps} 步",
        f"- 判据：top-5（类特异度排序）里 argmax 落入注入集 ≥{HIT_MIN}，"
        f"且 ≥{PASS_MIN_FRAC:.0%} 的原子余弦超过零对照 p95",
        "",
        "| 配置 | SNR (dB) | 实得 SIR (dB) | 命中 | 最佳余弦 | 覆盖率 max | 零对照 p95 | "
        "频率误差 | oracle 峰导 | oracle 余弦 | 判定 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    n_pass_by_snr = {s: 0 for s in snrs}
    for rec in out["configs"]:
        for r in rec["snr"]:
            n_pass_by_snr[r["snr_db"]] += int(r["passed"])
            lines.append(
                f"| {rec['name']} | {r['snr_db']:+.0f} | {r['sir_db']:+.1f} | {r['n_hit']}/5 | "
                f"{r['cos_best']:+.2f} | {r['dict_cov_max']:.2f} | {r['null_cos_p95']:.2f} | "
                f"{r['freq_err']:.2f} Hz | "
                f"{r['oracle_ch']} | {r['oracle_cos']:+.2f} | "
                f"{'**PASS**' if r['passed'] else '—'} |")
    lines += ["", "## 通过数 vs SNR", "", "| SNR (dB) | 通过配置数 |", "|---|---|"]
    for s in snrs:
        lines.append(f"| {s:+.0f} | {n_pass_by_snr[s]}/{len(out['configs'])} |")
    lines += ["", f"原始数据 `probe/c_synth_{tag}.json`。"]
    rp = ROOT / "reports" / f"T0_C_synth{('_' + tag) if tag else ''}.md"
    rp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写出 {jp.name} / {rp.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
