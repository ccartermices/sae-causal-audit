"""d20 —— T0 仪器校准（S0 闸门的唯一判据来源），频谱 token 版。

为什么不沿用 v0（CNN）
----------------------
v0（`d20_t0_ssvep_cnn_v0.py`）用 ShallowConvNet + 40 类，结果写在
`reports/T0_S0_cnn_v0.md`：训练 100%、测试 2.5%（机会 2.5%）。这不是数据的问题
——同一个数据上固定频谱检测器（3 个枕导取 argmax）能到 58.8%，说明是分类器
容量/样本量失衡（200 试次/类 × 40 类）。T0 的任务是「校准仪器」，不是「打赢
SSVEP 竞赛」，所以仪器改成可解释性天然需要的形式：

  token = 「某一频点、某一时刻的 64 导幅度空间模式」

这样原子天生带三个属性，正好对上语义探针的三项：
  * 地形   = 原子方向的 64 维空间分布（取 |d_i| 的 argmax）
  * 频率   = 原子在哪些频点的 token 上发放（激活加权的频点均值）
  * 潜伏期 = 原子在哪些时刻的 token 上发放

为什么是 8 类而不是 40 类
-------------------------
40 个目标在 8–15.8 Hz 上以 0.2 Hz 布点（JFPM 分组），相邻间隔 0.2 Hz。
2 s 窗的频率分辨率是 0.5 Hz > 0.2 Hz，40 类在原理上不可分；6 s 窗能到 0.167 Hz
但只剩 1 帧，token 就没有时间维、失去「潜伏期」属性。
因此按基频聚成 8 类（`class = target % 8`，每类 5 个相位偏移，300 试次/类），
用 2 s 窗。**这是对判据的收窄，已记入 DECISIONS D20**，40 类准确率仍作为
次要指标一并报告（`decoder_acc_40`），不参与 S0 判定。

判据（`03_判据与闸门` v1.0-final §1，S0）
----------------------------------------
对每个基频 f：
  1. 该频率**留出试次**上的平均原子激活取 top-5；
  2. 地形：这 5 个原子的 |d_i| argmax 落在 ROI_OCCIPITAL 的个数 ≥ 3；
  3. 频率：这 5 个原子的偏好频率与 f 的偏差（中位数）≤ 0.4 Hz。
两项都过的基频数 ≥ 6/8 记为 S0 PASS（v1.0-final 的 40 类原文为 7/40 时按比例
放大到 8 类，此处从严取 6/8）。

为什么取 |d_i| 而不是有符号 argmax
----------------------------------
SSVEP 的试次间相位不锁定，空间模式的**符号不可辨识**（同一原子在不同试次
可以是反相的）。因此地形判据只能建在幅度上，这是先验的、不是事后挑的。

产出
----
probe/t0_s0.json         机器读（registry 用）
reports/T0_S0.md         人读
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "sae"))
import sig_layer as SL  # noqa: E402
import token_sae as TS  # noqa: E402

EPO = ROOT / "data" / "interim" / "epochs"
OUT = ROOT / "probe"
REP = ROOT / "reports"

# ---------------------------------------------------------------- 冻结的仪器参数
# 全部先验确定，唯一来源是 `sig_layer.py`（改那里才会让 registry 的 C5 BLOCK）。
STFT_WIN_S = SL.SSVEP_STFT_WIN_S          # 2.0 s → 0.5 Hz 分辨率
STFT_HOP_S = SL.SSVEP_STFT_HOP_S          # 0.5 s 跳（75% 重叠）
TOKEN_BAND = SL.SSVEP_TOKEN_BAND          # 6.5–17.3 Hz = 刺激频段 ±1.5 Hz
TOKEN_BAND_HARMONIC = SL.SSVEP_TOKEN_BAND_HARMONIC   # 次要：含二次谐波
T0_CLASSES = SL.T0_CLASSES                # class = target % 8
S0_PASS_MIN = SL.S0_PASS_MIN              # 8 类里至少 6 类两项都过


def load_ssvep(n_subjects: int = 10):
    files = sorted((EPO / "ssvep").glob("*.npz"),
                   key=lambda p: int(re.search(r"S(\d+)\.npz", p.name).group(1)))
    Xs, ys, bs, ss = [], [], [], []
    freq = None
    for i, p in enumerate(files[:n_subjects]):
        z = np.load(p)
        Xs.append(z["X"])
        ys.append(z["y"])
        bs.append(z["block"])
        ss.append(np.full(len(z["y"]), i, dtype=np.int16))
        freq = z["freq"]
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(bs),
            np.concatenate(ss), np.asarray(freq).ravel(), float(z["sfreq"]))


def ssvep_ch_names() -> list[str]:
    """64 导名，唯一来源是 `sig_layer.SSVEP_CH_NAMES`。

    早期版本在这里直接解析 `64-channels.loc`，把 `PO3` 读成 `Po3`，
    结果 S0 的地形判据只能匹配 O1/Oz/O2 三个导（D20-8）。
    """
    assert len(SL.SSVEP_CH_NAMES) == SL.SSVEP_N_CHANS == SL.SPEC["ssvep"]["n_chans"]
    return list(SL.SSVEP_CH_NAMES)


def spectral_tokens(X: np.ndarray, fs: float, band: tuple[float, float],
                    nperseg_s: float = STFT_WIN_S, hop_s: float = STFT_HOP_S,
                    need_amp: bool = True):
    """(n, C, T) → token 张量。

    返回
    ----
    tok   : (n*Tf*F, C) float32，原始幅度（先不做归一，交给 zscore_bins）
    amp   : (n, C, F, Tf) 未归一幅度（解码器用）；need_amp=False 时为 None
    fbin  : (F,) 频点中心
    n_tok_per_trial : int
    tvec  : (Tf,) 帧中心时刻（相对**刺激起点**，s）
    """
    from scipy.signal import stft

    nperseg = int(round(nperseg_s * fs))
    hop = int(round(hop_s * fs))
    f, t, Z = stft(X, fs=fs, nperseg=nperseg, noverlap=nperseg - hop, axis=-1)
    sel = (f >= band[0]) & (f <= band[1])
    fbin = f[sel]
    A = np.abs(Z[:, :, sel, :]).astype(np.float32)     # (n, C, F, Tf)
    n, C, F, Tf = A.shape
    del Z
    tok = A.transpose(0, 3, 2, 1).reshape(n * Tf * F, C)  # 索引 = (trial*Tf+tf)*F+fi
    amp = A if need_amp else None
    del A
    # 帧中心：scipy 的 t 相对**试次起点**（实测：方波 2.4–3.6 s 落在 t=2.5/3.5 的帧上），
    # 换算到相对刺激起点要加 EPOCHS 的起点（-0.5 s）。
    tvec = t + SL.EPOCHS["ssvep"][0]
    return tok, amp, fbin, Tf * F, tvec


def zscore_bins(tok: np.ndarray, tr_token: np.ndarray, n_freq: int) -> np.ndarray:
    """逐 (频点, 导) z 化：每个频点等权，原子学到的是**形状**而不是 1/f 功率。

    为什么必须做（T0 第一次运行的实测，`reports/T0_diagnosis.md` §2）
    ---------------------------------------------------------------
    8–17 Hz 上幅度的 1/f 落差有好几倍，直接喂幅度时低频频点的 token 幅度大、
    激活也大，条件频率读数被拽到频带中段（≈11.5 Hz），8 个类的频率偏差呈现
    「离 11.5 越远越大」的单调序列（3.3/2.2/1.2/0.4/0.7/1.5/2.5/3.5 Hz），
    这不是原子偏好频率，而是能量权重。
    归一化统计量只用**训练试次**的 token 算，不用留出集。

    为什么不是逐 token L2（第一版的做法）
    -------------------------------------
    L2 把「这个频点是否有信号」这件事整体抹掉：unit 范数下噪声 token 与信号
    token 一样长，原子激活退化成纯形状匹配，而形状在单帧单频点上由宽带噪声主导。
    实测 L2 版原子的通道 PR 中位数 28（满分 64），occ 命中 0–2/5。
    """
    n = len(tok) // n_freq
    X2 = tok.reshape(n, n_freq, -1)                    # (n*Tf, F, C)
    mu = X2[tr_token].mean(0)
    sd = X2[tr_token].std(0) + 1e-6
    return ((X2 - mu) / sd).reshape(len(tok), -1).astype(np.float32)


def decode(feat: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
           C: float = 0.05):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(feat[tr])
    clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(feat[tr]), y[tr])
    pred = clf.predict(sc.transform(feat[te]))
    return pred, clf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=10)
    ap.add_argument("--sae-steps", type=int, default=6000)
    # 与主链路同配置（D=4096, k=32）：S0 要验的就是 T1/T2 上用的那个字典本身，
    # 换小字典会让「仪器校准」和「正式实验」不是同一台仪器。
    ap.add_argument("--sae-dict", type=int, default=SL.SAE_DICT)
    ap.add_argument("--sae-k", type=int, default=SL.SAE_K)
    ap.add_argument("--max-tokens", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)

    X, y, blk, sub, freq, fs = load_ssvep(a.n_subjects)
    ch = ssvep_ch_names()
    cls8 = (y % T0_CLASSES).astype(np.int64)
    f_ref = np.array([freq[np.arange(len(freq)) % T0_CLASSES == c].mean()
                      for c in range(T0_CLASSES)], dtype=np.float64)
    tr, te = blk < 5, blk == 5
    print(f"SSVEP: {len(X)} 试次 | 训练 {tr.sum()} 测试 {te.sum()} | "
          f"8 类（= target%8），类参考频率 {np.round(f_ref, 2)}")
    print(f"类别样本数 {np.bincount(cls8)}")

    # ---------------------------------------------------------------- token
    tok, amp, fbin, per_trial, tvec = spectral_tokens(X, fs, TOKEN_BAND)
    F, Tf = len(fbin), len(tvec)
    tr_frame = np.repeat(tr, Tf)                 # zscore_bins 按 (试次, 帧) 取统计量
    tok = zscore_bins(tok, tr_frame, F)
    print(f"token: 每试次 {per_trial}（F={F} × Tf={Tf}），单 token 维 {tok.shape[1]}")
    print(f"频点 {fbin[0]:.1f}–{fbin[-1]:.1f} Hz 步长 {fbin[1] - fbin[0]:.2f}；"
          f"帧中心 {tvec[0]:.2f}–{tvec[-1]:.2f} s；逐 (频点,导) z 化（统计量只用训练集）")

    trial_of = np.repeat(np.arange(len(X)), per_trial)
    fbin_of = np.tile(np.tile(np.arange(F), Tf), len(X))

    # ---------------------------------------------------------------- 解码器
    feat_amp = amp.mean(-1).reshape(len(X), -1)          # 时间平均幅度谱
    pred8, _ = decode(feat_amp, cls8, tr, te)
    acc8 = float((pred8 == cls8[te]).mean())
    per_sub = [float((pred8[sub[te] == k] == cls8[te][sub[te] == k]).mean())
               for k in range(a.n_subjects)]
    pred40, _ = decode(feat_amp, y.astype(np.int64), tr, te)
    acc40 = float((pred40 == y[te]).mean())
    print(f"解码器：8 类 {acc8:.3f}（中位 {np.median(per_sub):.3f}，机会 0.125）| "
          f"40 类 {acc40:.3f}（机会 0.025）")

    # ---------------------------------------------------------------- SAE
    tok_tr = tok[tr[trial_of]]
    if len(tok_tr) > a.max_tokens:
        keep = rng.choice(len(tok_tr), a.max_tokens, replace=False)
        tok_tr = tok_tr[keep]
    print(f"SAE 训练 token 数 {len(tok_tr)} | D={a.sae_dict} k={a.sae_k}")
    sae = TS.TopKSAE(d_in=tok.shape[1], d_dict=a.sae_dict, k=a.sae_k, seed=a.seed)
    sae.fit(tok_tr, steps=a.sae_steps)
    r2 = TS.r2(tok_tr[:50000], sae.reconstruct(tok_tr[:50000]))
    print(f"SAE 重构 R²（训练 token 前 5 万）{r2:.4f}")

    idx, val = sae.encode_sparse(tok)
    dirs = sae.atom_dirs()                                # (D_dict, C)
    act = TS.atom_trial_activation(idx, val, trial_of, len(X), a.sae_dict)
    pred_sae, _ = decode(act, cls8, tr, te)
    acc_sae = float((pred_sae == cls8[te]).mean())
    print(f"SAE 特征读出：8 类 {acc_sae:.3f}")

    # ---------------------------------------------------------------- S0 判据
    # 地形：|d_i| 的 argmax 导联
    occ = set(SL.ROI_OCCIPITAL)
    topo_arg = np.array(ch)[np.abs(dirs).argmax(1)]
    is_occ = np.array([t in occ for t in topo_arg])
    # 仪器质量诊断：原子的空间集中度（通道 PR）与字典的枕区占比。
    # 可解释的空间滤波器 PR 应该小（个位数）；PR≈C 说明原子是弥散的基向量。
    def _pr(v):
        v = np.abs(v) ** 2
        return float(v.sum() ** 2 / max((v ** 2).sum(), 1e-12))
    pr_atom = np.array([_pr(d) for d in dirs])
    occ_rate = float(is_occ.mean())

    # 频率：激活加权的频点均值。**按条件读**（判据原文是「当次刺激频率」，
    # 即在该类试次上看原子偏好何处），另算一个全局版作诊断：
    # 全局版回答「原子本身是否频率特异」——若原子是纯空间滤波器，全局版会塌到
    # 频带中心（≈11.5 Hz），这正是 T1/T2 上「频带属性只能按条件读」的证据。
    tok_te = te[trial_of]
    idx_te, val_te = idx[tok_te], val[tok_te]
    fbin_te = fbin_of[tok_te]
    cls_te = cls8[trial_of[tok_te]]
    D = a.sae_dict

    def fpref_of(mask_tok):
        w = np.zeros(D)
        s = np.zeros(D)
        np.add.at(w, idx_te[mask_tok].ravel(), val_te[mask_tok].ravel())
        np.add.at(s, idx_te[mask_tok].ravel(),
                  (val_te[mask_tok] * fbin_te[mask_tok, None]).ravel())
        fp = np.full(D, np.nan)
        nz = w > 1e-9
        fp[nz] = s[nz] / w[nz]
        return fp

    fpref_glob = fpref_of(np.ones(len(idx_te), dtype=bool))
    fpref_c = {c: fpref_of(cls_te == c) for c in range(T0_CLASSES)}

    rows = []
    for c in range(T0_CLASSES):
        m = te & (cls8 == c)
        mean_act = act[m].mean(0)
        top5 = np.argsort(-mean_act)[:5]
        hits = int(is_occ[top5].sum())
        fp = fpref_c[c][top5]
        err = float(np.nanmedian(np.abs(fp - f_ref[c])))
        fpg = fpref_glob[top5]
        err_g = float(np.nanmedian(np.abs(fpg - f_ref[c])))
        rows.append(dict(cls=c, f_ref=float(f_ref[c]), hits=hits,
                         f_pref=[None if np.isnan(v) else float(v) for v in fp],
                         freq_err=err,
                         f_pref_global=[None if np.isnan(v) else float(v) for v in fpg],
                         freq_err_global=err_g,
                         top5_ch=[str(topo_arg[i]) for i in top5],
                         pass_occipital=bool(hits >= SL.S0_MIN_OCCIPITAL_HITS),
                         pass_freq=bool(err <= SL.SSVEP_FREQ_TOL),
                         passed=bool(hits >= SL.S0_MIN_OCCIPITAL_HITS
                                     and err <= SL.SSVEP_FREQ_TOL)))
    n_pass = sum(r["passed"] for r in rows)
    med_hits = float(np.median([r["hits"] for r in rows]))
    med_err = float(np.median([r["freq_err"] for r in rows]))
    med_err_g = float(np.median([r["freq_err_global"] for r in rows]))
    s0_pass = bool(n_pass >= S0_PASS_MIN)
    print(f"S0：枕区命中中位数 {med_hits:.1f}/5 | 频率偏差中位数 {med_err:.2f} Hz"
          f"（全局读数 {med_err_g:.2f} Hz，仅诊断）| "
          f"两项都过 {n_pass}/{T0_CLASSES} → {'PASS' if s0_pass else 'FAIL'}")

    # ---------------------------------------------------------------- 次要：含谐波
    tok_h, _, fbin_h, pt_h, _ = spectral_tokens(X, fs, TOKEN_BAND_HARMONIC,
                                                need_amp=False)
    tok_h = zscore_bins(tok_h, tr_frame, len(fbin_h))
    trial_of_h = np.repeat(np.arange(len(X)), pt_h)
    fbin_of_h = np.tile(np.tile(np.arange(len(fbin_h)), len(tvec)), len(X))
    tok_h_tr = tok_h[tr[trial_of_h]]
    if len(tok_h_tr) > a.max_tokens:
        keep = rng.choice(len(tok_h_tr), a.max_tokens, replace=False)
        tok_h_tr = tok_h_tr[keep]
    sae_h = TS.TopKSAE(d_in=tok_h.shape[1], d_dict=a.sae_dict, k=a.sae_k, seed=a.seed)
    sae_h.fit(tok_h_tr, steps=a.sae_steps)
    idx_h, val_h = sae_h.encode_sparse(tok_h)
    act_h = TS.atom_trial_activation(idx_h, val_h, trial_of_h, len(X), a.sae_dict)
    dirs_h = sae_h.atom_dirs()
    is_occ_h = np.array([t in occ for t in np.array(ch)[np.abs(dirs_h).argmax(1)]])
    w_h = np.zeros(a.sae_dict); f_h = np.zeros(a.sae_dict)
    m_te = te[trial_of_h]
    np.add.at(w_h, idx_h[m_te].ravel(), val_h[m_te].ravel())
    np.add.at(f_h, idx_h[m_te].ravel(),
              (val_h[m_te] * fbin_of_h[m_te, None]).ravel())
    fpref_h = np.where(w_h > 1e-9, f_h / np.maximum(w_h, 1e-9), np.nan)
    cls_h = cls8[trial_of_h[m_te]]
    idx_h_te, val_h_te = idx_h[m_te], val_h[m_te]
    fbin_h_te = fbin_of_h[m_te]

    def fpref_of_h(mask_tok):
        w = np.zeros(a.sae_dict); s = np.zeros(a.sae_dict)
        np.add.at(w, idx_h_te[mask_tok].ravel(), val_h_te[mask_tok].ravel())
        np.add.at(s, idx_h_te[mask_tok].ravel(),
                  (val_h_te[mask_tok] * fbin_h_te[mask_tok, None]).ravel())
        fp = np.full(a.sae_dict, np.nan)
        nz = w > 1e-9
        fp[nz] = s[nz] / w[nz]
        return fp

    fpref_h_c = {c: fpref_of_h(cls_h == c) for c in range(T0_CLASSES)}
    rows_h = []
    for c in range(T0_CLASSES):
        m = te & (cls8 == c)
        top5 = np.argsort(-act_h[m].mean(0))[:5]
        hits = int(is_occ_h[top5].sum())
        err = float(np.nanmedian(np.abs(fpref_h_c[c][top5] - f_ref[c])))
        rows_h.append(dict(cls=c, hits=hits, freq_err=err,
                           passed=bool(hits >= SL.S0_MIN_OCCIPITAL_HITS
                                       and err <= SL.SSVEP_FREQ_TOL)))
    n_pass_h = sum(r["passed"] for r in rows_h)
    print(f"次要（含二次谐波 {TOKEN_BAND_HARMONIC[0]:.1f}–{TOKEN_BAND_HARMONIC[1]:.1f} Hz）："
          f"两项都过 {n_pass_h}/{T0_CLASSES} | 枕区命中中位数 "
          f"{np.median([r['hits'] for r in rows_h]):.1f}/5")
    del tok_h, idx_h, val_h, act_h

    # ---------------------------------------------------------------- 落盘
    OUT.mkdir(exist_ok=True)
    res = dict(
        n_subjects=a.n_subjects, n_trials=int(len(X)),
        n_train=int(tr.sum()), n_test=int(te.sum()),
        n_classes=T0_CLASSES, class_rule="target % 8",
        f_ref=[float(v) for v in f_ref],
        decoder_acc=acc8, decoder_acc_median=float(np.median(per_sub)),
        decoder_acc_per_subject=per_sub, decoder_acc_40=acc40,
        sae_readout_acc=acc_sae,
        stft_win_s=STFT_WIN_S, stft_hop_s=STFT_HOP_S,
        token_band=[float(v) for v in TOKEN_BAND],
        token_dim=int(tok.shape[1]), tokens_per_trial=int(per_trial),
        n_freq_bins=int(F), n_frames=int(Tf),
        n_tokens_sae=int(len(tok_tr)),
        sae_dict=a.sae_dict, sae_k=a.sae_k, sae_seed=a.seed, sae_r2=r2,
        s0_median_hits=med_hits, s0_median_freq_err=med_err,
        atom_pr_channel_median=float(np.median(pr_atom)),
        atom_pr_channel_p10=float(np.percentile(pr_atom, 10)),
        atom_pr_channel_p90=float(np.percentile(pr_atom, 90)),
        dict_occipital_rate=occ_rate, chance_occipital_rate=len(occ) / len(ch),
        s0_median_freq_err_global=med_err_g,
        s0_n_pass=n_pass, s0_pass_min=S0_PASS_MIN,
        s0_targets_passing_occipital=int(sum(r["pass_occipital"] for r in rows)),
        s0_targets_passing_freq=int(sum(r["pass_freq"] for r in rows)),
        s0_pass=s0_pass, per_class=rows,
        harmonic=dict(band=[float(v) for v in TOKEN_BAND_HARMONIC],
                      n_pass=n_pass_h, per_class=rows_h),
        ch_names=ch, seconds=round(time.time() - t0, 1),
    )
    p = OUT / f"t0_s0{a.tag}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {p.relative_to(ROOT)}  ({res['seconds']} s)")

    # ---------------------------------------------------------------- 人读报告
    lines = [
        "# T0 · S0 仪器校准（SSVEP，频谱 token）",
        "",
        "> 由 `scripts/d20_t0_ssvep.py` 生成，勿手改。判据取 `03_判据与闸门` v1.0-final §1，",
        "> 按 DECISIONS D20 收窄为 8 类基频（40 类的频率间隔 0.2 Hz 小于 2 s 窗的 0.5 Hz 分辨率）。",
        "",
        f"- 数据：{a.n_subjects} 名被试，{tr.sum()} 训练 / {te.sum()} 测试（留出 block 5）",
        f"- token：每试次 {per_trial} 个（{F} 频点 × {Tf} 帧），单 token {tok.shape[1]} 维",
        f"- 字典：D={a.sae_dict}，k={a.sae_k}，重构 R² {r2:.3f}",
        "",
        "| 项 | 实测 | 判据 | 结论 |",
        "|---|---|---|---|",
        f"| 解码器 8 类准确率 | {acc8:.3f}（中位 {np.median(per_sub):.3f}，机会 0.125） | — | — |",
        f"| 解码器 40 类准确率（次要） | {acc40:.3f}（机会 0.025） | — | — |",
        f"| SAE 特征读出 8 类 | {acc_sae:.3f} | — | — |",
        f"| top-5 枕区命中中位数 | {med_hits:.1f} | ≥{SL.S0_MIN_OCCIPITAL_HITS} | "
        f"{'通过' if med_hits >= SL.S0_MIN_OCCIPITAL_HITS else '不通过'} |",
        f"| 谱峰偏差中位数 | {med_err:.2f} Hz | ≤{SL.SSVEP_FREQ_TOL} Hz | "
        f"{'通过' if med_err <= SL.SSVEP_FREQ_TOL else '不通过'} |",
        f"| 同上的全局读数（诊断，不判） | {med_err_g:.2f} Hz | — | — |",
        f"| 两项都过 | {n_pass}/{T0_CLASSES} | ≥{S0_PASS_MIN} | — |",
        f"| *原子通道 PR 中位数（诊断）* | {np.median(pr_atom):.1f}（p10 {np.percentile(pr_atom, 10):.1f}，满分 {len(ch)}） | — | — |",
        f"| *字典的枕区占比（诊断）* | {occ_rate:.3f}（随机 {len(occ) / len(ch):.3f}） | — | — |",
        "",
        f"**S0 判定：{'通过' if s0_pass else '不通过'}**",
        "",
        f"次要分析（token 频带扩到含二次谐波 {TOKEN_BAND_HARMONIC[0]:.1f}–"
        f"{TOKEN_BAND_HARMONIC[1]:.1f} Hz）：两项都过 {n_pass_h}/{T0_CLASSES}。",
        "谐波原子若挤进 top-5，会让「偏好频率 = 基频」这一条判据变严，故单列。",
        "",
        "## 逐基频",
        "",
        "| 类 | 参考频率 | 枕区命中 | 偏好转数（top-5，按条件） | 偏差 | 全局偏差 | top-5 导联 | 判定 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fps = ", ".join("—" if v is None else f"{v:.1f}" for v in r["f_pref"])
        lines.append(
            f"| {r['cls']} | {r['f_ref']:.1f} | {r['hits']} | {fps} | "
            f"{r['freq_err']:.2f} | {r['freq_err_global']:.2f} | "
            f"{', '.join(r['top5_ch'])} | "
            f"{'过' if r['passed'] else '不过'} |")
    lines += [
        "",
        "## 读数说明",
        "",
        "「按条件」= 只在该类自己的留出 token 上算原子的激活加权频点均值，对应判据原文的",
        "「当次刺激频率」。「全局」= 在全部留出 token 上算，回答的是**原子本身是否频率特异**：",
        "若原子只是空间滤波器，全局读数会塌到频带中心（≈11.5 Hz）。两者差距大并不是矛盾，",
        "而是说明「频带属性属于原子的发放分布，不属于原子方向」——这一条会直接决定 T1/T2 的",
        "语义探针怎么读（见 DECISIONS D20-7）。",
    ]
    (REP / "T0_S0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"→ reports/T0_S0.md")
    return 0 if s0_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
