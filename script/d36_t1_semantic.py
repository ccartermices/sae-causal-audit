"""d36 — 语义探针：SAE 的「自称」与「实测」是否对得上（表 1 的数据来源）。

`03_判据与闸门` §4 把语义探针写成三项（地形 / 频带 / 潜伏期）。在**当前架构**
（`TOKEN_LAYER="pre_pool"`，token 是 $F=40$ 维卷积特征图上某一时刻的切片）上，
$d_i\in\mathbb R^{40}$ **不含时间轴**，因此「潜伏期」没有权重侧的对应物——
$d_i$ 不是 $(C,T)$ 图，无法从中读出该特征在第几秒最强。故本脚本只落实**两项**
（地形、频带），语义得分 $0\!-\!2$；潜伏期降为描述量（只报实测分布）。
这条口径偏离记在 `DECISIONS.md` D36。

「自称」怎么从权重算出来
------------------------
`ShallowFBCSPNet` 的 `conv_time_spat` 是两层串联：
$$\tilde h_j(t)=\sum_{f}W_s[j,f,c]\sum_\tau k_f(\tau)x_c(t-\tau)
  \;\Longrightarrow\;
  \tilde h_j(t)=\sum_c\sum_\tau K_j(c,\tau)x_c(t-\tau),\quad
  K_j(c,\tau)=\sum_f W_s[j,f,c]\,k_f(\tau)$$
$K_j$ 是第 $j$ 个特征通道的等效时空核 $(C,25)$。`bnorm` 在 eval 下是逐通道对角仿射，
对方向只贡献 $\gamma_j$ 一个缩放。于是原子 $i$（方向 $d_i\in\mathbb R^{40}$）自称的
时空模式是 $A_i(c,\tau)=\sum_j d_{ij}\gamma_j K_j(c,\tau)$。

* 地形：$p_c=\sum_\tau|A_i(c,\tau)|$ 的 argmax 落在哪个 ROI；
* 频带：$|\mathrm{FFT}(A_i(c^\ast,\cdot))|$ 在 **1–45 Hz**（分析带）内的峰，
  $c^\ast=\arg\max_c p_c$。不限带会把峰推到 100 Hz 以上（25 抽头的核在 250 Hz 下
  高频分量很多，实测核峰从 1.5 铺到 44.9 Hz），那是采样率伪影不是频带。

「实测」怎么算
--------------
$$g_i=\mathbb E_{n,t}\frac{\partial a_i(n,t)}{\partial x}=
  \frac{1}{NT}\frac{\partial}{\partial x}\sum_{n,t}\mathrm{ReLU}\big(w_i\!\cdot\!h(n,t)+b_i\big)$$
因此**一次反传**就得到它（把 $a_i$ 全求和），不需要逐 token 反传。

产物：`probe/t1_semantic_{tag}.json`、`reports/T1_semantic_{tag}.md`
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
SPEC_BAND = (1.0, 45.0)     # 分析带：与 SL.BANDPASS 一致
FFT_N = 512                 # 核只有 25 抽头，补零到 512 才能定位峰


# ------------------------------------------------------------------ 权重侧
def equiv_kernels(model) -> np.ndarray:
    """$K_j(c,\\tau)$ 与 $\\gamma_j$：(40,C,25)、(40,)。"""
    ct = model.conv_time_spat.conv_time.weight.detach().cpu().numpy()[:, 0, :, 0]
    ws = model.conv_time_spat.conv_spat.weight.detach().cpu().numpy()[:, :, 0, :]
    K = np.einsum("jfc,ft->jct", ws, ct)
    gam = model.bnorm.weight.detach().cpu().numpy()
    return K.astype(np.float64), gam.astype(np.float64)


def peak_band(sig: np.ndarray, sfreq: float) -> tuple[float, str | None]:
    """1–45 Hz 内 |FFT| 的峰频与所属频带。"""
    P = np.abs(np.fft.rfft(sig, n=FFT_N))
    f = np.fft.rfftfreq(FFT_N, d=1.0 / sfreq)
    m = (f >= SPEC_BAND[0]) & (f <= SPEC_BAND[1])
    fp = float(f[m][P[m].argmax()])
    return fp, SL.band_of(fp)


def claimed(dirs: np.ndarray, K: np.ndarray, gam: np.ndarray, ch: list[str],
            sfreq: float):
    """返回每个原子的 (通道剖面 (D,C), 自称导, 自称频带, 自称峰频)。"""
    A = np.einsum("dj,jct->dct", dirs * gam[None, :], K)   # (D,C,25)
    p = np.abs(A).sum(2)                                   # (D,C)
    ch_star = p.argmax(1)
    band, fpk = [], []
    for i in range(len(A)):
        fp, b = peak_band(A[i, ch_star[i]], sfreq)
        band.append(b)
        fpk.append(fp)
    return p, ch_star, np.array(band, dtype=object), np.array(fpk)


# ------------------------------------------------------------------ 数据侧
def input_grads(model, X: np.ndarray, dirs: np.ndarray):
    """$g_i=(1/NT)\\,\\partial\\sum a_i/\\partial x$，返回 (D,C,T)。

    **必须逐原子反传**：$\\sum_i a_i$ 的梯度是所有原子梯度的和，反传一次只能得到
    总和。$a_i$ 对 $x$ 的依赖是 $\\sum_t\\mathbf 1[a_i(t)>0]\\,\\partial(w_i\\!\\cdot\\!h(t))/\\partial x$，
    掩码随 $i$ 变，无法用一次反传拆开。全部试次放进一个 batch，图只建一次
    （`retain_graph=True`），每个原子只多一次反传。
    """
    import torch
    dev = next(model.parameters()).device
    xb = torch.tensor(X, dtype=torch.float32, device=dev).requires_grad_(True)
    h = model.bnorm(model.conv_time_spat(
        model.dimshuffle(model.ensuredims(xb)))).squeeze(-1)          # (b,F,T')
    W = torch.tensor(dirs, dtype=torch.float32, device=dev)           # (D,F)
    D = W.shape[0]
    acc = np.zeros((D,) + tuple(X.shape[1:]), dtype=np.float32)       # float32：720 MB→360 MB
    n_tok = X.shape[0] * h.shape[2]
    for i in range(D):
        a = torch.relu(torch.einsum("bft,f->bt", h, W[i]))
        g, = torch.autograd.grad(a.sum(), xb, retain_graph=(i < D - 1))
        acc[i] = g.sum(0).cpu().numpy()
        if (i + 1) % 1024 == 0:
            print(f"      grad {i + 1}/{D}", flush=True)
    return acc / n_tok


def roi_of(chan: str) -> str | None:
    for name, roi in (("occ", SL.ROI_OCCIPITAL), ("L", SL.ROI_LEFT_MOTOR),
                      ("R", SL.ROI_RIGHT_MOTOR), ("mid", SL.ROI_MIDLINE),
                      ("front", SL.ROI_FRONTAL_ARTIFACT)):
        if chan in roi:
            return name
    return None


# ------------------------------------------------------------------ 单被试
def run_subject(sub: str, seed: int = 0, steps: int = 3000, k: int = SL.SAE_K,
                d_dict: int = SL.SAE_DICT, n_trials: int = 60,
                src: str = "k8", verbose: bool = True) -> dict:
    import torch

    tr = D31.load_session(sub, "T")
    te = D31.load_session(sub, "E")
    eeg = [i for i, c in enumerate(tr["ch"]) if c not in SL.T1_EOG_CH]
    Xtr, Xte = D31.zscore(tr["X"][:, eeg], te["X"][:, eeg])
    ytr, yte = tr["y"], te["y"]
    ch_names = [tr["ch"][i] for i in eeg]

    dec = json.loads((PROBE / "t1_decoder_main.json").read_text(encoding="utf-8"))
    epochs = [r for r in dec["runs"] if r["subject"] == sub and r["chans"] == "eeg"
              and r["seed"] == seed][0]["best_epoch"]
    model, _ = D31.train_once(Xtr, ytr, Xtr[:1], ytr[:1], epochs=epochs, seed=seed)

    prev = json.loads((PROBE / f"t1_sae_ablate_{src}.json").read_text(encoding="utf-8"))
    rec = [r for r in prev["runs"] if r["subject"] == sub][0]
    steps = int(prev.get("steps", steps))
    d_dict = int(prev.get("d_dict", d_dict))

    Htr = D32.feature_map(model, Xtr)
    sae = TopKSAE(Htr.shape[1], d_dict, k, seed=seed)
    sae.fit(D32.tokens_of(Htr), steps=steps, bs=1024, lr=2e-3)

    # --- 侧化采样 60 个测试试次（每类均衡），只为降成本：E_n 在 60 试次上已稳定
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(SL.T1_N_CLASSES):
        cand = np.where(yte == c)[0]
        idx.append(rng.choice(cand, min(n_trials // SL.T1_N_CLASSES, len(cand)),
                              replace=False))
    idx = np.sort(np.concatenate(idx))

    sfreq = float(tr["sfreq"])
    dirs = sae.atom_dirs()
    K, gam = equiv_kernels(model)
    p_cl, ch_star, band_cl, fpk_cl = claimed(dirs, K, gam, ch_names, sfreq)

    t0 = time.time()
    G = input_grads(model, Xte[idx], dirs)
    p_me = np.abs(G).mean(2)                       # (D,C)：对时间求平均
    ch_me = p_me.argmax(1)
    band_me, fpk_me, lat_me = [], [], []
    for i in range(len(dirs)):
        c = ch_me[i]
        fp, b = peak_band(G[i, c], sfreq)
        band_me.append(b); fpk_me.append(fp)
        lat_me.append(float(np.abs(G[i, c]).argmax()) / sfreq + SL.EPOCHS["bci2a"][0])
    band_me = np.array(band_me, dtype=object)

    # --- 三项 → 两项打分
    topo_ok = np.array([roi_of(ch_names[ch_me[i]]) == roi_of(ch_names[ch_star[i]])
                        and roi_of(ch_names[ch_me[i]]) is not None
                        for i in range(len(dirs))])
    band_ok = np.array([band_me[i] is not None and band_me[i] == band_cl[i]
                        for i in range(len(dirs))])
    score = topo_ok.astype(int) + band_ok.astype(int)

    # --- 置换零对照：把「实测」属性随机换给别的原子，得到一致率的偶然水平
    rng2 = np.random.RandomState(seed + 7)
    roi_me = np.array([roi_of(c) for c in (ch_names[i] for i in ch_me)], dtype=object)
    roi_cl = np.array([roi_of(c) for c in (ch_names[i] for i in ch_star)], dtype=object)
    topo_null, band_null = [], []
    for _ in range(20):
        pm = rng2.permutation(len(dirs))
        topo_null.append(np.mean([roi_me[i] == roi_cl[pm[i]] and roi_me[i] is not None
                                  for i in range(len(dirs))]))
        band_null.append(np.mean([band_me[pm[i]] is not None and band_me[pm[i]] == band_cl[i]
                                  for i in range(len(dirs))]))

    eff = np.array(rec["eff"])
    null = np.array(rec["null_eff"])
    thr = float(np.percentile(null, 95))

    if verbose:
        print(f"  {sub} | topo {topo_ok.mean():.1%} band {band_ok.mean():.1%} | "
              f"得分分布 {np.bincount(score, minlength=3).tolist()} | "
              f"g 用时 {time.time() - t0:.0f}s", flush=True)

    return dict(subject=sub, seed=seed, steps=steps, d_dict=d_dict, k=k,
                n_trials=int(len(idx)), epochs=int(epochs),
                score=score.tolist(), topo_ok=topo_ok.tolist(), band_ok=band_ok.tolist(),
                topo_null=float(np.mean(topo_null)), band_null=float(np.mean(band_null)),
                ch_star=[ch_names[i] for i in ch_star],
                ch_meas=[ch_names[i] for i in ch_me],
                band_claimed=band_cl.tolist(), band_measured=band_me.tolist(),
                fpeak_claimed=[float(x) for x in fpk_cl], fpeak_measured=[float(x) for x in fpk_me],
                latency_measured=lat_me,
                eff=eff.tolist(), null_p95=thr, margin_sd=float(rec["margin_sd"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str,
                    default="A01,A02,A03,A04,A05,A06,A07,A08,A09")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--from-ablate", type=str, default="k8")
    ap.add_argument("--tag", type=str, default="main")
    a = ap.parse_args()

    info = det_lock.lock(0)
    print(f"[det] {info}", flush=True)
    subs = [s.strip() for s in a.subjects.split(",") if s.strip()]
    t0 = time.time()
    out = dict(spec_band=list(SPEC_BAND), fft_n=FFT_N, det=info, runs=[])
    for sub in subs:
        out["runs"].append(run_subject(sub, a.seed, a.steps, n_trials=a.n_trials,
                                       src=a.from_ablate))
    out["elapsed_s"] = round(time.time() - t0, 1)
    (PROBE / f"t1_semantic_{a.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    summarize(out, a.tag)
    print(f"[done] {out['elapsed_s']} s", flush=True)
    return 0


def summarize(out: dict, tag: str) -> None:
    from scipy.stats import spearmanr
    _s = "" if tag == "main" else f"_{tag}"   # main 写规范名

    lines = ["# T1 语义探针：SAE 的「自称」与输入梯度实测", "",
             f"自动生成。分析带 {out['spec_band'][0]:.0f}–{out['spec_band'][1]:.0f} Hz，"
             f"FFT 补零 {out['fft_n']}；确定性锁 `{out['det']}`。", "",
             "语义得分**只有两项**（地形、频带）：$d_i$ 是 40 维 token 方向、不含时间轴，",
             "潜伏期在该架构下没有权重侧对应物（D36）。", "",
             "## 表 1 语义–因果联合分布", "",
             "| 被试 | 得分 | 特征数 | 消融效应中位数 | IQR | 超零比例 |",
             "|---|---|---|---|---|---|"]
    for r in out["runs"]:
        s = np.array(r["score"]); e = np.array(r["eff"]); thr = r["null_p95"]
        for v in (0, 1, 2):
            m = s == v
            if m.sum() == 0:
                continue
            ev = e[m]
            q1, q3 = np.percentile(ev, [25, 75])
            lines.append(f"| {r['subject']} | {v} | {int(m.sum())} | {np.median(ev):+.4f} | "
                         f"[{q1:+.4f}, {q3:+.4f}] | {(ev > thr).mean():.1%} |")
    lines += ["", "## 逐被试单指标通过率与 S1", "",
              "「置换零」= 把实测属性随机换给别的原子后的一致率，即偶然水平。", "",
              "| 被试 | 地形 | 地形零 | 频带 | 频带零 | ρ(得分,效应) | ρ(地形,效应) | "
              "ρ(频带,效应) | 实测潜伏期中位 (s) |",
              "|---|---|---|---|---|---|---|---|---|"]
    rho_all, topo_all, band_all = [], [], []
    for r in out["runs"]:
        s = np.array(r["score"]); e = np.array(r["eff"])
        t = np.array(r["topo_ok"]).astype(float)
        b = np.array(r["band_ok"]).astype(float)
        rho = spearmanr(s, e).statistic
        rho_all.append(rho); topo_all.append(t.mean()); band_all.append(b.mean())
        lines.append(f"| {r['subject']} | {t.mean():.1%} | {r['topo_null']:.1%} | "
                     f"{b.mean():.1%} | {r['band_null']:.1%} | {rho:+.3f} | "
                     f"{spearmanr(t, e).statistic:+.3f} | {spearmanr(b, e).statistic:+.3f} | "
                     f"{np.median(r['latency_measured']):.2f} |")
    rho_all = np.array(rho_all)
    tp = np.array([np.mean(r["topo_ok"]) for r in out["runs"]])
    tn = np.array([r["topo_null"] for r in out["runs"]])
    bp = np.array([np.mean(r["band_ok"]) for r in out["runs"]])
    bn = np.array([r["band_null"] for r in out["runs"]])
    from scipy.stats import wilcoxon
    # --- 覆盖受限的地形口径：只在「实测与自称所指导联都有 ROI 定义」的原子内比。
    # 22 导里只有 16 个落在 5 个 ROI 内，全导联口径的分母因此混进了「ROI 未定义」的原子，
    # 把一致率压低。受限口径给出的是「在能判的原子中对上了多少」，可与 T2 直接并列。
    from collections import Counter as _C
    _both, _obs, _nul = [], [], []
    for r in out["runs"]:
        rm = np.array([roi_of(c) for c in r["ch_meas"]], dtype=object)
        rc = np.array([roi_of(c) for c in r["ch_star"]], dtype=object)
        m = (rm != None) & (rc != None)  # noqa: E711
        _both.append(float(m.mean()))
        if not m.any():
            continue
        a_, b_ = rm[m], rc[m]
        _obs.append(float((a_ == b_).mean()))
        pa, pb, n_ = _C(a_), _C(b_), len(a_)
        _nul.append(sum(pa[kk] / n_ * pb[kk] / n_ for kk in set(pa) & set(pb)))
    _both = float(np.median(_both))

    lines += ["", f"**S1**：ρ(语义得分, 消融效应) = {rho_all.mean():+.3f} ± {rho_all.std(ddof=1):.3f}"
                  f"（判据 |ρ| ≤ {SL.S1_RHO_MAX}）→ "
                  f"{'通过（自称不预测因果）' if abs(rho_all.mean()) <= SL.S1_RHO_MAX else '失败'}。", "",
              "| 项 | 实测一致率 | 置换零 | 差 | Wilcoxon p | 高于零的被试数 |",
              "|---|---|---|---|---|---|"]
    for nm, a, b in (("地形", tp, tn), ("频带", bp, bn)):
        d = a - b
        lines.append(f"| {nm} | {a.mean():.1%} | {b.mean():.1%} | {d.mean():+.1%} | "
                     f"{wilcoxon(d).pvalue:.4f} | {int((d > 0).sum())}/{len(d)} |")
    if _obs:
        lines.append(f"| 地形（仅双方 ROI 有定义的原子，占 {_both:.0%}） | "
                     f"{np.mean(_obs):.1%} | {np.mean(_nul):.1%} | "
                     f"{(np.mean(_obs) - np.mean(_nul)):+.1%} | — | "
                     f"{int(np.sum(np.array(_obs) > np.array(_nul)))}/{len(_obs)} |")
        lines += ["", "上面最后一行是**覆盖受限口径**：22 导里只有 16 个落在 5 个 ROI 内，"
                  "全导联口径把「ROI 未定义」的原子也记成不一致，会低估一致率。"
                  "受限口径是「在能判的原子中对上了多少」，可与 `T2_semantic.md` 直接并列"
                  "（那边覆盖只有 15%，两份报告都用这一行做跨数据集比较）。", ""]
    lines.append("")
    (REPORTS / f"T1_semantic{_s}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
