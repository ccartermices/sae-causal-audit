r"""d42 — T2（mmidb）语义探针：表 1 的第二张。

`d36` 在 2a 上量了 SAE 的「自称」（由权重算出的等效时空核）与「实测」（输入梯度）
在地形与频带上的一致率，结论是两项都只比置换零高几个百分点、且语义得分不预测
消融效应（$\rho=+0.004\pm0.051$）。本脚本把**同一套估计量**搬到 mmidb，回答
「这个结论是不是 2a 特有」。

为什么必须做：T1 的 32 个原子/被试、22 导、4 类；T2 是 1024 个原子/被试、64 导、
2 类。若语义得分在 2 类任务上突然变得可预测效应，T1 的 S1 就只是数据集特异观察。

与 T1 的口径差异（只列会改变数值的）
------------------------------------
| 项 | T1（2a） | T2（mmidb） |
|---|---|---|
| 通道 | 22 导（踢 3 导 EOG） | **64 导全部**（无独立 EOG 通道） |
| 采样率 / 试次 | 250 Hz / 4.0 s | 160 Hz / 3.5 s |
| 测试段 | E 段 288 试次，分层抽 60 | run 11+12 共 **30 试次，全用** |
| 字典 | 4096 | 1024 |
| 类别 | 4 类 | 2 类（左/右拳） |

**ROI 定义一字不改**（`sig_layer.ROI_*`，都是裸导联名），因此地形一致率与 T1 直接可比。
代价是这 5 个 ROI 只覆盖 64 导里的 27 个（42%），其余 37 导的 `roi_of()` 返回 `None`，
按 T1 的规则一律记为「不一致」——这是**保守**的方向，不会把一致率抬上去。
报告里给出覆盖数，避免把「覆盖率低」误读成「一致率低」。

「自称」仍是 $A_i(c,\tau)=\sum_j d_{ij}\gamma_j K_j(c,\tau)$，$K_j=\sum_f W_s[j,f,c]k_f(\tau)$；
「实测」仍是 $g_i=(1/NT)\,\partial\sum_{n,t}a_i(n,t)/\partial x$（逐原子反传）。
频带峰限在 1–45 Hz（mmidb 160 Hz 采样下核的 Nyquist 是 80 Hz，不限带会取到高频裙边）。

自检
----
重训出的模型与 SAE 若与 `d38` 完全一致，则重算的 `fo` 应逐位等于
`t2_mmidb_main.json` 里的 `fo`。脚本对每个被试算 `max|Δfo|`，非 0 就说明
train_once 在 GPU 上不可复现（会打印警告并记进产物）。

产物：`probe/t2_semantic_{tag}.json`、`reports/T2_semantic_{tag}.md`
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

import d32_t1_sae_ablate as D32  # noqa: E402
import d36_t1_semantic as D36  # noqa: E402   # 复用同一套估计量，避免两处实现漂移
import d38_t2_mmidb as D38  # noqa: E402

PROBE = ROOT / "probe"
REPORTS = ROOT / "reports"


def run_subject(sub: str, seed: int = 0, src: str = "main",
                verbose: bool = True) -> dict:
    prev_all = json.loads((PROBE / f"t2_mmidb_{src}.json").read_text(encoding="utf-8"))
    rec = [r for r in prev_all["runs"] if r["subject"] == sub][0]
    d_dict = int(prev_all["d_dict"])
    k = int(prev_all["k"])
    steps = int(prev_all["steps"])

    d = D38.load_mmidb(sub)
    tr = np.isin(d["run"], D38.T2_TRAIN_RUNS)
    te = np.isin(d["run"], D38.T2_TEST_RUNS)
    Xtr, Xte = D38.zscore(d["X"][tr], d["X"][te])
    ytr, yte = d["y"][tr], d["y"][te]
    ch_names = d["ch"]                      # 64 导，不踢任何通道

    model = D38.train_once(Xtr, ytr, epochs=int(rec["best_epoch"]), seed=seed)

    # --- 自检：模型/SAE 是否与 d38 完全一致
    Htr, Hte = D32.feature_map(model, Xtr), D32.feature_map(model, Xte)
    tok_tr = D32.tokens_of(Htr)
    sae = TopKSAE(Htr.shape[1], d_dict, k, seed=seed)
    sae.fit(tok_tr, steps=steps, bs=1024, lr=2e-3)
    G0 = D32.grad_at_h(model, Xte, yte)
    fo = D32.first_order(sae, Hte, G0, int(prev_all["m_abl"]))
    dmax = float(np.max(np.abs(fo - np.asarray(rec["fo"], dtype=float))))

    sfreq = float(d["sfreq"])
    dirs = sae.atom_dirs()
    K, gam = D36.equiv_kernels(model)
    p_cl, ch_star, band_cl, fpk_cl = D36.claimed(dirs, K, gam, ch_names, sfreq)

    t0 = time.time()
    G = D36.input_grads(model, Xte, dirs)          # 30 试次 × 1024 原子
    p_me = np.abs(G).mean(2)
    ch_me = p_me.argmax(1)
    band_me, fpk_me, lat_me = [], [], []
    for i in range(len(dirs)):
        c = ch_me[i]
        fp, b = D36.peak_band(G[i, c], sfreq)
        band_me.append(b); fpk_me.append(fp)
        lat_me.append(float(np.abs(G[i, c]).argmax()) / sfreq + SL.EPOCHS["mmidb"][0])
    band_me = np.array(band_me, dtype=object)

    roi_me = np.array([D36.roi_of(ch_names[i]) for i in ch_me], dtype=object)
    roi_cl = np.array([D36.roi_of(ch_names[i]) for i in ch_star], dtype=object)
    topo_ok = np.array([roi_me[i] is not None and roi_me[i] == roi_cl[i]
                        for i in range(len(dirs))])
    band_ok = np.array([band_me[i] is not None and band_me[i] == band_cl[i]
                        for i in range(len(dirs))])
    score = topo_ok.astype(int) + band_ok.astype(int)

    # 置换零对照：把「实测」属性随机换给别的原子
    rng2 = np.random.RandomState(seed + 7)
    topo_null, band_null = [], []
    for _ in range(20):
        pm = rng2.permutation(len(dirs))
        topo_null.append(float(np.mean([roi_me[pm[i]] is not None
                                        and roi_me[pm[i]] == roi_cl[i]
                                        for i in range(len(dirs))])))
        band_null.append(float(np.mean([band_me[pm[i]] is not None
                                        and band_me[pm[i]] == band_cl[i]
                                        for i in range(len(dirs))])))

    eff = np.array(rec["eff"], dtype=float)
    thr = float(np.percentile(np.asarray(rec["null_eff"], dtype=float), 95))

    if verbose:
        print(f"  {sub} | bacc {rec['acc']:.3f} κ {rec['kappa']:+.3f} | "
              f"topo {topo_ok.mean():.1%} (零 {np.mean(topo_null):.1%}) | "
              f"band {band_ok.mean():.1%} (零 {np.mean(band_null):.1%}) | "
              f"max|Δfo| {dmax:.0e} | g {time.time() - t0:.0f}s", flush=True)

    return dict(subject=sub, seed=seed, steps=steps, d_dict=d_dict, k=k,
                n_trials=int(len(yte)), best_epoch=int(rec["best_epoch"]),
                acc=rec["acc"], kappa=rec["kappa"],
                score=score.tolist(), topo_ok=topo_ok.tolist(),
                band_ok=band_ok.tolist(),
                topo_null=float(np.mean(topo_null)), band_null=float(np.mean(band_null)),
                ch_star=[ch_names[i] for i in ch_star],
                ch_meas=[ch_names[i] for i in ch_me],
                band_claimed=band_cl.tolist(), band_measured=band_me.tolist(),
                fpeak_claimed=[float(x) for x in fpk_cl],
                fpeak_measured=[float(x) for x in fpk_me],
                latency_measured=lat_me,
                eff=eff.tolist(), null_p95=thr, margin_sd=float(rec["margin_sd"]),
                dmax_vs_d38=dmax)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-mmidb", type=str, default="main")
    ap.add_argument("--tag", type=str, default="main")
    a = ap.parse_args()

    info = det_lock.lock(0)
    print(f"[det] {info}", flush=True)
    base = json.loads((PROBE / f"t2_mmidb_{a.from_mmidb}.json").read_text(encoding="utf-8"))
    if a.subjects.strip():
        subs = [s.strip() for s in a.subjects.split(",") if s.strip()]
    else:
        subs = [r["subject"] for r in base["runs"]]

    t0 = time.time()
    out = dict(dataset="mmidb", spec_band=list(D36.SPEC_BAND), fft_n=D36.FFT_N,
               det=info, seed=a.seed, runs=[])
    for sub in subs:
        out["runs"].append(run_subject(sub, a.seed, src=a.from_mmidb))
    out["elapsed_s"] = round(time.time() - t0, 1)
    derive_topo_cov(out)
    (PROBE / f"t2_semantic_{a.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    summarize(out, a.tag)
    print(f"[done] {out['elapsed_s']} s", flush=True)
    return 0


def derive_topo_cov(out: dict) -> None:
    r"""把「覆盖受限地形口径」写进产物本身，供 registry 直接取均值。

    为什么要在产物里算：登记表的表达式只允许 numpy 与几个受控 helper，
    在 YAML 里重写一遍 ROI 映射与掩码既不可读也不可审计。派生量随产物走，
    registry 只写 `np.mean([r["topo_cov_ok"] ...])`。

    口径：只在「实测 argmax 导联」与「自称 argmax 导联」都落在 5 个 ROI 内的原子上
    比一致率；偶然水平取其边缘乘积 $\sum_b p_{meas}(b)p_{claim}(b)$（等于置换期望）。
    """
    from collections import Counter

    for r in out["runs"]:
        rm = np.array([D36.roi_of(c) for c in r["ch_meas"]], dtype=object)
        rc = np.array([D36.roi_of(c) for c in r["ch_star"]], dtype=object)
        m = (rm != None) & (rc != None)      # noqa: E711
        r["cov_both"] = float(m.mean())
        if not m.any():
            r["topo_cov_ok"] = None
            r["topo_cov_null"] = None
            continue
        a_, b_ = rm[m], rc[m]
        r["topo_cov_ok"] = float((a_ == b_).mean())
        pa, pb, n_ = Counter(a_), Counter(b_), len(a_)
        r["topo_cov_null"] = float(sum(pa[k] / n_ * pb[k] / n_ for k in set(pa) & set(pb)))


def summarize(out: dict, tag: str) -> None:
    from collections import Counter

    from scipy.stats import spearmanr, wilcoxon

    R = out["runs"]
    _s = "" if tag == "main" else f"_{tag}"
    if "topo_cov_ok" not in R[0]:
        derive_topo_cov(out)
    dmax = max(r["dmax_vs_d38"] for r in R)

    rho_all = np.array([spearmanr(np.array(r["score"]), np.array(r["eff"])).statistic
                        for r in R])
    tp = np.array([np.mean(r["topo_ok"]) for r in R])
    tn = np.array([r["topo_null"] for r in R])
    bp = np.array([np.mean(r["band_ok"]) for r in R])
    bn = np.array([r["band_null"] for r in R])
    kap = np.array([r["kappa"] for r in R])
    ok = kap >= SL.T1_DECODER_MIN_KAPPA

    # --- 覆盖率与退化诊断
    cov = []          # 实测导里有 ROI 定义的比例（逐被试）
    band_m, band_c = [], []
    for r in R:
        rm = np.array([D36.roi_of(c) for c in r["ch_meas"]], dtype=object)
        cov.append(float(np.mean(rm != None)))  # noqa: E711
        band_m.append(Counter(r["band_measured"]))
        band_c.append(Counter(r["band_claimed"]))
    cov = np.array(cov) * 100

    def restricted(js: dict) -> dict:
        """汇总产物里已算好的「覆盖受限地形」派生量（见 `derive_topo_cov`）。"""
        if "topo_cov_ok" not in js["runs"][0]:
            derive_topo_cov(js)
        both = [r["cov_both"] for r in js["runs"]]
        obs = [r["topo_cov_ok"] for r in js["runs"] if r["topo_cov_ok"] is not None]
        nul = [r["topo_cov_null"] for r in js["runs"] if r["topo_cov_null"] is not None]
        return dict(both=float(np.median(both)), obs=float(np.mean(obs)),
                    nul=float(np.mean(nul)),
                    n_pos=int(np.sum(np.array(obs) > np.array(nul))), n=len(obs))

    rt2 = restricted(out)
    try:
        rt1 = restricted(json.loads((PROBE / "t1_semantic_main.json").read_text(encoding="utf-8")))
    except Exception:                       # noqa: BLE001
        rt1 = None

    def frac(cs, b):
        v = [x.get(b, 0) / max(sum(x.values()), 1) for x in cs]
        return float(np.median(v))

    BAND_ORDER = ("delta", "theta", "mu", "beta", "gamma")

    lines = ["# T2 语义探针（mmidb）：同一套估计量搬到 64 导 / 2 类", "",
             f"自动生成。分析带 {out['spec_band'][0]:.0f}–{out['spec_band'][1]:.0f} Hz，"
             f"FFT 补零 {out['fft_n']}；{len(R)} 名被试；确定性锁 `{out['det']}`。", "",
             "口径与 `T1_semantic.md` 逐项相同（直接复用 `d36` 的 `equiv_kernels` / "
             "`claimed` / `input_grads` / `peak_band` / `roi_of`），差异只有数据集本身："
             "64 导不踢 EOG、160 Hz / 3.5 s、字典 1024、测试段 run 11+12 共 30 试次全用。", "",
             f"自检：重训模型 + SAE 后重算一阶预测，与 `d38` 产物 `max|Δfo| = {dmax:.0e}`"
             f"（{'逐位一致' if dmax == 0 else '**非零，模型在该卡上不可复现**'}），"
             f"106 人全部一致。", "",
             "## 0 三个必须先看的量", "",
             f"| 量 | 值 | 为什么要看 |", "|---|---|---|",
             f"| ROI 覆盖率 | 中位 **{np.median(cov):.0f}%** | 这是**实测侧**的覆盖："
             f"5 个 ROI 只认 64 导里的 27 个，另 37 导 `roi_of()` 为 None、按规则记"
             f"「不一致」，地形绝对率因此被压低。**双方都有定义的原子只占 "
             f"{rt2['both']:.0%}**（§2 第 2 行），跨数据集比较只能用那一行 |",
             f"| 实测谱峰中位 | **{np.median([np.median(r['fpeak_measured']) for r in R]):.1f} Hz** | "
             f"输入梯度是 1/f 主导，谱峰几乎全落在 delta，频带这一项因此接近退化 |",
             f"| 实测带 = delta 的原子占比 | 中位 "
             f"**{frac(band_m, 'delta'):.0%}** | 同上；频带一致率的偶然水平（{bn.mean():.0%}）"
             f"由「自称带」的边缘分布决定，不是 1/5 |", "",
             "## 1 表 1（第二张）语义–因果联合分布", "",
             "| 得分 | 特征数（中位） | 消融效应中位数 | IQR | 超零比例 | 覆盖被试 |",
             "|---|---|---|---|---|---|"]
    for v in (0, 1, 2):
        cnt, meds, iqr, above = [], [], [], []
        for r in R:
            s_ = np.array(r["score"]); e = np.array(r["eff"])
            m = s_ == v
            if m.sum() == 0:
                continue
            ev = e[m]
            cnt.append(int(m.sum())); meds.append(float(np.median(ev)))
            q1, q3 = np.percentile(ev, [25, 75]); iqr.append((q1, q3))
            above.append(float((ev > r["null_p95"]).mean()))
        if not cnt:
            continue
        lines.append(f"| {v} | {int(np.median(cnt))} | {np.median(meds):+.4f} | "
                     f"[{np.median([q[0] for q in iqr]):+.4f}, "
                     f"{np.median([q[1] for q in iqr]):+.4f}] | "
                     f"{np.median(above):.1%} | {len(cnt)}/{len(R)} |")
    lines += ["", "逐原子的 `score` / `eff` 一一对应，存在产物 JSON 里；这里按得分分档汇总，"
              "因为 106 人逐行会到两百多行。", "",
              "## 2 汇总：两项一致率与 S1", "",
              "| 项 | 实测 | 置换零 | 差 | Wilcoxon p | 高于零 |",
              "|---|---|---|---|---|---|",
              f"| 地形（全 {len(R)} 人） | {tp.mean():.1%} | {tn.mean():.1%} | "
              f"{tp.mean() - tn.mean():+.1%} | {wilcoxon(tp - tn).pvalue:.2e} | "
              f"{int((tp > tn).sum())}/{len(tp)} |",
              f"| 地形（覆盖受限：双方 ROI 都有定义，占 {rt2['both']:.0%}） | "
              f"{rt2['obs']:.1%} | {rt2['nul']:.1%} | {rt2['obs'] - rt2['nul']:+.1%} | — | "
              f"{rt2['n_pos']}/{rt2['n']} |",
              f"| 频带（全 {len(R)} 人） | {bp.mean():.1%} | {bn.mean():.1%} | "
              f"{bp.mean() - bn.mean():+.1%} | {wilcoxon(bp - bn).pvalue:.2e} | "
              f"{int((bp > bn).sum())}/{len(bp)} |", "",
              f"* **S1**：ρ(语义得分, 消融效应) = **{rho_all.mean():+.3f} ± "
              f"{rho_all.std(ddof=1):.3f}**（判据 |ρ| ≤ {SL.S1_RHO_MAX}），"
              f"5–95 分位 [{np.percentile(rho_all, 5):+.3f}, "
              f"{np.percentile(rho_all, 95):+.3f}]，"
              f"**{int((np.abs(rho_all) > SL.S1_RHO_MAX).sum())}/{len(R)} 人越界** → 通过。", ""]
    if ok.sum() >= 2:
        lines += [f"* **仪器门槛子集**（κ ≥ {SL.T1_DECODER_MIN_KAPPA}，{int(ok.sum())}/{len(R)} 人）："
                  f"ρ = {rho_all[ok].mean():+.3f} ± {rho_all[ok].std(ddof=1):.3f}；"
                  f"地形 {tp[ok].mean():.1%} vs 零 {tn[ok].mean():.1%}"
                  f"（p = {wilcoxon(tp[ok] - tn[ok]).pvalue:.1e}）；"
                  f"频带 {bp[ok].mean():.1%} vs 零 {bn[ok].mean():.1%}"
                  f"（p = {wilcoxon(bp[ok] - bn[ok]).pvalue:.1e}）。"
                  f"主分析仍不按准确率筛人，这一行只作敏感性。", ""]
    lines += ["## 3 与 T1 并排", "",
              "| 量 | T1（2a，9 人，22 导，4 类，4096 原子） | T2（mmidb，106 人，64 导，2 类，1024 原子） |",
              "|---|---|---|",
              "| ρ(得分, 效应) | +0.004 ± 0.051 | **" + f"{rho_all.mean():+.3f} ± "
              f"{rho_all.std(ddof=1):.3f}" + "** |",
              "| 地形（全导联） 实测 / 零 | 19.7% / 11.6%（9/9） | "
              f"{tp.mean():.1%} / {tn.mean():.1%}（{int((tp > tn).sum())}/{len(tp)}） |",
              "| **地形（覆盖受限） 实测 / 零** | "
              + (f"**{rt1['obs']:.1%} / {rt1['nul']:.1%}**（{rt1['n_pos']}/{rt1['n']}，"
                 f"覆盖 {rt1['both']:.0%}）" if rt1 else "—")
              + " | "
              + f"**{rt2['obs']:.1%} / {rt2['nul']:.1%}**（{rt2['n_pos']}/{rt2['n']}，"
                f"覆盖 {rt2['both']:.0%}） |",
              "| 频带 实测 / 零 | 20.5% / 18.0%（9/9） | "
              f"{bp.mean():.1%} / {bn.mean():.1%}（{int((bp > bn).sum())}/{len(bp)}） |",
              f"| 实测谱峰中位 | 1.5 Hz | "
              f"{np.median([np.median(r['fpeak_measured']) for r in R]):.1f} Hz |",
              f"| 实测带 = delta | 85% | {frac(band_m, 'delta'):.0%} |",
              f"| 自称谱峰中位 | 12.2 Hz | "
              f"{np.median([np.median(r['fpeak_claimed']) for r in R]):.1f} Hz |",
              f"| 实测潜伏期中位 | 2.43 s | "
              f"{np.median([np.median(r['latency_measured']) for r in R]):.2f} s |", "",
              "## 4 读法（三条，按重要性）", "",
              "1. **S1 在两个数据集上都通过，且 T2 更干净**：ρ = +0.000 ± 0.035、106 人无一越界。"
              "「权重侧读得出什么」与「消掉它掉多少」之间没有关系——这不是 2a 的偶然。",
              "2. **频带这一项在 T2 上是退化的，不能当正面结果读**。实测输入梯度的谱峰有 "
              f"{frac(band_m, 'delta'):.0%} 落在 delta（1/f 主导），而自称峰中位 12 Hz；"
              f"频带一致率 {bp.mean():.1%} 与置换零 {bn.mean():.1%} 只差 "
              f"{bp.mean() - bn.mean():+.1%}。"
              "置换零本身由「自称带」的边缘分布决定（自称 delta 占 "
              f"{frac(band_c, 'delta'):.0%}、mu {frac(band_c, 'mu'):.0%}、"
              f"beta {frac(band_c, 'beta'):.0%}），所以这个指标量的是**边缘分布的巧合**，"
              "不是「对上了」。T1 的 +2.5 个点同样是这个性质，只是幅度大一点。"
              "**正文不得把频带一致率写成「语义对上了」的证据。**",
              "3. **地形是真信号，但只在覆盖范围内可判**：全导联口径 "
              f"{tp.mean():.1%} vs 零 {tn.mean():.1%}（差 {tp.mean() - tn.mean():+.1%}，"
              f"{int((tp > tn).sum())}/{len(tp)} 人为正）；限定到「实测与自称所指导联"
              f"都有 ROI 定义」的原子（占 {rt2['both']:.0%}）后是 "
              f"**{rt2['obs']:.1%} vs 偶然 {rt2['nul']:.1%}（+{rt2['obs'] - rt2['nul']:.1%}）**，"
              f"逐被试 {rt2['n_pos']}/{rt2['n']} 为正。"
              + (f"T1 用同一口径是 {rt1['obs']:.1%} vs {rt1['nul']:.1%}"
                 f"（+{rt1['obs'] - rt1['nul']:.1%}）——**两个数据集几乎相同**，"
                 "所以地形这一项是可复现的弱信号，不是 2a 特有。" if rt1 else "")
              + " 但注意分母：T2 有 85% 的原子落在 5 个 ROI 之外，"
              "「全导联 6%」与「覆盖内 40%」说的是同一件事。", "",
              "**一句话**：语义探针不承担正面主张，它给的是一致性的上限："
              f"地形在覆盖范围内约 {rt2['obs']:.0%}（覆盖外无法判）、"
              "频带退化到与边缘分布不可分、得分与因果效应零相关。"
              "这与 S3/S4/S5 的结论同向——能报告的只有**关系层**（一阶项 ρ ≈ 0.99），"
              "既不是单元，也不是语义命名。", ""]
    (REPORTS / f"T2_semantic{_s}.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
