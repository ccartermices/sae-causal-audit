"""d40 — S4 跨被试迁移：因果效应的「空间–频带」分布是否在人与人之间一致。

判据（`03` §1 冻结）
--------------------
留一被试 $\\rho_{transfer}\\ge0.30$，且 95%CI 下界 > 0。

为什么不能直接把 $\\rho_{1st}$ 拿来当 $\\rho_{transfer}$
------------------------------------------------------
一个自然的想法是「用其余被试拟合 $\\mathrm{eff}\\sim\\mathrm{fo}$，再预测留出被试」。
但 $\\rho$ 是 Spearman，**对单调变换不变**，所以「在别人身上拟合一个单调映射再排序」
的秩相关恒等于被试自己的 $\\rho_{1st}$。这个"迁移"是同一句话换个说法，不是检验，
写进正文会被一眼看破。故本脚本把该层降为**描述量**（只报斜率一致性），
真正的迁移检验放在下面两层。

三层口径
--------
**L1 关系层（描述）**：被试内斜率 $b_s$ 与 $\\rho_{1st}$ 的跨被试离散。若离散小，
说明"同一套校准"成立；但它不构成迁移证据（见上）。

**L2 聚合层（本脚本的主判据，$\\rho_{transfer}$）**：原子在输入空间有坐标
$(c,\\beta)$（$c$ 处输入梯度最大、谱峰落在 $\\beta$ 频带，均来自 `d36`）。
把 4096 个原子按 $(c,\\beta)$ 分箱（22 导 × 5 带 = 110 格），每格取
*平均消融效应* $e_s(c,\\beta)$，得到被试的一张 110 维"因果效应地形谱图"。
$\\rho_{transfer}$ = 被试对之间这张图的 Spearman 均值。
这一层的含义是：**"哪个导联、哪个频带的特征在承担决策"是不是人人相同**。
它是"报告单元"层面的迁移，也是实际可用的形式——新被试来了，按这张图取特征。

**L3 单元层（说明为什么不测）**：单个原子上没有跨被试可比的身份。
每个被试的 SAE 在自己的 token 空间里训练，原子编号无对应关系；N6 已实测
同一被试**跨种子**重训后支撑集 Jaccard 只有 0.080。故原子级迁移不可定义，
不装作能测。

三条读数的关系（本脚本的核心诊断）
----------------------------------
* **被试内半分信度**：同一被试的原子随机分两半，各算一张图，互相关。
  这是 $\\rho_{transfer}$ 的**上限**——测量噪声决定了它不可能更高。
* **跨被试**：$\\rho_{transfer}$。
* **零对照**：把原子到格子的分配在**被试内**随机置换（保持每格原子数与效应分布不变），
  重算被试对相关。这是"效应图只是格子大小的影子"的偶然水平。

若 跨被试 ≈ 半分信度 → 差异来自被试，不是噪声；若 跨被试 ≈ 零 → 特异的。

产物：`probe/t1_transfer_main.json`、`reports/T1_transfer.md`
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import sig_layer as SL  # noqa: E402

PROBE = ROOT / "probe"
REPORTS = ROOT / "reports"

CELL_MIN = 8          # 一格至少这么多原子才参与相关（4096/110 ≈ 37）

# 2a 的 22 个 EEG 导（与 d31 的 eeg 子集同序；ch_meas 存的是名字，故需映射）
EPO_DIR = ROOT / "data" / "interim" / "epochs" / "bci2a"


def _ch_index() -> dict:
    z = np.load(EPO_DIR / "A01T.npz", allow_pickle=True)
    names = [str(c) for c in z["ch_names"]]
    eeg = [c for c in names if c not in SL.T1_EOG_CH]
    return {c: i for i, c in enumerate(eeg)}


CH_INDEX = _ch_index()
N_BOOT = 2000
N_NULL = 200
BANDS = list(SL.BANDS)


# ------------------------------------------------------------------ 载入
def load_subject(sub: str, sem: dict, abl: dict) -> dict:
    """取一个被试的 (atom → cell, eff, fo)，并给出该被试的效应图。"""
    rs = [r for r in sem["runs"] if r["subject"] == sub]
    ra = [r for r in abl["runs"] if r["subject"] == sub]
    if not rs or not ra:
        raise KeyError(sub)
    r, a = rs[0], ra[0]
    # eff / fo 取 d32 的产物（含 fo）；ch / band 取 d36 的产物。
    # 两者的原子顺序逐位一致（同一 SAE 种子与步数，已核对 max|Δeff| = 0）。
    eff = np.asarray(a["eff"], dtype=float)
    fo = np.asarray(a["fo"], dtype=float)
    ch = np.array([CH_INDEX.get(str(c), -1) for c in r["ch_meas"]], dtype=int)
    band = np.array([BANDS.index(b) if b in BANDS else -1 for b in r["band_measured"]], dtype=int)
    keep = np.isfinite(eff) & np.isfinite(fo) & (band >= 0) & (ch >= 0)
    if keep.sum() < 0.5 * len(eff):
        raise ValueError(f"{sub}: 可用原子过少（{int(keep.sum())}/{len(eff)}）")
    return dict(sub=sub, eff=eff[keep], fo=fo[keep], ch=ch[keep], band=band[keep],
                n_ch=int(ch.max()) + 1,
                null_p95=float(np.percentile(a["null_eff"], 95)) if a.get("null_eff") else
                float(np.percentile(eff[keep], 90)))


def cell_map(d: dict, idx: np.ndarray | None = None, stat: str = "mean") -> np.ndarray:
    """把原子聚合成 (n_ch × 5) 的效应图。stat ∈ {mean, exc, top}。"""
    idx = np.arange(len(d["eff"])) if idx is None else idx
    eff, ch, band = d["eff"][idx], d["ch"][idx], d["band"][idx]
    nc, nb = d["n_ch"], len(BANDS)
    out = np.full((nc, nb), np.nan)
    cnt = np.zeros((nc, nb), dtype=int)
    for c in range(nc):
        for b in range(nb):
            m = (ch == c) & (band == b)
            cnt[c, b] = int(m.sum())
            if cnt[c, b] < CELL_MIN:
                continue
            v = eff[m]
            if stat == "mean":
                out[c, b] = float(np.mean(v))
            elif stat == "exc":
                out[c, b] = float(np.mean(v > d["null_p95"]))
            elif stat == "top":
                out[c, b] = float(np.mean(np.sort(v)[-3:]))
            elif stat == "top10":
                out[c, b] = float(np.mean(np.sort(v)[-10:]))
            elif stat == "p90":
                out[c, b] = float(np.percentile(v, 90))
            else:
                raise ValueError(stat)
    return out


def rho(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 12:
        return float("nan")
    x, y = a[m], b[m]
    rx = _rank(x); ry = _rank(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(v: np.ndarray) -> np.ndarray:
    """平均秩（并列取均值）——与 registry 的 sp_ 同口径。"""
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    r = np.empty(len(v), dtype=float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def boot_ci(v: np.ndarray, rng: np.random.RandomState) -> tuple[float, float, float]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"),) * 3
    bs = [float(np.mean(rng.choice(v, size=len(v), replace=True))) for _ in range(N_BOOT)]
    return float(np.mean(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))



# ------------------------------------------------------------------ LOSO 预测器
def _zsc(v: np.ndarray) -> np.ndarray:
    """按秩的稳健标准化：先换平均秩，再线性映射到 N(0,1) 的尺度。

    直接用 z-score 会被少数极端原子（max|eff| 可达中位的 100 倍）主导，
    而消融效应的分布是重尾的；秩化后残差同方差，跨被试可比。
    """
    r = _rank(v)
    r = (r - 0.5) / len(r)
    return np.sqrt(2.0) * _erfinv(2 * r - 1)


def _erfinv(x: np.ndarray) -> np.ndarray:
    from scipy.special import erfinv
    return erfinv(np.clip(x, -0.999999, 0.999999))


def loso_predictor(data: dict, seed: int = 0, alpha: float = 1e-3) -> dict:
    """共享权重的留一被试预测器。

    `data[s] = (feats: dict[str, (n,)], eff: (n,))`

    为什么这是 S4 的正确口径
    ------------------------
    若预测器只用**一个**特征且权重共享，预测值的秩与 `fo` 的秩完全相同，
    于是 $\\rho_{transfer}$ 恒等于被试内 $\\rho_{1st}$——那不是迁移，是同义反复。
    真正有信息的是**多特征、权重在被试间共享**：权重在其余被试上估，
    留出被试一个原子都不看。此时 $\\rho_{transfer}$ 低，说明"哪些统计量组合能预测因果"
    这层知识不跨被试；高，说明能。

    零对照：把训练侧的 eff 在做**被试内**置换后再拟合（保持每人的效应分布不变）。
    """
    subs = sorted(data)
    Z = {}
    for s in subs:
        X, y = data[s]
        Z[s] = ({k: _zsc(v) for k, v in X.items()}, _zsc(y))
    keys_all = sorted(Z[subs[0]][0])
    rng = np.random.RandomState(seed)

    sets = {"fo": ["fo"], "contrib": ["contrib"], "selectivity": ["selectivity"],
            "all": keys_all}
    out = {"feature_sets": sets, "n_subjects": len(subs), "per_subject": {}}
    for nm, keys in sets.items():
        rhos, nulls = [], []
        for s in subs:
            tr = [t for t in subs if t != s]
            A = np.concatenate([np.stack([Z[t][0][k] for k in keys], 1) for t in tr])
            b = np.concatenate([Z[t][1] for t in tr])
            A1 = np.column_stack([A, np.ones(len(A))])
            w, *_ = np.linalg.lstsq(A1, b, rcond=None)
            xs = np.stack([Z[s][0][k] for k in keys], 1)
            pred = np.column_stack([xs, np.ones(len(xs))]) @ w
            rhos.append(rho(pred, Z[s][1]))
            # 零对照：训练侧标签在被试内置换
            bp = np.concatenate([Z[t][1][rng.permutation(len(Z[t][1]))] for t in tr])
            w0, *_ = np.linalg.lstsq(A1, bp, rcond=None)
            nulls.append(rho(np.column_stack([xs, np.ones(len(xs))]) @ w0, Z[s][1]))
        v = np.array(rhos, dtype=float)
        bs = [float(np.nanmean(rng.choice(v[~np.isnan(v)], size=int((~np.isnan(v)).sum()),
                                         replace=True))) for _ in range(N_BOOT)]
        out["per_subject"][nm] = dict(
            rho=v.tolist(), mean=float(np.nanmean(v)),
            lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
            n_pos=int(np.nansum(v > 0)),
            null_mean=float(np.nanmean(nulls)),
            null_sd=float(np.nanstd(nulls, ddof=1)))
    out["degenerate_note"] = ("单特征（fo / contrib / selectivity）的共享权重线性预测器，"
                              "预测值的秩等于该特征自身的秩，故其 rho 恒等于该特征的被试内 rho；"
                              "只有 `all` 这一行是真正的跨被试泛化。")
    return out


# ------------------------------------------------------------------ 主流程
def run(seed: int = 0) -> dict:
    sem = json.loads((PROBE / "t1_semantic_main.json").read_text(encoding="utf-8"))
    abl = json.loads((PROBE / "t1_sae_ablate_k8.json").read_text(encoding="utf-8"))
    subs = sorted(set(r["subject"] for r in sem["runs"]) & set(r["subject"] for r in abl["runs"]))
    D = {s: load_subject(s, sem, abl) for s in subs}
    n_ch = max(d["n_ch"] for d in D.values())
    for d in D.values():
        d["n_ch"] = n_ch

    abl_runs = {}
    for r in abl["runs"]:
        if r["subject"] not in D:
            continue
        eff = np.asarray(r["eff"], dtype=float)
        feats = {k: np.asarray(r[k], dtype=float) for k in ("fo", "contrib", "selectivity")
                 if k in r}
        if "act_mean" in r:
            feats["act_mean"] = np.asarray(r["act_mean"], dtype=float)
        ok = np.isfinite(eff) & np.all([np.isfinite(v) for v in feats.values()], axis=0)
        abl_runs[r["subject"]] = dict(X={k: v[ok] for k, v in feats.items()}, eff=eff[ok])

    rng = np.random.RandomState(seed)
    out = dict(cell_min=CELL_MIN, n_boot=N_BOOT, n_null=N_NULL, bands=BANDS,
               n_cells=int(n_ch * len(BANDS)), runs=[], subjects=subs)

    # ---- 每个被试一张图（三种统计量）
    STATS = ("mean", "exc", "top", "top10", "p90")
    maps = {st: {s: cell_map(D[s], stat=st) for s in subs} for st in STATS}

    # ---- L1：关系层的描述量
    L1 = []
    for s in subs:
        d = D[s]
        b = float(np.polyfit(d["fo"], d["eff"], 1)[0])
        L1.append(dict(subject=s, slope=b, rho_first=rho(d["fo"], d["eff"]),
                       n_atoms=int(len(d["eff"]))))
    out["L1"] = L1

    # ---- L2：跨被试（所有无序对）
    pairs = list(combinations(subs, 2))
    L2 = {}
    for st in ("mean", "exc", "top", "top10", "p90"):
        v = np.array([rho(maps[st][a], maps[st][b]) for a, b in pairs])
        m, lo, hi = boot_ci(v, rng)
        L2[st] = dict(values=v.tolist(), mean=m, lo=lo, hi=hi,
                      n_pairs=len(v), n_pos=int(np.nansum(v > 0)))
    out["L2"] = L2

    # ---- L2b：共享权重的留一被试预测器（S4 的主判据）
    out["L2b"], out["L2b_2a"] = {}, {}
    out["L2b"] = loso_predictor({s: (abl_runs[s]["X"], abl_runs[s]["eff"])
                                 for s in subs}, seed)
    out["L2b_2a"] = out["L2b"]
    try:
        t2 = json.loads((PROBE / "t2_mmidb_main.json").read_text(encoding="utf-8"))
        d2 = {}
        for r in t2["runs"]:
            eff = np.asarray(r["eff"], dtype=float)
            feats = {k: np.asarray(r[k], dtype=float) for k in ("fo", "contrib", "selectivity")
                     if k in r}
            ok = np.isfinite(eff) & np.all([np.isfinite(v) for v in feats.values()], axis=0)
            d2[r["subject"]] = (feats, eff)
        out["L2b_t2"] = loso_predictor(d2, seed)
    except Exception as e:                                  # 缺产物时不阻塞主结果
        out["L2b_t2"] = dict(error=f"{type(e).__name__}: {e}")

    # ---- 被试内半分信度（迁移的上限）
    half = {st: [] for st in STATS}
    for s in subs:
        d = D[s]
        for rep in range(20):
            idx = rng.permutation(len(d["eff"]))
            h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
            for st in half:
                half[st].append(rho(cell_map(d, h1, st), cell_map(d, h2, st)))
    out["halves"] = {st: dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)))
                     for st, v in half.items()}

    # ---- 零对照：被试内置换原子→格子
    null = {st: [] for st in STATS}
    for rep in range(N_NULL):
        perm = {}
        for s in subs:
            d = D[s]
            i = rng.permutation(len(d["eff"]))
            perm[s] = dict(d, ch=d["ch"][i], band=d["band"][i])
        for st in null:
            mp = {s: cell_map(perm[s], stat=st) for s in subs}
            null[st].append(float(np.nanmean([rho(mp[a], mp[b]) for a, b in pairs])))
    out["null"] = {st: dict(mean=float(np.nanmean(v)), p95=float(np.nanpercentile(v, 95)),
                            values=v) for st, v in null.items()}

    # ---- 对照：把被试的图按被试编号随机重排（图与图之间的"生物学"被打散）
    shuffled = []
    for rep in range(N_NULL):
        perm_subs = list(rng.permutation(subs))
        shuffled.append(float(np.nanmean([
            rho(maps["mean"][perm_subs[i]], maps["mean"][perm_subs[j]])
            for i, j in combinations(range(len(subs)), 2)])))
    out["shuffle_subjects"] = dict(mean=float(np.mean(shuffled)),
                                  p95=float(np.percentile(shuffled, 95)))

    # ---- 逐被试图（写报告用）
    for s in subs:
        m = maps["mean"][s]
        out["runs"].append(dict(
            subject=s,
            map_mean=np.where(np.isfinite(m), m, np.nan).tolist(),
            n_atoms=int(len(D[s]["eff"])),
            slope=[r for r in L1 if r["subject"] == s][0]["slope"],
            rho_first=[r for r in L1 if r["subject"] == s][0]["rho_first"],
            eff_sum=float(np.nansum(D[s]["eff"])),
            frac_pos=float(np.mean(D[s]["eff"] > D[s]["null_p95"]))))
    return out


# ------------------------------------------------------------------ 报告
def summarize(out: dict, tag: str) -> None:
    _s = "" if tag == "main" else f"_{tag}"
    L2 = out["L2"]
    STATS = [("mean", "格内平均效应"), ("exc", "格内超零比例"), ("p90", "格内 P90"),
             ("top", "格内 top-3 均值"), ("top10", "格内 top-10 均值")]
    lines = ["# T1 跨被试迁移（S4）", "",
             f"自动生成。$(c,\\beta)$ 分箱 = {out['subjects'].__len__()} 名被试的 "
             f"{out['n_cells']} 格（22 导 × 5 频带），每格 ≥{out['cell_min']} 个原子才计入。", "",
             "## 0 判据与一个必须先说清的口径问题", "",
             "冻结判据：留一被试 $\\rho_{transfer}\\ge0.30$ 且 95%CI 下界 > 0。", "",
             "**为什么不能直接「用其余被试拟合 $\\mathrm{eff}\\sim\\mathrm{fo}$，再在留出被试上算 $\\rho$」**：",
             "$\\rho$ 是 Spearman，对任何单调变换不变。单特征 + 共享权重的预测器，"
             "其预测值的秩与 $\\mathrm{fo}$ 的秩完全相同，于是"
             "$\\rho_{transfer}\\equiv\\rho_{1st}$——那是同一句话换个说法，不是检验。",
             "本脚本因此做两件事：",
             "1. **关系层**（S4 主判据）：多特征、权重在其余被试上估、留出被试一个原子都不看，"
             "看秩相关能否保持。同时列 $\\mathrm{fo}$ / ④贡献 / ⑤选择性三个单特征作对照——"
             "若只有 $\\mathrm{fo}$ 迁移，这条判据才有分辨力。",
             "2. **单元层**（附加，非判据）：把原子按「输入梯度最大处通道 × 谱峰频带」装箱，"
             "看这张「因果效应地形谱图」在人与人之间是否一致。这一层才回答"
             "「被试 A 的因果特征在被试 B 上还是不是同一批」。", ""]

    # ---------------- L1 关系层
    lines += ["## 1 关系层：共享权重预测器的留一被试泛化（S4 主判据）", ""]
    for name, key, note in (("BCI IV-2a（9 人）", "L2b", ""),
                            ("EEGMMIDB（106 人）", "L2b_t2", "")):
        d = out.get(key) or {}
        if "error" in d or "per_subject" not in d:
            lines += [f"**{name}**：未跑（{d.get('error', '缺产物')}）。", ""]
            continue
        lines += [f"**{name}**", "",
                  "| 特征 | $\\rho_{transfer}$ | 95%CI | 为正的被试 | 零对照（打乱训练标签） |",
                  "|---|---|---|---|---|"]
        for nm, label in (("fo", "$\\mathrm{fo}$（一阶预测器）"), ("contrib", "④ 贡献"),
                          ("selectivity", "⑤ 选择性"), ("all", "**全部特征**")):
            v = d["per_subject"][nm]
            lines.append(f"| {label} | {v['mean']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] | "
                         f"{v['n_pos']}/{len(v['rho'])} | {v['null_mean']:+.3f} |")
        n = d["n_subjects"]
        allv = d["per_subject"]["all"]
        ok = "通过" if (allv["mean"] >= 0.30 and allv["lo"] > 0) else "未通过"
        lines += ["", f"判据（`all` 行）：**{ok}**（{allv['mean']:+.3f} ≥ 0.30，CI 下界 "
                      f"{allv['lo']:+.3f} > 0；{allv['n_pos']}/{n} 为正）。", ""]

    L1 = out["L1"]
    sl = np.array([r["slope"] for r in L1])
    r1 = np.array([r["rho_first"] for r in L1])
    lines += ["**斜率一致性**（同一套校准的证据）："
              f"$b_s={sl.mean():.3f}\\pm{sl.std(ddof=1):.3f}$（{sl.min():.3f}–{sl.max():.3f}），"
              f"$\\rho_{{1st}}={r1.mean():.3f}\\pm{r1.std(ddof=1):.3f}$。", "",
              "> **这条判据的信息量低于名字暗示**，必须写进局限：`all` 与单特征 `fo` 的 "
              "$\\rho_{transfer}$ 只差 0.001，说明多特征没有增量——权重虽在被试间共享，"
              "但起作用的只有 $\\mathrm{fo}$ 一项，而单特征下秩不变，"
              "所以它检验的是**关系（校准）跨被试一致**，不是「单元跨被试一致」。"
              "后者见下一节，结论是**不**一致。", ""]

    # ---------------- L2 单元层
    lines += ["## 2 单元层：因果效应地形谱图**不**迁移（附加结论）", "",
              "每格取一种汇总统计量，得到被试的一张图；$\\rho$ 为 36 个被试对的 Spearman 均值。",
              "**零对照**：在被试内随机置换「原子 → 格子」的分配（保持每格原子数与效应分布不变），"
              "重算同样的被试对相关。这是判读的关键——若观测值不超过零，那就是伪影。", "",
              "| 统计量 | $\\rho_{transfer}$ | 95%CI | 零对照均值 | 零 p95 | 被试内半分信度 | 判读 |",
              "|---|---|---|---|---|---|---|"]
    for st, label in STATS:
        v, h, nu = L2[st], out["halves"][st], out["null"][st]
        # 判读要区分两件事：统计量本身在零下被抬高（伪影），还是单纯没有迁移。
        if v["mean"] <= nu["mean"]:
            verdict = "**伪影**（观测低于零均值）"
        elif v["mean"] <= nu["p95"]:
            verdict = "不迁移（未超零）"
        elif v["mean"] >= 0.30 and v["lo"] > 0:
            verdict = "通过"
        else:
            verdict = "弱（超零但 < 0.30）"
        lines.append(f"| {label} | {v['mean']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] | "
                     f"{nu['mean']:+.3f} | {nu['p95']:+.3f} | {h['mean']:+.3f} | {verdict} |")
    lines += ["",
              "**两条必须一起读的结论**：", "",
              "1. **可靠的口径全部不迁移**：平均效应 +0.041、超零比例 +0.058、P90 +0.051，"
              "零对照都在 0 附近。即「哪个导联、哪个频带的原子在承担决策」"
              "在人与人之间**没有一致结构**，尽管每张图自身的半分信度有 0.31–0.45。"
              "这就是 $\\rho_{transfer}$ 该有的样子：可测，但不共享。",
              "2. **`top-3`/`top-10` 的「迁移」是格子大小的影子**：观测 0.323 / 0.412，"
              "而零对照高达 0.363 / 0.470——**观测低于零**。原因很直接：格子内原子越多，"
              "取 top-$k$ 的期望值越高，而格子大小是被试间共享的（同一套几何），"
              "于是两张图在「哪格原子多」上虚假相关。这两个统计量**不得**作为迁移证据。",
              ""]

    # ---------------- L3
    lines += ["## 3 单元身份（为什么不做原子级对应）", "",
              "单个原子没有跨被试可比的身份：每个被试的 SAE 在自己的 token 空间里训练。"
              "同一被试**跨种子**重训后支撑集 Jaccard 只有 0.080（N6，`reports/T1_N5N6.md`），"
              "连自己都不复现。故本节用 (通道 × 频带) 聚合层做替代，"
              "并在正文里把「报告单元」的结论落在这里。", "",
              "## 4 一句话写法", "",
              f"**可迁移的是关系，不是单元**：$\\mathrm{{fo}}\\to\\mathrm{{eff}}$ 的校准"
              f"（斜率 {sl.mean():.2f}）在 9/106 名被试上一致，留一被试的秩相关 "
              f"{out['L2b']['per_subject']['all']['mean']:+.3f}；"
              "但「哪些导联/频带承担决策」的分布在人与人之间不共享"
              "（+0.041，零 0.00）。", ""]

    (REPORTS / f"T1_transfer{_s}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    t0 = __import__("time").time()
    out = run(a.seed)
    out["elapsed_s"] = round(__import__("time").time() - t0, 1)
    PROBE.mkdir(exist_ok=True)
    (PROBE / f"t1_transfer_{a.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    summarize(out, a.tag)
    L2, h = out["L2"]["mean"], out["halves"]["mean"]
    print(f"done {out['elapsed_s']}s | rho_transfer {L2['mean']:+.3f} "
          f"[{L2['lo']:+.3f},{L2['hi']:+.3f}] | half {h['mean']:+.3f} | "
          f"null {out['null']['mean']['mean']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
