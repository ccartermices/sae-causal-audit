"""d35 — 把 T1 的三张表按 D30 的分层合并，并检验层内符号一致性。

为什么需要单独一步
------------------
T1_SAE / T1_joint / T1_S2 的明细表是**逐被试**的。9 个被试直接求均值会
把符号相反的两层（EEG 主导 vs EOG 主导）互相抵消——D33 已实测到
④/⑤ 选择器在 A07 上是负的。因此正文只能用**分层值**，合并值只能作为
「没有普适效应」的反证。

层定义（D30，由解码器的 κ 比值定，与 SAE 无关）
-----------------------------------------------
r = κ(仅 3 导 EOG) / κ(22 导 EEG)
* r ≥ 1.0 → **EOG 主导**
* r ≤ 0.7 → **EEG 主导**
* 其余 → **接近**

这里报的量
----------
1. 单原子：最大效应 / 最大翻转率 / 超零原子数（按层均值±SD）。
2. 联合（k=8）：④/⑤/一阶/causal_top/causal_bottom 的 Δbacc 按层均值±SD，
   以及「符号与层内多数一致」的被试数。
3. S2：π(EOG) / π(额区) 与 π_art 的层均值。
4. 层间比较：EEG 主导 vs EOG 主导的 Mann-Whitney（人数少，只作描述）。

产物：`reports/T1_layers.md`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

PROBE = ROOT / "probe"
REPORTS = ROOT / "reports"

R_EOG = 1.0      # r ≥ 此值 → EOG 主导
R_EEG = 0.7      # r ≤ 此值 → EEG 主导

LAYERS = ("EEG 主导", "接近", "EOG 主导")


def load(name: str) -> dict:
    return json.loads((PROBE / name).read_text(encoding="utf-8"))


def layer_of(kappa_eeg: float, kappa_eog: float) -> str:
    r = kappa_eog / max(kappa_eeg, 1e-9)
    if r >= R_EOG:
        return "EOG 主导"
    if r <= R_EEG:
        return "EEG 主导"
    return "接近"


def fmt(v: float, nd: int = 3, sign: bool = True) -> str:
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def msd(v, nd: int = 3, sign: bool = True) -> str:
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return "—"
    if len(v) == 1:
        return fmt(v[0], nd, sign)
    return f"{fmt(v.mean(), nd, sign)} ± {v.std(ddof=1):.{nd}f}"


def main() -> int:
    dec = load("t1_decoder_main.json")
    abl = load("t1_sae_ablate_k8.json")
    jnt = load("t1_joint_k8.json")
    s2 = load("t1_s2_main.json")

    # --- 每被试的 κ（EEG / EOG），用种子 0，与 SAE 分析同一模型
    kap = {}
    for r in dec["runs"]:
        if r["seed"] == 0:
            kap.setdefault(r["subject"], {})[r["chans"]] = r["kappa"]
    lay = {s: layer_of(k["eeg"], k["eog"]) for s, k in kap.items() if "eeg" in k and "eog" in k}

    subs = sorted(lay)
    by_layer = {L: [s for s in subs if lay[s] == L] for L in LAYERS}

    # --- 单原子
    abl_s = {r["subject"]: r for r in abl["runs"]}
    jnt_s = {r["subject"]: r for r in jnt["runs"]}
    s2_s = {r["subject"]: r for r in s2["runs"]}

    lines = ["# T1 分层汇总（9 被试）", "",
             "自动生成。层由**解码器**的 κ 比值定（D30），与 SAE 无关：",
             f"`r = κ(仅 3 导 EOG) / κ(22 导 EEG)`；r ≥ {R_EOG} → EOG 主导，"
             f"r ≤ {R_EEG} → EEG 主导，其余为接近。", "",
             "## 0 分层名单", "",
             "| 被试 | κ(EEG) | κ(EOG) | r | 层 |", "|---|---|---|---|---|"]
    for s in subs:
        k = kap[s]
        r = k["eog"] / max(k["eeg"], 1e-9)
        lines.append(f"| {s} | {fmt(k['eeg'])} | {fmt(k['eog'])} | {r:.2f} | **{lay[s]}** |")
    lines.append("")

    # --- 1 单原子
    lines += ["## 1 单原子消融（k=8）", "",
              "| 层 | n | 最大效应 | 最大翻转率 | 超零原子数（噪声期望 205） |",
              "|---|---|---|---|---|"]
    for L in LAYERS:
        rows = [abl_s[s] for s in by_layer[L] if s in abl_s]
        if not rows:
            continue
        eff_mx = [float(np.nanmax(np.abs(np.array(r["eff"])))) for r in rows]
        fl_mx = [float(np.nanmax(np.array(r["flip"]))) for r in rows]
        hi = []
        for r in rows:
            e = np.array(r["eff"]); ok = ~np.isnan(e); e = e[ok]
            nl = np.array(r["null_eff"])
            hi.append(int((e > np.percentile(nl, 95)).sum()))
        lines.append(f"| {L} | {len(rows)} | {msd(eff_mx)} | {msd(fl_mx, 3, False)} | "
                     f"{msd(hi, 0, False)} / 4096 |")
    lines.append("")

    # --- 2 联合消融
    names = ["contrib", "selectivity", "first_order", "causal_top", "causal_bottom"]
    lines += ["## 2 联合消融 top-50（k=8）：Δbacc，按层", "",
              "| 层 | n | random | " + " | ".join(names) + " |",
              "|---|---|---|" + "---|" * len(names)]
    per_layer = {}
    for L in LAYERS:
        rows = [jnt_s[s] for s in by_layer[L] if s in jnt_s]
        if not rows:
            continue
        d = {}
        d["random"] = [c["d_acc"] for r in rows for c in r["conds"] if c["sel"].startswith("random")]
        cells = [msd(d["random"])]
        for nm in names:
            v = np.array([np.mean([c["d_acc"] for c in r["conds"] if c["sel"] == nm]) for r in rows])
            d[nm] = v
            cell = fmt(v.mean())
            if len(v) > 1:
                npos = int((v > 0).sum())
                cell += f"（正 {npos}/{len(v)}）"
            cells.append(cell)
        per_layer[L] = d
        lines.append(f"| {L} | {len(rows)} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- 2b 配对检验：选择器 vs random（同被试配对，9 人）
    lines += ["### 2b 选择器 vs 随机（同被试配对）", ""]
    try:
        from scipy.stats import wilcoxon
    except Exception:                                    # pragma: no cover
        wilcoxon = None
    rnd = np.array([np.mean([c["d_acc"] for c in jnt_s[s]["conds"]
                             if c["sel"].startswith("random")]) for s in subs if s in jnt_s])
    lines += ["| 选择器 | 均值 Δ | 与 random 的配对差 | 正/负 | Wilcoxon p |",
              "|---|---|---|---|---|"]
    for nm in ["contrib", "selectivity", "first_order", "causal_top", "causal_bottom"]:
        v = np.array([np.mean([c["d_acc"] for c in jnt_s[s]["conds"] if c["sel"] == nm])
                      for s in subs if s in jnt_s])
        dd = v - rnd
        p = "—"
        if wilcoxon is not None and len(dd) >= 6:
            p = f"{wilcoxon(dd).pvalue:.4f}"
        lines.append(f"| `{nm}` | {fmt(v.mean())} | {fmt(dd.mean())} ± {dd.std(ddof=1):.3f} | "
                     f"{int((dd > 0).sum())}/{int((dd < 0).sum())} | {p} |")
    lines += ["", "> 这是唯一有推断意义的检验：9 名被试、同被试配对、只换选择器。"
              "`causal_top` 用 E 段标签选原子，是上界不是方法。", ""]

    # --- 3 S2
    lines += ["## 3 S2 伪迹拆分（k=8）", "",
              "| 层 | n | π(EOG) | π(额区) | π_art(EOG) | π_art(额区) | 主 acc | 替代 acc(EOG) |",
              "|---|---|---|---|---|---|---|---|"]
    for L in LAYERS:
        rows = [s2_s[s] for s in by_layer[L] if s in s2_s]
        if not rows:
            continue
        pe = [r["stat"]["eog"]["raw"] for r in rows]
        pf = [r["stat"]["frontal"]["raw"] for r in rows]
        ae = [float(np.mean([1.0 if x["pi_raw_eog"] >= 0.5 else 0.0 for x in r["rows"]])) for r in rows]
        af = [float(np.mean([1.0 if x["pi_raw_frontal"] >= 0.5 else 0.0 for x in r["rows"]])) for r in rows]
        ma = [r["main_acc"] for r in rows]
        sa = [r["sub_acc"]["eog"] for r in rows]
        lines.append(f"| {L} | {len(rows)} | {msd(pe, 3, False)} | {msd(pf, 3, False)} | "
                     f"{msd(ae, 2, False)} | {msd(af, 2, False)} | {msd(ma, 3, False)} | "
                     f"{msd(sa, 3, False)} |")
    lines.append("")

    # --- 4 层间检验
    lines += ["## 4 层间比较（EEG 主导 vs EOG 主导）", ""]
    try:
        from scipy.stats import mannwhitneyu
    except Exception:                                    # pragma: no cover
        mannwhitneyu = None
    a = by_layer["EEG 主导"]; b = by_layer["EOG 主导"]
    lines += ["| 量 | EEG 主导 | EOG 主导 | p（Mann-Whitney） |", "|---|---|---|---|"]
    rows_cmp = []
    for nm in names:
        va = np.array([np.mean([c["d_acc"] for c in jnt_s[s]["conds"] if c["sel"] == nm])
                       for s in a if s in jnt_s])
        vb = np.array([np.mean([c["d_acc"] for c in jnt_s[s]["conds"] if c["sel"] == nm])
                       for s in b if s in jnt_s])
        rows_cmp.append((nm, va, vb))
    for nm, va, vb in rows_cmp:
        p = "—"
        if mannwhitneyu is not None and len(va) >= 2 and len(vb) >= 2:
            p = f"{mannwhitneyu(va, vb, alternative='two-sided').pvalue:.3f}"
        lines.append(f"| ④/⑤ 选择器 `{nm}` Δbacc | {msd(va)} | {msd(vb)} | {p} |")
    for nm, lab in (("eog", "π(EOG)"), ("frontal", "π(额区)")):
        va = np.array([s2_s[s]["stat"][nm]["raw"] for s in a if s in s2_s])
        vb = np.array([s2_s[s]["stat"][nm]["raw"] for s in b if s in s2_s])
        p = "—"
        if mannwhitneyu is not None and len(va) >= 2 and len(vb) >= 2:
            p = f"{mannwhitneyu(va, vb, alternative='two-sided').pvalue:.3f}"
        lines.append(f"| S2 {lab} | {msd(va, 3, False)} | {msd(vb, 3, False)} | {p} |")
    lines += ["", "> 每组只有 3 人时 Mann-Whitney 的最小可能 p 为 0.10，"
              "因此 p 值只能当描述量，不能当推断。", ""]

    (REPORTS / "T1_layers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
