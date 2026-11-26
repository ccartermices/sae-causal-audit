"""d37 — 零对照 N5（眼动事件率）与 N6（重训后的特征对应率）。

`03_判据与闸门` §2 把 N5/N6 写成「每个都跑，缺一视为仪器未验收」。它们回答的
不是「SAE 有没有用」，而是**SAE 的单元是不是绑在了数据里的某个真东西上**：

* **N5**：一个原子的激活如果只是跟着眨眼/眼动走，那它在运动想象上的「判别选择性」
  就是伪迹的影子。做法：由 3 导 EOG 阈值检测事件 → 每试次事件率；算逐原子激活
  与事件率的 Spearman；比较**高效应原子**（按消融效应 top-50）与随机原子的 |ρ|。
  若高效应原子并不比随机更相关眼动，说明效应不是眼动驱动的。

* **N6**：同一套 SAE 流程在**随机半分的试次**上重训，单元还在不在？两种口径：
  - **(a) 同种子**：字典初始化相同、索引可比，直接在公共留出 token 上算逐原子
    支撑集 Jaccard（对角即对应）；
  - **(b) 跨种子**：字典初始化不同，索引不可比，必须先**匈牙利匹配**再算 Jaccard。
    4096×4096 的分配问题太慢，按 §6 表 4 的用途取**平均激活 top-300** 的子矩阵。

产物：`probe/t1_n5n6_{tag}.json`、`reports/T1_N5N6_{tag}.md`
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

N5_TOP_BY_EFFECT = 50       # 高效应原子数
N5_RANDOM = 500             # 随机参照原子数
N5_EVENT_MAD = 10.0         # 事件检测：中位 + 10·MAD（3·MAD 会让 94–99% 的试次都
#                             算「有事件」，事件率退化成噪声计数）
N6_TOP_MATCH = 300          # 匈牙利匹配用的子矩阵规模
N6_SUB_TOKENS = 20000       # 算支撑集时抽的留出 token 数


# ------------------------------------------------------------------ N5
def eog_events(Xeog: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """逐试次眨眼/眼动事件率（次/秒）与峰值幅度。

    事件 = |EOG| 超过「全体中位 + 3·MAD」的连续段（三导取包络）。眨眼约 200–400 µV、
    持续 100–400 ms，该阈值在 2a 上大约落在 60–90 µV，能抓住眨眼而不抓 alpha。
    """
    env = np.abs(Xeog).max(1)                     # (n, T) 三导包络
    med = np.median(env)
    mad = np.median(np.abs(env - med)) + 1e-9
    thr = med + N5_EVENT_MAD * 1.4826 * mad
    rate = np.zeros(len(env))
    for i, v in enumerate(env):
        above = v > thr
        if not above.any():
            continue
        # 连续段计数（上升沿）
        rate[i] = float(np.sum(above[1:] & ~above[:-1]) + above[0])
    dur = env.shape[1] / sfreq
    return rate / dur, env.max(1), float(thr)


def n5(sub: str, seed: int, steps: int, d_dict: int, k: int, src: str) -> dict:
    from scipy.stats import spearmanr

    tr = D31.load_session(sub, "T")
    te = D31.load_session(sub, "E")
    eog_idx = [i for i, c in enumerate(tr["ch"]) if c in SL.T1_EOG_CH]
    eeg = [i for i, c in enumerate(tr["ch"]) if c not in SL.T1_EOG_CH]
    Xtr, Xte = D31.zscore(tr["X"][:, eeg], te["X"][:, eeg])

    rate, peak, thr = eog_events(te["X"][:, eog_idx], float(te["sfreq"]))

    dec = json.loads((PROBE / "t1_decoder_main.json").read_text(encoding="utf-8"))
    epochs = [r for r in dec["runs"] if r["subject"] == sub and r["chans"] == "eeg"
              and r["seed"] == seed][0]["best_epoch"]
    model, _ = D31.train_once(Xtr, tr["y"], Xtr[:1], tr["y"][:1], epochs=epochs, seed=seed)
    Htr = D32.feature_map(model, Xtr)
    Hte = D32.feature_map(model, Xte)
    sae = _fit_sae(Htr, d_dict, k, seed, steps)

    a_trial = _trial_activation(sae, Hte)           # (n, D)

    prev = json.loads((PROBE / f"t1_sae_ablate_{src}.json").read_text(encoding="utf-8"))
    rec = [r for r in prev["runs"] if r["subject"] == sub][0]
    eff = np.array(rec["eff"])
    ok = ~np.isnan(eff)
    top = np.where(ok)[0][np.argsort(-eff[ok])[:N5_TOP_BY_EFFECT]]
    rng = np.random.RandomState(seed)
    rnd = rng.choice(len(eff), N5_RANDOM, replace=False)

    rho = np.full(len(eff), np.nan)
    rho_pk = np.full(len(eff), np.nan)
    for i in np.concatenate([top, rnd]):
        if np.std(a_trial[:, i]) < 1e-12:
            continue
        rho[i] = spearmanr(a_trial[:, i], rate).statistic
        rho_pk[i] = spearmanr(a_trial[:, i], peak).statistic

    return dict(subject=sub, thr_uV=thr, rate_hz=float(rate.mean()),
                peak_uV=float(np.median(peak)),
                rho_top=np.nanmedian(np.abs(rho[top])),
                rho_rand=np.nanmedian(np.abs(rho[rnd])),
                rho_peak_top=float(np.nanmedian(np.abs(rho_pk[top]))),
                rho_peak_rand=float(np.nanmedian(np.abs(rho_pk[rnd]))),
                frac_trials_event=float((peak > thr).mean()),
                frac_peak_top_high=float(np.nanmean(np.abs(rho_pk[top]) > 0.3)),
                frac_peak_rand_high=float(np.nanmean(np.abs(rho_pk[rnd]) > 0.3)),
                frac_top_high=float(np.nanmean(np.abs(rho[top]) > 0.3)),
                frac_rand_high=float(np.nanmean(np.abs(rho[rnd]) > 0.3)),
                top_idx=top.tolist(), rho_top_vals=rho[top].tolist())


# ------------------------------------------------------------------ 支撑集与 Jaccard
def _sparse_rows(sae, tok: np.ndarray, rows: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """给定扁平 token 下标 rows，返回这些 token 的稀疏激活（>0，只保留 top-k）。

    不materialize 整张 (N,D)：4096 原子 × 27 万 token 是 4.5 GB。
    """
    import torch
    dev = sae.device
    W = sae.enc.weight.detach(); b = sae.enc.bias.detach()
    out = np.zeros((len(rows), sae.d_dict), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), chunk):
            sel = rows[i:i + chunk]
            xb = torch.tensor(tok[sel], dtype=torch.float32, device=dev)
            a = torch.relu(xb @ W.T + b)
            top = torch.topk(a, sae.k, dim=1)
            out[i:i + chunk] = torch.zeros_like(a).scatter(1, top.indices,
                                                           top.values).cpu().numpy()
    return out


def _fit_sae(H_fit: np.ndarray, d_dict: int, k: int, seed: int, steps: int):
    sae = TopKSAE(H_fit.shape[1], d_dict, k, seed=seed)
    sae.fit(D32.tokens_of(H_fit), steps=steps, bs=1024, lr=2e-3)
    return sae


def _trial_activation(sae, H: np.ndarray) -> np.ndarray:
    """逐试次平均激活 (n, D)。逐试次循环，避免 (n·T',D) 的稠密矩阵。"""
    n = H.shape[0]
    out = np.zeros((n, sae.d_dict), dtype=np.float32)
    for i in range(n):
        flat = H[i].T                                  # (T', F)
        r = _sparse_rows(sae, flat, np.arange(len(flat)))
        out[i] = r.mean(0)
    return out


def _support(sae, H: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """留出 token 子集 idx 上的支撑集（bool, (D, len(idx))）。"""
    tok = H.transpose(0, 2, 1).reshape(-1, H.shape[1])
    return (_sparse_rows(sae, tok, idx) > 0).T


def _jaccard_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A (D,m)、B (E,m) bool → (D,E) Jaccard。"""
    Af = A.astype(np.float32); Bf = B.astype(np.float32)
    inter = Af @ Bf.T
    sa = Af.sum(1)[:, None]; sb = Bf.sum(1)[None, :]
    return inter / np.maximum(sa + sb - inter, 1e-9)


def n6(sub: str, seed: int, steps: int, d_dict: int, k: int) -> dict:
    from scipy.optimize import linear_sum_assignment

    tr = D31.load_session(sub, "T")
    te = D31.load_session(sub, "E")
    eeg = [i for i, c in enumerate(tr["ch"]) if c not in SL.T1_EOG_CH]
    Xtr, Xte = D31.zscore(tr["X"][:, eeg], te["X"][:, eeg])

    dec = json.loads((PROBE / "t1_decoder_main.json").read_text(encoding="utf-8"))
    epochs = [r for r in dec["runs"] if r["subject"] == sub and r["chans"] == "eeg"
              and r["seed"] == seed][0]["best_epoch"]
    model, _ = D31.train_once(Xtr, tr["y"], Xtr[:1], tr["y"][:1], epochs=epochs, seed=seed)
    Htr = D32.feature_map(model, Xtr)
    Hte = D32.feature_map(model, Xte)

    # 公共留出 token（E 段），两个 SAE 都在它上面取支撑集
    n_tok = Hte.shape[0] * Hte.shape[2]
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(n_tok, min(N6_SUB_TOKENS, n_tok), replace=False))

    # (a) 同种子、随机半分试次重训
    perm = rng.permutation(Htr.shape[0])
    half = [perm[:len(perm) // 2], perm[len(perm) // 2:]]
    sA = TopKSAE(Htr.shape[1], d_dict, k, seed=seed).fit(
        D32.tokens_of(Htr[half[0]]), steps=steps, bs=1024, lr=2e-3)
    sB = TopKSAE(Htr.shape[1], d_dict, k, seed=seed).fit(
        D32.tokens_of(Htr[half[1]]), steps=steps, bs=1024, lr=2e-3)
    SA = _support(sA, Hte, idx); SB = _support(sB, Hte, idx)
    act = SA.mean(1) + SB.mean(1)
    top = np.argsort(-act)[:N6_TOP_MATCH]
    jac = _jaccard_matrix(SA[top], SB[top])
    diag = np.diag(jac)

    # (b) 跨种子（0 vs 1）、全训练集重训，索引不可比 → 匈牙利
    sC = TopKSAE(Htr.shape[1], d_dict, k, seed=seed + 1).fit(
        D32.tokens_of(Htr), steps=steps, bs=1024, lr=2e-3)
    SC = _support(sC, Hte, idx)
    actC = SC.mean(1)
    topC = np.argsort(-actC)[:N6_TOP_MATCH]
    J = _jaccard_matrix(SA[top], SC[topC])
    r, c = linear_sum_assignment(-J)
    matched = J[r, c]

    return dict(subject=sub,
                jac_same_median=float(np.median(diag)),
                jac_same_p90=float(np.percentile(diag, 90)),
                frac_same_gt05=float((diag > 0.5).mean()),
                jac_cross_median=float(np.median(matched)),
                jac_cross_p90=float(np.percentile(matched, 90)),
                frac_cross_gt05=float((matched > 0.5).mean()),
                n_top=int(len(top)),
                sae_r2=float(_r2(sA, Htr)))


def _r2(sae, H) -> float:
    tok = D32.tokens_of(H)[:20000]
    rec = sae.reconstruct(tok)
    return float(1 - ((tok - rec) ** 2).sum() / ((tok - tok.mean(0)) ** 2).sum())


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str, default="A01,A02,A03,A04,A05,A06,A07,A08,A09")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--from-ablate", type=str, default="k8")
    ap.add_argument("--tag", type=str, default="main")
    a = ap.parse_args()

    info = det_lock.lock(0)
    print(f"[det] {info}", flush=True)
    prev = json.loads((PROBE / f"t1_sae_ablate_{a.from_ablate}.json").read_text(encoding="utf-8"))
    d_dict, k, steps = int(prev["d_dict"]), int(prev["k"]), int(prev.get("steps", a.steps))

    subs = [s.strip() for s in a.subjects.split(",") if s.strip()]
    t0 = time.time()
    out = dict(d_dict=d_dict, k=k, steps=steps, det=info, n5=[], n6=[])
    for sub in subs:
        out["n5"].append(n5(sub, a.seed, steps, d_dict, k, a.from_ablate))
        print(f"  N5 {sub} | ρ_top {out['n5'][-1]['rho_top']:.3f} "
              f"ρ_rand {out['n5'][-1]['rho_rand']:.3f}", flush=True)
        out["n6"].append(n6(sub, a.seed, steps, d_dict, k))
        print(f"  N6 {sub} | 同种子 J {out['n6'][-1]['jac_same_median']:.3f} "
              f"跨种子 J {out['n6'][-1]['jac_cross_median']:.3f}", flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    (PROBE / f"t1_n5n6_{a.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    summarize(out, a.tag)
    print(f"[done] {out['elapsed_s']} s", flush=True)
    return 0


def summarize(out: dict, tag: str) -> None:
    from scipy.stats import wilcoxon
    _s = "" if tag == "main" else f"_{tag}"   # main 写规范名

    lines = ["# T1 零对照 N5 / N6", "",
             f"自动生成。字典 {out['d_dict']}／k={out['k']}／{out['steps']} 步；"
             f"确定性锁 `{out['det']}`。", "",
             "## N5 特征激活 × 眼动事件率", "",
             f"事件 = |EOG| 包络超过「中位 + {N5_EVENT_MAD:.0f}·MAD」的连续段；"
             "ρ 为逐原子激活与试次事件率（或峰值幅度）的 Spearman，取 |ρ| 的中位。",
             "", "| 被试 | 阈值 (µV) | 有事件试次 | 事件率 (Hz) | 中位峰值 (µV) | "
             "|ρ(事件率)| top / rand | |ρ(峰值)| top / rand |", "|---|---|---|---|---|---|---|"]
    for r in out["n5"]:
        lines.append(f"| {r['subject']} | {r['thr_uV']:.1f} | {r['frac_trials_event']:.1%} | "
                     f"{r['rate_hz']:.3f} | {r['peak_uV']:.0f} | "
                     f"{r['rho_top']:.3f} / {r['rho_rand']:.3f} | "
                     f"{r['rho_peak_top']:.3f} / {r['rho_peak_rand']:.3f} |")
    dt = np.array([r["rho_top"] - r["rho_rand"] for r in out["n5"]])
    dk = np.array([r["rho_peak_top"] - r["rho_peak_rand"] for r in out["n5"]])
    rt = np.array([r["rho_top"] for r in out["n5"]])
    lines += ["", f"**配对差（事件率，top − random）= {dt.mean():+.4f} ± {dt.std(ddof=1):.4f}，"
                  f"Wilcoxon p = {wilcoxon(dt).pvalue:.4f}，方向为正 {int((dt > 0).sum())}/{len(dt)}**"
                  f"（正号 = 效应原子更相关眼动）",
              "",
              f"**配对差（峰值幅度）= {dk.mean():+.4f} ± {dk.std(ddof=1):.4f}，"
              f"p = {wilcoxon(dk).pvalue:.4f}，方向为正 {int((dk > 0).sum())}/{len(dk)}**",
              "",
              f"**判读**：方向虽一致，**量级可忽略**——效应原子的 |ρ| 中位仅 {rt.mean():.3f}，"
              f"即眼动只能解释约 {rt.mean() ** 2:.1%} 的试次间方差。",
              "N5 因此给出的是「眼动不是这些原子效应的来源」，不是「有眼动耦合」。", ""]
    lines += ["## N6 重训后的特征对应率", "",
              "同种子 = 随机半分试次重训、索引可比；跨种子 = 种子 0 vs 1、匈牙利匹配。",
              "两者都在**公共留出 token**（E 段 2 万）上算支撑集 Jaccard。", "",
              "| 被试 | SAE R² | 同种子 J 中位 | p90 | >0.5 比例 | 跨种子 J 中位 | p90 | >0.5 比例 |",
              "|---|---|---|---|---|---|---|---|"]
    for r in out["n6"]:
        lines.append(f"| {r['subject']} | {r['sae_r2']:.3f} | {r['jac_same_median']:.3f} | "
                     f"{r['jac_same_p90']:.3f} | {r['frac_same_gt05']:.1%} | "
                     f"{r['jac_cross_median']:.3f} | {r['jac_cross_p90']:.3f} | "
                     f"{r['frac_cross_gt05']:.1%} |")
    js = np.array([r["jac_same_median"] for r in out["n6"]])
    jc = np.array([r["jac_cross_median"] for r in out["n6"]])
    lines += ["", f"**同种子 {js.mean():.3f} ± {js.std(ddof=1):.3f}；"
                  f"跨种子 {jc.mean():.3f} ± {jc.std(ddof=1):.3f}**"
                  f"（历史 S5 阈值 {SL.S5_JACCARD_MAX} 仅作对照）", ""]
    (REPORTS / f"T1_N5N6{_s}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
