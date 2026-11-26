#!/usr/bin/env python
"""d27 — F：把事件坐标写进 token，看空间部分能否产生位置特异性。

诊断（D23 §7）
--------------
token = 「某 (试次, 帧, 频点) 上的 64 维空间模式」，SAE 逐 token 独立重建，
于是只能学到空间模式的**边缘分布**。「这个模式来自什么事件」是 token 之间的
**联合结构**，边缘里没有 —— 所以任何字典正则都刷不出位置特异性。

本脚本的直接检验：**把事件坐标显式拼进 token**，让「事件」变成可观测的输入。

    token = [ 空间模式(z 化, 64 维) , 帧 one-hot(Tf) , 频点 one-hot(F) ]

上下文里**不含任何关于注入位置的信息**（只有时间帧与频点），所以：
  - 若空间部分仍学不到 C3 ⇒ 特异性不是「信息缺失」造成的，另有原因
  - 若学得到 ⇒ 证明根因诊断成立，且给出修法（代价是可解释性依赖喂入的坐标）

两个关键对照
------------
  * **shuffled ctx**：把帧/频点 one-hot 逐 token 随机置换。若特异性还在，
    说明原子只是在记忆「token 索引」而非绑定事件。
  * **spatial_frac**：原子方向里空间部分的范数占比。若接近 0，说明原子被上下文
    主导，空间读数不可信。

输出
----
  probe/f_ctx_{tag}.json
  reports/T0_F_ctx_{tag}.md
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
import d25_e_l1_verify as D25   # noqa: E402
import d26_e_specificity as D26  # noqa: E402

SPEC_MIN = 0.15
SIGMA = 0.020
F0 = 10.0


def run(sae, cond, ctx, a, seed):
    sae_ = D24.RegSAE(sae["d_in"], a.sae_dict, a.sae_k, seed=seed,
                      ei=sae["ei"], ej=sae["ej"], groups=sae["groups"])
    Xtr = np.concatenate([cond["T"][cond["tr"]], ctx[cond["tr"]]], axis=1)
    Xte = np.concatenate([cond["T"][cond["te"]], ctx[cond["te"]]], axis=1)
    sae_.fit(Xtr, steps=a.sae_steps, lr=a.lr, reg=sae["reg"], lam=sae["lam"], tau=sae["tau"])
    dirs = sae_.atom_dirs().astype(np.float64)
    C = cond["C"]
    spat = dirs[:, :C]
    frac = np.linalg.norm(spat, axis=1) / (np.linalg.norm(dirs, axis=1) + 1e-12)
    Dm = spat - spat.mean(1, keepdims=True)
    Dm /= np.linalg.norm(Dm, axis=1, keepdims=True) + 1e-12
    tgt = float(np.abs(Dm @ cond["g"]).max())
    allc = D26.all_positions_cov(Dm, sae["P"], sae["sd_f"], SIGMA)
    idx, _ = sae_.encode_sparse(Xte)
    Xhat = sae_.reconstruct(Xte)
    r2 = 1 - float(((Xhat - Xte) ** 2).sum()) / max(
        float(((Xte - Xte.mean(0)) ** 2).sum()), 1e-12)
    return dict(cov=tgt, cov_median=float(np.median(allc)),
                cov_p90=float(np.percentile(allc, 90)),
                rank=int((allc > tgt).sum()) + 1, spec=tgt - float(np.median(allc)),
                r2=float(r2), spatial_frac=float(np.median(frac)),
                spatial_frac_min=float(frac.min()),
                alive=int((np.bincount(idx.ravel(), minlength=a.sae_dict) > 0).sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trial", type=int, default=1200)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--sae-dict", type=int, default=1024)
    ap.add_argument("--sae-k", type=int, default=16)
    ap.add_argument("--sae-steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-tokens", type=int, default=100_000)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--ctx", type=str, default="tf",
                    choices=["tf", "tf_shuf", "none", "trial"],
                    help="tf=帧+频点 one-hot；trial=试次级全局状态；tf_shuf=随机置换；none=无")
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)

    seeds = [int(v) for v in a.seeds.split(",") if v.strip()]
    X0, ch_names, fs = D22.load_trials(a.n_trial, 0)
    n = len(X0)
    is_inj = np.zeros(n, bool)
    is_inj[np.random.default_rng(0).permutation(n)[: n // 2]] = True
    P, d = D24.load_geometry(ch_names)
    w = np.exp(-(d ** 2) / (2 * SIGMA ** 2)); w /= w.max()
    sel = np.where(w > 1e-4)[0]
    cond = D25.build_condition(X0, is_inj, fs, w, sel, int(np.argmax(w)),
                               F0, a.snr, 0, a.max_tokens)
    ei, ej = D24.knn_edges(P, k=4)
    groups = D24.group_masks(ch_names)
    sd_f = cond["m"]["sd"][cond["fi"]]
    Tf, F = cond["m"]["Tf"], cond["m"]["F"]

    # 事件坐标：帧 one-hot + 频点 one-hot（不含任何位置信息）
    n_tok = len(cond["T"])
    tf_of, fbin_of = cond["m"]["tf_of"], cond["m"]["fbin_of"]
    if a.ctx == "none":
        ctx = np.zeros((n_tok, 0), np.float32)
    elif a.ctx == "trial":
        # 试次级全局状态：该试次在 f0 频点上的平均 64 维 z 模式，广播到它的全部 token。
        # 不含标签，是合法的无监督特征 —— 但它携带「本试次有没有 C3 的 10 Hz 爆发」。
        m_ = cond["m"]
        zf = ((cond["tok"].reshape(n, Tf, F, cond["C"])[:, :, cond["fi"], :]
               - m_["mu"][cond["fi"]]) / m_["sd"][cond["fi"]]).mean(1)   # (n, C)
        ctx = np.repeat(zf, Tf * F, axis=0).astype(np.float32)
    else:
        ctx = np.zeros((n_tok, Tf + F), np.float32)
        ctx[np.arange(n_tok), tf_of] = 1.0
        ctx[np.arange(n_tok), Tf + fbin_of] = 1.0
        if a.ctx == "tf_shuf":
            rng = np.random.default_rng(7)
            perm = rng.permutation(n_tok)
            ctx = ctx[perm]

    d_in = cond["C"] + ctx.shape[1]
    print(f"上下文 {a.ctx}：{ctx.shape[1]} 维 | "
          f"token 总维 {d_in} | 注入 N_eff {float(w.sum()**2/(w**2).sum()):.1f}", flush=True)

    base = dict(d_in=d_in, ei=ei, ej=ej, groups=groups, P=P, sd_f=sd_f)
    designs = [
        ("base+ctx", dict(base, reg="base", lam=0.0, tau=0.0)),
        ("l1_1.0+ctx", dict(base, reg="l1", lam=1.0, tau=0.0)),
        ("pr2+ctx", dict(base, reg="pr", lam=3.0, tau=2.0)),
        ("pr6+ctx", dict(base, reg="pr", lam=3.0, tau=6.0)),
    ]

    out = dict(ctx=a.ctx, n_trial=int(n), sigma=SIGMA, snr_db=float(a.snr),
               spec_min=SPEC_MIN, d_in=int(d_in), results=[], seed=0)
    print(f"\n{'设计':14s} {'目标cov':>8s} {'全脑中位':>8s} {'排名':>7s} {'特异性':>8s} "
          f"{'空间占比':>8s} {'R²':>7s}")
    for name, cfg in designs:
        rs = []
        for s in seeds:
            t0 = time.time()
            r = run(cfg, cond, ctx, a, s)
            rs.append(r)
            out["results"].append(dict(design=name, seed=s, **r))
            print(f"  {name:14s} seed={s} | {r['cov']:.2f} | 中位 {r['cov_median']:.2f} "
                  f"| 排 {r['rank']:2d} | spec {r['spec']:+.2f} | "
                  f"空间占比 {r['spatial_frac']:.2f} | R² {r['r2']:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        print(f"{name:14s} {np.mean([r['cov'] for r in rs]):8.2f} "
              f"{np.mean([r['cov_median'] for r in rs]):8.2f} "
              f"{np.median([r['rank'] for r in rs]):7.0f} "
              f"{np.mean([r['spec'] for r in rs]):+8.2f} "
              f"{np.mean([r['spatial_frac'] for r in rs]):8.2f} "
              f"{np.mean([r['r2'] for r in rs]):7.3f}", flush=True)

    tag = a.tag or a.ctx
    (ROOT / "probe").mkdir(exist_ok=True)
    jp = ROOT / "probe" / f"f_ctx_{tag}.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        f"# T0-F：把事件坐标写进 token（上下文 = `{a.ctx}`）",
        "",
        f"token = [空间 z 化 64 维, {ctx.shape[1]} 维事件坐标]。"
        "上下文**不含任何注入位置信息**（只有帧与频点）。",
        f"σ={SIGMA*1000:.0f} mm（$N_{{eff}}$≈3.1），SNR {a.snr:+.0f} dB，"
        f"字典 {a.sae_dict} × k={a.sae_k}，{a.sae_steps} 步，lr {a.lr:g}，"
        f"{len(seeds)} 种子均值。",
        "",
        f"判据：特异性 > {SPEC_MIN}（覆盖率减去全脑 64 位置的中位）。",
        "",
        "| 设计 | 目标 cov | 全脑中位 | 排名 | **特异性** | 空间占比 | R² | 判定 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, _cfg in designs:
        rs = [r for r in out["results"] if r["design"] == name]
        spec = np.mean([r["spec"] for r in rs])
        lines.append(
            f"| {name} | {np.mean([r['cov'] for r in rs]):.2f} | "
            f"{np.mean([r['cov_median'] for r in rs]):.2f} | "
            f"{np.median([r['rank'] for r in rs]):.0f}/64 | **{spec:+.2f}** | "
            f"{np.mean([r['spatial_frac'] for r in rs]):.2f} | "
            f"{np.mean([r['r2'] for r in rs]):.3f} | "
            f"{'**PASS**' if spec > SPEC_MIN else '—'} |")
    rp = ROOT / "reports" / f"T0_F_ctx_{tag}.md"
    rp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写出 {jp.name} / {rp.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
