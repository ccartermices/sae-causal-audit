"""d20 —— T0 仪器校准（S0 闸门的唯一判据来源）。

为什么先做这个
--------------
S0 是整篇的一票否决项：如果仪器在「答案毫无争议」的 SSVEP 上都找不回
「枕区 + 刺激频率」这两个已知事实，那么 T1/T2 上关于运动想象的任何结论都不成立。
所以先跑它，而且判据按 `03_判据与闸门` v1.0-final 冻结版执行，不在这里调。

流程
----
1. 10 名被试 × 240 试次，按 block 切分：block 0–4 训练、block 5 测试（严格留出）。
2. 训 40 类解码器（ShallowConvNet）。
3. 取池化前 conv 的 40 维 token，训 TopK SAE（D=4096, k=32）。
4. 对每个刺激频率 f：该频率试次上的平均原子激活取 top-5，检查
   (a) 原子的通道地形 argmax 是否落在预注册枕区集合（≥3/5 记为命中）；
   (b) 原子的「偏好频率」与 f 的偏差是否 ≤ 0.4 Hz。

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


def load_ssvep(n_subjects: int = 10):
    files = sorted((EPO / "ssvep").glob("*.npz"),
                   key=lambda p: int(re.sub(r"\D", "", p.stem)))
    Xs, ys, bs = [], [], []
    for p in files[:n_subjects]:
        z = np.load(p)
        Xs.append(z["X"]); ys.append(z["y"]); bs.append(z["block"])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(bs)


def ssvep_ch_names() -> list[str]:
    """从 64-channels.loc 解析导联名，规范成 10-05 写法（OZ→Oz）。"""
    out = []
    for line in (ROOT / "data" / "raw" / "ssvep" / "64-channels.loc").read_text(
            encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 4:
            n = parts[-1]
            out.append(n[0].upper() + n[1:].lower() if len(n) > 1 else n.upper())
    return out


def build_model(n_times: int, n_out: int):
    from braindecode.models import ShallowFBCSPNet
    return ShallowFBCSPNet(n_chans=64, n_outputs=n_out, n_times=n_times)


def train_model(model, Xtr, ytr, Xte, yte, epochs=80, bs=64, lr=1e-3):
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            xb, yb = Xt[j].to(dev), yt[j].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f"    ep {ep + 1}/{epochs}", flush=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 128):
            xb = torch.tensor(Xte[i:i + 128], dtype=torch.float32, device=dev)
            preds.append(model(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    return model, float((pred == yte).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--sae-steps", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token-trials", type=int, default=800,
                    help="用于 token/SAE 的训练试次数上限")
    a = ap.parse_args()
    t0 = time.time()

    X, y, blk = load_ssvep(a.n_subjects)
    tr, te = blk < 5, blk == 5
    print(f"SSVEP: {len(X)} 试次 | 训练 {tr.sum()} 测试 {te.sum()} | 40 类")

    mu = X[tr].mean((0, 2), keepdims=True)
    sd = X[tr].std((0, 2), keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    model, acc = train_model(build_model(X.shape[2], 40), Xn[tr], y[tr], Xn[te], y[te],
                             epochs=a.epochs)
    print(f"解码器 40 类准确率 {acc:.3f}（{time.time() - t0:.0f}s）")

    # ---- token 与 SAE
    rng0 = np.random.RandomState(a.seed)
    tr_idx = np.where(tr)[0]
    if a.token_trials and len(tr_idx) > a.token_trials:
        tr_idx = rng0.choice(tr_idx, a.token_trials, replace=False)
    tok_tr, topo = TS.tokens_from_model(model, Xn[tr_idx])
    T, F = tok_tr.shape[1], tok_tr.shape[2]   # (n, T', F)
    flat = tok_tr.reshape(-1, F)
    rng = np.random.RandomState(a.seed)
    if len(flat) > a.max_tokens:
        flat = flat[rng.choice(len(flat), a.max_tokens, replace=False)]
    print(f"token: {F} 维 × {T} 个/试次；训练用 {len(flat)} 个")

    sae = TS.TopKSAE(F, SL.SAE_DICT, SL.SAE_K, seed=a.seed)
    sae.fit(flat, steps=a.sae_steps, bs=512)
    rec = sae.reconstruct(flat[:20000])
    r2_tok = TS.r2(flat[:20000], rec)
    print(f"SAE 重构 R2={r2_tok:.4f}（{len(flat)} token 训练）")

    # ---- 测试集的原子–试次激活
    tok_te, _ = TS.tokens_from_model(model, Xn[te])
    n_te = tok_te.shape[0]
    idx, val = sae.encode_sparse(tok_te.reshape(-1, F))
    trial_of = np.repeat(np.arange(n_te), T)
    A = TS.atom_trial_activation(idx, val, trial_of, n_te, SL.SAE_DICT)

    # ---- S0
    ch = ssvep_ch_names()
    occ = {c.lower() for c in SL.ROI_OCCIPITAL}
    topo_n = topo / (np.abs(topo).max(1, keepdims=True) + 1e-9)
    dirs = sae.atom_dirs()                       # (D_dict, F)
    atom_topo = dirs @ topo_n                    # (D_dict, C)
    peak_ch = np.abs(atom_topo).argmax(1)

    y_te = y[te]
    # 真实频率表（分组 JFPM），不是 8+0.2i
    freq_map = np.load(EPO / "ssvep" / "S1.npz")["freq"]
    freq_of_atom, rows = [], []
    for f_idx in range(40):
        m = y_te == f_idx
        if m.sum() == 0:
            continue
        mean_act = A[m].mean(0)
        top5 = np.argsort(mean_act)[::-1][:5]
        hits = int(sum(ch[peak_ch[i]].lower() in occ for i in top5))
        prof = np.array([A[y_te == g][:, top5].mean() for g in range(40)])
        pref = int(np.argmax(prof))                    # 该原子组最偏好的刺激频率
        f_err = abs(float(freq_map[pref]) - float(freq_map[f_idx]))
        freq_of_atom.append(dict(target=f_idx, freq=round(float(freq_map[f_idx]), 2),
                                 hits=hits, pref_freq=round(float(freq_map[pref]), 2),
                                 freq_err=round(float(f_err), 2),
                                 atoms=[ch[peak_ch[i]] for i in top5]))
        rows.append((f_idx, hits, f_err))

    hits_arr = np.array([r[1] for r in rows])
    err_arr = np.array([r[2] for r in rows])
    n_pass_occ = int((hits_arr >= SL.S0_MIN_OCCIPITAL_HITS).sum())
    n_pass_freq = int((err_arr <= SL.SSVEP_FREQ_TOL).sum())
    s0_pass = bool(np.median(hits_arr) >= SL.S0_MIN_OCCIPITAL_HITS
                   and np.median(err_arr) <= SL.SSVEP_FREQ_TOL)

    out = dict(n_subjects=a.n_subjects, n_train=int(tr.sum()), n_test=int(te.sum()),
               decoder_acc=round(acc, 4), token_dim=int(F), tokens_per_trial=int(T),
               n_tokens_sae=int(len(flat)), sae_dict=SL.SAE_DICT, sae_k=SL.SAE_K,
               sae_r2=round(float(r2_tok), 4),
               s0_median_hits=float(np.median(hits_arr)),
               s0_median_freq_err=float(np.median(err_arr)),
               s0_targets_passing_occipital=n_pass_occ,
               s0_targets_passing_freq=n_pass_freq,
               s0_pass=s0_pass, per_target=freq_of_atom,
               ch_names=ch, seconds=round(time.time() - t0, 1))
    (ROOT / "probe").mkdir(exist_ok=True)
    (ROOT / "probe" / "t0_s0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# T0 · S0 仪器校准（SSVEP）", "",
         "> 由 `scripts/d20_t0_ssvep.py` 生成，勿手改。判据取 `03_判据与闸门` v1.0-final。", "",
         f"- 数据：{a.n_subjects} 名被试，{out['n_train']} 训练 / {out['n_test']} 测试（按 block 留出）",
         f"- 解码器 40 类准确率 **{acc:.3f}**", "",
         "| 项 | 实测 | 判据 | 结论 |", "|---|---|---|---|",
         f"| token 维 / 每试次 | {F} / {T} | — | — |",
         f"| SAE 重构 R² | {r2_tok:.4f} | — | — |",
         f"| top-5 枕区命中中位数 | {np.median(hits_arr):.1f} | ≥3 | "
         f"{'通过' if np.median(hits_arr) >= SL.S0_MIN_OCCIPITAL_HITS else '不通过'} |",
         f"| 谱峰偏差中位数 | {np.median(err_arr):.2f} Hz | ≤0.4 | "
         f"{'通过' if np.median(err_arr) <= SL.SSVEP_FREQ_TOL else '不通过'} |",
         f"| 40 个目标里两项都过 | {n_pass_occ} / {n_pass_freq} | — | — |", "",
         f"**S0 判定：{'通过' if s0_pass else '不通过'}**", "",
         "## 逐目标", "", "| 目标 | 频率 (Hz) | 枕区命中 | 偏好频率 | 偏差 (Hz) | top-5 通道 |",
         "|---|---|---|---|---|---|"]
    for r in freq_of_atom:
        L.append(f"| {r['target']} | {r['freq']} | {r['hits']} | {r['pref_freq']} | "
                 f"{r['freq_err']} | {', '.join(r['atoms'])} |")
    (ROOT / "reports" / "T0_S0.md").write_text("\n".join(L), encoding="utf-8")
    print(f"S0 {'通过' if s0_pass else '不通过'} | 枕区中位 {np.median(hits_arr):.1f} "
          f"| 频率偏差中位 {np.median(err_arr):.2f} Hz | 用时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
