"""d32 — T1 观察性分析：真实标签下的 SAE 特征、消融效应与零分布。

与 T0/T1 边界的区别
--------------------
T0-C/E/G 与 T1 边界都靠**注入已知源**：那里真值已知，测的是仪器能力。
本脚本换成**真实标签**（2a 四类运动想象），问的是论文的主问题：
SAE 的某个特征被消掉，解码能力掉多少？掉得多的那些特征，是不是「自称」解释得
最清楚的那些（语义得分，留给 d33）？

流程
----
1. 用 d31 选定的训练轮数，在 T 段重训解码器（确定性已锁）。
2. 取 `bnorm` 之后的特征图当 token：每试次 T' 个 40 维 token。
3. 在 **T 段** token 上训 TopK SAE（字典 1024，k=8）。
4. 在 **E 段**逐个原子消融：只减该原子激活最强的 m 个时刻的贡献 $a_i d_i$（D15-2），
   测真实类 margin 掉多少。
5. 对照：
   * **一阶预测器**（S3 的锚）：grad_h(margin) 与 (a_i d_i) 在 top-m 上的内积；
   * **N2 同范数随机方向**：同样的幅度剖面、随机方向，效应应居中于 0。

产物：`probe/t1_sae_ablate_{tag}.json`、`reports/T1_SAE_{tag}.md`
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

PROBE = ROOT / "probe"
REPORTS = ROOT / "reports"


# ------------------------------------------------------------------ 特征图
def feature_map(model, X, bs: int = 64) -> np.ndarray:
    """bnorm 之后的特征图：(n, F, T')。"""
    import torch
    dev = next(model.parameters()).device
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs], dtype=torch.float32, device=dev)
            h = model.bnorm(model.conv_time_spat(
                model.dimshuffle(model.ensuredims(xb))))
            outs.append(h.squeeze(-1).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def tokens_of(H: np.ndarray) -> np.ndarray:
    """(n, F, T') → (n·T', F)，行是 token。"""
    return H.transpose(0, 2, 1).reshape(-1, H.shape[1])


# ------------------------------------------------------------------ 消融
def tail_logits(model, h):
    """从 bnorm 输出往后跑（braindecode 的 forward 就是按子模块顺序串行）。"""
    names = [n for n, _ in model.named_children()]
    mods = list(model.children())[names.index("bnorm") + 1:]
    x = h
    for m in mods:
        x = m(x)
    return x


class Ablater:
    """单原子消融：只减每个试次里该原子激活最强的 m 个时刻（D15-2）。

    实现要点：不重跑前段卷积。特征图 H 只算一次，逐原子只需
    a_i = ReLU(tok · w_i + b_i)（一次内积），再走尾部。
    """

    def __init__(self, model, sae, H: np.ndarray, m: int):
        import torch
        self.model, self.sae, self.m = model, sae, m
        self.device = next(model.parameters()).device
        self.H = torch.tensor(H, dtype=torch.float32, device=self.device)  # (n,F,T')
        self.n, self.F, self.Tp = self.H.shape
        self.tok = self.H.transpose(1, 2).contiguous()                     # (n,T',F)
        self.W = sae.enc.weight.detach()                                   # (D,F)
        self.b = sae.enc.bias.detach()                                     # (D,)

    def logits(self, atom: int, direction=None, chunk: int = 128) -> np.ndarray:
        import torch
        d = (self.sae.atom_dirs()[atom] if direction is None else direction)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        m = min(self.m, self.Tp)
        outs = []
        with torch.no_grad():
            for i in range(0, self.n, chunk):
                tk = self.tok[i:i + chunk]                                  # (b,T',F)
                ai = torch.relu(tk @ self.W[atom] + self.b[atom])            # (b,T')
                top = torch.topk(ai, m, dim=1)
                tk = tk.clone()
                tk.scatter_add_(1, top.indices.unsqueeze(-1).expand(-1, -1, self.F),
                                -top.values.unsqueeze(-1) * d)
                h = tk.transpose(1, 2).unsqueeze(-1)                        # (b,F,T',1)
                outs.append(tail_logits(self.model, h).cpu().numpy())
        return np.concatenate(outs)


def margin(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = logits.copy()
    z[np.arange(len(y)), y] = -np.inf
    return logits[np.arange(len(y)), y] - z.max(1)


def grad_at_h(model, X, y, bs: int = 64):
    """返回 ∇_h margin（bnorm 输出处的梯度），(n, F, T')。"""
    import torch
    dev = next(model.parameters()).device
    gs = []
    model.eval()
    cache = {}

    def hook(m, i, o):
        o.retain_grad()
        cache["h"] = o

    hd = model.bnorm.register_forward_hook(hook)
    try:
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs], dtype=torch.float32, device=dev)
            yb = torch.tensor(y[i:i + bs], dtype=torch.long, device=dev)
            logit = model(xb)
            other = logit.clone()
            other[torch.arange(len(yb)), yb] = -1e9
            mi = logit[torch.arange(len(yb)), yb] - other.max(1).values
            model.zero_grad(set_to_none=True)
            mi.sum().backward()
            gs.append(cache["h"].grad.squeeze(-1).cpu().numpy().copy())
    finally:
        hd.remove()
    return np.concatenate(gs)


def first_order(sae, H: np.ndarray, G: np.ndarray, m: int = 8) -> np.ndarray:
    """一阶预测：$\\Delta\\mathrm{margin}\\approx\\sum_t a_i(t)\\,(\\nabla_h(t)\\cdot d_i)$，
    与消融口径严格对齐（也只在 $a_i$ 的 top-m 时刻求和），否则两者不可比。

    **$a_i$ 必须由 token $H$ 算，梯度 $G$ 只用于方向投影。**
    早期版本把 $G$ 直接喂进 `sae.enc` 求激活，梯度尺度比 token 小两个数量级
    （实测 0.0034 vs 0.89），`relu(enc(G))` 有 99.999% 为 0——
    $\\mathrm{fo}$ 退化成几乎全零的常向量，选择器实际等于「取前 50 个原子」。
    见 D35。
    """
    import torch
    D = sae.d_dict
    dev = sae.device
    dirs = sae.atom_dirs()                                           # (D, F)
    pred = np.zeros(D, dtype=np.float64)
    for h, g in zip(H, G):                                           # h (F,T'), g (F,T')
        with torch.no_grad():
            ht = torch.tensor(h.T, dtype=torch.float32, device=dev)    # (T', F)
            a = torch.relu(sae.enc(ht)).cpu().numpy()                 # (T', D)
        gg = g.T @ dirs.T                                             # (T', D) 每 token 每原子
        m2 = min(m, a.shape[0])
        top = np.argpartition(-a, m2 - 1, axis=0)[:m2]                # (m, D)
        av = np.take_along_axis(a, top, axis=0)
        gv = np.take_along_axis(gg, top, axis=0)
        pred += (av * gv).sum(0)
    return pred / len(G)


# ------------------------------------------------------------------ 特征统计
def feature_stats(a: np.ndarray, y_tok: np.ndarray, dirs: np.ndarray) -> dict:
    """①②③④⑤（⑥ 自解释得分在 d33 做，需要权重剖面）。"""
    D = a.shape[1]
    act_mean = a.mean(0)
    act_freq = (a > 0).mean(0)
    act_mag = np.array([np.median(v[v > 0]) if (v > 0).any() else 0.0 for v in a.T])
    dnorm = np.linalg.norm(dirs, axis=1)                            # (D,)
    contrib = (np.abs(a).mean(0)) * dnorm                           # 每原子的平均重建幅度
    contrib = contrib / max(contrib.sum(), 1e-12)
    cls = np.unique(y_tok)
    sel = np.zeros(D)
    for i in range(D):
        v = a[:, i]
        grand = v.mean()
        between = np.mean([(v[y_tok == c].mean() - grand) ** 2 for c in cls])
        within = np.mean([v[y_tok == c].var() for c in cls])
        sel[i] = between / max(within, 1e-12)
    return dict(act_mean=act_mean, act_freq=act_freq, act_mag=act_mag,
                contrib=contrib, selectivity=sel)


def dense_topk(sae, tok, n_sub: int = 30000, seed: int = 0):
    """在随机子样本上算稀疏稠密激活；同时返回对应的类别标签索引。

    全量 288×537×1024 的稠密矩阵要 630 MB，没必要：统计量在 3 万 token 上已经稳定
    （每原子平均 3 万 × k/D ≈ 234 次激活）。
    """
    import torch
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(tok), min(n_sub, len(tok)), replace=False)
    sub = tok[idx]
    dev = sae.device
    out = np.zeros((len(sub), sae.d_dict), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(sub), 8192):
            xb = torch.tensor(sub[i:i + 8192], dtype=torch.float32, device=dev)
            a = torch.relu(sae.enc(xb))
            top = torch.topk(a, sae.k, dim=1)
            sp = torch.zeros_like(a).scatter(1, top.indices, top.values)
            out[i:i + 8192] = sp.cpu().numpy()
    return out, idx


# ------------------------------------------------------------------------ 单被试
def run_subject(sub: str, seed: int = 0, m_abl: int = 8, d_dict: int = 1024,
                k: int = 32, steps: int = 3000, n_null: int = 30,
                max_atoms: int = 0, verbose=True) -> dict:
    import torch

    tr = D31.load_session(sub, "T")
    te = D31.load_session(sub, "E")
    eeg = [i for i, c in enumerate(tr["ch"]) if c not in SL.T1_EOG_CH]
    Xtr, Xte = D31.zscore(tr["X"][:, eeg], te["X"][:, eeg])
    ytr, yte = tr["y"], te["y"]

    dec = json.loads((PROBE / "t1_decoder_main.json").read_text(encoding="utf-8"))
    cand = [r for r in dec["runs"] if r["subject"] == sub and r["chans"] == "eeg"
            and r["seed"] == seed]
    epochs = cand[0]["best_epoch"] if cand else 60
    model, _ = D31.train_once(Xtr, ytr, Xtr[:1], ytr[:1], epochs=epochs, seed=seed)
    logit0 = _logits(model, Xte)
    acc0 = float(np.mean([(logit0.argmax(1)[yte == c] == c).mean()
                          for c in range(SL.T1_N_CLASSES)]))
    m0 = margin(logit0, yte)

    # --- token 与 SAE
    Htr, Hte = feature_map(model, Xtr), feature_map(model, Xte)
    F, Tp = Htr.shape[1], Htr.shape[2]
    tok_tr, tok_te = tokens_of(Htr), tokens_of(Hte)

    sae = TopKSAE(F, d_dict, k, seed=seed)
    sae.fit(tok_tr, steps=steps, bs=1024, lr=2e-3)
    rec = sae.reconstruct(tok_tr[:20000])
    r2 = float(1 - ((tok_tr[:20000] - rec) ** 2).sum()
               / ((tok_tr[:20000] - tok_tr.mean(0)) ** 2).sum())

    a_sub, idx_sub = dense_topk(sae, tok_te, seed=seed)
    y_sub = np.repeat(yte, Tp)[idx_sub]
    fs = feature_stats(a_sub, y_sub, sae.atom_dirs())

    # --- 一阶预测（a_i 用 token 算，梯度只做方向投影）
    G = grad_at_h(model, Xte, yte)
    fo = first_order(sae, Hte, G, m_abl)

    # --- 逐原子消融（可只跑前 max_atoms 个，用于冒烟）
    n_atom = d_dict if max_atoms <= 0 else min(max_atoms, d_dict)
    abl = Ablater(model, sae, Hte, m_abl)
    # 一致性自检：不消融时，尾部前向必须复现整网 logits
    chk = np.abs(abl.logits(0, direction=np.zeros(F, dtype=np.float32)) - logit0).max()
    if chk > 1e-3:
        raise RuntimeError(f"尾部前向与整网不一致（max|Δlogit|={chk:.3g}）")
    eff = np.full(d_dict, np.nan)
    dacc = np.full(d_dict, np.nan)
    flip = np.full(d_dict, np.nan)
    y0 = logit0.argmax(1)
    for i in range(n_atom):
        lgi = abl.logits(i)
        eff[i] = float((m0 - margin(lgi, yte)).mean())
        dacc[i] = acc0 - float(np.mean([(lgi.argmax(1)[yte == c] == c).mean()
                                        for c in range(SL.T1_N_CLASSES)]))
        flip[i] = float((lgi.argmax(1) != y0).mean())
        if verbose and (i + 1) % 256 == 0:
            print(f"    [{sub}] 消融 {i + 1}/{n_atom}", flush=True)

    # --- N2：同范数随机方向（用 0 号原子的幅度剖面，只换方向）
    rng = np.random.RandomState(seed)
    null_eff = []
    for _ in range(n_null):
        u = rng.randn(F).astype(np.float32)
        u /= max(np.linalg.norm(u), 1e-12)
        null_eff.append(float((m0 - margin(abl.logits(0, direction=u), yte)).mean()))
    null_eff = np.array(null_eff)

    if verbose:
        print(f"  {sub} | acc {acc0:.3f} | SAE R² {r2:.3f} | "
              f"最大 |效应| {np.nanmax(np.abs(eff)):.4f} | 随机方向 "
              f"{null_eff.mean():+.4f}±{null_eff.std():.4f}", flush=True)

    return dict(subject=sub, seed=seed, acc=round(acc0, 4), sae_r2=round(r2, 4),
                n_tokens=int(len(tok_tr)), Tp=int(Tp), F=int(F), epochs=int(epochs),
                eff=eff.tolist(), dacc=dacc.tolist(), flip=flip.tolist(),
                fo=fo.tolist(), margin_sd=float(m0.std()),
                null_eff=null_eff.tolist(),
                act_mean=fs["act_mean"].tolist(), act_freq=fs["act_freq"].tolist(),
                act_mag=fs["act_mag"].tolist(), contrib=fs["contrib"].tolist(),
                selectivity=fs["selectivity"].tolist())


def _logits(model, X, bs: int = 128) -> np.ndarray:
    import torch
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.tensor(X[i:i + bs], dtype=torch.float32,
                                          device=dev)).cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str, default="A01,A02,A03")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--m-abl", type=int, default=SL.SAE_TOPK_ABLATE)
    ap.add_argument("--d-dict", type=int, default=SL.SAE_DICT)
    ap.add_argument("--k", type=int, default=SL.SAE_K)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-null", type=int, default=30)
    ap.add_argument("--max-atoms", type=int, default=0, help=">0 时只消融前若干个（冒烟）")
    ap.add_argument("--tag", type=str, default="main")
    a = ap.parse_args()

    info = det_lock.lock(0)
    print(f"[det] {info}", flush=True)

    subs = [s.strip() for s in a.subjects.split(",") if s.strip()]
    t0 = time.time()
    out = dict(d_dict=a.d_dict, k=a.k, m_abl=a.m_abl, steps=a.steps, det=info, runs=[])
    for sub in subs:
        out["runs"].append(run_subject(sub, seed=a.seed, m_abl=a.m_abl,
                                       d_dict=a.d_dict, k=a.k, steps=a.steps,
                                       n_null=a.n_null, max_atoms=a.max_atoms))
    out["elapsed_s"] = round(time.time() - t0, 1)
    PROBE.mkdir(exist_ok=True)
    (PROBE / f"t1_sae_ablate_{a.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    summarize(out, a.tag)
    print(f"[done] {out['elapsed_s']} s", flush=True)
    return 0


def summarize(out: dict, tag: str) -> None:
    from scipy.stats import spearmanr

    lines = ["# T1 SAE 消融（真实标签，BCI2a）", "",
             f"自动生成。字典 {out['d_dict']}／k={out['k']}／每原子消融 top-{out['m_abl']} 时刻。",
             f"确定性锁：`{out['det']}`。", "",
             "效应 = 真实类 margin 的平均下降；`z` = 效应 / margin 的试次间标准差；",
             f"`超零` = 效应超过 N2 随机方向 95 分位的原子数"
             f"（{out['d_dict']} 个原子在纯噪声下的期望约 {out['d_dict'] * 0.05:.0f} 个）。",
             "",
             "| 被试 | 解码 bacc | SAE R² | 最大 z | 最大翻转率 | 随机方向均值 | 随机 p95 | 超零原子数 |"
             " ρ(贡献,效应) | ρ(选择性,效应) | ρ(一阶,实测) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["runs"]:
        eff = np.array(r["eff"]); ok = ~np.isnan(eff)
        e = eff[ok]
        z = e / max(r.get("margin_sd", 1.0), 1e-9)
        fl = np.array(r["flip"])[ok]
        ct = np.array(r["contrib"])[ok]
        sl = np.array(r["selectivity"])[ok]
        fo = np.array(r["fo"])[ok]
        nl = np.array(r["null_eff"])
        n_hi = int((e > np.percentile(nl, 95)).sum())
        lines.append(
            f"| {r['subject']} | {r['acc']:.3f} | {r['sae_r2']:.3f} | {z.max():+.3f} | "
            f"{fl.max():.1%} | {nl.mean():+.4f} | {np.percentile(nl, 95):+.4f} | "
            f"{n_hi}/{len(e)} | {spearmanr(ct, e).statistic:+.3f} | "
            f"{spearmanr(sl, e).statistic:+.3f} | {spearmanr(fo, e).statistic:+.3f} |")
    lines += ["", "## 效应最强的 15 个原子（各被试分别）", ""]
    for r in out["runs"]:
        eff = np.array(r["eff"]); ok = ~np.isnan(eff)
        order = np.argsort(-eff[ok])[:15]
        idx = np.where(ok)[0][order]
        lines += [f"### {r['subject']}", "",
                  "| 原子 | 效应 z | 翻转率 | 平均激活 | 激活频率 | 选择性 | 一阶预测 |",
                  "|---|---|---|---|---|---|---|"]
        for i in idx:
            lines.append(
                f"| {i} | {eff[i] / max(r.get('margin_sd', 1.0), 1e-9):+.3f} | "
                f"{r['flip'][i]:.1%} | {r['act_mean'][i]:.3f} | {r['act_freq'][i]:.3f} | "
                f"{r['selectivity'][i]:.4f} | {r['fo'][i]:+.4f} |")
        lines.append("")
    (REPORTS / f"T1_SAE_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
