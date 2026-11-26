"""d11 —— 宽度体检（R1：唯一可能让整个题目作废的风险）。

问题
----
计划里写的是「EEGNet 的 latent 只有几十~几百维，在上面训过完备 SAE 是空转，
所以要挂到 conv 特征图展平后的上千维」。这句话有两个未经验证的前提：

  (a) conv 特征图展平后真有「上千维」；
  (b) 维度高就等于可稀疏分解（即参与率 PR 也高）。

本脚本把两个前提都测出来，并用量化的饱和规则决定字典尺寸——**规则先于结果写死**：

  R1 判定规则
  ------------
  若 PR < 64 且 R²(D=4096) − R²(D=1024) < 0.01  ⇒  4096 空转，字典取
  min{D : R²(D) ≥ 0.99·max R²}；
  否则保留 4096。

产出
----
reports/P1_width.md      人读
data/width_probe.json    机器读（registry 用）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sig_layer as SL  # noqa: E402

EPO = ROOT / "data" / "interim" / "epochs"


# --------------------------------------------------------------------- 数据
def load_hands(n_subjects: int, seed: int = 0):
    """取「想象双手」二类任务：cond=1（想象）, kind=0（手）。按 run 分训练/测试。"""
    subs = sorted(p.stem for p in (EPO / "mmidb").glob("*.npz"))
    subs = [s for s in subs if s not in SL.EXCLUDE_STRUCT_SUBJECTS]
    rng = np.random.RandomState(seed)
    subs = list(rng.permutation(subs)[:n_subjects])
    Xtr, ytr, Xte, yte, gtr, gte = [], [], [], [], [], []
    for s in subs:
        z = np.load(EPO / "mmidb" / f"{s}.npz")
        m = (z["cond"] == 1) & (z["kind"] == 0)
        X, y, run = z["X"][m], z["y"][m], z["run"][m]
        tr = np.isin(run, (4, 8))
        te = run == 12
        if X.shape[1] != 64:
            continue
        Xtr.append(X[tr]); ytr.append(y[tr]); gtr.append(np.full(tr.sum(), len(gte)))
        Xte.append(X[te]); yte.append(y[te]); gte.append(np.full(te.sum(), len(gte)))
    return (np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(gtr),
            np.concatenate(Xte), np.concatenate(yte), np.concatenate(gte), subs)


def zscore(Xtr, *others):
    mu = Xtr.mean((0, 2), keepdims=True)
    sd = Xtr.std((0, 2), keepdims=True) + 1e-6
    return tuple([(Xtr - mu) / sd] + [(o - mu) / sd for o in others])


# ------------------------------------------------------------------ 特征提取
def build_model(name: str, n_times: int):
    # braindecode 1.12 起 `EEGNetv4` 改名 `EEGNet`，2.0 里旧名已删；两边都要能跑。
    from braindecode.models import ShallowFBCSPNet

    if name == "eegnet":
        try:
            from braindecode.models import EEGNet
        except ImportError:                     # braindecode < 1.12
            from braindecode.models import EEGNetv4 as EEGNet
        return EEGNet(n_chans=64, n_outputs=2, n_times=n_times)
    if name == "shallow":
        return ShallowFBCSPNet(n_chans=64, n_outputs=2, n_times=n_times)
    raise ValueError(name)


def forward_to_features(model, x):
    """跑到 final_layer 之前，返回展平后的特征（SAE 要挂的东西）。"""
    x = x.to(next(model.parameters()).device)
    h = x
    for name, mod in model.named_children():
        if name == "final_layer":
            break
        h = mod(h)
    return h.flatten(1).detach(), h.detach()


def train_decoder(model, Xtr, ytr, Xte, yte, epochs=60, bs=32, lr=1e-3, seed=0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte, dtype=torch.float32).to(dev)).argmax(1).cpu().numpy()
    acc = float((pred == yte).mean())
    return model, acc


# ---------------------------------------------------------------------- SAE
class TopKSAE:
    """TopK 稀疏自编码器：每样本只保留 k 个原子，避免跨字典尺寸调 L1 系数。"""

    def __init__(self, d_in: int, d_dict: int, k: int, seed: int = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.k = k
        self.enc = nn.Linear(d_in, d_dict)
        self.dec = nn.Linear(d_dict, d_in, bias=False)
        nn.init.kaiming_uniform_(self.enc.weight)
        with torch.no_grad():
            self.dec.weight.copy_(self.enc.weight.t())

    def fit(self, X, steps=3000, bs=128, lr=3e-3, l1=1e-4, seed=0):
        import torch

        torch.manual_seed(seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.enc.to(dev); self.dec.to(dev)
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=lr)
        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        n = len(Xt)
        for _ in range(steps):
            idx = torch.randint(0, n, (min(bs, n),), device=dev)
            xb = Xt[idx]
            a = torch.relu(self.enc(xb))
            top = torch.topk(a, self.k, dim=1)
            a_sp = torch.zeros_like(a).scatter(1, top.indices, top.values)
            rec = self.dec(a_sp)
            loss = ((rec - xb) ** 2).mean() + l1 * a_sp.abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def transform(self, X):
        import torch

        dev = next(self.enc.parameters()).device
        with torch.no_grad():
            a = torch.relu(self.enc(torch.tensor(X, dtype=torch.float32, device=dev)))
            top = torch.topk(a, self.k, dim=1)
            a_sp = torch.zeros_like(a).scatter(1, top.indices, top.values)
            rec = self.dec(a_sp).cpu().numpy()
            act = (a_sp > 0).cpu().numpy()
        return rec, act


def r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def pca_stats(F, ks=(10, 50, 100)):
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-6)
    lam = np.linalg.eigvalsh(np.cov(Fz.T))[::-1]
    lam = np.clip(lam, 0, None)
    cum = np.cumsum(lam) / lam.sum()
    C = np.corrcoef(Fz.T)
    iu = np.triu_indices(C.shape[0], 1)
    return dict(pr=float(lam.sum() ** 2 / (lam ** 2).sum()),
                evr_top10=[round(float(x), 4) for x in lam[:10] / lam.sum()],
                cum_at={int(k): round(float(cum[k - 1]), 4) for k in ks if k <= len(cum)},
                median_abs_r=round(float(np.median(np.abs(C[iu]))), 4))


def whiten(Ftr, Fte, eps=1e-6):
    """PCA 白化：把特征空间的各向异性拉平，再交给 SAE。

    动机：若原始特征的主成分谱极度倾斜（第一大主成分占 80%+），字典原子会挤在
    少数方向上，跨种子不可复现。白化是可预注册的线性预处理，不改解码器。
    """
    mu = Ftr.mean(0)
    A = Ftr - mu
    C = A.T @ A / max(len(A) - 1, 1)
    w, V = np.linalg.eigh(C)
    keep = w > eps * w.max()
    W = V[:, keep] / np.sqrt(w[keep])
    return (Ftr - mu) @ W, (Fte - mu) @ W


def participation_ratio(F):
    """PR = (Σλ)² / Σλ²，协方差特征值的参与率：有效维度。"""
    Fc = F - F.mean(0, keepdims=True)
    C = Fc.T @ Fc / max(len(Fc) - 1, 1)
    lam = np.linalg.eigvalsh(C)
    lam = lam[lam > 0]
    return float(lam.sum() ** 2 / (lam ** 2).sum())


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["eegnet", "shallow"], default="shallow")
    ap.add_argument("--n-subjects", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--dict-sizes", default="128,256,512,1024,2048,4096")
    ap.add_argument("--whiten", action="store_true", help="SAE 前做 PCA 白化")
    a = ap.parse_args()

    Xtr, ytr, gtr, Xte, yte, gte, subs = load_hands(a.n_subjects)
    Xtr, Xte = zscore(Xtr, Xte)
    print(f"被试 {len(subs)} | 训练 {Xtr.shape} 测试 {Xte.shape} | "
          f"类别 {np.bincount(ytr)}/{np.bincount(yte)}")

    model = build_model(a.model, Xtr.shape[2])
    n_par = sum(p.numel() for p in model.parameters())
    model, acc = train_decoder(model, Xtr, ytr, Xte, yte, epochs=a.epochs)
    print(f"解码器 {a.model}: 参数 {n_par} | 留 run 测试准确率 {acc:.3f}")

    import torch
    with torch.no_grad():
        Ftr_raw, htr = forward_to_features(model, torch.tensor(Xtr, dtype=torch.float32))
        Fte_raw, hte = forward_to_features(model, torch.tensor(Xte, dtype=torch.float32))
    Ftr_raw, Fte_raw = Ftr_raw.cpu().numpy(), Fte_raw.cpu().numpy()
    D = int(Ftr_raw.shape[1])
    map_shape = tuple(htr.shape[1:])
    print(f"末端特征图形状 {map_shape} → 展平 D={D}")

    # 逐维 z 化（均值/尺度都取训练集），SAE 与 PR 都在 z 化后的空间里算
    mu, sd = Ftr_raw.mean(0), Ftr_raw.std(0) + 1e-6
    Ftr = (Ftr_raw - mu) / sd
    Fte = (Fte_raw - mu) / sd
    pr_raw = participation_ratio(Ftr_raw)
    pstats = pca_stats(Ftr)
    if a.whiten:
        Ftr, Fte = whiten(Ftr, Fte)
        pstats_w = pca_stats(Ftr)
        print(f"白化后 D={Ftr.shape[1]} PR={pstats_w['pr']:.2f}")
        D = int(Ftr.shape[1])   # 白化会丢掉数值零方向，字典尺寸按实际维数走
    else:
        pstats_w = None

    pr = participation_ratio(Ftr)
    print(f"参与率 PR={pr:.2f}（z 化后），PR_raw={pr_raw:.2f}，D={D}")

    rows = []
    for Dk in [int(x) for x in a.dict_sizes.split(",")]:
        k = max(8, min(Dk // 32, 64))
        sae = TopKSAE(D, Dk, k, seed=0).fit(Ftr, steps=2500)
        rec_te, act_te = sae.transform(Fte)
        rec_tr, act_tr = sae.transform(Ftr)
        rows.append(dict(dict_size=Dk, k=k,
                         r2_train=round(r2(Ftr, rec_tr), 4),
                         r2_test=round(r2(Fte, rec_te), 4),
                         dead_frac=round(float((act_tr.sum(0) == 0).mean()), 4),
                         active_per_sample=round(float(act_te.sum(1).mean()), 2)))
        print(f"  D={Dk:5d} k={k:3d} R2tr={rows[-1]['r2_train']:.4f} "
              f"R2te={rows[-1]['r2_test']:.4f} dead={rows[-1]['dead_frac']:.3f}")

    # ---- R1 规则（先于结果写死）
    r2map = {r["dict_size"]: r["r2_test"] for r in rows}
    best = max(r2map.values())
    if pr < 64 and (4096 in r2map and 1024 in r2map) and (r2map[4096] - r2map[1024] < 0.01):
        chosen = min(d for d, v in r2map.items() if v >= 0.99 * best)
        decision = "4096 空转"
    else:
        chosen = max(r2map)
        decision = "保留 4096"

    out = dict(model=a.model, params=n_par, n_subjects=len(subs),
               n_train=int(len(Xtr)), n_test=int(len(Xte)),
               test_acc=round(acc, 4), feature_map_shape=list(map_shape),
               feature_dim=D, participation_ratio=round(pr, 3),
               participation_ratio_raw=round(pr_raw, 3),
               pca=pstats, pca_whitened=pstats_w, whiten=bool(a.whiten),
               sweep=rows, chosen_dict=chosen, r1_decision=decision,
               rule="PR<64 且 R2(4096)-R2(1024)<0.01 ⇒ 4096 空转，取 min{D:R2≥0.99max}")
    tag = a.model + ("_whiten" if a.whiten else "")
    (ROOT / "data" / f"width_probe_{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    L = [f"# P1 · 宽度体检（{a.model}）", "",
         "> 由 `scripts/d11_width.py` 生成，勿手改。规则先于结果写死（见脚本文档串）。", "",
         f"- 被试 {len(subs)} 人（已排除结构非标准 3 人），训练 {len(Xtr)} 试次 / 测试 {len(Xte)} 试次",
         f"- 解码器参数 {n_par}，留 run 测试准确率 **{acc:.3f}**",
         f"- 末端特征图形状 `{map_shape}` → 展平 **D={D}**",
         f"- 参与率 **PR={pr:.1f}**（有效维度）；逐维相关中位 |r|={pstats['median_abs_r']:.2f}",
         f"- 主成分解释比前 10：{pstats['evr_top10']}",
         f"- 累计解释比 {pstats['cum_at']}", "",
         "| 字典尺寸 D | k | R²(训练) | R²(测试) | 死原子比例 | 每样本激活数 |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dict_size']} | {r['k']} | {r['r2_train']:.4f} | "
                 f"{r['r2_test']:.4f} | {r['dead_frac']:.3f} | {r['active_per_sample']:.1f} |")
    L += ["", f"**判定：{decision}，字典尺寸取 {chosen}。**", "",
          "## 对计划的影响", "",
          "- 若 `D` 只有几百维而 PR 更低，则「过完备字典 + 单义原子」的前提不成立，",
          "  S1 的语义–因果解耦只能在**低维、强相关**的原子上测，结论必须限定为",
          "  「在可分解性受限的表征上，语义与因果仍然解耦/耦合」。",
          "- 若饱和点远低于 4096，正文里的字典尺寸必须改成实测值，不能写 4096。"]
    (ROOT / "reports" / f"P1_width_{tag}.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n判定 {decision} → 字典 {chosen}；写出 reports/P1_width_{tag}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
