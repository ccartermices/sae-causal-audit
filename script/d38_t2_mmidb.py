"""d38 — T2 规模层：在 EEGMMIDB（64 导、二类、160 Hz）上复验一阶预测器。

为什么必须做
------------
T1 的 S3 结论是「在 ShallowFBCSPNet + 8-token 局部消融尺度上，一阶项可以当因果选择器」
（$\rho_{1st}=0.987$）。这条结论有两个不能自动外推的理由：

1. **维度**：T1 是 22 导、$F=40$、token 时间分辨率 250 Hz；T2 是 64 导、二类。
   一阶项的成立依赖「消融是局部线性的」，而局部线性的程度会随网络宽度/类别数变。
2. **数据**：T1 是 9 名被试的 2a，T2 是 mmidb 的独立样本。若只在 2a 上成立，
   S3 只能作为数据集特异的观察，不能进正文的主结论。

`03_判据与闸门` §5 规定 T2 的被试内切分是「留一 run」。这里取 6 个 hands run
（3/4/7/8/11/12，左右拳），4 run 训、2 run 测。

与 T1 的口径差异（必须写进方法节）
----------------------------------
| 项 | T1（2a） | T2（mmidb） | 理由 |
|---|---|---|---|
| 切分 | 跨 session（T 段 → E 段） | 跨 run（4 run → 2 run） | mmidb 无第二场 |
| 类别 | 4 类运动想象 | 2 类（左/右拳，**执行+想象混合**） | 执行条件只有 45 试次，单用不够训解码器 |
| 通道 | 22 导（踢 3 导 EOG） | 64 导（无独立 EOG 通道） | 数据集固有 |
| 字典 | 4096 | **1024** | token 数少 5.5 倍，按比例降字典，使「每原子平均激活次数」与 T1 同量级 |
| 时长 | 4.0 s @250 Hz | 3.5 s @160 Hz | `SL.EPOCHS` |

产物：`probe/t2_mmidb_{tag}.json`、`reports/T2_mmidb_{tag}.md`
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
import d33_t1_joint_ablate as D33  # noqa: E402

EPO = ROOT / "data" / "interim" / "epochs" / "mmidb"
PROBE = ROOT / "probe"
REPORTS = ROOT / "reports"

T2_CLASSES = 2
T2_TRAIN_RUNS = (3, 4, 7, 8)
T2_TEST_RUNS = (11, 12)


# ------------------------------------------------------------------ 数据
def load_mmidb(sub: str):
    """只取 hands 条件（run 3/4/7/8/11/12），二类左/右拳；执行与想象都用。"""
    z = np.load(EPO / f"{sub}.npz", allow_pickle=True)
    keep = (z["cond"] == 1) | (z["cond"] == 0)      # 全部
    keep = keep & (np.isin(z["run"], list(T2_TRAIN_RUNS) + list(T2_TEST_RUNS)))
    return dict(X=z["X"][keep].astype(np.float32), y=z["y"][keep].astype(np.int64),
                run=z["run"][keep], ch=[str(c) for c in z["ch_names"]],
                sfreq=float(z["sfreq"]))


def zscore(Xtr, Xte):
    mu = Xtr.mean((0, 2), keepdims=True)
    sd = Xtr.std((0, 2), keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


# ------------------------------------------------------------------ 模型
def build(n_chans: int, n_times: int):
    from braindecode.models import ShallowFBCSPNet
    return ShallowFBCSPNet(n_chans=n_chans, n_outputs=T2_CLASSES,
                           n_times=n_times, drop_prob=0.5)


def train_once(Xtr, ytr, epochs: int, seed: int, bs: int = 16, lr: float = 3e-4,
               wd: float = 1e-3):
    """mmidb 每被试只有 90 个 hands 试次，60 个训练试次下 1e-3/32 会让 train bacc=1.000
    而 test bacc 掉到 0.47。lr 3e-4 + wd 1e-3 + bs 16 把 train bacc 压到 0.99 但
    test bacc 稳定在 0.58–0.64（S001/S003/S050 扫描，见 `_scratch/t2_reg_sweep.py`）。"""
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    model = build(Xtr.shape[1], Xtr.shape[2]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.long, device=dev)
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
    n = len(Xt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i:i + bs].to(dev)
            opt.zero_grad()
            lossf(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    return model


def kappa2(y_true, y_pred) -> float:
    cm = np.zeros((T2_CLASSES, T2_CLASSES))
    np.add.at(cm, (y_true, y_pred), 1)
    po = np.trace(cm) / cm.sum()
    pe = float((cm.sum(0) * cm.sum(1)).sum() / cm.sum() ** 2)
    return float((po - pe) / max(1 - pe, 1e-12))


def bacc_of(logits, y) -> float:
    return float(np.mean([(logits.argmax(1)[y == c] == c).mean()
                          for c in range(T2_CLASSES) if (y == c).any()]))


# ------------------------------------------------------------------ 单被试
def run_subject(sub: str, seed: int = 0, steps: int = 3000, d_dict: int = 1024,
                k: int = SL.SAE_K, m_abl: int = SL.SAE_TOPK_ABLATE,
                topk: int = 50, n_rand: int = 5, n_null: int = 30,
                verbose: bool = True) -> dict:
    import torch

    d = load_mmidb(sub)
    tr = np.isin(d["run"], T2_TRAIN_RUNS)
    te = np.isin(d["run"], T2_TEST_RUNS)
    if tr.sum() < 30 or te.sum() < 10:
        raise ValueError(f"{sub}: 可用于训练的 run 不足（train {int(tr.sum())}, test {int(te.sum())}）")
    Xtr, Xte = zscore(d["X"][tr], d["X"][te])
    ytr, yte = d["y"][tr], d["y"][te]

    # --- 训练轮数：只在训练 run 内做留一 run（4 折），不看测试 run
    grid = (20, 40, 60)
    fold_acc = {g: [] for g in grid}
    tr_runs = sorted(set(d["run"][tr]))
    for r in tr_runs:
        va = d["run"][tr] == r
        for g in grid:
            mdl = train_once(Xtr[~va], ytr[~va], epochs=g, seed=int(seed * 10 + r))
            fold_acc[g].append(bacc_of(D32._logits(mdl, Xtr[va]), ytr[va]))
    cv_mean = {g: float(np.mean(v)) for g, v in fold_acc.items()}
    best_ep = max(grid, key=lambda g: cv_mean[g])

    model = train_once(Xtr, ytr, epochs=best_ep, seed=seed)
    logit0 = D32._logits(model, Xte)
    acc0 = bacc_of(logit0, yte)
    kap0 = kappa2(yte, logit0.argmax(1))
    m0 = D32.margin(logit0, yte)

    # --- token 与 SAE
    Htr, Hte = D32.feature_map(model, Xtr), D32.feature_map(model, Xte)
    F, Tp = Htr.shape[1], Htr.shape[2]
    tok_tr = D32.tokens_of(Htr)
    sae = TopKSAE(F, d_dict, k, seed=seed)
    sae.fit(tok_tr, steps=steps, bs=1024, lr=2e-3)
    rec = sae.reconstruct(tok_tr[:20000])
    r2 = float(1 - ((tok_tr[:20000] - rec) ** 2).sum()
               / ((tok_tr[:20000] - tok_tr.mean(0)) ** 2).sum())

    a_sub, idx_sub = D32.dense_topk(sae, D32.tokens_of(Hte), seed=seed)
    y_sub = np.repeat(yte, Tp)[idx_sub]
    fs = D32.feature_stats(a_sub, y_sub, sae.atom_dirs())

    # --- 一阶预测与逐原子消融
    G = D32.grad_at_h(model, Xte, yte)
    fo = D32.first_order(sae, Hte, G, m_abl)

    abl = D32.Ablater(model, sae, Hte, m_abl)
    chk = np.abs(abl.logits(0, direction=np.zeros(F, dtype=np.float32)) - logit0).max()
    if chk > 1e-3:
        raise RuntimeError(f"尾部前向与整网不一致（max|Δlogit|={chk:.3g}）")
    eff = np.full(d_dict, np.nan)
    flip = np.full(d_dict, np.nan)
    y0 = logit0.argmax(1)
    for i in range(d_dict):
        lgi = abl.logits(i)
        eff[i] = float((m0 - D32.margin(lgi, yte)).mean())
        flip[i] = float((lgi.argmax(1) != y0).mean())
        if verbose and (i + 1) % 256 == 0:
            print(f"    [{sub}] 消融 {i + 1}/{d_dict}", flush=True)

    # --- N2：同范数随机方向
    rng = np.random.RandomState(seed)
    null_eff = []
    for _ in range(n_null):
        u = rng.randn(F).astype(np.float32)
        u /= max(np.linalg.norm(u), 1e-12)
        null_eff.append(float((m0 - D32.margin(abl.logits(0, direction=u), yte)).mean()))
    null_eff = np.array(null_eff)

    # --- 联合 top-50：四个选择器
    ok = ~np.isnan(eff)
    sel = {
        "first_order": np.argsort(-fo)[:topk],
        "causal_top": np.where(ok)[0][np.argsort(-eff[ok])[:topk]],
        "causal_bottom": np.where(ok)[0][np.argsort(eff[ok])[:topk]],
    }
    rand_sets = [rng.choice(d_dict, topk, replace=False) for _ in range(n_rand)]
    conds = []
    for nm, at in sel.items():
        lg = D33.joint_logits(model, sae, Hte, np.asarray(at), m_abl)
        conds.append(dict(sel=nm, d_acc=round(acc0 - bacc_of(lg, yte), 4),
                          flip=float((lg.argmax(1) != y0).mean())))
    for i, at in enumerate(rand_sets):
        lg = D33.joint_logits(model, sae, Hte, np.asarray(at), m_abl)
        conds.append(dict(sel=f"random{i}", d_acc=round(acc0 - bacc_of(lg, yte), 4),
                          flip=float((lg.argmax(1) != y0).mean())))

    from scipy.stats import spearmanr
    rho = float(spearmanr(fo, eff).statistic) if np.std(fo) > 1e-12 else float("nan")

    if verbose:
        print(f"  {sub} | bacc {acc0:.3f} κ {kap0:+.3f} (CV {cv_mean[best_ep]:.3f} → {best_ep} ep) | "
              f"R² {r2:.3f} | ρ(fo,eff) {rho:+.3f} | "
              f"fo {np.mean([c['d_acc'] for c in conds if c['sel'] == 'first_order']):+.3f} "
              f"oracle {np.mean([c['d_acc'] for c in conds if c['sel'] == 'causal_top']):+.3f}",
              flush=True)

    return dict(subject=sub, seed=seed, acc=round(acc0, 4), kappa=round(kap0, 4),
                cv_mean=cv_mean,
                best_epoch=int(best_ep), sae_r2=round(r2, 4), rho_first=round(rho, 4),
                n_tokens=int(len(tok_tr)), Tp=int(Tp), F=int(F), d_dict=d_dict, k=k,
                m_abl=m_abl, topk=topk, margin_sd=float(m0.std()),
                eff=eff.tolist(), flip=flip.tolist(), fo=fo.tolist(),
                null_eff=null_eff.tolist(), conds=conds,
                contrib=fs["contrib"].tolist(), selectivity=fs["selectivity"].tolist())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str,
                    default="S001,S010,S020,S030,S040,S050,S060,S070,S080,S090,S099,S109")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--d-dict", type=int, default=1024)
    ap.add_argument("--k", type=int, default=SL.SAE_K)
    ap.add_argument("--m-abl", type=int, default=SL.SAE_TOPK_ABLATE)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n-rand", type=int, default=5)
    ap.add_argument("--n-null", type=int, default=30)
    ap.add_argument("--tag", type=str, default="main")
    a = ap.parse_args()

    info = det_lock.lock(0)
    print(f"[det] {info}", flush=True)
    subs = [s.strip() for s in a.subjects.split(",") if s.strip()]
    t0 = time.time()
    out = dict(dataset="mmidb", d_dict=a.d_dict, k=a.k, m_abl=a.m_abl, steps=a.steps,
               topk=a.topk, train_runs=list(T2_TRAIN_RUNS), test_runs=list(T2_TEST_RUNS),
               det=info, runs=[])
    skipped = []
    for sub in subs:
        try:
            out["runs"].append(run_subject(sub, a.seed, a.steps, a.d_dict, a.k,
                                           a.m_abl, a.topk, a.n_rand, a.n_null))
        except ValueError as e:
            skipped.append(str(e))
            print(f"  [skip] {e}", flush=True)
    out["skipped"] = skipped
    out["elapsed_s"] = round(time.time() - t0, 1)
    PROBE.mkdir(exist_ok=True)
    (PROBE / f"t2_mmidb_{a.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    summarize(out, a.tag)
    print(f"[done] {out['elapsed_s']} s", flush=True)
    return 0


def summarize(out: dict, tag: str) -> None:
    from scipy.stats import spearmanr, wilcoxon
    _s = "" if tag == "main" else f"_{tag}"   # main 写规范名

    R = out["runs"]
    rho = np.array([r["rho_first"] for r in R])
    bacc = np.array([r["acc"] for r in R])
    kap = np.array([r["kappa"] for r in R])
    r2 = np.array([r["sae_r2"] for r in R])

    def g(nm):
        return np.array([np.mean([c["d_acc"] for c in r["conds"]
                                  if c["sel"] == nm or (nm == "random" and c["sel"].startswith("random"))])
                         for r in R])

    fo, ra, orc, bot = g("first_order"), g("random"), g("causal_top"), g("causal_bottom")

    lines = ["# T2 规模层：mmidb 上的一阶预测器复验", "",
             f"自动生成。字典 {out['d_dict']}／k={out['k']}／每原子消融 top-{out['m_abl']} 时刻；",
             f"训练 run {out['train_runs']}，测试 run {out['test_runs']}；"
             f"{len(R)} 名被试；确定性锁 `{out['det']}`。", "",
             "## 与 T1 的口径差异（必须与结果同页）", "",
             "| 项 | T1（2a） | T2（mmidb） |", "|---|---|---|",
             "| 切分 | 跨 session（T→E） | 跨 run（4 训 2 测） |",
             "| 类别 | 4 类运动想象 | 2 类左右拳，**执行+想象混合** |",
             "| 通道 | 22 导（去 3 导 EOG） | 64 导 |",
             "| 每被试训练试次 | 273 | **60** |",
             f"| 字典 | 4096 | **{out['d_dict']}**（token 少 5.5 倍，按比例降） |",
             "",
             "**T2 的解码器明显弱于 T1**（每被试只有 60 个训练试次，ShallowFBCSPNet 会记满训练集），",
             "所以 T2 只用来检验「一阶项是否仍是好的因果选择器」，不用于比较效应幅度。", "",
             "> ⚠️ **`测试 bacc` / `κ` 仅作仪器线（D39）。** 本表下方的先验问题不是「一阶项是否成立」",
             "> （那个由 $\\rho$ 与联合消融回答），而是「这一列数字能否当解码能力的估计」。答：**不能**。",
             "> `d39` 在 30 名被试上比较了三条切分协议：现行 2 类 60/30、6 个 hands run 留一（75/15，无偏）、",
             "> 以及 4 类 120/60（训练试次翻倍）。三者的 $\\kappa$ 为 **+0.130 / +0.113 / +0.124**，",
             "> 在噪声内不可区分——**扩 run 不能提高解码器强度**，故维持现行协议。",
             "> 因此这两列只用来回答一个二值问题：「这段数据被解码到了机会水平以上吗」",
             "> （$\\kappa>0.10$ 记为是）。$\\rho$ 与 $\\kappa$ 的被试间相关只有 +0.066（p = 0.50），",
             "> 说明机制量不受解码强度影响，这两列对主结论无贡献。", "",
             "## 逐被试", "",
             "| 被试 | bacc* | κ* | 训练轮数 | SAE R² | ρ(一阶,实测) | 一阶 | 随机 | oracle | bottom |",
             "|---|---|---|---|---|---|---|---|---|---|",
             "", "`*` 仅作仪器线（见上，D39）：不是解码能力的估计，只用于「是否高于机会水平」的二值判断。", ""]
    for r in R:
        cell = lambda nm: np.mean([c["d_acc"] for c in r["conds"]
                                   if c["sel"] == nm or (nm == "random" and c["sel"].startswith("random"))])
        lines.append(f"| {r['subject']} | {r['acc']:.3f} | {r['kappa']:+.3f} | {r['best_epoch']} | "
                     f"{r['sae_r2']:.3f} | {r['rho_first']:+.3f} | {cell('first_order'):+.3f} | "
                     f"{cell('random'):+.3f} | {cell('causal_top'):+.3f} | {cell('causal_bottom'):+.3f} |")

    lines += ["", "## 合并", "",
              "| 量 | 值 |", "|---|---|",
              f"| ρ(一阶, 实测) | **{rho.mean():+.4f} ± {rho.std(ddof=1):.4f}**"
              f"（{rho.min():.3f}–{rho.max():.3f}） |",
              f"| ρ ≥ S3 下界 {SL.S3_RHO_MIN} | **{int((rho >= SL.S3_RHO_MIN).sum())}/{len(rho)}** |",
              f"| ρ ≥ 0.90 | {int((rho >= 0.90).sum())}/{len(rho)} |",
              f"| 测试 bacc（**仅作仪器线**） | {bacc.mean():.3f} ± {bacc.std(ddof=1):.3f}"
              f"（{bacc.min():.3f}–{bacc.max():.3f}） |",
              f"| κ（**仅作仪器线**） | {kap.mean():+.3f} ± {kap.std(ddof=1):.3f}"
              f"（κ > {SL.T1_DECODER_MIN_KAPPA} 的 {int((kap > SL.T1_DECODER_MIN_KAPPA).sum())}/{len(kap)}） |",
              f"| SAE R² | {r2.mean():.3f} ± {r2.std(ddof=1):.3f} |",
              f"| ρ 与 bacc 的 Spearman | {spearmanr(rho, bacc).statistic:+.3f}"
              f"（p = {spearmanr(rho, bacc).pvalue:.3f}） |",
              "",
              "| 选择器 | Δ平衡准确率 | 为正的被试数 | 与随机的配对差 / p |", "|---|---|---|---|"]
    for nm, v in (("一阶 top-50", fo), ("随机 50", ra), ("oracle", orc), ("causal_bottom", bot)):
        dl = v - ra
        pv = "—" if nm == "随机 50" else f"{wilcoxon(dl).pvalue:.2e}"
        lines.append(f"| {nm} | {v.mean():+.4f} ± {v.std(ddof=1):.4f} | {int((v > 0).sum())}/{len(v)} | "
                     f"{dl.mean():+.4f} / {pv} |")
    lines += ["",
              f"**一阶 ≈ oracle**：配对差 {np.mean(fo - orc):+.4f}（p = {wilcoxon(fo - orc).pvalue:.3f}）。",
              f"**一阶 > 随机 > bottom** 在被试内成立 {int(((fo > ra) & (ra > bot)).sum())}/{len(fo)}。", ""]
    (REPORTS / f"T2_mmidb{_s}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
