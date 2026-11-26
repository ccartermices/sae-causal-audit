"""d33 — T1 联合消融：不同「选择器」挑出的 top-50 原子，一起消掉会怎样。

为什么需要这一步
----------------
d32 的单原子消融效应极小（一个原子只占 4096 分之一的重建，且只动 8/537 个时刻）。
即使方向对，单原子也动不了行为——这不是「SAE 没用」，是**剂量不足**。
`03_判据与闸门` §6 表 2 要的正是「top-50 集合联合效应」：
用不同选择器各挑 50 个原子，一起消融，比谁挑得更狠。

选择器（同一批原子、同一批测试试次，只换挑选依据）
--------------------------------------------------
| 选择器 | 依据 | 说明 |
|---|---|---|
| `random` | 均匀随机 50 个（重复 5 次） | 下界 |
| `contrib` | ④ 重建贡献 | SAE 自己的「重要度」 |
| `selectivity` | ⑤ 判别选择性 | 监督统计量 |
| `first_order` | 一阶预测器 $\nabla_h\mathrm{margin}\cdot a_i d_i$ | S3 的锚 |
| `causal_top` | d32 实测的消融效应（用 E 段标签） | **上界／oracle**，不是可用方法 |
| `causal_bottom` | 实测效应的最负 50 个 | 反向对照 |

产物：`probe/t1_joint_{tag}.json`、`reports/T1_joint_{tag}.md`
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


def joint_logits(model, sae, H, atoms, m: int, chunk: int = 64) -> np.ndarray:
    """同时消融一组原子：每个原子各减自己 top-m 时刻的贡献。"""
    import torch
    dev = next(model.parameters()).device
    tok_all = torch.tensor(H.transpose(0, 2, 1).copy(), dtype=torch.float32, device=dev)
    n, Tp, F = tok_all.shape
    W = sae.enc.weight.detach()[atoms]        # (S,F)
    b = sae.enc.bias.detach()[atoms]          # (S,)
    D = torch.tensor(sae.atom_dirs()[atoms], dtype=torch.float32, device=dev)  # (S,F)
    m = min(m, Tp)
    outs = []
    with torch.no_grad():
        for i in range(0, n, chunk):
            tk = tok_all[i:i + chunk].clone()                      # (b,T',F)
            A = torch.relu(tk @ W.T + b)                            # (b,T',S)
            for j in range(len(atoms)):
                ai = A[:, :, j]
                top = torch.topk(ai, m, dim=1)
                idx = top.indices.unsqueeze(-1).expand(-1, -1, F)
                tk.scatter_add_(1, idx, -top.values.unsqueeze(-1) * D[j])
            h = tk.transpose(1, 2).unsqueeze(-1)
            outs.append(D32.tail_logits(model, h).cpu().numpy())
    return np.concatenate(outs)


def run_subject(sub: str, seed: int, steps: int, topk: int, m_abl: int,
                n_rand: int = 5, src: str = "main", verbose=True) -> dict:
    import torch

    tr = D31.load_session(sub, "T")
    te = D31.load_session(sub, "E")
    eeg = [i for i, c in enumerate(tr["ch"]) if c not in SL.T1_EOG_CH]
    Xtr, Xte = D31.zscore(tr["X"][:, eeg], te["X"][:, eeg])
    ytr, yte = tr["y"], te["y"]

    prev = json.loads((PROBE / f"t1_sae_ablate_{src}.json").read_text(encoding="utf-8"))
    rec = [r for r in prev["runs"] if r["subject"] == sub][0]
    epochs = rec["epochs"]

    model, _ = D31.train_once(Xtr, ytr, Xtr[:1], ytr[:1], epochs=epochs, seed=seed)
    logit0 = D32._logits(model, Xte)
    acc0 = float(np.mean([(logit0.argmax(1)[yte == c] == c).mean()
                          for c in range(SL.T1_N_CLASSES)]))
    m0 = D32.margin(logit0, yte)

    Htr, Hte = D32.feature_map(model, Xtr), D32.feature_map(model, Xte)
    F = Htr.shape[1]
    # 步数必须与 d32 一致，否则这里重训的 SAE 与 eff/contrib 的来源不同源
    steps = int(prev.get("steps", steps))
    sae = TopKSAE(F, prev["d_dict"], prev["k"], seed=seed)
    sae.fit(D32.tokens_of(Htr), steps=steps, bs=1024, lr=2e-3)

    eff = np.array(rec["eff"])
    ok = ~np.isnan(eff)
    sel = {
        "contrib": np.argsort(-np.array(rec["contrib"]))[:topk],
        "selectivity": np.argsort(-np.array(rec["selectivity"]))[:topk],
        "first_order": np.argsort(-np.array(rec["fo"]))[:topk],
        "causal_top": np.where(ok)[0][np.argsort(-eff[ok])[:topk]],
        "causal_bottom": np.where(ok)[0][np.argsort(eff[ok])[:topk]],
    }
    rng = np.random.RandomState(seed)
    rand_sets = [rng.choice(prev["d_dict"], topk, replace=False) for _ in range(n_rand)]

    out = dict(subject=sub, acc0=round(acc0, 4), margin_sd=float(m0.std()),
               topk=topk, conds=[])
    def add(name, atoms):
        lg = joint_logits(model, sae, Hte, np.asarray(atoms), m_abl)
        bacc = float(np.mean([(lg.argmax(1)[yte == c] == c).mean()
                              for c in range(SL.T1_N_CLASSES)]))
        out["conds"].append(dict(sel=name, d_acc=round(acc0 - bacc, 4),
                                 d_margin=float((m0 - D32.margin(lg, yte)).mean()),
                                 flip=float((lg.argmax(1) != logit0.argmax(1)).mean())))
    for nm, at in sel.items():
        add(nm, at)
    for k, at in enumerate(rand_sets):
        add(f"random{k}", at)

    if verbose:
        print(f"  {sub} | acc0 {acc0:.3f} | " + " ".join(
            f"{c['sel']} {c['d_acc']:+.3f}" for c in out["conds"] if not c["sel"].startswith("random")),
            flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str, default="A01,A03,A07")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--m-abl", type=int, default=SL.SAE_TOPK_ABLATE)
    ap.add_argument("--n-rand", type=int, default=5)
    ap.add_argument("--from-ablate", type=str, default="main")
    ap.add_argument("--tag", type=str, default="main")
    a = ap.parse_args()

    info = det_lock.lock(0)
    print(f"[det] {info}", flush=True)
    subs = [s.strip() for s in a.subjects.split(",") if s.strip()]
    t0 = time.time()
    out = dict(topk=a.topk, m_abl=a.m_abl, src=a.from_ablate, det=info, runs=[])
    for sub in subs:
        out["runs"].append(run_subject(sub, a.seed, a.steps, a.topk, a.m_abl,
                                       a.n_rand, a.from_ablate))
    out["elapsed_s"] = round(time.time() - t0, 1)
    (PROBE / f"t1_joint_{a.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    summarize(out, a.tag)
    print(f"[done] {out['elapsed_s']} s", flush=True)
    return 0


def summarize(out: dict, tag: str) -> None:
    lines = ["# T1 联合消融：不同选择器的 top-50", "",
             f"自动生成。每条件消融 {out['topk']} 个原子，每原子按 top-{out['m_abl']} 时刻。",
             "`causal_top` 用测试集标签选原子，是**上界**，不可作为方法使用。", ""]
    names = ["random", "contrib", "selectivity", "first_order", "causal_top", "causal_bottom"]
    lines += ["| 被试 | 基线 bacc | " + " | ".join(names) + " |",
              "|---|---|" + "---|" * len(names)]
    for r in out["runs"]:
        cells = []
        for nm in names:
            v = [c["d_acc"] for c in r["conds"]
                 if c["sel"] == nm or (nm == "random" and c["sel"].startswith("random"))]
            cells.append(f"{np.mean(v):+.3f}" + ("±" + f"{np.std(v):.3f}" if len(v) > 1 else ""))
        lines.append(f"| {r['subject']} | {r['acc0']:.3f} | " + " | ".join(cells) + " |")
    lines += ["", "## 翻转率（预测类别改变的试次占比）", "",
              "| 被试 | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
    for r in out["runs"]:
        cells = []
        for nm in names:
            v = [c["flip"] for c in r["conds"]
                 if c["sel"] == nm or (nm == "random" and c["sel"].startswith("random"))]
            cells.append(f"{np.mean(v):.1%}")
        lines.append(f"| {r['subject']} | " + " | ".join(cells) + " |")
    (REPORTS / f"T1_joint_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
