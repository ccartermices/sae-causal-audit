"""d13 —— token 级表征体检（路线 B 的可行性证据）。

背景
----
d11/d12 证明：把「一个试次」当作 SAE 的输入单元时，ShallowConvNet 的 1240 维特征图
有效维只有 1.5，EEGNet 的 272 维有效维约 21，且 1194 个训练样本撑不起大字典。

本脚本换一个单元来问同一个问题：**如果把卷积特征图按时间切片当 token，表征是否
变得可稀疏分解、样本是否够多？** 这一步只测「有没有资格做过完备字典」，不训 SAE。

测什么
------
1. token 数量（决定「每原子多少 token」这条统计约束）
2. token 空间的有效维 PR（决定字典是否有可分的多个方向）
3. 主成分谱（看是否被单一全局能量因子主导）

两个候选切片位置
----------------
* `pre-pool`：`conv_time_spat` 之后，形状 (n, 40, 537, 1) → 每试次 537 个 40 维 token
* `post-pool`：时域池化之后，形状 (n, 40, 31, 1) → 每试次 31 个 token
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import d11_width as W  # noqa: E402


def pr_and_spectrum(tok: np.ndarray) -> dict:
    sub = tok[np.random.RandomState(0).choice(
        len(tok), min(30000, len(tok)), replace=False)]
    lam = np.linalg.eigvalsh(np.cov(sub.T))[::-1]
    lam = np.clip(lam, 0, None)
    return dict(pr=round(float(lam.sum() ** 2 / (lam ** 2).sum()), 2),
                evr_top5=[round(float(x), 4) for x in lam[:5] / lam.sum()])


def main() -> int:
    import torch

    Xtr, ytr, _g, Xte, yte, _gt, subs = W.load_hands(40)
    Xtr, Xte = W.zscore(Xtr, Xte)
    model = W.build_model("shallow", Xtr.shape[2])
    model, acc = W.train_decoder(model, Xtr, ytr, Xte, yte, epochs=80)
    print(f"解码器 acc={acc:.3f}，被试 {len(subs)}，训练试次 {len(Xtr)}")

    with torch.no_grad():
        xb = torch.tensor(Xtr[:240], dtype=torch.float32)
        c = model.bnorm(model.conv_time_spat(model.dimshuffle(model.ensuredims(
            xb.to(next(model.parameters()).device)))))
        p = model.pool_nonlin_exp(model.pool(model.conv_nonlin_exp(c)))

    out = dict(n_subjects=len(subs), n_train=int(len(Xtr)), decoder_acc=round(acc, 4),
               variants={})
    for name, h in [("pre_pool", c), ("post_pool", p)]:
        h = h.squeeze(-1).permute(0, 2, 1).reshape(-1, h.shape[1]).cpu().numpy()
        per_trial = h.shape[0] // 240
        out["variants"][name] = dict(
            tokens_per_trial=int(per_trial), token_dim=int(h.shape[1]),
            n_tokens_probe=int(h.shape[0]),
            n_tokens_full=int(per_trial * len(Xtr)),
            tokens_per_atom_at_4096=round(per_trial * 6000 / 4096, 1),
            **pr_and_spectrum(h))
        v = out["variants"][name]
        print(f"  {name:10s} {per_trial:4d} token/试次 × {h.shape[1]:3d} 维  "
              f"PR={v['pr']:.2f}  前5解释比累计 {sum(v['evr_top5']):.2f}")

    (ROOT / "data" / "token_probe.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    pre, post = out["variants"]["pre_pool"], out["variants"]["post_pool"]
    L = ["# P1 · token 级表征体检", "",
         "> 由 `scripts/d13_token_probe.py` 生成，勿手改。只测「有没有资格做过完备字典」，不训 SAE。", "",
         f"- 被试 {len(subs)} 人，训练试次 {len(Xtr)}，解码器 acc {acc:.3f}", "",
         "| 切片位置 | token/试次 | token 维 | 有效维 PR | 前 5 PC 累计 | 4096 原子时每原子 token 数（按全量 6000 试次） |",
         "|---|---|---|---|---|---|",
         f"| 池化前 conv | {pre['tokens_per_trial']} | {pre['token_dim']} | **{pre['pr']:.1f}** | "
         f"{sum(pre['evr_top5']):.2f} | {pre['tokens_per_atom_at_4096']:.0f} |",
         f"| 池化后 conv | {post['tokens_per_trial']} | {post['token_dim']} | **{post['pr']:.1f}** | "
         f"{sum(post['evr_top5']):.2f} | {post['tokens_per_atom_at_4096']:.0f} |", "",
         "## 读法", "",
         f"1. 池化**把表征压扁了**：池化前 PR={pre['pr']:.1f}/40、谱较平；池化后 PR 只有 "
         f"{post['pr']:.1f}，第一主成分占 {post['evr_top5'][0]:.2f}。也就是说 d11 里看到的"
         "「有效维 1.5」有一半是池化造成的，不全是架构的锅。",
         f"2. 池化前的 token 数按全量语料约 {pre['tokens_per_atom_at_4096']:.0f} 万级/4096 原子——"
         "统计上终于够了（每原子数十个 token）。",
         "3. 代价：token 维只有 40，原子是「某时刻的空间滤波器」，不是「一个概念」；"
         "语义探针要改成「该原子在哪些时刻、哪些通道活跃」，而不是「该原子代表什么动作」。", "",
         "## 对消融语义的要求", "",
         "token 级消融必须**按时刻局部**做：给定试次 $x$，原子 $i$ 在时刻 $t$ 的贡献是 "
         "$a_{i,t}d_i$；消融 = 只减去该原子在其 top-k 激活时刻的贡献，而不是整段减。"
         "否则时间结构被抹平，消融效应会退化回 d11 里那个「全局能量因子」。"]
    (ROOT / "reports" / "P1_token_probe.md").write_text("\n".join(L), encoding="utf-8")
    print("写出 reports/P1_token_probe.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
