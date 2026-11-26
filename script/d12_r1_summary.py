"""d12 —— 把三份宽度体检合成一张判定表（R1 结论页）。

读 data/width_probe_{shallow,eegnet,shallow_whiten}.json，输出
reports/P1_R1_decision.md：一张跨配置表 + 三条可选路线及其代价。

判定规则与 d11 一致，不在本脚本里重定：
  若 PR<64 且 R2(4096)-R2(1024)<0.01  ⇒ 该配置下 4096 空转。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAGS = [("shallow", "ShallowConvNet（末端池化后）"),
        ("eegnet", "EEGNetv4（末端卷积后）"),
        ("shallow_whiten", "ShallowConvNet + PCA 白化")]


def main() -> int:
    rows, missing = [], []
    for tag, label in TAGS:
        p = ROOT / "data" / f"width_probe_{tag}.json"
        if not p.exists():
            missing.append(tag)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        sw = {r["dict_size"]: r for r in d["sweep"]}
        rows.append(dict(tag=tag, label=label, D=d["feature_dim"], PR=d["participation_ratio"],
                         R=d["pca"]["median_abs_r"], cum10=d["pca"]["cum_at"].get("10"),
                         acc=d["test_acc"], n_train=d["n_train"],
                         r2_128=sw.get(128, {}).get("r2_test"),
                         r2_1024=sw.get(1024, {}).get("r2_test"),
                         r2_4096=sw.get(4096, {}).get("r2_test"),
                         dead4096=sw.get(4096, {}).get("dead_frac"),
                         chosen=d["chosen_dict"], verdict=d["r1_decision"]))
    if missing:
        print("缺少:", missing)

    L = ["# P1 · R1 宽度体检结论", "",
         "> 由 `scripts/d12_r1_summary.py` 从三份 `data/width_probe_*.json` 合成，勿手改。", "",
         "## 1 三个配置的实测", "",
         "| 配置 | 特征维 D | 有效维 PR | 逐维相关中位 \\|r\\| | 前 10 PC 累计 | 解码 acc | "
         "SAE R²(D=128) | R²(1024) | R²(4096) | 判定 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['label']} | {r['D']} | {r['PR']:.1f} | {r['R']:.2f} | "
                 f"{r['cum10']:.2f} | {r['acc']:.3f} | {r['r2_128']:.3f} | "
                 f"{r['r2_1024']:.3f} | {r['r2_4096']:.3f} | {r['verdict']} → {r['chosen']} |")

    L += ["", "## 2 读法（这一节决定了整篇的方法走向）", "",
          "1. **维度不等于自由度。** ShallowConvNet 末端是 1240 维，但第一大主成分就占 "
          "83%，逐维相关中位 0.82，有效维 PR 只有 1.6——「上千维特征图」在统计上近似 1 维。",
          "2. **在低秩空间里，R² 是个会骗人的指标。** 128 个原子、每样本 8 个激活就能把 R² "
          "做到 0.88，因为它只需重建那一个主导方向。R² 高≠字典在做事。",
          "3. **白化不是解药。** 白化把 D 拉到 956、PR 拉到 956（构造使然），但 1194 个训练"
          "样本撑不起这个维数：所有字典尺寸的测试 R² 全为负。样本量是硬约束，不是维数。",
          "4. **EEGNet 反而更健康**（PR=33，D=272），但同样受样本量限制：训练 R² 0.97、"
          "测试 0.67，2048 个原子已经明显过拟合。",
          "", "## 3 三条路线（需拍板）", "",
          "| 路线 | 做法 | 代价 | 能保住的主张 |",
          "|---|---|---|---|",
          "| **A 缩表** | 字典 ≤ 512，k=8–16，挂在 EEGNet 的 272 维上；全量 106 人 ≈ 6000 试次 | "
          "「过完备字典」的叙事没了，改说「统计可辨识的字典」 | 语义–因果解耦仍可测，但原子数少，"
          "稳定性结论会变弱 |",
          "| **B token 级 SAE** | 把卷积特征图按时间切片当 token（每试次 ~30–500 个），"
          "token 数从千级升到百万级；字典 512–4096 才站得住 | 每个原子变成「某时刻的空间滤波器」，"
          "消融语义要重写；工程量最大 | 保住过完备 + 单义原子 + 跨种子可复现 |",
          "| **C 负结果** | 把「EEG 解码器表征低秩 + 样本少 ⇒ 过完备 SAE 不可辨识」本身作为结论，"
          "给出 PR / N/D / R² 饱和三条判据 | 需要更多架构与数据集来支撑普适性 | 与 TMLR 那篇"
          "「SAE 因果效度缺口」接成方法学序列，审稿风险最低 |",
          "",
          "**我的建议：B 为主线，C 作为 B 的对照章节**（既然 B 要按 token 重做，"
          "顺手把 A/C 的可辨识性判据当作前置门槛，成本很低）。理由是只有 B 同时保住"
          "「过完备」与「可解释单元」两个卖点，而 A 会把论文缩成一篇小样本审计，"
          "C 单独成篇则与你已有的 TMLR 稿重叠。", ""]

    tp = ROOT / "data" / "token_probe.json"
    if tp.exists():
        t = json.loads(tp.read_text(encoding="utf-8"))["variants"]
        pre, post = t["pre_pool"], t["post_pool"]
        L[-1:] = [f"> **补充证据（`data/token_probe.json`）**：换成 token 级后，池化前 conv 的 "
                  f"{pre['tokens_per_trial']} 个/试次、{pre['token_dim']} 维 token，有效维 "
                  f"PR={pre['pr']:.1f}、前 5 PC 只占 {sum(pre['evr_top5']):.2f}；而池化后 "
                  f"PR 掉到 {post['pr']:.1f}、第一主成分就占 {post['evr_top5'][0]:.2f}。"
                  f"按全量 6000 试次算，4096 原子时每原子约 "
                  f"{pre['tokens_per_atom_at_4096']:.0f} 个 token——统计上第一次站得住。"
                  f"详见 `reports/P1_token_probe.md`。", ""]

    (ROOT / "reports" / "P1_R1_decision.md").write_text("\n".join(L), encoding="utf-8")
    print("写出 reports/P1_R1_decision.md")
    for r in rows:
        print(f"  {r['tag']:16s} D={r['D']:5d} PR={r['PR']:7.2f} R2@1024={r['r2_1024']:.3f} "
              f"R2@4096={r['r2_4096']:.3f} → {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
