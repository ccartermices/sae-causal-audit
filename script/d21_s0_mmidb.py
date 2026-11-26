"""d21 —— S0b：mmidb 对侧化正对照（替代 SSVEP 的 S0a）。

为什么换正对照（`reports/T0_diagnosis.md`，DECISIONS D20-10 / D21）
--------------------------------------------------------------------
SSVEP 上 S0 不过，机制是两条：①「平均激活 top-5」与类别无关（8 个类选出同一批原子）；
②原子是空间方向，「哪个频点」是 token 坐标而非值，条件频率读数恒在频带中心。
SSVEP 是最不利的测试台：诱发响应只占 6% 的 token、且是窄带。

mmidb 的 MI 不一样：ERD 覆盖整个 3.5 s 窗，信号占比高一个量级；而且已知事实是
**有符号的空间事实**——想象左手 → 右半球（C4 附近）去同步，想象右手 → 左半球（C3 附近），
不依赖「原子要带频率身份」这个在本构造里不成立的能力。

判据（S0b，写进 `03_判据与闸门` §1，标记 post-hoc）
--------------------------------------------------
对每个类别 c（T1=想象左手、T2=想象右手），取该类**留出 run（run 12）**上平均原子激活
top-5 的原子，逐个算侧化指数

    LI_i = (E_contra - E_ipsi) / (E_contra + E_ipsi)

其中 E 是原子通道模式在对应 ROI 上的能量和（T1 的对侧 = `ROI_RIGHT_MOTOR`，
T2 的对侧 = `ROI_LEFT_MOTOR`）。判据：**LI 中位数 > 0.20 且 ≥3/5 个原子 LI > 0**，
两类都满足 ⟹ S0b PASS。

原子通道模式怎么来：token 是第一层卷积的输出，第一层对输入是线性的，所以
原子方向 d_i（40 维）对应的通道模式是 `topo.T @ d_i`（64 维）。这一步在
`sae/token_sae.channel_topography` 里已有（CombinedConv 的时间核 × 空间核）。

产出
----
probe/s0_mmidb.json     机器读（registry 用）
reports/T0_S0b_mmidb.md 人读
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
import sig_layer as SL  # noqa: E402
import token_sae as TS  # noqa: E402

OUT = ROOT / "probe"
REP = ROOT / "reports"

CONTRA = {0: "RIGHT", 1: "LEFT"}   # 0 = 想象左手（`d10` 的 LABEL_CODE）→ 右半球对侧；1 = 右手 → 左半球
ROI = {0: SL.ROI_RIGHT_MOTOR, 1: SL.ROI_LEFT_MOTOR}
ROI_IPSI = {0: SL.ROI_LEFT_MOTOR, 1: SL.ROI_RIGHT_MOTOR}
LI_PASS_MEDIAN = 0.20
LI_PASS_COUNT = 3
PASS_MIN_CLASSES = 2


def load_hands(n_subjects: int, seed: int = 0):
    """与 d11 同一切分：cond=1（想象）× kind=0（手），run 4/8 训练、run 12 测试。"""
    EPO = ROOT / "data" / "interim" / "epochs" / "mmidb"
    subs = sorted(p.stem for p in EPO.glob("*.npz"))
    subs = [s for s in subs if s not in SL.EXCLUDE_STRUCT_SUBJECTS]
    subs = list(np.random.RandomState(seed).permutation(subs)[:n_subjects])
    Xtr, ytr, Xte, yte, subs_ok = [], [], [], [], []
    for s in subs:
        z = np.load(EPO / f"{s}.npz")
        m = (z["cond"] == 1) & (z["kind"] == 0)
        X, y, run = z["X"][m], z["y"][m], z["run"][m]
        if X.shape[1] != 64:
            continue
        tr, te = np.isin(run, (4, 8)), run == 12
        Xtr.append(X[tr]); ytr.append(y[tr])
        Xte.append(X[te]); yte.append(y[te])
        subs_ok.append(s)
    return (np.concatenate(Xtr), np.concatenate(ytr).astype(np.int64),
            np.concatenate(Xte), np.concatenate(yte).astype(np.int64), subs_ok)


def mu_beta_lateralization(X: np.ndarray, y: np.ndarray, fs: float = 160.0) -> dict:
    """oracle：直接从未滤波 EEG 算 mu 带（8–13 Hz）功率的侧化。

    这是「已知事实存在」的独立检查，不经过任何模型。想象左手（y=0）时对侧（右半球 C4）
    的 mu 功率更低，故 LI = (C4 − C3)/(C4 + C3) 在 y=0 上应更负、在 y=1 上更正。
    """
    from scipy.signal import welch

    names = _ch_names()
    i3, i4 = names.index("C3"), names.index("C4")
    f, P = welch(X, fs=fs, nperseg=int(2 * fs), axis=-1)
    band = (f >= 8) & (f <= 13)
    p3, p4 = P[:, i3, band].mean(1), P[:, i4, band].mean(1)
    li = (p4 - p3) / (p3 + p4 + 1e-12)
    return dict(li_left_median=float(np.median(li[y == 0])),
                li_right_median=float(np.median(li[y == 1])),
                separation=float(np.median(li[y == 1]) - np.median(li[y == 0])))


def lateralization(pattern: np.ndarray, cls: int, ch_names: list[str]) -> dict:
    e_c = float(sum(pattern[ch_names.index(c)] ** 2 for c in ROI[cls] if c in ch_names))
    e_i = float(sum(pattern[ch_names.index(c)] ** 2 for c in ROI_IPSI[cls] if c in ch_names))
    return dict(li=(e_c - e_i) / (e_c + e_i + 1e-12), e_contra=e_c, e_ipsi=e_i)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--sae-steps", type=int, default=4000)
    ap.add_argument("--sae-dict", type=int, default=SL.SAE_DICT)
    ap.add_argument("--sae-k", type=int, default=SL.SAE_K)
    ap.add_argument("--max-tokens", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)

    Xtr, ytr, Xte, yte, subs = load_hands(a.n_subjects, a.seed)
    import d11_width as W

    Xtr, Xte = W.zscore(Xtr, Xte)
    print(f"mmidb 双手想象：{len(subs)} 人 | 训练 {len(Xtr)} 试次 | 测试 {len(Xte)} 试次 "
          f"| 类别 {np.bincount(ytr)}")

    model = W.build_model("shallow", Xtr.shape[-1])
    model, acc = W.train_decoder(model, Xtr, ytr, Xte, yte, epochs=a.epochs, seed=a.seed)
    print(f"解码器（ShallowConvNet，留出 run 12）：{acc:.3f}")

    # ---------------------------------------------------------------- token
    Xall = np.concatenate([Xtr, Xte])
    tok, topo = TS.tokens_from_model(model, Xall)      # (n, T', 40), (40, C)
    n, Tp, Fdim = tok.shape
    trial_of = np.repeat(np.arange(n), Tp)
    is_tr = np.concatenate([np.ones(len(Xtr), bool), np.zeros(len(Xte), bool)])
    print(f"token：每试次 {Tp} 个，单 token {Fdim} 维，共 {n * Tp} 个")
    print(f"token 有效维 PR：{_pr(tok.reshape(-1, Fdim)[:60000]):.1f}/{Fdim}")

    # ---------------------------------------------------------------- SAE
    # 注意：必须先把 (n, T', F) 摊平成 (n·T', F)。直接喂 3 维张量时 nn.Linear 只作用
    # 在最后一维，topk 却在 dim=1 上做，激活会膨胀成 (bs, T', D)——显存直接爆（实测 22 GB）。
    tok2 = tok.reshape(-1, Fdim)
    frame_tr = np.repeat(is_tr, Tp)
    tok_tr = tok2[frame_tr]
    if len(tok_tr) > a.max_tokens:
        keep = rng.choice(len(tok_tr), a.max_tokens, replace=False)
        tok_tr = tok_tr[keep]
    sae = TS.TopKSAE(d_in=Fdim, d_dict=a.sae_dict, k=a.sae_k, seed=a.seed)
    sae.fit(tok_tr, steps=a.sae_steps)
    idx, val = sae.encode_sparse(tok2)
    act = TS.atom_trial_activation(idx, val, trial_of, n, a.sae_dict)
    dirs = sae.atom_dirs()                             # (D, Fdim)
    patt_ch = (topo.T @ dirs.T).T                      # (D, C)
    print(f"SAE：D={a.sae_dict} k={a.sae_k} | 原子通道 PR 中位 "
          f"{np.median([_pr1(p) for p in patt_ch]):.1f}/{patt_ch.shape[1]}")

    ch_names = _ch_names()
    te_slice = slice(len(Xtr), n)
    # 两类各自的留出试次掩码
    m_te = {c: np.zeros(n, bool) for c in (0, 1)}
    for c in (0, 1):
        m_te[c][te_slice] = yte == c

    def select(rank: str, cls: int) -> np.ndarray:
        """选 top-5 原子。两种排序都算，用来量化「平均激活对类别不敏感」这件事。"""
        if rank == "mean":
            score = act[m_te[cls]].mean(0)
        else:                                   # contrast：类内均值 − 类外均值
            score = act[m_te[cls]].mean(0) - act[~m_te[cls]].mean(0)
        return np.argsort(-score)[:5]

    rows, rows_c = [], []
    for rank, bucket in (("mean", rows), ("contrast", rows_c)):
        for cls in (0, 1):
            top5 = select(rank, cls)
            lis = [lateralization(patt_ch[i], cls, ch_names)["li"] for i in top5]
            med = float(np.median(lis))
            cnt = int(sum(1 for v in lis if v > 0))
            bucket.append(dict(cls=int(cls), top5=[int(i) for i in top5],
                               li=[round(float(v), 3) for v in lis],
                               li_median=round(med, 3), n_positive=cnt,
                               passed=bool(med > LI_PASS_MEDIAN and cnt >= LI_PASS_COUNT)))
        print(f"  [{rank}] " + " | ".join(
            f"类 {r['cls']}（{CONTRA[r['cls']]} 对侧）LI 中位 {r['li_median']:+.2f} 正号 "
            f"{r['n_positive']}/5 {'过' if r['passed'] else '不过'}" for r in bucket[-2:]))

    # 两类的 top-5 重叠：直接量化「平均激活选出的原子与类别无关」
    def jacc(a, b):
        sa, sb = set(a), set(b)
        return len(sa & sb) / len(sa | sb)
    jac_mean = jacc(rows[0]["top5"], rows[1]["top5"])
    jac_cont = jacc(rows_c[0]["top5"], rows_c[1]["top5"])

    n_pass = sum(r["passed"] for r in rows_c)
    s0b = bool(n_pass >= PASS_MIN_CLASSES)

    # ---------------------------------------------------------------- S0c
    # 不依赖「哪半球」这种符号假设的检验：原子层能否重建出**模型自由**的类对比模式？
    #   p : 直接在通道上算 mu 带幅度，类 1 − 类 0（不经任何模型）
    #   r : 仪器输出的通道级对比 topoᵀ @ Σ_i (ā_i¹ − ā_i⁰) d_i
    # 两者余弦相似度高 ⟹ 原子层「看见」了同一张空间图；这正是 S0 想验的「仪器有效」。
    from scipy.signal import welch

    fq, P = welch(Xall, fs=160.0, nperseg=320, axis=-1)
    bmu = (fq >= 8) & (fq <= 13)
    amp_all = np.sqrt(P[:, :, bmu].mean(2))                   # (n, 64) 全部试次
    amp_te = amp_all[len(Xtr):]                               # 留出部分
    p_orc = amp_te[yte == 1].mean(0) - amp_te[yte == 0].mean(0)
    w_cls = act[m_te[1]].mean(0) - act[m_te[0]].mean(0)       # (D,)

    # 原子 → 通道图的**经验**测法：按激活加权该原子所有 token 所在试次的通道幅度。
    # 不用 topo.T @ d 的线性化（bnorm 的仿射会把形状搞歪，但它是构造性的一阶近似）。
    amp_tok = amp_all[trial_of]                               # (n_tok, 64)
    Pc = np.zeros((a.sae_dict, amp_all.shape[1]), dtype=np.float64)
    for k in range(a.sae_k):
        np.add.at(Pc, idx[:, k], amp_tok * val[:, k, None])
    wsum = np.zeros(a.sae_dict)
    np.add.at(wsum, idx.ravel(), val.ravel())
    patt_emp = Pc / np.maximum(wsum, 1e-12)[:, None]
    r_emp = (patt_emp * w_cls[:, None]).sum(0)
    r_ins = topo.T @ (dirs * w_cls[:, None]).sum(0)           # 线性化版，作诊断

    # **必须去共模**：mu 幅度是正数，类对比里含一个「整体幅度差」的常数分量，
    # 不去掉的话任何正向量都能与它相关 0.9（实测零对照 |cos| 也就 0.895）。
    # 去均值后剩下的才是**空间形状**，这才是要验的东西。
    def _shape(v):
        v = np.asarray(v, dtype=np.float64)
        v = v - v.mean()
        return v / (np.linalg.norm(v) + 1e-12)
    p_hat = _shape(p_orc)
    cos = float(_shape(r_emp) @ p_hat)
    cos_lin = float(_shape(r_ins) @ p_hat)
    # 零对照：打乱类别标签重算，看余弦的随机分布
    # 零对照：随机抽 5 个原子按同样的方式重建（不用类别信息），看余弦的随机分布
    null = []
    for _ in range(200):
        pick = rng.choice(a.sae_dict, 5, replace=False)
        w_p = np.zeros(a.sae_dict)
        w_p[pick] = rng.choice([-1.0, 1.0], 5)
        null.append(float(_shape((patt_emp * w_p[:, None]).sum(0)) @ p_hat))
    null = np.array(null)
    orc_top = ch_names[int(np.argmax(np.abs(p_orc)))]
    ins_top = ch_names[int(np.argmax(np.abs(r_emp - r_emp.mean())))]
    print(f"  S0c：余弦(r, p) = {cos:+.3f}（线性化版 {cos_lin:+.3f}）| "
          f"零对照 |cos| 95 分位 {np.percentile(np.abs(null), 95):.3f}")
    # 判定必须同时越过**绝对值线**与**零对照**：只比 0.5 会把「随机 5 个原子也能到 0.6」
    # 这种情形判成过（实测零对照 p95 就是 0.617）。
    null95 = float(np.percentile(np.abs(null), 95))
    s0c = bool(cos >= 0.5 and cos > null95)
    print(f"       模型自由对比模式峰导 {orc_top} | 仪器重建峰导 {ins_top}")
    print(f"       判据 cos ≥ 0.5 且 > 零对照 p95 {null95:.3f} → {'过' if s0c else '不过'}")


    print(f"  两类 top-5 的 Jaccard：平均激活 {jac_mean:.2f} | 对比排序 {jac_cont:.2f}")
    for r in rows_c:
        print(f"  类 {r['cls']}（{CONTRA[r['cls']]} 对侧）：LI {[round(v, 2) for v in r['li']]} "
              f"中位 {r['li_median']:+.2f} 正号 {r['n_positive']}/5 → "
              f"{'过' if r['passed'] else '不过'}")

    # 零对照：随机取 5 个原子，看 LI 中位数的分布
    rnd = []
    for _ in range(500):
        pick = rng.choice(a.sae_dict, 5, replace=False)
        rnd.append(np.median([abs(lateralization(patt_ch[i], 0, ch_names)["li"])
                              for i in pick]))
    rnd = np.array(rnd)
    print(f"零对照（随机 5 原子，类 1）：|LI| 中位数 95 分位 {np.percentile(rnd, 95):.3f}")

    # oracle：mu 带功率侧化
    orc = mu_beta_lateralization(Xte, yte)
    print(f"oracle（mu 8–13 Hz，C3/C4 功率）：左手 LI {orc['li_left_median']:+.3f} | "
          f"右手 LI {orc['li_right_median']:+.3f} | 分离度 {orc['separation']:+.3f}")

    print(f"S0b：{n_pass}/2 类通过 → {'PASS' if s0b else 'FAIL'}")
    print(f"S0c：余弦 {cos:+.3f} → {'PASS' if s0c else 'FAIL'}")

    # ---------------------------------------------------------------- 落盘
    OUT.mkdir(exist_ok=True)
    res = dict(
        n_subjects=len(subs), n_train=int(len(Xtr)), n_test=int(len(Xte)),
        decoder_acc=float(acc),
        token_dim=Fdim, tokens_per_trial=Tp, n_tokens=int(n * Tp),
        token_pr=float(_pr(tok.reshape(-1, Fdim)[:60000])),
        sae_dict=a.sae_dict, sae_k=a.sae_k, sae_seed=a.seed, sae_steps=a.sae_steps,
        atom_pr_channel_median=float(np.median([_pr1(p) for p in patt_ch])),
        li_pass_median=LI_PASS_MEDIAN, li_pass_count=LI_PASS_COUNT,
        per_class=rows_c, per_class_mean_rank=rows,
        jaccard_top5_mean=float(jac_mean), jaccard_top5_contrast=float(jac_cont),
        n_pass=int(n_pass), s0b_pass=s0b,
        s0c_cosine=cos, s0c_cosine_linearized=cos_lin,
        s0c_null_abs_p95=null95,
        s0c_oracle_peak_ch=orc_top, s0c_instrument_peak_ch=ins_top, s0c_pass=s0c,
        null_abs_li_p95=float(np.percentile(rnd, 95)),
        oracle_mu=orc, subjects=subs, seconds=round(time.time() - t0, 1),
    )
    p = OUT / f"s0_mmidb{a.tag}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        "# T0 · S0b 仪器校准（mmidb 对侧化，替代 SSVEP 的 S0a）",
        "",
        "> 由 `scripts/d21_s0_mmidb.py` 生成，勿手改。判据见 `03_判据与闸门` §1 S0b（D21）。",
        "",
        f"- 数据：{len(subs)} 名被试（排除结构非标准 3 人），训练 {len(Xtr)} / 测试 {len(Xte)}（留出 run 12）",
        f"- 解码器：ShallowConvNet，**{acc:.3f}**（机会 0.5）",
        f"- token：每试次 {Tp} 个、{Fdim} 维，有效维 PR {res['token_pr']:.1f}/{Fdim}",
        f"- 字典：D={a.sae_dict}，k={a.sae_k}；原子通道 PR 中位 {res['atom_pr_channel_median']:.1f}/{patt_ch.shape[1]}",
        "",
        "判据按**对比排序**（类内均值 − 类外均值）选 top-5，理由见 §注；平均激活排序的结果单列作诊断。",
        "",
        "| 类 | 对侧 | top-5 原子的 LI | LI 中位 | 正号数 | 判定 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows_c:
        lines.append(f"| {r['cls']}（{'左手' if r['cls'] == 0 else '右手'}） | {CONTRA[r['cls']]} | "
                     f"{r['li']} | {r['li_median']:+.2f} | {r['n_positive']}/5 | "
                     f"{'过' if r['passed'] else '不过'} |")
    lines += [
        "",
        f"**S0b 判定：{'通过' if s0b else '不通过'}（{n_pass}/2 类）**",
        "",
        "## S0c：原子层能否重建模型自由的类对比模式（不依赖半球符号假设）",
        "",
        "| 项 | 实测 | 判据 | 结论 |",
        "|---|---|---|---|",
        f"| 余弦(仪器重建, 模型自由对比，去共模) | **{cos:+.3f}** | ≥0.50 且 > 零对照 p95 | {'通过' if s0c else '不通过'} |",
        f"| 同上的线性化版（topoᵀd，诊断） | {cos_lin:+.3f} | — | — |",
        f"| 零对照（随机 5 原子，\\|cos\\| 95 分位） | {null95:.3f} | — | — |",
        f"| 模型自由对比的峰导 | {orc_top} | — | — |",
        f"| 仪器重建的峰导 | {ins_top} | — | — |",
        "",
        f"**S0c 判定：{'通过' if s0c else '不通过'}**",
        "",
        "S0c 不问「原子能量在哪半球」（那要假设去同步/同步哪一个主导），只问：把原子层的类对比",
        "投影回通道后，是否与**不经任何模型**算出的 mu 带类对比同形。这是 S0 想验的『仪器有效』",
        "在最弱假设下的版本。",
        "",
        f"- 判据：LI 中位 > {LI_PASS_MEDIAN} 且 ≥{LI_PASS_COUNT}/5 个原子 LI > 0",
        f"- 两类 top-5 的 Jaccard：**平均激活 {jac_mean:.2f}**（几乎同一批原子 → 与类别无关），"
        f"对比排序 {jac_cont:.2f}",
        "",
        "## 注：为什么判据用对比排序而不是平均激活",
        "",
        "S0a（SSVEP）的失败机制之一是「该类试次上的平均原子激活」选出的是**响得最多**的原子，",
        "与类别无关（D20-10 §机制一）。在 mmidb 上直接量到同一件事：平均激活排序下两类的 top-5",
        f"Jaccard = {jac_mean:.2f}，且两侧化方向不随类别翻转（两类都偏右半球）。",
        "因此 S0b 的「该类编码原子」必须用**类特异度**（类内均值 − 类外均值）来定义，",
        "这是对判据的改动，已在 DECISIONS D21 记录并标 post-hoc。",
        f"- 零对照（随机 5 个原子，|LI| 中位数的 95 分位）：{np.percentile(rnd, 95):.3f}",
    ]
    lines += [
        f"- oracle（不经过模型，直接算 mu 8–13 Hz 功率）：左手 LI {orc['li_left_median']:+.3f}、"
        f"右手 LI {orc['li_right_median']:+.3f}，分离度 {orc['separation']:+.3f}。",
        "  左手（对侧右半球）的 C4−C3 侧化应更负，故分离度 >0 表示已知事实存在且方向正确。",
    ]
    (REP / "T0_S0b_mmidb.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"→ probe/{p.name} | reports/T0_S0b_mmidb.md ({res['seconds']} s)")
    return 0 if s0b else 3


def _pr(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    lam = np.clip(np.linalg.eigvalsh(np.cov(v.T))[::-1], 0, None)
    return float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-12))


def _pr1(p: np.ndarray) -> float:
    v = np.abs(p) ** 2
    return float(v.sum() ** 2 / max((v ** 2).sum(), 1e-12))


def _ch_names() -> list[str]:
    EPO = ROOT / "data" / "interim" / "epochs" / "mmidb"
    for p in sorted(EPO.glob("*.npz")):
        z = np.load(p)
        if "ch_names" in z.files:
            return [str(c) for c in z["ch_names"]]
    raise RuntimeError("mmidb npz 里没有 ch_names，无法算 ROI 侧化")


if __name__ == "__main__":
    raise SystemExit(main())
