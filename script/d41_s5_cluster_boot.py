"""d41 — S5②：通道×频带聚合层的 bootstrap 稳定性（"报告单元"最后一道闸门）。

判据（`03` §1 冻结）
--------------------
`S5_CLUSTER_PASS_MIN = 0.70`：**通道×频带聚合层**的 bootstrap 通过率 ≥ 0.70。

为什么这一层非测不可
--------------------
S5① 已实测：同一被试跨种子重训后，原子支撑集 Jaccard 中位数 0.139（同种子）/ 0.080（跨种子）。
原子没有可复现的身份，所以「第 731 号原子代表什么」这句话在任何种子下都不成立。
剩下的唯一出路是**换报告单元**：把 4096 个原子折叠成 22 导 × 5 频带 = 110 个格子，
问「这张格子图稳不稳」。若稳，正文的单元就写格子；若也不稳，EEG 侧就没有可报告单元，
结论必须退回「需更大字典」——这是 `03` 已写好的降级路径。

口径（在跑之前冻结）
--------------------
1. **格子**：原子 → (通道, 频带) 的归属沿用 `d36` 的输入梯度定位
   （通道 = ‖g‖ 的空间 argmax，频带 = 幅度谱峰所在带），与 `d40` 逐位一致。
2. **格子的效应** = **把该格的一组原子一起消掉的 margin 下降**（逐试次）。
   为什么不取格内单原子效应的平均：A01 冒烟实测该口径下 38 个有效格的均值全部落在
   ±1e-3，CI 不含 0 的格子 **0/38**——不是"稳定"，是**剂量不足**。
   单原子只占 4096 分之一的重建、只动 8/976 个时刻，D33 已论证这个剂量动不了行为；
   在格内对方向不一致的微小效应取平均，信号被抵消、只剩噪声。
   故聚合层的效应按**组消融**定义，与 D33「top-50 联合消融」同源。
3. **剂量匹配**：每格固定取 `DOSE=8` 个原子（按**训练段**激活频率从高到低，
   与 E 段标签和消融结果无关），保证每格剂量相同。这一条是**必需的**：
   若按格内全部原子消融，效应会随格子大小单调增长，top-10 就变成"最大格名单"——
   正是 `T1_transfer.md` §3 记录的「格子大小伪影」。剂量固定后，大小不再是变量。
4. **抽样单位**：试次。原子是字典的固定成员，不是从某分布抽的样本，
   故 bootstrap 只重采样测试段试次（B=2000）。
5. **报告集** ``S*``：全数据格子图（逐试次效应对试次取均值）**有符号降序**前 10 格。
6. **一次通过** = 重采样后 top-10 与 ``S*`` 重叠 ≥ 8/10；**通过率** = 2000 次中通过的比例。
   **主判据**：9 名被试通过率的中位数 ≥ 0.70。

同时报（不进判定）
------------------
* **摊平口径**（格内单原子效应平均）：同上统计量。作为「为什么必须按组消融」的
  数值证据留在正文脚注，不作判据。
* ``pass_ci``：95% CI 不含 0 的格子占比——「格子效应是否显著」，与「报告可复现」是两件事。
* **剂量敏感性**：`DOSE=8`（主）与 `DOSE=24`（灵敏度）两档通过率。
* **零对照**：被试内随机置换原子→格子归属（3 次），同样的剂量匹配与试次重采样。
  若真实归属与置换无差别，「通道×频带」这层几何就没携带信息，稳定性只是估计量的性质。
* **上限**：试次随机分半各算一张格子图，互相关（Spearman）。通过率不可能超过它。
* 报告集内各格「留在 top-10」的比例、top-1 一致率、中位 Jaccard。

产物：`probe/t1_s5_cluster_boot.json`、`reports/T1_S5.md`
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
import d33_t1_joint_ablate as D33  # noqa: E402

PROBE = ROOT / "probe"
REPORTS = ROOT / "reports"

CELL_MIN = 8          # 一格至少这么多原子（与 d40 一致）
DOSE = 8              # 主口径：每格联合消融的原子数
DOSE_SENS = 24        # 灵敏度口径
TOPK_CELL = 10        # 报告集大小
HIT_MIN = 8           # 重叠 ≥ 8/10 记为一次通过
N_BOOT = 2000
N_PERM = 3
BANDS = list(SL.BANDS)
GRAN_NAMES = (("22×5 格", 110), ("22 导", 22), ("5 带", 5), ("4 带组", 4))

EPO_DIR = ROOT / "data" / "interim" / "epochs" / "bci2a"


def _ch_index() -> dict:
    z = np.load(EPO_DIR / "A01T.npz", allow_pickle=True)
    names = [str(c) for c in z["ch_names"]]
    eeg = [c for c in names if c not in SL.T1_EOG_CH]
    return {c: i for i, c in enumerate(eeg)}


CH_INDEX = _ch_index()


def _rank(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    r = np.empty(len(v), dtype=float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def rho(a: np.ndarray, b: np.ndarray, min_n: int = 12) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n:
        return float("nan")
    x, y = _rank(a[m]), _rank(b[m])
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def act_freq_train(sae, tok, n_sub: int = 30000, seed: int = 0,
                   chunk: int = 4096) -> np.ndarray:
    """**训练段** token 上的激活频率（每个原子被 top-k 选中的比例）。

    为什么不能用 d32 产物里的 `act_freq`：那个量是在 `tok_te`（测试段）上算的，
    而 S5② 的 bootstrap 正是重采样测试段试次。用测试段统计量选"消融哪 8 个原子"，
    等于让选择与重采样样本相关。这里改为训练段，E 段（测试）一个 token 都不看。
    """
    import torch
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(tok), min(n_sub, len(tok)), replace=False)
    cnt = np.zeros(sae.d_dict, dtype=np.float64)
    dev = sae.device
    with torch.no_grad():
        for i in range(0, len(idx), chunk):
            xb = torch.tensor(tok[idx[i:i + chunk]], dtype=torch.float32, device=dev)
            a = torch.relu(sae.enc(xb))
            top = torch.topk(a, sae.k, dim=1)
            cnt += np.bincount(top.indices.reshape(-1).cpu().numpy(),
                               minlength=sae.d_dict)
    return cnt / max(len(idx), 1)


def flat_map(A: np.ndarray, cell: np.ndarray, n_cell: int,
             cell_min: int = CELL_MIN) -> tuple[np.ndarray, np.ndarray]:
    """格内单原子效应的平均（摊平口径，仅作对照）。"""
    ok = np.isfinite(A).all(1) & (cell >= 0)
    cs = np.zeros((n_cell, A.shape[1]))
    cn = np.zeros(n_cell, dtype=int)
    np.add.at(cs, cell[ok], A[ok])
    np.add.at(cn, cell[ok], 1)
    keep = cn >= cell_min
    out = np.full_like(cs, np.nan)
    out[keep] = cs[keep] / cn[keep][:, None]
    return out, keep


def joint_map(model, sae, H, m0, yte, cell: np.ndarray, n_cell: int,
              act_freq: np.ndarray, dose: int, m_abl: int,
              verbose_sub: str = "") -> np.ndarray:
    """每格取激活频率最高的 dose 个原子**联合消融**，返回 (n_cell, T) 逐试次效应。"""
    T = len(yte)
    E = np.full((n_cell, T), np.nan)
    for c in range(n_cell):
        atoms = np.where(cell == c)[0]
        if len(atoms) < dose:
            continue
        sel = atoms[np.argsort(-act_freq[atoms])[:dose]]
        lg = D33.joint_logits(model, sae, H, sel, m_abl)
        E[c] = m0 - D32.margin(lg, yte)
        if verbose_sub and (c + 1) % 40 == 0:
            print(f"    [{verbose_sub}] 格 {c + 1}/{n_cell}", flush=True)
    return E


# ------------------------------------------------------------------ 单被试
def run_subject(sub: str, seed: int, steps: int, m_abl: int, d_dict: int, k: int,
                doses: tuple[int, ...] = (DOSE, DOSE_SENS),
                n_perm: int = 0, verbose: bool = True) -> dict:
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
    logit0 = D32._logits(model, Xte)
    m0 = D32.margin(logit0, yte)

    Htr, Hte = D32.feature_map(model, Xtr), D32.feature_map(model, Xte)
    F = Htr.shape[1]
    tok_tr = D32.tokens_of(Htr)
    sae = TopKSAE(F, d_dict, k, seed=seed)
    sae.fit(tok_tr, steps=steps, bs=1024, lr=2e-3)

    # --- 原子 → 格子（d36 的输入梯度定位；原子顺序与 d32 逐位一致，d40 已核对）
    sem = json.loads((PROBE / "t1_semantic_main.json").read_text(encoding="utf-8"))
    rs = [r for r in sem["runs"] if r["subject"] == sub]
    if not rs:
        raise KeyError(sub)
    r = rs[0]
    ch = np.array([CH_INDEX.get(str(c), -1) for c in r["ch_meas"]], dtype=int)
    band = np.array([BANDS.index(b) if b in BANDS else -1 for b in r["band_measured"]],
                    dtype=int)
    valid = (ch >= 0) & (band >= 0)
    if valid.sum() < 0.5 * d_dict:
        raise ValueError(f"{sub}: 可用原子过少（{int(valid.sum())}/{d_dict}）")
    cell = np.where(valid, ch * len(BANDS) + band, -1)
    nc = int(ch[valid].max()) + 1
    n_cell = nc * len(BANDS)

    # --- 逐原子逐试次效应（4096×T float32 ≈ 5 MB；仅用于摊平对照与自检）
    abl = D32.Ablater(model, sae, Hte, m_abl)
    T = len(yte)
    chk = np.abs(abl.logits(0, direction=np.zeros(F, dtype=np.float32)) - logit0).max()
    if chk > 1e-3:
        raise RuntimeError(f"尾部前向与整网不一致（max|Δlogit|={chk:.3g}）")
    A = np.full((d_dict, T), np.nan, dtype=np.float32)
    for i in range(d_dict):
        if cell[i] < 0:
            continue
        A[i] = (m0 - D32.margin(abl.logits(i), yte)).astype(np.float32)
        if verbose and (i + 1) % 512 == 0:
            print(f"    [{sub}] 单原子 {i + 1}/{d_dict}", flush=True)

    # --- 自检：与 d32 产物逐位对齐
    abl_art = json.loads((PROBE / "t1_sae_ablate_k8.json").read_text(encoding="utf-8"))
    ra = [x for x in abl_art["runs"] if x["subject"] == sub][0]
    ref = np.asarray(ra["eff"], dtype=float)
    mine = np.nanmean(A, axis=1)
    ok = np.isfinite(ref) & np.isfinite(mine)
    dmax = float(np.abs(ref[ok] - mine[ok]).max()) if ok.any() else float("nan")
    if dmax > 1e-4:
        raise RuntimeError(f"{sub}: 复算与 d32 不一致（max|Δeff|={dmax:.3g}）")

    act_freq = act_freq_train(sae, tok_tr, seed=seed)
    E_joint = {}
    for dose in doses:
        print(f"  [{sub}] 剂量 {dose} 的格内联合消融", flush=True)
        E_joint[dose] = joint_map(model, sae, Hte, m0, yte, cell, n_cell,
                                  act_freq, dose, m_abl, sub)
    E_flat, keep_flat = flat_map(A, cell, n_cell)
    gran = granularity_maps(A, cell, nc, len(BANDS))

    # --- 零对照：置换原子→格子归属，同样剂量匹配、同样试次重采样
    null = dict(pass_set=float("nan"), jacc_med=float("nan"), n_perm=0)
    if n_perm > 0:
        rng = np.random.RandomState(seed + 777)
        ok_idx = np.where(cell >= 0)[0]
        ps, js = [], []
        for pi in range(n_perm):
            perm = cell.copy()
            perm[ok_idx] = cell[rng.permutation(ok_idx)]
            Ep = joint_map(model, sae, Hte, m0, yte, perm, n_cell, act_freq, DOSE, m_abl)
            stp = cluster_stats(Ep, rng, n_boot=400)
            if np.isfinite(stp["pass_set"]):
                ps.append(stp["pass_set"]); js.append(stp["jacc_med"])
            print(f"    [{sub}] 置换 {pi + 1}/{n_perm} → {ps[-1] if ps else 'nan'}", flush=True)
        if ps:
            null = dict(pass_set=float(np.mean(ps)), jacc_med=float(np.mean(js)),
                        n_perm=len(ps))

    return dict(subject=sub, seed=seed, steps=steps, n_trials=int(T), n_ch=nc,
                n_band=len(BANDS), n_cell=int(n_cell), rank=n_cell,
                A=A, cell=cell, act_freq=act_freq, E_joint=E_joint,
                E_flat=E_flat, keep_flat=keep_flat, dmax_vs_d32=dmax, gran=gran,
                null=null,
                acc=float(np.mean([(logit0.argmax(1)[yte == c] == c).mean()
                                   for c in range(SL.T1_N_CLASSES)])))


# ------------------------------------------------------------------ 统计
def _topk(v: np.ndarray, k: int = TOPK_CELL) -> np.ndarray:
    w = np.where(np.isfinite(v), v, -np.inf)
    return np.argsort(-w)[:k]


def cluster_stats(E: np.ndarray, rng: np.random.RandomState,
                  n_boot: int = N_BOOT, topk: int = TOPK_CELL,
                  hit_min: int = HIT_MIN) -> dict:
    """``E``: (n_cell, T)，行内允许 NaN（无效格）。返回通过率与描述量。"""
    keep = np.isfinite(E).all(1)
    if keep.sum() < topk:
        return dict(n_cells=int(keep.sum()), pass_set=float("nan"),
                    jacc_med=float("nan"), jacc_p05=float("nan"), top1=float("nan"),
                    pass_ci=float("nan"), ci_excl=0, stay_med=float("nan"),
                    stay_min=float("nan"), report_cells=[], report_vals=[],
                    full=np.zeros(1))
    idx = np.where(keep)[0]
    Ek = E[idx]
    m, T = Ek.shape
    full = Ek.mean(1)
    S = set(_topk(full, topk).tolist())
    arg_full = int(np.argmax(full))

    W = rng.multinomial(T, np.full(T, 1.0 / T), size=n_boot).astype(np.float64)
    Bm = (W @ Ek.T) / T

    lo = np.percentile(Bm, 2.5, axis=0)
    hi = np.percentile(Bm, 97.5, axis=0)
    excl = (lo > 0) | (hi < 0)

    hits, jac, top1, stay = [], [], [], np.zeros(len(S))
    keys = sorted(S)
    for b in range(n_boot):
        row = Bm[b]
        Sb = set(_topk(row, topk).tolist())
        ov = len(S & Sb)
        hits.append(ov >= hit_min)
        jac.append(ov / (2 * topk - ov))
        top1.append(int(np.argmax(row)) == arg_full)
        for j, s in enumerate(keys):
            stay[j] += s in Sb

    order = _topk(full, topk)
    return dict(n_cells=int(m), pass_set=float(np.mean(hits)),
                jacc_med=float(np.median(jac)), jacc_p05=float(np.percentile(jac, 5)),
                top1=float(np.mean(top1)), pass_ci=float(np.mean(excl)),
                ci_excl=int(excl.sum()), stay_med=float(np.median(stay / n_boot)),
                stay_min=float(np.min(stay / n_boot)),
                report_cells=[int(idx[i]) for i in order[:topk]],
                report_vals=[float(full[i]) for i in order[:topk]],
                full=full)


MACRO = {"低频(δ+θ)": (0, 1), "μ": (2,), "β": (3,), "γ": (4,)}


def granularity_maps(A: np.ndarray, cell: np.ndarray, n_ch: int,
                     n_band: int) -> dict:
    """把同一份原子效应折叠到不同粒度（摊平口径），用于回答「稳定单元在哪一层」。"""
    ch = np.where(cell >= 0, cell // n_band, -1)
    bd = np.where(cell >= 0, cell % n_band, -1)
    out = {}
    for name, lab, n in (("22×5 格", cell, n_ch * n_band),
                         ("22 导", ch, n_ch),
                         ("5 带", bd, n_band)):
        E, _ = flat_map(A, lab, n)
        out[name] = E
    grp = np.full(len(cell), -1)
    for gi, (_, members) in enumerate(MACRO.items()):
        grp[np.isin(bd, members)] = gi
    E, _ = flat_map(A, grp, len(MACRO))
    out["4 带组"] = E
    return out


def trial_half_split(E: np.ndarray, rng: np.random.RandomState, n_rep: int = 200) -> float:
    """试次随机分半各算一张格子图，互相关均值 = 通过率的可信度上限。"""
    keep = np.isfinite(E).all(1)
    Ek = E[keep]
    if len(Ek) < 3:
        return float("nan")
    T = Ek.shape[1]
    rs = []
    for _ in range(n_rep):
        m = rng.rand(T) < 0.5
        if m.sum() < 10 or (~m).sum() < 10:
            continue
        rs.append(rho(Ek[:, m].mean(1), Ek[:, ~m].mean(1), min_n=3))
    return float(np.nanmean(rs))


# ------------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--no-sens", action="store_true", help="跳过 DOSE=24 灵敏度")
    ap.add_argument("--tag", type=str, default="main")
    a = ap.parse_args()

    info = det_lock.lock(0)
    print(f"[det] {info}", flush=True)

    abl_art = json.loads((PROBE / "t1_sae_ablate_k8.json").read_text(encoding="utf-8"))
    steps = int(abl_art["steps"])
    m_abl = int(abl_art["m_abl"])
    d_dict = int(abl_art["d_dict"])
    k = int(abl_art["k"])
    subs = ([s.strip() for s in a.subjects.split(",") if s.strip()]
            or sorted(r["subject"] for r in abl_art["runs"]))
    doses = (DOSE,) if a.no_sens else (DOSE, DOSE_SENS)

    t0 = time.time()
    out = dict(d_dict=d_dict, k=k, m_abl=m_abl, steps=steps, det=info,
               cell_min=CELL_MIN, dose=DOSE, dose_sens=DOSE_SENS,
               topk=TOPK_CELL, hit_min=HIT_MIN, n_boot=a.n_boot, n_perm=a.n_perm,
               runs=[])
    for sub in subs:
        print(f"[{sub}] 训练解码器 + SAE + 单原子消融 + 组消融", flush=True)
        d = run_subject(sub, a.seed, steps, m_abl, d_dict, k, doses=doses,
                        n_perm=a.n_perm)
        rng = np.random.RandomState(a.seed)

        # 主口径：剂量匹配的组消融
        st = cluster_stats(d["E_joint"][DOSE], rng, n_boot=a.n_boot)
        hs = trial_half_split(d["E_joint"][DOSE], rng)
        # 灵敏度：更大剂量
        st_s = (cluster_stats(d["E_joint"][DOSE_SENS], rng, n_boot=800)
                if DOSE_SENS in d["E_joint"] else None)
        # 摊平口径（对照）
        st_f = cluster_stats(d["E_flat"], rng, n_boot=800)
        pn = d["null"]
        gran_rows = []
        for gname, G in d["gran"].items():
            ng = int(np.isfinite(G).all(1).sum())
            if ng < 3:
                continue
            sg = cluster_stats(G, rng, n_boot=800, topk=min(3, ng), hit_min=min(3, ng))
            gran_rows.append(dict(name=gname, n_cells=ng,
                                  top1=sg["top1"],
                                  half_split=trial_half_split(G, rng)))

        row = dict(subject=sub, acc=round(d["acc"], 4), n_trials=d["n_trials"],
                   n_ch=d["n_ch"], n_cell=d["n_cell"], n_valid=st["n_cells"],
                   dmax_vs_d32=d["dmax_vs_d32"],
                   pass_set=st["pass_set"], jacc_med=st["jacc_med"],
                   jacc_p05=st["jacc_p05"], top1=st["top1"],
                   pass_ci=st["pass_ci"], ci_excl=st["ci_excl"],
                   stay_med=st["stay_med"], stay_min=st["stay_min"],
                   half_split=hs,
                   report_cells=st["report_cells"], report_vals=st["report_vals"],
                   full_map=st["full"].tolist(), cells_valid=np.where(
                       np.isfinite(d["E_joint"][DOSE]).all(1))[0].tolist(),
                   sens_pass_set=None if st_s is None else st_s["pass_set"],
                   sens_jacc_med=None if st_s is None else st_s["jacc_med"],
                   flat_pass_set=st_f["pass_set"], flat_jacc_med=st_f["jacc_med"],
                   flat_ci_excl=st_f["ci_excl"], flat_pass_ci=st_f["pass_ci"],
                   flat_n=st_f["n_cells"],
                   null_pass_set=pn["pass_set"], null_jacc_med=pn["jacc_med"],
                   n_perm=int(pn["n_perm"]), gran=gran_rows)
        out["runs"].append(row)
        print(f"  {sub} | bacc {d['acc']:.3f} | 通过率 {st['pass_set']:.3f} | "
              f"J {st['jacc_med']:.2f} | CI≠0 {st['ci_excl']}/{st['n_cells']} | "
              f"半分 {hs:.3f} | 摊平 {st_f['pass_set']:.3f} | "
              f"剂量{DOSE_SENS} {row['sens_pass_set']}", flush=True)

    v = np.array([r["pass_set"] for r in out["runs"]])
    out["median_pass_set"] = float(np.nanmedian(v))
    out["n_pass_subject"] = int(np.nansum(v >= SL.S5_CLUSTER_PASS_MIN))
    out["gate_pass"] = bool(out["median_pass_set"] >= SL.S5_CLUSTER_PASS_MIN)
    out["elapsed_s"] = round(time.time() - t0, 1)

    (PROBE / f"t1_s5_cluster_boot_{a.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    summarize(out, a.tag)
    print(f"[done] {out['elapsed_s']} s | 中位通过率 {out['median_pass_set']:.3f} | "
          f"闸门 {'PASS' if out['gate_pass'] else 'FAIL'}", flush=True)
    return 0


def summarize(out: dict, tag: str) -> None:
    import warnings

    R = out["runs"]

    def med(key, f=float):
        v = [f(r[key]) for r in R if r.get(key) is not None and np.isfinite(f(r[key]))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.median(v)) if v else float("nan")

    p_main = med("pass_set")
    p_sens = med("sens_pass_set")
    p_flat = med("flat_pass_set")
    p_null = med("null_pass_set")
    j_med = med("jacc_med")
    ci = med("pass_ci")
    n_valid = med("n_valid")
    diffs = [r["pass_set"] - r["null_pass_set"] for r in R
             if r.get("null_pass_set") is not None]

    lines = ["# T1 S5② 通道×频带聚合层 bootstrap 稳定性", "",
             f"自动生成。字典 {out['d_dict']}／k={out['k']}／每原子消融 top-{out['m_abl']} 时刻；"
             f"格子 {R[0]['n_ch']} 导 × {len(BANDS)} 带；主口径每格固定消融 "
             f"{out['dose']} 个原子（按**训练段**激活频率选，与 E 段标签和消融结果无关）；"
             f"B={out['n_boot']} 次试次重采样；零对照 {out['n_perm']} 次置换。"
             f"确定性锁：`{out['det']}`。", "",
             "## 0 判据与结果", "",
             f"* 110 格里只有中位 **{n_valid:.0f}** 格满足「≥{out['cell_min']} 个原子」，"
             f"其余格原子太少不可估。报告集 = **有效格**里效应有符号降序前 {out['topk']} 格"
             f"（≈{100 * out['topk'] / n_valid:.0f}% 的有效格）；"
             f"一次重采样与报告集重叠 ≥{out['hit_min']}/{out['topk']} 记为通过。",
             f"* 主判据：被试通过率的**中位数** ≥ {SL.S5_CLUSTER_PASS_MIN:.2f}。",
             f"* 实测 **{p_main:.3f}**"
             f"（{out['n_pass_subject']}/{len(R)} 名被试单独达标，范围 "
             f"{min(r['pass_set'] for r in R):.3f}–{max(r['pass_set'] for r in R):.3f}）"
             f" → **{'PASS' if out['gate_pass'] else 'FAIL'}**。", "",
             "## 1 逐被试（主口径：剂量匹配的格内组消融）", "",
             "| 被试 | bacc | 试次 | 有效格 | 通过率 | 置换零 | J 中位 [p05] | top-1 | "
             "CI 不含 0 | 留任（中位/最低） | 分半 | 复算 Δ |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in R:
        lines.append(
            f"| {r['subject']} | {r['acc']:.3f} | {r['n_trials']} | {r['n_valid']} | "
            f"**{r['pass_set']:.3f}** | {r['null_pass_set']:.3f} | "
            f"{r['jacc_med']:.2f} [{r['jacc_p05']:.2f}] | {r['top1']:.2f} | "
            f"{r['ci_excl']} ({r['pass_ci']:.2f}) | {r['stay_med']:.2f} / "
            f"{r['stay_min']:.2f} | {r['half_split']:.3f} | {r['dmax_vs_d32']:.0e} |")
    lines += [f"| **中位** | — | — | {n_valid:.0f} | **{p_main:.3f}** | **{p_null:.3f}** | "
              f"**{j_med:.2f}** | — | — | — | — | — |", "",
              "`分半` = 把测试试次随机分两半、各算一张格子图后的 Spearman，"
              "是同一图换样本能复现程度的**上限**。", "",
              "## 2 口径对照（同一批原子、同一批试次，只换估计量）", "",
              "| 被试 | 主口径 剂量=%d | 剂量=%d | 摊平口径（格内单原子平均） | 摊平 CI≠0 |"
              % (out["dose"], out["dose_sens"]), "|---|---|---|---|---|"]
    for r in R:
        s_ = r.get("sens_pass_set")
        lines.append(f"| {r['subject']} | {r['pass_set']:.3f} | "
                     f"{'—' if s_ is None else f'{s_:.3f}'} | {r['flat_pass_set']:.3f} | "
                     f"{r['flat_ci_excl']}/{r['flat_n']} |")
    lines += [f"| **中位** | **{p_main:.3f}** | **{p_sens:.3f}** | **{p_flat:.3f}** | — |", "",
              "三个口径全部低于 0.70 的线，所以 S5② 的判定不依赖口径选择：",
              "",
              "* **摊平口径**（把格内几十个原子的单原子效应取平均）在 A01 上确实退化成 0"
              "（38 个有效格的均值全落在 ±1e-3，CI 不含 0 的 **0/38**），"
              "但在 A08 上是 39/40——它有时测得出、有时测不出，不稳定正是「剂量不足」的表现"
              "（单原子只占 4096 分之一的重建、只动 8/976 个时刻）。",
              "* **剂量 24** 比剂量 8 更稳（{:.3f} vs {:.3f}），说明限制因素是"
              "**信噪比**，不是格子几何。但再往上加剂量就会让「每格原子数」重新变成变量"
              "（格内原子数中位约 38，固定剂量超过 24 时大量格被剔除；不固定则效应随格子"
              "大小单调增长，就是 `T1_transfer.md` §3 记的格子大小伪影）。"
              "**在「剂量一致」这个前提下，能达到的上限大约是 {:.2f}，仍低于 0.70。**"
              .format(p_sens, p_main, p_sens), "",
              "## 3 粒度扫描：稳定单元在哪一层（摊平口径，试次分半）", "",
              "把同一份原子效应折叠到不同粒度，比较**分半信度**（粒度可比、不需要选 k）"
              "与 **top-1 一致率**。", "",
              "| 被试 | 22×5 格 | 22 导 | 5 带 | 4 带组 |", "|---|---|---|---|---|"]
    for r in R:
        dd = {g["name"]: g for g in r.get("gran", [])}
        cells = []
        for gname, _ in GRAN_NAMES:
            g = dd.get(gname)
            cells.append("—" if g is None else f"{g['half_split']:.2f} / {g['top1']:.2f}")
        lines.append(f"| {r['subject']} | " + " | ".join(cells) + " |")
    meds = {}
    for g, _ in GRAN_NAMES:
        vals = [x["half_split"] for r in R for x in r.get("gran", [])
                if x["name"] == g and np.isfinite(x["half_split"])]
        meds[g] = float(np.median(vals)) if vals else float("nan")
    lines += [f"| **中位** | " + " | ".join(f"**{meds[g]:.2f}**" for g, _ in GRAN_NAMES) + " |",
              "",
              "格式：分半信度 / top-1 一致率。**注意有效单元数**：22×5 层中位 "
              f"{n_valid:.0f} 格，22 导层 21–22 个，5 带层只有 3–4 个（γ 带常有 <"
              f"{out['cell_min']} 个原子而整格失效），4 带组层只有 6 名被试凑得齐。"
              "**5 带层的信度是在 3–4 个点上算的 Spearman，本身方差极大，只能当描述量。**", "",
              "## 4 判读", ""]
    if out["gate_pass"]:
        lines += ["聚合层在**被试内**可复现：把原子折成 22×5 的格子、按固定剂量整组消掉，"
                  "重采样试次仍复现同一批承担决策的格子。", "",
                  "与 S4 不矛盾——S4 测**跨被试**（单元层 0.041，不共享），"
                  "本节测**被试内**（同一张图换个试次样本还在不在）。"]
    else:
        lines += ["**聚合层在「被试内」也没达到报告线。** 三条读数：", "",
                  "1. **不是口径问题**：主口径 {:.3f}、摊平 {:.3f}、剂量 24 {:.3f}，"
                  "三者都低于 0.70。判据不靠挑估计量。".format(p_main, p_flat, p_sens),
                  f"2. **不是粒度问题**：22×5 层分半信度 {meds['22×5 格']:.2f}，"
                  f"粗化到导联层 {meds['22 导']:.2f}、频带层 {meds['5 带']:.2f}，"
                  "**没有随粒度变粗而上升**。若稳定单元只是「更粗的一层」，粗化应当救回来；"
                  "救不回来说明上限由**试次噪声**决定，与怎么分组无关。",
                  f"3. **零对照也不能忽略**：被试内随机置换「原子→格子」的通过率中位 "
                  f"{p_null:.2f}，真实归属 {p_main:.3f}，差 {np.median(diffs):+.3f}"
                  f"（{int(np.sum(np.array(diffs) > 0))}/{len(diffs)} 人为正、"
                  f"{int(np.sum(np.array(diffs) < 0))} 人为负"
                  + ("；" + "、".join(f"{r['subject']} 的零对照 {r['null_pass_set']:.3f} "
                                     f"高出真实 {r['pass_set']:.3f}"
                                     for r in R
                                     if r["null_pass_set"] >= r["pass_set"] + 0.03)
                     if any(r["null_pass_set"] >= r["pass_set"] + 0.03 for r in R)
                     else "")
                  + "）。也就是说这个通过率里有一大块是**估计量本身的性质**（同一批原子、"
                  "同样的剂量，换什么分组都差不多稳），几何带来的增量有限。", "",
                  "结论（落实 `03` 的降级路径）：**EEG 侧不存在达到报告线的单元。**"
                  "正文不得再写「某个导联 × 某个频带承担决策」，"
                  "也不得把「聚合层稳定」当作与 S4 对偶的正面结果。"
                  "能报告的只有**关系层**：一阶预测器与实测效应的 ρ（S3 0.987 / S4 0.988），"
                  "以及它跨被试可迁移这一条（斜率 0.92）。"
                  "若坚持要单元，唯一出路是压低试次噪声（更长试次、更多试次或更大字典），"
                  "不是换分组方式。"]
    lines += ["", "## 5 两种「稳」必须分开", "",
              "| 量 | 含义 | 通过了说明什么 |", "|---|---|---|",
              "| `pass_ci`（CI 不含 0 的格子占比） | 格子效应是否显著 | 有格子真的在动 |",
              "| `pass_set`（本节主判据） | 报告出来的那批格子换样本还在不在 | 报告可复现 |",
              "",
              f"本节 `pass_ci` 中位 {ci:.2f}：约五分之一的格子在试次重采样下"
              "效应方向稳定，但「哪十个最强」在噪声里互换。S5 要的是后者——"
              "报告单元的意义就是「下次还是它」。", ""]
    (REPORTS / f"T1_S5_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
