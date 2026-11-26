"""sig_layer —— 冻结层：预处理与语义探针的全部数值定义（v1.0-final, 2026-11-18）。

为什么单独一个文件
------------------
判据 S0–S6 的语言是「枕区集合」「mu 带」「0.5–4.0 s 窗口」。如果这些边界散落在
各个脚本里，改一个数字就能让闸门通过，而论文里看不出来。放在这里之后：

  1. 任何引用它的登记表条目，都会把本文件的哈希记为来源；
  2. 一旦改动，`check_registry.py` 的 C5「未过期」立刻 BLOCK，必须重建并重新解释；
  3. 论文方法节可以直接引用本文件的常量名（`SL.ROI_OCCIPITAL` 等），不存在两处不一致。

改动纪律：任何修改都要在 `DECISIONS.md` 里记一条，并注明是否 post-hoc。

本版冻结依据（每条都不是文献抄来的猜测）
----------------------------------------
* 2a 的试次窗：镜像里 `trial` 字段给出逐试次起点，试次间隔约 2015 样点（8.06 s @250 Hz），
  不是常见的「4 s 试次」。滑窗 CSP+LDA（A01T，282 试次，5 折）实测判别信息：
  1.5–2.5 s 0.287 → **2.5–3.5 s 0.752** → 4.5–5.5 s 0.607 → 6.0–7.0 s 0.397。
  故 MI 窗取 2.0–6.0 s，基线取 0.5–2.0 s（提示期）。证据见 `reports/P0_2a_window.md`。
* SSVEP 的窗：数据集 Readme 明确「500 ms pre-stimulus, 5.5 s post-stimulus onset」，
  1500 点 @250 Hz。故刺激起点 = 样点 125，稳态窗取 0.5–5.0 s post-stim（样点 250–1375）。
* EEGMMIDB：T1/T2 事件时长 4.1–4.2 s，取 0.5–4.0 s，基线 -0.2–0 s。
  三名被试（S088/S092/S100）的试次结构非标准（6.5 s 周期、128 Hz、无独立静息段），
  进 `EXCLUDE_STRUCT_SUBJECTS`，主分析 n=106，敏感性分析单列。
"""

from __future__ import annotations

# ---------------------------------------------------------------- 频带（Hz）
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
N_BANDS = len(BANDS)

# SSVEP 的刺激频段（JFPM，8–15.8 Hz，步长 0.2 Hz）与谱峰容差
SSVEP_STIM_RANGE = (8.0, 15.8)
SSVEP_FREQ_TOL = 0.4  # Hz，判据 S0 的一半（S0 用 ≤0.4 Hz）

# ------------------------------------------------------- T0 仪器（频谱 token，D20）
# token = 「某一频点、某一时刻的 C 导幅度空间模式」。下面四个常量由原理定死：
#   * 窗 2.0 s → 0.5 Hz 分辨率，基频间隔 1 Hz 有两倍余量；
#     更短则 8 类也分不开，更长（如满 6 s）只剩 1 帧、token 失去时间维。
#   * 频带 = 刺激频段 ±1.5 Hz（8.0–15.8 → 6.5–17.3），留出泄漏裙边又不收进谐波。
#   * 8 类 = target % 8（每类 5 个相位偏移）；40 类在 0.5 Hz 分辨率下不可分。
SSVEP_STFT_WIN_S = 2.0
SSVEP_STFT_HOP_S = 0.5
SSVEP_TOKEN_BAND = (6.5, 17.3)
SSVEP_TOKEN_BAND_HARMONIC = (6.5, 33.0)   # 次要分析：含二次谐波（16–31.6 Hz）
T0_CLASSES = 8
S0_PASS_MIN = 6            # 8 类中至少 6 类「地形 + 频率」两项都过（从严于 40 类原文的 7/40）


# ---------------------------------------------------------------- 时间窗（s）
# 逐数据集定义，均相对「事件起点」；不是相对刺激起点的地方会在注释里写明
EPOCHS = {
    "mmidb": (0.5, 4.0),    # 相对 T1/T2 注解 onset（注解时长 4.1–4.2 s）
    "bci2a": (2.0, 6.0),    # 相对 run.trial 起点；滑窗实测判别峰在 2.5–3.5 s
    # 相对刺激起点（= 样点 125）。必须用满 6 s（含刺激前 0.5 s）：
    # 目标间隔 0.2 Hz，窗长 4.5 s 的分辨率只有 0.222 Hz > 0.2 Hz，相邻目标不可分；
    # 实测 4.5 s 窗下 FFT 分类掉到 2%（机会 2.5%），满窗时 40 个目标全部排第一。
    "ssvep": (-0.5, 5.5),
}
BASELINES = {              # None = 不做基线校正
    "mmidb": (-0.2, 0.0),
    "bci2a": (0.5, 2.0),    # 提示期，位于 MI 窗之前
    "ssvep": None,          # 刺激前 0.5 s 已在数据里，留给 S0 的谱峰判据自行使用
}
RESAMPLE_TO = {"mmidb": 160.0, "bci2a": 250.0, "ssvep": 250.0}
BANDPASS = (0.5, 45.0)
NOTCH = {"mmidb": 60.0, "bci2a": 50.0, "ssvep": 50.0}
REJECT_UV = 150.0          # 2a 的剔除阈值（该数据集以 uV 存储且已清理）
# mmidb 的「粗大伪迹」标记阈值：只用于敏感性分析分层，不参与剔除。
# 该数据集 0.5–45 Hz 下峰峰值中位数 266 uV、p90 653 uV，150 uV 会标记 100% 的试次。
ARTIFACT_FLAG_UV = {"mmidb": 500.0, "bci2a": 150.0}

# 语义探针的潜伏期容差
LATENCY_BAND = (0.5, 2.5)
LATENCY_TOL = 0.5          # s，语义得分「潜伏期」一项的判对容差

# ---------------------------------------------------------------- 头皮区域
# 10-10 命名；EEGMMIDB 的 64 导直接可用，2a 的 22 导是其子集
ROI_OCCIPITAL = ["O1", "O2", "Oz", "PO3", "PO4", "POz", "PO5", "PO6"]
ROI_LEFT_MOTOR = ["C3", "FC3", "CP3", "C1", "FC1", "CP1"]
ROI_RIGHT_MOTOR = ["C4", "FC4", "CP4", "C2", "FC2", "CP2"]
ROI_MIDLINE = ["Cz", "FCz", "CPz"]
ROI_FRONTAL_ARTIFACT = ["Fp1", "Fp2", "AF3", "AF4", "F7", "F8"]

# 2a 镜像没有 clab 字段，通道顺序按 BNCI 001-2014 官方定义（与 moabb 一致，已核）
BCI2A_CH_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
    "EOG1", "EOG2", "EOG3",
]
BCI2A_N_EEG = 22
BCI2A_N_EOG = 3

# SSVEP（清华 benchmark）的 64 导，数据集自带 `64-channels.loc` 的顺序。
# .loc 里的大小写不规范（`Fp1 Po3 Poz OZ CB1`），必须规范化后才能与 ROI 集合求交——
# 否则 `PO3/PO4/POz/PO5/PO6` 一个都匹配不上，S0 的地形判据会静默残废（D20-8）。
SSVEP_CH_NAMES = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
    "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1",
    "O1", "Oz", "O2", "CB2",
]
SSVEP_N_CHANS = len(SSVEP_CH_NAMES)

# ---------------------------------------------------------------- token 级 SAE
# 路线 B（DECISIONS D15）：SAE 的输入单元是「卷积特征图的某一个时刻切片」，
# 不是「一个试次」。这三个常量决定字典与稀疏度，改动需重跑 d11/d13 体检。
TOKEN_LAYER = "pre_pool"     # 池化前的 conv 输出（PR=19.8 vs 池化后 1.3）
SAE_DICT = 4096
SAE_K = 8                    # 每 token 激活原子数（D32：40 维 token 上 k=32 等于 80% 维度，
#                              重建 R²=0.984 近恒等；k=8 时 R²=0.885，原子才构成单元）
SAE_TOPK_ABLATE = 8          # 消融时取该原子激活最强的 8 个时刻（局部化，见 D15-2）
SAE_SEEDS = (0, 1, 2)        # 跨种子稳定性（S5 / N6）

# ---------------------------------------------------------------- 判据数值
SEMANTIC_MAX_SCORE = 3          # 地形 / 频带 / 潜伏期 各 1 分
S0_MIN_OCCIPITAL_HITS = 3       # top-5 中至少 3 个落在枕区
S1_RHO_MAX = 0.20               # |rho_sem| 上界
S3_RHO_MIN = 0.45               # 一阶预测器下界
S3_MARGIN = 0.15                # 相对最好 SAE 统计量的领先量
S4_RHO_TRANSFER_MIN = 0.30      # 留一被试下界
# S5 旧口径（跨种子特征级 Jaccard）：仅用于 N6 的对照报数，不再是 S5 的闸门。
# D26：per-trial 框架下因子没有跨试次的固定身份，Jaccard 失去对象，S5 改为下方 PT_* 口径。
S5_JACCARD_MAX = 0.30           # 特征级跨种子 Jaccard 上界（历史值，保留供 N6 报数）
S5_CLUSTER_PASS_MIN = 0.70      # 聚合层 bootstrap 通过率下界（同上）

# S5 新口径（D26）：试次级定位命中率 + 可分辨边界
PT_HIT_COS = 0.80               # 单试次读出与参照方向（经验签名）的 |cos| 命中线
PT_HIT_MIN = 0.60               # 注入试次命中率下界（三选二）
PT_FAR_MAX = 0.05               # 干净试次误报率上界
PT_SVD_K = 24                   # per-trial SVD 对照的成分数（D25/D27）
PT_K_RES = 24                   # per-trial 残差字典追加的方向数
PT_SNR_FLOOR_MIN_DB = 10.0      # 可分辨边界只报「≥ 该值」；低于它不算通过（D26）
PT_SNR_GRID_DB = (None, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0)   # T1 边界扫描网格

# T1 口径（D28）：绝对命中线随维度漂移（22 导下 0.80 只相当于 2.6σ），
# 因此判据改为 AUC + 分离度，绝对命中率降级为描述量。
T1_AUC_MIN = 0.95               # 试次级读出的 AUC 下界
T1_SEP_MIN = 0.60               # 分离度 = 命中(注入) − 命中(干净) 的下界
T1_NULL_AUC_LO = 0.40           # 零对照 AUC 区间
T1_NULL_AUC_HI = 0.60
T1_NULL_SEP_MAX = 0.10          # 零对照分离度上界
T1_BOUND_BAND_DB = (10.0, 15.0)  # 实测边界：+15 稳过、+10 为界

# competent 子集：仅作敏感性分析，主分析不按准确率剔除（避免选择偏倚抬高效应的老问题）
COMPETENT_MIN_KAPPA = 0.20      # 二类 kappa，等价于平衡准确率 0.60
COMPETENT_ALPHA = 0.05          # 同时要求单侧二项检验 p < 0.05

# T1 解码协议（D29）：超参只在 T 段内选，E 段只做一次前向。
T1_CV_FOLDS = 5                 # T 段内分层折数，仅用于选 epoch 数
T1_EPOCH_GRID = (30, 60, 100)   # 候选训练轮数；CV 上取平均验证准确率最高者
T1_N_CLASSES = 4                # 2a：左手/右手/双脚/舌头；机会水平 0.25
T1_EOG_CH = ("EOG1", "EOG2", "EOG3")
# 仪器有效的最低门槛（不是科学结论）：低于它说明解码器没接上标签，
# 后续任何消融读出的「效应」都无法与噪声区分。
T1_DECODER_MIN_KAPPA = 0.10

# ---------------------------------------------------------------- 被试与数据
# 结构非标准：6.5 s 试次周期 / 128 Hz / 无独立静息段，12 个任务 run 全部如此
EXCLUDE_STRUCT_SUBJECTS = ("S088", "S092", "S100")

# ---------------------------------------------------------------- 数据规格
SPEC = {
    "mmidb": dict(n_subjects=109, n_subjects_primary=106, n_chans=64, sfreq=160.0,
                  n_runs=14, n_task_runs=12),
    "bci2a": dict(n_subjects=9, n_chans_eeg=22, n_chans_eog=3, sfreq=250.0,
                  n_sessions=2, n_labeled_runs=6, n_classes=4,
                  n_trials_per_session=288, trial_len_s=8.06),
    "ssvep": dict(n_subjects=35, n_subjects_downloaded=10, n_chans=64, sfreq=250.0,
                  n_targets=40, n_blocks=6, n_samples=1500),
}


def band_of(freq: float) -> str | None:
    """把频率判到频带名；落在边界上取上界所在带（与 numpy 的右开区间一致）。"""
    for name, (lo, hi) in BANDS.items():
        if lo <= freq < hi:
            return name
    return None
