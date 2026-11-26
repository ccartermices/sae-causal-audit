"""P0 标注审计：把「每个被试到底有多少可用试次、采样率是多少」变成事实文件。

为什么必须单独做
----------------
计划里原本假设「109 人 × 14 run × 每类 90 试次 × 160 Hz」。实测发现三处都不对：

  1. 三个被试（S088/S092/S100）的 12 个任务 run 是 **128 Hz**，只有基线 run 是 160 Hz；
  2. 每被试每类试次数在 72–116 之间浮动，不是整齐的 90；
  3. S100 每个 run 每类只有 6 试次（同长度 123 s），S088/S092 是 9–10 试次。

任何「109×90 试次」式的表述都是错的。这些数必须由脚本从原始 EDF 的注解算出来，
写进 registry，正文只许引用登记表的键。

产出
----
  reports/P0_annot_audit.md      人读的审计报告（本脚本生成，不手改）
  data/registry_facts.json       追加 annotations 段（数值事实，供登记表取用）

用法
----
    python scripts/d02_annot_checks.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def load(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    mm = load("audit_annotations.csv")
    ba = load("audit_2a_runs.csv")

    # ---------------------------------------------------------------- mmidb
    per_sub_runs = Counter(r["subject"] for r in mm)
    sf_by_sub: dict[str, set[str]] = defaultdict(set)
    t1: dict[str, int] = defaultdict(int)
    t2: dict[str, int] = defaultdict(int)
    t0: dict[str, int] = defaultdict(int)
    bad = [r for r in mm if int(r["n_chans"]) < 0]

    for r in mm:
        s = r["subject"]
        sf_by_sub[s].add(r["sfreq"])
        t1[s] += int(r["n_T1"])
        t2[s] += int(r["n_T2"])
        t0[s] += int(r["n_T0"])

    subs = sorted(per_sub_runs)
    n_sub = len(subs)
    runs_per_sub = sorted(set(per_sub_runs.values()))

    low_sf = sorted(s for s, v in sf_by_sub.items() if "128.0" in v)
    per_class_tot = {s: t1[s] + t2[s] for s in subs}
    # 「标准」被试：每类各 90 试次
    canonical = sorted(s for s in subs if t1[s] == 90 and t2[s] == 90)

    sf_counter = Counter(next(iter(v)) if len(v) == 1 else "混合" for v in sf_by_sub.values())

    # ------------------------------------------------------------------ 2a
    per_file: dict[str, dict] = {}
    for r in ba:
        d = per_file.setdefault(r["file"], dict(runs=0, labeled=0, trials=0, art=0))
        d["runs"] += 1
        if r["labeled"] == "1":
            d["labeled"] += 1
            d["trials"] += int(r["n_trials"])
            d["art"] += int(r["n_artifacts"])
    lab_runs = sorted({v["labeled"] for v in per_file.values()})
    trials = sorted({v["trials"] for v in per_file.values()})
    art_tot = sum(v["art"] for v in per_file.values())
    trial_tot = sum(v["trials"] for v in per_file.values())
    short = sorted(k for k, v in per_file.items() if v["runs"] != 9)

    facts_p = DATA / "registry_facts.json"
    facts = json.loads(facts_p.read_text(encoding="utf-8"))
    facts["annotations"] = dict(
        mmidb=dict(
            n_subjects=n_sub,
            n_error_rows=len(bad),
            runs_per_subject=runs_per_sub[0] if len(runs_per_sub) == 1 else -1,
            runs_per_subject_values=runs_per_sub,
            n_subjects_with_128hz=len(low_sf),
            subjects_with_128hz=low_sf,
            n_subjects_canonical_180=len(canonical),
            t1_per_subject_min=min(t1.values()),
            t1_per_subject_max=max(t1.values()),
            t2_per_subject_min=min(t2.values()),
            t2_per_subject_max=max(t2.values()),
            t1_min_subject=min(t1, key=lambda s: t1[s]),
            t1_max_subject=max(t1, key=lambda s: t1[s]),
            trials_total=sum(per_class_tot.values()),
        ),
        bci2a=dict(
            n_files=len(per_file),
            labeled_runs_per_file=lab_runs[0] if len(lab_runs) == 1 else -1,
            labeled_runs_values=lab_runs,
            trials_per_file=trials[0] if len(trials) == 1 else -1,
            trials_values=trials,
            n_trials_total=trial_tot,
            n_artifacts_total=art_tot,
            artifact_rate=round(art_tot / trial_tot, 4) if trial_tot else None,
            files_with_fewer_than_9_runs=short,
        ),
    )
    facts_p.write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")

    # --------------------------------------------------------------- 报告
    lines: list[str] = []
    A = lines.append
    A("# P0 标注审计报告")
    A("")
    A("> 本文件由 `scripts/d02_annot_checks.py` 从 `data/audit_annotations.csv` 与")
    A("> `data/audit_2a_runs.csv` 生成，**不得手改**。所有数字都可以用一条命令复算。")
    A("")
    A("## 1 EEGMMIDB：三处与官方描述的偏差")
    A("")
    A(f"- 被试数 **{n_sub}**，每被试 run 数 **{runs_per_sub}**，解析失败 **{len(bad)}** 行。")
    A(f"- 采样率不是统一的 160 Hz：**{len(low_sf)} 名被试（{', '.join(low_sf)}）的 12 个任务 run 是 128 Hz**，")
    A("  其基线 run（R01/R02）仍是 160 Hz。混采样必须在分段前统一重采样，否则事件对齐会整体错位。")
    A(f"- 每类试次并非整齐 90：T1 在 {min(t1.values())}–{max(t1.values())} 之间，"
      f"T2 在 {min(t2.values())}–{max(t2.values())} 之间；"
      f"恰好各 90 的被试只有 **{len(canonical)}/{n_sub}** 人。")
    A(f"- 试次总数 {sum(per_class_tot.values())}（T1+T2），不是 109×180。")
    A("")
    A("### 逐被试（按试次总数排序，前 10 与后 10）")
    A("")
    A("| 被试 | T0 | T1 | T2 | T1+T2 | 采样率 |")
    A("|---|---|---|---|---|---|")
    order = sorted(subs, key=lambda s: per_class_tot[s])
    pick = order[:10] + order[-10:]
    for s in pick:
        sf = "/".join(sorted(sf_by_sub[s]))
        A(f"| {s} | {t0[s]} | {t1[s]} | {t2[s]} | {per_class_tot[s]} | {sf} |")
    A("")
    A("## 2 BCI IV-2a（HF 镜像 = Brunner 2008 Graz A）")
    A("")
    A(f"- 文件 {len(per_file)} 个，每文件有标签 run **{lab_runs}** 段，试次 **{trials}**/文件。")
    A(f"- 试次合计 **{trial_tot}**，标注伪迹 **{art_tot}**（{facts['annotations']['bci2a']['artifact_rate']:.1%}）。")
    A(f"- run 数不足 9 的文件：**{', '.join(short) if short else '无'}**"
      "（缺的是无标签校准 run，有标签部分完整，不影响分析）。")
    A("")
    A("## 3 对计划的直接后果")
    A("")
    A("1. 预处理锁必须写死「按被试重采样到统一率」，并在 registry 记 `n_subjects_with_128hz`。")
    A("2. 任何按试次计的统计量（每类试次、总试次）只能引用登记表键，不许在正文写 90。")
    A("3. `competent subject` 的判据要按**实际试次数**算 kappa，不能用固定 90 试次的名义量。")
    A("")

    (REPORTS / "P0_annot_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote reports/P0_annot_audit.md ({len(lines)} 行)")
    print(f"128Hz 被试: {low_sf} | canonical(90/90): {len(canonical)}/{n_sub} | T1 {min(t1.values())}-{max(t1.values())}")
    print(f"2a: {trial_tot} trials, artifact rate {facts['annotations']['bci2a']['artifact_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
