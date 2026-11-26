# 数据可用性（D0 闸门产出 · 已实测）

> 判定规则：**A 可直接分析 / B 需修补 / C 不可用（写明卡在哪一步）**。
> 只有 A 档能进正式分析；B 档进附录稳健性表；C 档写进正文的数据可获得性说明。
> 本文所有数字来自 `data/registry_facts.json`（由 `d01_audit.py` + `d02_annot_checks.py` 生成），
> 并受 `check_registry.py` 的 C10（原始文件清单）与 C7/C8（毒化）约束。

## 判定汇总

| # | 源 | 层级角色 | 实测状态 | 结论 |
|---|---|---|---|---|
| 1 | PhysioNet EEGMMIDB v1.0.0（HF 镜像 `raei/Schalk-2009-...`） | T2 主源 | 109 人 / 1526 EDF / 3066 文件 / 4.67 GB / 64 导 | **A** |
| 2 | BCI IV-2a（HF 镜像 `raei/Brunner-2008-...GrazA`，实为 Graz A） | T1 伪迹层主源 | 9 人 × 2 session / 18 文件 / 25 导（22 EEG + 3 EOG）/ 250 Hz / 5184 试次 | **A** |
| 3 | BCI IV-2b | T1 扩展 | 官方包需表单 POST（直连 404），`bnci-horizon` 站点 000 不可达，HF 无镜像 | **C**（放弃） |
| 4 | SSVEP Benchmark（清华 Wang 2017） | T0 正对照 | 10 人 / `S*.mat.7z` / 64 导 / 1500 点 / 40 目标 / 6 block / 250 Hz | **A** |
| 5 | BETA（Liu 2020） | T0 重复校准 | 清华下载页未见 BETA 入口（仅 chenyonghao 160-target、liangliyan 三组实验） | **C**（放弃） |
| 6 | OpenBMI / Lee2019 | T1 第二 MI 队列 | HF 搜索 `Lee2019`/`BNCI2014` 均 0 结果；仅 `Felipe717/openbmi-ica`（ICA 衍生产物，非原始） | **C**（放弃） |

**一句话结论**：三层各有 A 档源，题目成立；但**第二个运动想象队列拿不到**，所以
"跨数据集迁移"必须改成"同数据集内的留一被试迁移"，这条已写进 `DECISIONS.md` D5。

## 逐源记录

### 1 EEGMMIDB — A

- 取数路径：`https://hf-mirror.com/datasets/raei/Schalk-2009-EEGMotorMovementImagery/resolve/main/S{nnn}/S{nnn}R{nn}.edf`
  （磁盘直连 + 6 并发 + 429 指数退避；24 并发会被限流，实测 429）。
- 体量：3066 个文件 / 4.67 GB；`MANIFEST_raw.json` 记录每个文件的 sha256。
- 实测事实：
  - 每被试 **14 run**，全表一致；解析失败 0 行；通道恒为 64。
  - **采样率不统一**：`S088`、`S092`、`S100` 的 12 个任务 run 是 **128 Hz**（基线 R01/R02 仍 160 Hz）。
  - 每类试次**不是 90**：T1 ∈ [72, 116]，T2 ∈ [72, 116]；恰好 90/90 的被试只有 **25/109**。
  - 逐被试试次数见 `reports/P0_annot_audit.md` 的表。
- 后果：预处理锁必须写"按被试重采样到统一率"；正文禁止出现"109×90"这类表述。

### 2 BCI IV-2a — A

- 取数路径：`https://hf-mirror.com/datasets/raei/Brunner-2008-BCICompetition2008GrazDataSetA/resolve/main/A{nn}{T,E}.mat`
- 注意：该镜像实为 **BCI Competition 2008 Graz A**，通道/采样/范式与 2a 一致（22 EEG + 3 EOG、250 Hz、4 类），
  同样 9 人 × 2 session，但不是官方 2a 原包。正文必须写成"Graz A（2a 同源范式）"，不能冒充官方 2a。
- 实测事实：每文件 **6 个有标签 run**、**288 试次**（48/run）；`A04T.mat` 只有 7 个 run（缺 2 个无标签校准 run，有标签部分完整）。
- 伪迹：逐试次 0/1 标记求和 = **488 / 5184 = 9.41%**。这是 S1 伪迹层的起点，不是我们设定的阈值。

### 3 BCI IV-2b — C

- `http://bnci-horizon-2020.eu` → HTTP 000（站点已下线）。
- `https://www.bbci.de/competition/iv/download/BCICIV_2b_gdf.zip` → 404；官方下载页需填表 POST，无法脚本化。
- HF 镜像搜索 `IV-2b`/`BNCI2014` → 0 结果。
- 影响：可接受。T1 层的核心需求是**有 EOG 通道**，2a 已提供 3 导 EOG；2b 原本只是"第二个小样本"扩展。

### 4 SSVEP Benchmark — A

- 取数路径：`https://bci.med.tsinghua.edu.cn/upload/yijun/S{n}.mat.7z`（10 人，`S1..S10`）。
- 实测结构：`data` 张量 `(64, 1500, 40, 6)` = 通道 × 采样点 × 目标 × block，与官方规格逐项吻合（250 Hz × 6 s）。
- 用途：T0 仪器校准（S0 闸门）。**只跑 10 人**，不追全量 35 人。

### 5 BETA — C

- 清华下载页当前只有 `chenyonghao`（SSVEP 160-target）与 `liangliyan`（三组实验）两组数据，无 BETA 入口。
- 影响：可接受。BETA 的作用是"换一套 64 导 SSVEP 重复 S0"，属稳健性加分项；
  S0 只要在 SSVEP Benchmark 上过闸，仪器可信度即成立。

### 6 OpenBMI / Lee2019 — C

- HF 镜像无原始数据（只有 ICA 衍生产物）；moabb 的下载端点在国内网络不可达。
- 影响：**这是本次唯一真正损失的能力**——没有第二个 MI 队列，"跨数据集迁移"降级为"同数据集留一被试"。
  109 人的规模仍足以支撑跨被试主张，但正文不能写"跨数据集"。

## 收口结论

- **进入正式分析**：EEGMMIDB（T2）、BCI IV-2a 镜像（T1）、SSVEP Benchmark 10 人（T0）。
- **降级到附录**：无（B 档为空——三个 A 档源都不需要修补即可分析）。
- **放弃**：2b、BETA、OpenBMI（原因见上，均属"扩展"而非"必需"）。
- **对计划 §3 的调整**：
  1. T1 层只做 2a，伪迹层用其 3 导 EOG + 9.41% 已标注伪迹试次；
  2. S4 的表述从"跨数据集迁移"改为"留一被试迁移（109 人）"；
  3. 新增预处理锁：三源重采样到统一率后，EEGMMIDB 的 128 Hz 被试必须单独核验事件对齐。

## 复算方式

```bash
python scripts/d01_audit.py                    # 原始 → data/audit_*.csv + registry_facts.json
python scripts/d02_annot_checks.py             # → reports/P0_annot_audit.md + annotations 段
python scripts/check_registry.py --rebuild    # 10 项校验（含毒化）
```
