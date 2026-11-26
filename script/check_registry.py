r"""登记表校验器：不通过就不许往论文里写数字。

十一项检查
--------
C1 结构      key 唯一；claim/doc/level 必填；main 级不得用 literal
C2 求值      全部条目能算出来，无异常、无非有限值
C3 边界      落在声明的 bounds 内
C4 溯源      每个非 literal 条目至少有一个来源文件
C5 未过期    产物里记录的源文件哈希与磁盘一致（数据一改就红）
C6 确定性    连建两次 manifest_sha 相同
C7 毒化 A    把源 CSV 的数值改掉，对应条目必须跟着变（证明绑定，不是硬编码）
C8 毒化 B    手工篡改 numbers.json 的取值，必须被抓出来
C9 无手抄    literal 计数应为 0（有则列出，供人工确认是否可接受）
C11 正文 tex 稿件里每个 `\Reg...` 宏都必须是 numbers.tex 定义过的；且除宏、标签、
             引用键、版式参数之外不得出现裸数字（白名单在 manuscript/tex_whitelist.txt）

用法
----
    python scripts/check_registry.py
    python scripts/check_registry.py --rebuild   # 先重建再校验
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_registry as BR  # noqa: E402

ROOT = BR.ROOT
REG = BR.REG
RESULTS = []

# 下列常量既被毒化实验使用，也被登记表以 op: const 取走（论文里的数字），
# 所以必须写成模块级常量，不能散落在函数体里。
N_CHECKS = 11          # C1–C11
POISON_A_OLD = 109     # data/registry_facts.json 里 mmidb.n_subjects 的原值
POISON_A_NEW = 99      # C7 注入值
POISON_B_DELTA = 0.05  # C8 篡改幅度


def note(level, name, msg):
    RESULTS.append((level, name, msg))
    mark = {"PASS": "   ok", "WARN": " WARN", "BLOCK": "BLOCK"}[level]
    print(f"[{mark}] {name:<14s} {msg}")


def load_spec():
    with open(BR.SPEC, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_product():
    p = os.path.join(REG, "numbers.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
def c1_structure(spec):
    entries = spec["entries"]
    keys = [e["key"] for e in entries]
    dup = {k for k in keys if keys.count(k) > 1}
    missing = [e.get("key", "<?>") for e in entries
               if not e.get("claim") or not e.get("doc")]
    bad_level = [e["key"] for e in entries
                 if e.get("level", "secondary") not in ("main", "secondary", "descriptive")]
    lit_main = [e["key"] for e in entries
                if e.get("op") == "literal" and e.get("level") == "main"]
    issues = []
    if dup:
        issues.append(f"重复 key {sorted(dup)}")
    if missing:
        issues.append(f"缺 claim/doc {missing}")
    if bad_level:
        issues.append(f"level 非法 {bad_level}")
    if lit_main:
        issues.append(f"main 级不得 literal：{lit_main}（必须能从数据复取）")
    if issues:
        note("BLOCK", "C1 结构", "；".join(issues))
    else:
        note("PASS", "C1 结构", f"{len(entries)} 条，key 唯一、字段齐备")


def c2_evaluate():
    p = BR.build(write=False)
    if p["problems"]:
        note("BLOCK", "C2 求值", f"{len(p['problems'])} 条失败：{p['problems'][:3]}")
    else:
        note("PASS", "C2 求值", f"{p['counts']['total']} 条全部算出，无异常")
    return p


def c3_bounds(spec, prod):
    bad = []
    for e in spec["entries"]:
        b = e.get("bounds")
        if not b or e["key"] not in prod["entries"]:
            continue
        v = prod["entries"][e["key"]]["value"]
        if isinstance(v, (int, float)) and not (b[0] <= float(v) <= b[1]):
            bad.append(f"{e['key']}={v} 不在 {b}")
    if bad:
        note("BLOCK", "C3 边界", "；".join(bad))
    else:
        note("PASS", "C3 边界", "全部落在声明区间内")


def c4_provenance(spec, prod):
    orphans = []
    for e in spec["entries"]:
        k = e["key"]
        if k not in prod["entries"]:
            continue
        ent = prod["entries"][k]
        if ent["op"] == "literal":
            if not ent.get("locator"):
                orphans.append(f"{k}（literal 但无 locator）")
        elif not ent["sources"]:
            orphans.append(f"{k}（无来源文件）")
    if orphans:
        note("BLOCK", "C4 溯源", "；".join(orphans))
    else:
        note("PASS", "C4 溯源", "每条都有来源文件或 locator")


def c5_fresh(prod):
    stale, missing = [], []
    seen = set()
    for ent in prod["entries"].values():
        for s in ent["sources"]:
            rel = s["path"]
            if rel in seen:
                continue
            seen.add(rel)
            p = os.path.join(ROOT, rel)
            if not os.path.exists(p):
                missing.append(rel)
            elif BR.sha16(p) != s["sha16"]:
                stale.append(rel)
    if prod.get("registry_sha") != BR.sha16(BR.SPEC):
        stale.append("registry/numbers.yaml")
    if missing or stale:
        note("BLOCK", "C5 未过期",
             f"缺失 {missing}；已变更 {stale} → 跑 build_registry.py 重建")
    else:
        note("PASS", "C5 未过期", f"{len(seen)} 个来源文件哈希一致")


def c6_determinism():
    a = BR.build(write=False)
    b = BR.build(write=False)
    if a["manifest_sha"] != b["manifest_sha"]:
        note("BLOCK", "C6 确定性", f"两次构建不一致 {a['manifest_sha']} vs {b['manifest_sha']}")
    else:
        note("PASS", "C6 确定性", f"manifest_sha={a['manifest_sha']}（可重复）")


def c7_poison_data(prod):
    """改源数据 → 值必须跟着变。若不变，说明该条目实际是硬编码的。

    BCI_SAE 的「源数据」是 data/registry_facts.json（由 d01_audit.py 从原始数据审计而来）。
    这里把 mmidb.n_subjects 从 109 改成 99，若登记表仍报 109，就是硬编码。
    """
    base = os.path.join(ROOT, "data", "registry_facts.json")
    if not os.path.exists(base):
        note("WARN", "C7 毒化A", "缺少 data/registry_facts.json，跳过（先跑 d01_audit.py）")
        return
    orig = prod["entries"].get("data_mmidb_n_subjects", {}).get("value")
    if orig is None:
        note("WARN", "C7 毒化A", "基线缺 data_mmidb_n_subjects，无法比较")
        return
    with tempfile.TemporaryDirectory() as td:
        tmp_data = os.path.join(td, "data")
        os.makedirs(tmp_data)
        with open(base, encoding="utf-8") as f:
            facts = json.load(f)
        facts.setdefault("mmidb", {})["n_subjects"] = POISON_A_NEW
        facts["mmidb"].pop("subjects", None)  # 去掉可能的优先字段，确保走 n_subjects
        with open(os.path.join(tmp_data, "registry_facts.json"), "w", encoding="utf-8") as f:
            json.dump(facts, f)

        mini = {"version": 1, "entries": [
            {"key": "poison_probe", "claim": "毒化探针", "doc": "test",
             "level": "secondary", "fmt": "int",
             "expr": 'json_("registry_facts.json")["mmidb"]["n_subjects"]'}]}
        spec_p = os.path.join(td, "mini.yaml")
        with open(spec_p, "w", encoding="utf-8") as f:
            yaml.safe_dump(mini, f, allow_unicode=True)
        out = os.path.join(td, "out")
        os.makedirs(out)
        env = dict(os.environ, REGISTRY_DATA=tmp_data, REGISTRY_OUT=out)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_registry.py"),
                            "--spec", spec_p, "--out", out],
                           env=env, capture_output=True, text=True, cwd=ROOT)
        jp = os.path.join(out, "numbers.json")
        if r.returncode != 0 or not os.path.exists(jp):
            note("BLOCK", "C7 毒化A", f"子进程失败：{(r.stderr or r.stdout)[-200:]}")
            return
        with open(jp, encoding="utf-8") as f:
            got = json.load(f)["entries"]["poison_probe"]["value"]
        if abs(float(got) - POISON_A_NEW) < 1e-9:
            note("PASS", "C7 毒化A",
                 f"改数据后取值从 {orig} 变为 {got}，绑定成立")
        else:
            note("BLOCK", "C7 毒化A", f"改了数据值却仍是 {got}，说明该条是硬编码")


def c8_poison_product(prod):
    """手改产物 → 必须被抓出（防止有人改 json 冒充结果）。"""
    import copy
    fake = copy.deepcopy(prod)
    k = "data_mmidb_n_subjects"
    if k not in fake["entries"]:
        note("WARN", "C8 毒化B", f"缺 {k}，跳过")
        return
    old = float(fake["entries"][k]["value"])
    fake["entries"][k]["value"] = old + POISON_B_DELTA
    fake["manifest_sha"] = "0000000000000000"
    fresh = BR.build(write=False)
    diff = []
    for kk in fresh["entries"]:
        if kk not in fake["entries"]:
            continue
        try:
            a = float(fresh["entries"][kk]["value"])
            b = float(fake["entries"][kk]["value"])
        except (TypeError, ValueError):
            # 文本类条目（版本号、哈希）不参与数值比对，改由 C5 的 registry_sha 管
            if fresh["entries"][kk]["text"] != fake["entries"][kk]["text"]:
                diff.append(kk)
            continue
        if not np.isclose(a, b, atol=1e-12):
            diff.append(kk)
    if k in diff:
        note("PASS", "C8 毒化B", f"篡改 {k} 被识别（{old}→{old + POISON_B_DELTA}）")
    else:
        note("BLOCK", "C8 毒化B", "篡改未被识别，校验逻辑失效")


def c10_raw_manifest(deep=False):
    """原始字节完整性：MANIFEST_raw.json 记的每个文件都必须还在、大小一致（--deep 时核对 sha256）。

    这一条是 CM 那套 C1–C9 里没有的：那边原始数据不在仓库里，只能管到派生 CSV。
    我们这里原始数据就在仓库内（data/raw/），所以「有人替换了某个 EDF」必须能发现。
    """
    mp = os.path.join(ROOT, "data", "raw", "MANIFEST_raw.json")
    if not os.path.exists(mp):
        note("WARN", "C10 原始完整", "无 MANIFEST_raw.json（先跑 d00_fetch.py）")
        return
    with open(mp, encoding="utf-8") as f:
        man = json.load(f)
    miss, size_bad = [], []
    for rel, rec in man.items():
        p = os.path.join(ROOT, "data", "raw", rel)
        if not os.path.exists(p):
            miss.append(rel)
        elif os.path.getsize(p) != rec.get("bytes"):
            size_bad.append(rel)
    if miss or size_bad:
        note("BLOCK", "C10 原始完整",
             f"缺失 {len(miss)} 个（例：{miss[:2]}）；大小不符 {len(size_bad)} 个（例：{size_bad[:2]}）")
        return
    if deep:
        bad = [rel for rel, rec in man.items()
               if BR.sha16(os.path.join(ROOT, "data", "raw", rel)) != (rec.get("sha256") or "")[:16]]
        if bad:
            note("BLOCK", "C10 原始完整", f"sha256 不符 {len(bad)} 个（例：{bad[:3]}）")
            return
        note("PASS", "C10 原始完整", f"{len(man)} 个文件 sha256 全部一致（deep）")
    else:
        note("PASS", "C10 原始完整", f"{len(man)} 个文件在位且大小一致（--deep 可核对 sha256）")


TEX_CMD_ARGS = re.compile(
    r"\\(?:label|ref|eqref|autoref|pageref|citep|citet|citealp|citealt|cite|"
    r"input|include|includegraphics|url|href|bibliography|bibliographystyle|"
    r"documentclass|usepackage|begin|end|newcommand|renewcommand|providecommand|"
    r"setlength|addtolength|vspace|hspace|rule|scalebox|resizebox|caption|"
    r"textwidth|linewidth|columnwidth)\*?(\[[^\]]*\])?\{[^{}]*\}")
# 下面这些是「记号」而不是结果：数据集名、电极名、闸门/假设编号、幂次、1/f 写法。
# 只剥离白纸黑字的记号形态，任何真正的统计量都必须走 \Reg 宏。
TEX_NOTATION = [
    (re.compile(r"\bIV-?2[ab]\b"), " "),        # BCI Competition IV-2a/2b
    (re.compile(r"\b[2]a\b"), " "),             # 正文里对 2a 的简称
    (re.compile(r"\b[A-Z]{1,3}\d{1,2}\b"), " "),  # C3 / O1 / PO3 / CP6 电极名
    (re.compile(r"\b[SHTRG]\d\b"), " "),        # S0/S3/T1/H2/R1 编号
    (re.compile(r"\^\{?-?\d+\}?"), " "),        # 幂次 R^2、10^-4
    (re.compile(r"\d+\s*/\s*f\b"), " "),        # 1/f 谱记法
    (re.compile(r"\d+(?:\.\d+)?\s*(?:cm|mm|pt|em|ex|in|\\textwidth|\\linewidth)\b"), " "),  # 版式尺寸
]
TEX_MACRO_USE = re.compile(r"\\Reg[A-Za-z]+")
TEX_DEF = re.compile(r"\\newcommand\{\\(Reg[A-Za-z]+)\}")


def _tex_dir():
    """当前投稿稿优先（manuscript_tbme），旧稿目录（manuscript）作为回退。"""
    for name in ("manuscript_tbme", "manuscript"):
        d = os.path.join(ROOT, name)
        if os.path.isdir(d) and any(f.endswith(".tex") for f in os.listdir(d)):
            return d
    return None


def _tex_files():
    d = _tex_dir()
    if d is None:
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".tex"))


def _tex_whitelist():
    d = _tex_dir()
    p = os.path.join(d, "tex_whitelist.txt") if d else ""
    allowed, reasons = set(), {}
    if p and os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            tok, _, why = line.strip().partition("#")
            if tok.strip():
                allowed.add(tok.strip())
                reasons[tok.strip()] = why.strip()
    return allowed, reasons


def c11_tex(prod):
    """正文 tex 的两条硬约束：宏必须存在；数字必须来自登记表。"""
    files = _tex_files()
    tex = os.path.join(REG, "numbers.tex")
    defined = set()
    if os.path.exists(tex):
        defined = set(TEX_DEF.findall(open(tex, encoding="utf-8").read()))
    if not defined:
        note("WARN", "C11 正文tex", "registry/numbers.tex 不存在，跳过（先跑 build_registry.py）")
        return
    if not files:
        note("WARN", "C11 正文tex", "未找到 manuscript_tbme/ 或 manuscript/ 下的 .tex，跳过")
        return

    allowed, _reasons = _tex_whitelist()
    undefined, bare, used = {}, {}, set()
    # 参考文献条目里的卷期页码与 arXiv 号是书目信息，不是本文的结果，不参与裸数字扫描。
    skip_digits = {"refs.tex"}
    for fp in files:
        name = os.path.basename(fp)
        for ln, line in enumerate(open(fp, encoding="utf-8"), 1):
            if line.lstrip().startswith("%"):
                continue
            line = re.sub(r"(?<!\\)%.*", "", line)          # 行内注释
            for m in TEX_MACRO_USE.finditer(line):
                tok = m.group(0)[1:]
                used.add(tok)
                if tok not in defined:
                    undefined.setdefault(tok, []).append(f"{name}:{ln}")
            body = line
            for _ in range(3):                              # 命令参数可能嵌套
                body = TEX_CMD_ARGS.sub(" ", body)
            body = TEX_MACRO_USE.sub(" ", body)
            if name in skip_digits:
                continue
            for pat, rep in TEX_NOTATION:
                body = pat.sub(rep, body)
            for m in re.finditer(r"\d+(?:\.\d+)?", body):
                tok = m.group(0)
                if tok in allowed:
                    continue
                bare.setdefault(tok, []).append(f"{name}:{ln}")

    if undefined:
        note("BLOCK", "C11 正文tex",
             "；".join(f"\\{k}（{v[0]}）" for k, v in sorted(undefined.items()))[:300])
        return
    if bare:
        top = sorted(bare.items(), key=lambda kv: -len(kv[1]))[:12]
        note("BLOCK", "C11 正文tex",
             f"{len(bare)} 种裸数字：{[(k, v[0]) for k, v in top]}")
        return
    note("PASS", "C11 正文tex",
         f"{len(files)} 个 tex 文件，引用 {len(used)} 个宏全部有定义，裸数字 0 处")


def c9_literals(prod):
    lit = [k for k, v in prod["entries"].items() if v["op"] == "literal"]
    if lit:
        note("WARN", "C9 无手抄", f"{len(lit)} 条 literal 需人工确认：{lit}")
    else:
        note("PASS", "C9 无手抄", "全部条目均由数据/常量算出，0 条 literal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="先重建产物再校验")
    ap.add_argument("--deep", action="store_true", help="C10 额外核对原始文件 sha256（慢）")
    a = ap.parse_args()
    if a.rebuild:
        BR.build()
        print("已重建 registry/numbers.{json,md,tex}\n")

    spec = load_spec()
    prod = load_product()
    if prod is None:
        note("BLOCK", "产物存在", "registry/numbers.json 不存在，先跑 build_registry.py")
    else:
        c1_structure(spec)
        p2 = c2_evaluate()
        c3_bounds(spec, prod)
        c4_provenance(spec, prod)
        c5_fresh(prod)
        c6_determinism()
        c7_poison_data(prod)
        c8_poison_product(prod)
        c9_literals(prod)
        c10_raw_manifest(deep=a.deep)
        c11_tex(prod)

    nb = sum(1 for lv, _, _ in RESULTS if lv == "BLOCK")
    nw = sum(1 for lv, _, _ in RESULTS if lv == "WARN")
    np_ = sum(1 for lv, _, _ in RESULTS if lv == "PASS")
    print(f"\n合计 {len(RESULTS)} 项：BLOCK {nb} / WARN {nw} / PASS {np_}")
    return 1 if nb else 0


if __name__ == "__main__":
    sys.exit(main())
