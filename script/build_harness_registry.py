"""把 registry/numbers.tex 的最终渲染值导成 SharpenNeedles 审计用的 registry。

为什么需要这一步而不是直接读 numbers.yaml：numbers.yaml 存的是取数表达式，
审计器要的是"排版后落在纸面上的那个数"。两者只差 fmt 与四舍五入，
而 R3.NUM.TRACE 比较的正是纸面数字——用 yaml 的值会因为格式化差异误报。

输出：registry/harness_registry.json
  {"metrics": {<宏名>: {"value": float, "fmt": "", "key": <yaml key>, "unit": str}}}
"""
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "registry" / "numbers.tex"
YAML = ROOT / "registry" / "numbers.yaml"
OUT = ROOT / "registry" / "harness_registry.json"


def main() -> int:
    tex = TEX.read_text(encoding="utf-8")
    macros = re.findall(r"\\newcommand\{\\(Reg[A-Za-z0-9]+)\}\{([^{}]*)\}", tex)
    meta = {}
    for e in (yaml.safe_load(YAML.read_text(encoding="utf-8")) or {}).get("entries", []):
        meta[str(e.get("key"))] = {
            "unit": e.get("unit", ""),
            "claim": e.get("claim", ""),
            "section": e.get("section", ""),
        }

    metrics = {}
    skipped = []
    for name, raw in macros:
        s = (raw.strip().replace("\\,", "").replace("$", "").replace("−", "-")
             .replace("\\%", "").replace("%", "").strip())
        try:
            v = float(s)
        except ValueError:
            skipped.append((name, raw))
            continue
        key = "Reg" + name if not name.startswith("Reg") else name
        metrics[key] = {
            "value": v,
            "fmt": "",
            "macro": "\\" + name,
            "rendered": s,
            "unit": meta.get(name, {}).get("unit", ""),
        }

    payload = {
        "note": "由 scripts/build_harness_registry.py 从 registry/numbers.tex 生成；"
                "只收录数值型宏，供 SharpenNeedles 的 R3.NUM.TRACE 复算。",
        "source": TEX.relative_to(ROOT).as_posix(),
        "metrics": metrics,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {OUT}  {len(metrics)} numeric macros, {len(skipped)} non-numeric skipped")
    if skipped:
        for n, r in skipped[:8]:
            print("   skip", n, "=", r[:40])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
