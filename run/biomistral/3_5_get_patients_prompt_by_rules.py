import json
from pathlib import Path
import pandas as pd

# ====== Global switches ======
TOP_K = 5                  # Take topK after sorting by confidence
SHOW_CONFIDENCE = True     # True: append (confidence=...) after each rule line
DEDUP_BY_CLASS = True      # True: within topK, keep only the highest-confidence rule per class

RELATIONS_JSON = Path("llm_prompt/relations.json")

TASKS = [
    {
        "name": "pressure_low",
        "test_csv": Path("datasets/pressureLow_test.csv"),
        "target_col": "pressure_low",

        "template_txt": Path("llm_prompt/prompt_pressure_low_rules.txt"),

        "rules_json": Path("results/rule/pressure_low/rule/rules.json"),

        "llm_rules_candidates": [
            Path("results/rule/pressure_low/rule/llm_rules_low.json"),
        ],

        "output_dir": Path("llm_prompt/prompts/pressure_low"),

        "select_classes": ["Low_Pressure_Low", "Low_Pressure_Mid", "Low_Pressure_High"],
    },
    {
        "name": "pressure_high",
        "test_csv": Path("datasets/pressureHigh_test.csv"),
        "target_col": "pressure_high",

        "template_txt": Path("llm_prompt/prompt_pressure_high_rules.txt"),
        "rules_json": Path("results/rule/pressure_high/rule/rules.json"),

        "llm_rules_candidates": [
            Path("results/rule/pressure_high/rule/llm_rules_high.json"),
        ],

        "output_dir": Path("llm_prompt/prompts/pressure_high"),
        "select_classes": ["High_Pressure_Low", "High_Pressure_Mid", "High_Pressure_High"],
    },
]


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Template does not exist: {path}")
    return path.read_text(encoding="utf-8")


def load_llm_rules_map(candidates):
    for p in candidates or []:
        p = Path(p)
        if p.exists() and p.is_file():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                print(f"[INFO] Using LLM rules: {p}")
                return data, str(p)
            else:
                raise ValueError(f"llm_rules JSON must be a dict (JSON object). Got {type(data)}: {p}")
    print("[WARN] No llm_rules JSON found. Will fallback to rules.json only.")
    return {}, ""


def is_effective_label(v) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip()
    if not s:
        return False
    if s in ("0", "0.0", "0.00"):
        return False
    if s.lower() in ("nan", "<na>"):
        return False
    return True


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def find_matched_rules(patient_features_set: set, rules_data: dict):
    candidates = []
    for r in rules_data.get("rules", []):
        vals = set(r.get("values") or [])
        if not vals:
            continue
        if vals.issubset(patient_features_set):
            rid = str(r.get("id", "")).strip()
            if not rid:
                continue
            conf = _safe_float(r.get("confidence", 0.0), 0.0)
            cls = str(r.get("class", "")).strip()
            candidates.append((rid, conf, cls, list(r.get("values") or [])))

    candidates.sort(key=lambda x: x[1], reverse=True)
    topk = candidates[:TOP_K]

    if not DEDUP_BY_CLASS:
        return topk

    unique = []
    seen = set()
    for rid, conf, cls, vals in topk:
        key = cls if cls else f"__rid__:{rid}"
        if key in seen:
            continue
        seen.add(key)
        unique.append((rid, conf, cls, vals))
    return unique


def build_rule_text_fallback(values: list, rule_class: str, target_col: str) -> str:
    values_text = "、".join(str(v) for v in values)
    return (
        f"If the characteristics of a patient include {values_text}, "
        f"then the {target_col} category is likely to be {rule_class}."
    )


def build_rules_text(matched_rules: list, llm_rules_map: dict, show_conf: bool, target_col: str) -> str:
    lines = []
    for rid, conf, cls, vals in matched_rules:
        entry = llm_rules_map.get(str(rid), None)

        text = None
        conf_llm = None
        if isinstance(entry, dict):
            text = entry.get("text", None)
            conf_llm = entry.get("confidence", None)

        if not text:
            text = build_rule_text_fallback(vals, cls, target_col)

        if show_conf:
            use_conf = conf_llm if conf_llm is not None else conf
            try:
                use_conf = float(use_conf)
                lines.append(f"- {text} (confidence={use_conf:.6f})")
            except Exception:
                lines.append(f"- {text} (confidence={use_conf})")
        else:
            lines.append(f"- {text}")

    return "\n".join(lines)


def english_and_join(items):
    items = list(items)
    n = len(items)
    if n == 0:
        return ""
    if n == 1:
        return items[0]
    if n == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])} and {items[-1]}"


def build_select_string(matched_rules: list, llm_rules_map: dict, allowed_classes: list) -> str:
    allowed = list(allowed_classes or [])

    raw_classes = []
    for rid, _, cls, _ in matched_rules:
        cls_val = None
        entry = llm_rules_map.get(str(rid), None)
        if isinstance(entry, dict) and entry.get("class"):
            cls_val = str(entry.get("class")).strip()
        if not cls_val:
            cls_val = str(cls).strip() if cls else None
        if cls_val:
            raw_classes.append(cls_val)

    raw_set = set(raw_classes)
    picked = [c for c in allowed if c in raw_set]
    if not picked:
        picked = allowed[:]
    return english_and_join(picked)


def explain_features_with_relations(features_set, relations_map):
    descs = [relations_map[f] for f in sorted(set(features_set)) if f in relations_map]
    return "\n".join(descs)


def apply_template(template: str, mapping: dict) -> str:
    out = template
    for k in ("patient_features", "relations", "rules", "select"):
        out = out.replace("{" + k + "}", str(mapping.get(k, "")))
    return out


def run_task(task: dict, relations_map: dict):
    name = task["name"]
    test_csv: Path = task["test_csv"]
    target_col: str = task["target_col"]
    template_txt: Path = task["template_txt"]
    rules_json: Path = task["rules_json"]
    llm_rules_candidates = task["llm_rules_candidates"]
    output_dir: Path = task["output_dir"]
    select_classes = task.get("select_classes", [])

    print("\n" + "=" * 80)
    print(f"Task: {name}")
    print(f"  test_csv     : {test_csv}")
    print(f"  target_col   : {target_col}")
    print(f"  template_txt : {template_txt}")
    print(f"  rules_json   : {rules_json}")
    print(f"  output_dir   : {output_dir}")
    print("=" * 80)

    df = pd.read_csv(test_csv, encoding="utf-8", low_memory=False)
    if target_col not in df.columns:
        raise ValueError(f"Test set is missing target column '{target_col}': {test_csv}")

    rules_data = load_json(rules_json)

    llm_rules_map, llm_used = load_llm_rules_map(llm_rules_candidates)

    template_str = read_text(template_txt)
    output_dir.mkdir(parents=True, exist_ok=True)

    for N, (_, row) in enumerate(df.iterrows(), start=1):
        patient_features = []
        for col in df.columns:
            if col == target_col:
                continue
            v = row[col]
            if is_effective_label(v):
                patient_features.append(str(v).strip())

        patient_features_set = set(patient_features)
        patient_features_sorted = sorted(patient_features_set)
        patient_features_text = "、".join(patient_features_sorted) if patient_features_sorted else "None"

        matched = find_matched_rules(patient_features_set, rules_data)
        rules_text = build_rules_text(
            matched_rules=matched,
            llm_rules_map=llm_rules_map,
            show_conf=SHOW_CONFIDENCE,
            target_col=target_col,
        )
        if not rules_text:
            rules_text = "No matching rules for this patient."

        relations_text = explain_features_with_relations(patient_features_set, relations_map)
        if not relations_text:
            relations_text = "No relation descriptions found for this patient."

        select_text = build_select_string(
            matched_rules=matched,
            llm_rules_map=llm_rules_map,
            allowed_classes=select_classes,
        )
        if not select_text:
            select_text = english_and_join(select_classes) if select_classes else ""

        mapping = {
            "patient_features": patient_features_text,
            "relations": relations_text,
            "rules": rules_text,
            "select": select_text,
        }
        prompt_filled = apply_template(template_str, mapping)

        (output_dir / f"prompt_{N}.txt").write_text(prompt_filled, encoding="utf-8")

    print(f"[{name}] ✅ Done! Generated prompts.")
    print(f"[{name}]   Test CSV      : {test_csv}")
    print(f"[{name}]   Target col    : {target_col}")
    print(f"[{name}]   Template      : {template_txt}")
    print(f"[{name}]   Rules         : {rules_json}")
    print(f"[{name}]   LLM rules     : {llm_used if llm_used else '(not found, fallback to rules.json)'}")
    print(f"[{name}]   Relations     : {RELATIONS_JSON}")
    print(f"[{name}]   Output dir    : {output_dir}")
    print(f"[{name}]   Total prompts : {len(df)}")
    print(f"[{name}]   TOP_K={TOP_K} | SHOW_CONFIDENCE={SHOW_CONFIDENCE} | DEDUP_BY_CLASS={DEDUP_BY_CLASS}")


def main():
    relations_map = load_json(RELATIONS_JSON)
    if not isinstance(relations_map, dict):
        raise ValueError("relations.json must be a dict: {feature: description}")

    for task in TASKS:
        run_task(task, relations_map)


if __name__ == "__main__":
    main()
