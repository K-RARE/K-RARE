import json
import re
from pathlib import Path
import pandas as pd

# ===================== Global switches =====================
TOP_K = 5                  # Keep top-K after sorting by confidence (may be < K after skipping missing example_text)
SHOW_CONFIDENCE = False    # True: append "(confidence=...)" after each example; False: do not append
ENABLE_CLASS_DEDUP = True  # True: keep only one rule per class within top-K; False: keep all top-K

RELATIONS_JSON = Path("llm_prompt/relations.json")

TASKS = [
    {
        "name": "pressure_low",
        "test_csv": Path("datasets/pressureLow_test.csv"),
        "target_col": "pressure_low",
        
        "template_txt": Path("llm_prompt/prompt_pressure_low_examples_top3_answer.txt"),

        "rules_json": Path("results/rule/pressure_low/rule/rules.json"),

        "llm_rules_candidates": [
            Path("results/rule/pressure_low/rule/llm_rules_low.json"),
        ],

        "output_dir": Path("llm_prompt/prompts/pressure_low"),
    },
    {
        "name": "pressure_high",
        "test_csv": Path("datasets/pressureHigh_test.csv"),
        "target_col": "pressure_high",

        "template_txt": Path("llm_prompt/prompt_pressure_high_examples_top3_answer.txt"),

        "rules_json": Path("results/rule/pressure_high/rule/rules.json"),

        "llm_rules_candidates": [
            Path("results/rule/pressure_high/rule/llm_rules_high.json"),
        ],

        "output_dir": Path("llm_prompt/prompts/pressure_high"),
    },
]


def _read_csv_safely(path: Path) -> pd.DataFrame:
    path_str = str(path)
    for enc in ("utf-8", "utf-8-sig", "gbk", "cp936"):
        try:
            return pd.read_csv(path_str, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"Read failed: {path_str}")


def _read_text_safely(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk", "cp936"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"Template read failed (all encodings tried): {path}")


def load_json(path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File does not exist: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_llm_rules_map(candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                print(f"[INFO] Using LLM rules: {p}")
                return data, str(p)
    print("[WARN] No llm_rules JSON found. Will fallback to rules.json only.")
    return {}, ""


def is_effective_value(v) -> bool:
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


def index_rules_by_id(rules_data: dict) -> dict:
    idx = {}
    for r in rules_data.get("rules", []):
        rid = str(r.get("id"))
        if rid:
            idx[rid] = r
    return idx


def find_matched_rules(patient_features_set: set, rules_data: dict, top_k: int, dedup_by_class: bool):
    candidates = []
    for r in rules_data.get("rules", []):
        vals = set(r.get("values") or [])
        if not vals:
            continue
        if vals.issubset(patient_features_set):
            rid = str(r.get("id"))
            cls = str(r.get("class", "")).strip()
            try:
                conf = float(r.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0
            if rid:
                candidates.append((rid, conf, cls))

    candidates.sort(key=lambda x: x[1], reverse=True)
    topk = candidates[:top_k]

    if not dedup_by_class:
        return topk

    unique = []
    seen = set()
    for rid, conf, cls in topk:
        key = cls if cls else f"__rid__:{rid}"
        if key in seen:
            continue
        seen.add(key)
        unique.append((rid, conf, cls))
    return unique


def _tokenize_example_values(raw_example):
    if isinstance(raw_example, list):
        out = []
        for x in raw_example:
            s = str(x).strip()
            if is_effective_value(s):
                out.append(s)
        return out

    if isinstance(raw_example, str):
        parts = re.split(r"[、,，;；\s]+", raw_example.strip())
        return [p for p in parts if is_effective_value(p)]
    return []


def build_examples_text_and_values(top_rule_items, llm_rules_map, rules_index, show_conf: bool):
    lines = []
    example_values_set = set()

    fallback_conf = {str(rid): float(conf) for (rid, conf, *_rest) in top_rule_items}

    for rid, conf, _cls in top_rule_items:
        rid = str(rid)

        entry = llm_rules_map.get(rid, None)
        example_text = None
        llm_conf = None
        example_raw = None

        if isinstance(entry, dict):
            example_text = entry.get("example_text", None)
            llm_conf = entry.get("confidence", None)
            example_raw = entry.get("example", None)

        rdict = rules_index.get(rid, None)
        if not example_text and isinstance(rdict, dict):
            example_text = rdict.get("example_text", None)
        if example_raw is None and isinstance(rdict, dict):
            example_raw = rdict.get("example", None)

        if not example_text:
            continue

        for v in _tokenize_example_values(example_raw):
            if v:
                example_values_set.add(v)

        if show_conf:
            used_conf = llm_conf if llm_conf is not None else fallback_conf.get(rid, None)
            if used_conf is None:
                line = f"- {example_text}"
            else:
                try:
                    line = f"- {example_text} (confidence={float(used_conf):.6f})"
                except Exception:
                    line = f"- {example_text} (confidence={used_conf})"
        else:
            line = f"- {example_text}"

        lines.append(line)

    return "\n".join(lines), example_values_set


def explain_features_with_relations(features_set, relations_map):
    feats = sorted(set(features_set))
    descs = [relations_map[f] for f in feats if f in relations_map]
    return "\n".join(descs)


def apply_template(template: str, mapping: dict) -> str:
    out = template
    for k in ("patient_features", "relations", "examples"):
        out = out.replace("{" + k + "}", str(mapping.get(k, "")))
    return out


def apply_strict_read_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Template does not exist: {path}")
    return _read_text_safely(path)


def run_task(task: dict, relations_map: dict):
    name = task["name"]
    test_csv: Path = task["test_csv"]
    target_col: str = task["target_col"]
    template_txt: Path = task["template_txt"]
    rules_json: Path = task["rules_json"]
    llm_rules_candidates = task["llm_rules_candidates"]
    output_dir: Path = task["output_dir"]

    print("\n" + "=" * 80)
    print(f"Task: {name} (use examples -> get top3 answer)")
    print(f"  test_csv   : {test_csv}")
    print(f"  target_col : {target_col}")
    print(f"  template   : {template_txt}")
    print(f"  rules_json : {rules_json}")
    print(f"  output_dir : {output_dir}")
    print("=" * 80)

    template_str = apply_strict_read_template(template_txt)

    df = _read_csv_safely(test_csv)
    if target_col not in df.columns:
        raise ValueError(f"Test set is missing target column '{target_col}': {test_csv}")

    rules_data = load_json(rules_json)
    rules_index = index_rules_by_id(rules_data)

    llm_rules_map, llm_used = load_llm_rules_map(llm_rules_candidates)

    output_dir.mkdir(parents=True, exist_ok=True)

    for N, (_, row) in enumerate(df.iterrows(), start=1):
        patient_features = []
        for col in df.columns:
            if col == target_col:
                continue
            v = row[col]
            if is_effective_value(v):
                patient_features.append(str(v).strip())

        patient_features_set = set(patient_features)
        patient_features_sorted = sorted(patient_features_set)
        patient_features_text = "、".join(patient_features_sorted) if patient_features_sorted else "None"

        matched = find_matched_rules(
            patient_features_set=patient_features_set,
            rules_data=rules_data,
            top_k=TOP_K,
            dedup_by_class=ENABLE_CLASS_DEDUP,
        )

        examples_text, example_values_set = build_examples_text_and_values(
            top_rule_items=matched,
            llm_rules_map=llm_rules_map,
            rules_index=rules_index,
            show_conf=SHOW_CONFIDENCE,
        )
        if not examples_text:
            examples_text = "No matching examples for this patient."

        union_features_set = patient_features_set | example_values_set
        relations_text = explain_features_with_relations(union_features_set, relations_map)
        if not relations_text:
            relations_text = "No relation descriptions found for this patient or examples."

        mapping = {
            "patient_features": patient_features_text,
            "relations": relations_text,
            "examples": examples_text,
        }
        prompt_filled = apply_template(template_str, mapping)
        (output_dir / f"prompt_{N}.txt").write_text(prompt_filled, encoding="utf-8")

    print(f"[{name}] ✅ Done! Generated prompts.")
    print(f"[{name}]   LLM rules    : {llm_used if llm_used else '(not found, fallback to rules.json)'}")
    print(f"[{name}]   Relations    : {RELATIONS_JSON}")
    print(f"[{name}]   Output dir   : {output_dir}")
    print(f"[{name}]   Total prompts: {len(df)}")
    print(f"[{name}]   TOP_K={TOP_K} | SHOW_CONFIDENCE={SHOW_CONFIDENCE} | ENABLE_CLASS_DEDUP={ENABLE_CLASS_DEDUP}")


def main():
    relations_map = load_json(RELATIONS_JSON)
    if not isinstance(relations_map, dict):
        raise ValueError("relations.json must be a dict: {feature: description}")

    for task in TASKS:
        run_task(task, relations_map)


if __name__ == "__main__":
    main()
