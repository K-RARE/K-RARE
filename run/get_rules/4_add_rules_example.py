import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
import pandas as pd

TOP_K = 5  # Top-K relaxation for Step2

TASKS = [
    {
        "name": "pressure_low",
        "test_csv": "datasets/pressureLow_train.csv",
        "target_col": "pressure_low",
        "rules_json": Path("results/rule/pressure_low/rule/rules.json"),
    },
    {
        "name": "pressure_high",
        "test_csv": "datasets/pressureHigh_train.csv",
        "target_col": "pressure_high",
        "rules_json": Path("results/rule/pressure_high/rule/rules.json"),
    },
]


def row_to_filtered_values(row: pd.Series, target_col: str) -> Tuple[List[str], Set[str]]:
    vals: List[str] = []
    for col, v in row.items():
        if col == target_col:
            continue
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s in ("", "0", "0.0", "0.00"):
            continue
        vals.append(s)
    return vals, set(vals)


def is_class_match(rule_class, label_value) -> bool:
    if label_value is None:
        return False
    rc = str(rule_class).strip().lower()
    lv = str(label_value).strip().lower()
    return rc == lv


def build_maps_for_rules(
    rules_list: List[Dict],
    row_sets: List[Set[str]],
    rule_indices: List[int] = None,
) -> Tuple[Dict[int, List[int]], Dict[int, List[Tuple[int, float]]]]:
    if rule_indices is None:
        rule_indices = list(range(len(rules_list)))

    rules_patients_map: Dict[int, List[int]] = {i: [] for i in rule_indices}
    patient_rules_map: Dict[int, List[Tuple[int, float]]] = {}

    for rule_index in rule_indices:
        rule = rules_list[rule_index]
        values_set = set(rule.get("values", []))
        if not values_set:
            continue
        conf = float(rule.get("confidence", 0.0))
        for patient_index, patient_values in enumerate(row_sets):
            if values_set.issubset(patient_values):
                rules_patients_map[rule_index].append(patient_index)
                patient_rules_map.setdefault(patient_index, []).append((rule_index, conf))

    return rules_patients_map, patient_rules_map


def step1_fill_examples(
    df: pd.DataFrame,
    rules_list: List[Dict],
    patient_vals_list: List[List[str]],
    row_sets: List[Set[str]],
    target_col: str,
) -> int:
    rules_patients_map, patient_rules_map = build_maps_for_rules(rules_list, row_sets)

    rule_highest_map: Dict[int, List[int]] = {}
    for pidx, rc_list in patient_rules_map.items():
        if not rc_list:
            continue
        highest_conf = max(conf for (_, conf) in rc_list)
        highest_rules = [r for (r, c) in rc_list if c == highest_conf]
        for r in highest_rules:
            rule_highest_map.setdefault(r, []).append(pidx)

    added = 0
    for rule_index, patient_list in rule_highest_map.items():
        if not patient_list:
            continue
        if "example" in rules_list[rule_index] and rules_list[rule_index]["example"]:
            continue

        rule_class = rules_list[rule_index].get("class", "")
        matched_patients = []
        if target_col in df.columns:
            for p in patient_list:
                if is_class_match(rule_class, df.iloc[p][target_col]):
                    matched_patients.append(p)

        if not matched_patients:
            continue

        example_patient_idx = min(matched_patients)
        rules_list[rule_index]["example"] = patient_vals_list[example_patient_idx]
        added += 1

    return added


def step2_fill_examples_topk(
    df: pd.DataFrame,
    rules_list: List[Dict],
    patient_vals_list: List[List[str]],
    row_sets: List[Set[str]],
    target_col: str,
    top_k: int = TOP_K,
) -> int:
    all_rules_patients_map, all_patient_rules_map = build_maps_for_rules(rules_list, row_sets)

    remaining_rule_indices = [
        i for i, r in enumerate(rules_list)
        if not ("example" in r and r["example"])
    ]
    if not remaining_rule_indices:
        return 0

    patient_topk_rules: Dict[int, Set[int]] = {}
    for p, rc_list in all_patient_rules_map.items():
        if not rc_list:
            continue
        rc_sorted = sorted(rc_list, key=lambda x: x[1], reverse=True)
        kth_index = min(top_k - 1, len(rc_sorted) - 1)
        kth_conf = rc_sorted[kth_index][1]
        topk_rules = {r for (r, c) in rc_sorted if c >= kth_conf}
        patient_topk_rules[p] = topk_rules

    added = 0
    for rule_index in remaining_rule_indices:
        if "example" in rules_list[rule_index] and rules_list[rule_index]["example"]:
            continue

        candidates = []
        rule_class = rules_list[rule_index].get("class", "")
        for p in all_rules_patients_map.get(rule_index, []):
            if p not in patient_topk_rules or rule_index not in patient_topk_rules[p]:
                continue
            if target_col in df.columns and is_class_match(rule_class, df.iloc[p][target_col]):
                candidates.append(p)

        if not candidates:
            continue

        chosen_p = min(candidates)
        rules_list[rule_index]["example"] = patient_vals_list[chosen_p]
        added += 1

    return added


def process_task(task: Dict) -> None:
    name = task["name"]
    data_path = task["test_csv"]
    rules_path: Path = task["rules_json"]
    target_col = task["target_col"]

    print(f"\n===== Processing task: {name} =====")

    df = pd.read_csv(data_path, dtype=str)
    print(f"Total patients (rows) in {name} dataset: {len(df)}")

    with open(rules_path, "r", encoding="utf-8") as f:
        rules_data = json.load(f)
    rules_list: List[Dict] = rules_data.get("rules", [])
    print(f"Total rules in {name}: {len(rules_list)}")

    patient_vals_list: List[List[str]] = []
    row_sets: List[Set[str]] = []
    for _, row in df.iterrows():
        vals_list, vals_set = row_to_filtered_values(row, target_col)
        patient_vals_list.append(vals_list)
        row_sets.append(vals_set)

    added_step1 = step1_fill_examples(
        df=df,
        rules_list=rules_list,
        patient_vals_list=patient_vals_list,
        row_sets=row_sets,
        target_col=target_col,
    )
    print(f"Step1 added examples: {added_step1}")

    added_step2 = step2_fill_examples_topk(
        df=df,
        rules_list=rules_list,
        patient_vals_list=patient_vals_list,
        row_sets=row_sets,
        target_col=target_col,
        top_k=TOP_K,
    )
    print(f"Step2 (Top-{TOP_K}) added examples: {added_step2}")

    total_with_example = sum(1 for r in rules_list if "example" in r and r["example"])
    print(f"Total rules now with example: {total_with_example} / {len(rules_list)}")

    rules_data["rules"] = rules_list
    with open(rules_path, "w", encoding="utf-8") as f_out:
        json.dump(rules_data, f_out, ensure_ascii=False, indent=4)

    print(f"Updated rules in place: {rules_path}")


def main():
    for task in TASKS:
        process_task(task)


if __name__ == "__main__":
    main()
