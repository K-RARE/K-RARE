import json
from pathlib import Path
import pandas as pd
from math import isclose
from collections import defaultdict

NA_TOKEN = "<NA>"

# ===== Global default thresholds (used when not specified in a task) =====
GLOBAL_LIFT_THRESHOLD = 1.06
GLOBAL_CHI2_THRESHOLD = 1.0  # e.g., 3.841 (p<0.05) / 6.635 (p<0.01); None means disabled

# ===== New: global switch for 2nd-round dedup (same class: remove longer rule if subset exists) =====
GLOBAL_ENABLE_SAME_CLASS_SUBSET_DEDUP = True

TASKS = [
    {
        "name": "pressure_low",
        "data_csv": "datasets/pressureLow_train.csv",
        "target_col": "pressure_low",
        "rule_values_json": Path("results/rule/pressure_low/rule/rule_values.json"),
        "output_rules_json": Path("results/rule/pressure_low/rule/rules.json"),
        "lift_threshold": 1.08,
        "chi2_threshold": 2.3,
        "skip_filter_classes": ["Low_Pressure_High"],
        "enable_same_class_subset_dedup": True,
    },
    {
        "name": "pressure_high",
        "data_csv": "datasets/pressureHigh_train.csv",
        "target_col": "pressure_high",
        "rule_values_json": Path("results/rule/pressure_high/rule/rule_values.json"),
        "output_rules_json": Path("results/rule/pressure_high/rule/rules.json"),
        "lift_threshold": 1.16,
        "chi2_threshold": 4.95,
        "skip_filter_classes": ["High_Pressure_Low"],
        "enable_same_class_subset_dedup": True,
    },
]


def clean_series(s: pd.Series) -> pd.Series:
    s = s.where(s.notna(), NA_TOKEN)
    mask_zero = (s == 0) | (s.astype(str).str.strip() == "0")
    s = s.mask(mask_zero, NA_TOKEN)
    return s.astype(str)


def build_row_value_sets(df_clean: pd.DataFrame, feature_cols):
    row_value_sets = []
    for _, row in df_clean[feature_cols].iterrows():
        vals = set(v for v in row.astype(str).tolist() if v != NA_TOKEN and v != "")
        row_value_sets.append(vals)
    return row_value_sets


def _is_better_rule(r1: dict, r2: dict) -> bool:
    c1, c2 = r1.get("confidence", 0.0), r2.get("confidence", 0.0)
    if not isclose(c1, c2):
        return c1 > c2
    l1, l2 = r1.get("lift", 0.0), r2.get("lift", 0.0)
    if not isclose(l1, l2):
        return l1 > l2
    q1, q2 = r1.get("chi2", 0.0), r2.get("chi2", 0.0)
    if not isclose(q1, q2):
        return q1 > q2
    return False


def _fmt_rule_for_print(r: dict) -> str:
    vals = r.get("values", [])
    return (
        f"class={r.get('class','')} | "
        f"conf={r.get('confidence',0.0):.6f}, lift={r.get('lift',0.0):.6f}, "
        f"chi2={r.get('chi2',0.0):.6f}, laplace={r.get('laplace',0.0):.6f} | "
        f"values({len(vals)}): {vals}"
    )


def dedupe_by_values_and_class_max_confidence(rules):
    groups = defaultdict(list)
    for r in rules:
        cls = r.get("class", "")
        key = (frozenset(r.get("values", [])), cls)
        groups[key].append(r)

    removed_logs = []
    kept_all = []

    for key, lst in groups.items():
        if len(lst) == 1:
            kept_all.append(lst[0])
            continue
        best = lst[0]
        for cand in lst[1:]:
            if _is_better_rule(cand, best):
                best = cand
        kept_all.append(best)
        for cand in lst:
            if cand is best:
                continue
            removed_logs.append({
                "phase": 1,
                "reason": "same_values_same_class",
                "class": best.get("class", ""),
                "relation": "equal_to_kept",
                "kept_rule": {k: best.get(k) for k in ("class","values","confidence","lift","chi2","laplace")},
                "removed_rule": {k: cand.get(k) for k in ("class","values","confidence","lift","chi2","laplace")},
            })

    removed_count = len(rules) - len(kept_all)
    return kept_all, removed_count, removed_logs


def dedupe_by_subset_short_wins_if_higher_conf(rules):
    by_cls = defaultdict(list)
    for idx, r in enumerate(rules):
        by_cls[r.get("class", "")].append({
            "idx": idx,
            "rule": r,
            "set": frozenset(r.get("values", [])),
            "conf": r.get("confidence", 0.0)
        })

    to_remove = set()
    removed_logs = []

    for cls, lst in by_cls.items():
        m = len(lst)
        for j in range(m):
            if lst[j]["idx"] in to_remove:
                continue
            set_B = lst[j]["set"]
            conf_B = lst[j]["conf"]

            best_A = None
            for i in range(m):
                if i == j:
                    continue
                set_A = lst[i]["set"]
                conf_A = lst[i]["conf"]
                if set_A.issubset(set_B) and set_A != set_B and (conf_A > conf_B):
                    if (best_A is None) or (conf_A > best_A["conf"]):
                        best_A = lst[i]

            if best_A is not None:
                to_remove.add(lst[j]["idx"])
                removed_logs.append({
                    "phase": 2,
                    "reason": "subset_short_higher_conf",
                    "class": cls,
                    "relation": "A_subset_B_and_conf_A_gt_B",
                    "kept_rule": {k: best_A["rule"].get(k) for k in ("class","values","confidence","lift","chi2","laplace")},
                    "removed_rule": {k: lst[j]["rule"].get(k) for k in ("class","values","confidence","lift","chi2","laplace")},
                })

    deduped = [r for idx, r in enumerate(rules) if idx not in to_remove]
    removed_count = len(rules) - len(deduped)
    return deduped, removed_count, removed_logs


def compute_rules_for_task(
    data_csv: str,
    target_col: str,
    rule_values_json: Path,
    output_rules_json: Path,
    lift_threshold: float,
    chi2_threshold,
    skip_filter_classes=None,
    enable_same_class_subset_dedup=None
):
    if enable_same_class_subset_dedup is None:
        enable_same_class_subset_dedup = GLOBAL_ENABLE_SAME_CLASS_SUBSET_DEDUP

    skip_set = set(skip_filter_classes or [])

    df = pd.read_csv(data_csv, encoding="utf-8", low_memory=False)
    if target_col not in df.columns:
        raise ValueError(f"Input CSV is missing target column '{target_col}': {data_csv}")

    feature_cols = [c for c in df.columns if c != target_col]
    df_clean = df.copy()
    for col in feature_cols:
        df_clean[col] = clean_series(df_clean[col])
    df_clean[target_col] = df[target_col].astype(str)

    row_value_sets = build_row_value_sets(df_clean, feature_cols)

    with open(rule_values_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("values", [])

    n = len(df_clean)
    class_counts = df_clean[target_col].value_counts().astype(int).to_dict()
    num_classes = max(1, len(class_counts))

    rules = []
    seen = set()
    for it in items:
        cls = str(it.get("class", ""))
        values = [str(x) for x in it.get("values", []) if str(x) != ""]
        key_seen = (cls, tuple(values))
        if not cls or not values or key_seen in seen:
            continue
        seen.add(key_seen)

        mask_match = [set(values).issubset(s) for s in row_value_sets]
        total_match = int(sum(mask_match))
        matched_idx = [i for i, ok in enumerate(mask_match) if ok]

        if total_match == 0:
            in_class = 0
            confidence = 0.0
        else:
            cls_series = df_clean.iloc[matched_idx][target_col].astype(str)
            in_class = int((cls_series == cls).sum())
            confidence = in_class / total_match

        class_count = int(class_counts.get(cls, 0))
        if n > 0:
            supp_X  = total_match / n
            supp_Y  = class_count / n
            supp_XY = in_class / n
        else:
            supp_X = supp_Y = supp_XY = 0.0

        lift = (supp_XY / (supp_X * supp_Y)) if (supp_X > 0 and supp_Y > 0) else 0.0
        if (cls not in skip_set) and (lift < lift_threshold):
            continue

        a = in_class
        b = total_match - in_class
        c = class_count - in_class
        d = n - (a + b + c)

        chi2 = 0.0
        if n > 0:
            E_a = (total_match) * (class_count) / n
            E_b = (total_match) * (n - class_count) / n
            E_c = (n - total_match) * (class_count) / n
            E_d = (n - total_match) * (n - class_count) / n
            for O, E in ((a, E_a), (b, E_b), (c, E_c), (d, E_d)):
                if E > 0 and not isclose(E, 0.0):
                    chi2 += (O - E) ** 2 / E

        if (cls not in skip_set) and (chi2_threshold is not None) and (chi2 < chi2_threshold):
            continue

        support = supp_XY
        if n == 0:
            conviction = None
        else:
            p_not_y = 1.0 - supp_Y
            p_x_and_not_y = (b / n) if n > 0 else 0.0
            if p_x_and_not_y > 0:
                conviction = (supp_X * p_not_y) / p_x_and_not_y
            else:
                conviction = None

        laplace = (in_class + 1) / (total_match + num_classes)
        wra_value = supp_X * (confidence - supp_Y)

        rules.append({
            "class": cls,
            "values": values,
            "confidence": round(float(confidence), 6),
            "lift": round(float(lift), 6),
            "chi2": round(float(chi2), 6),
            "support": round(float(support), 6),
            "conviction": (None if conviction is None else round(float(conviction), 6)),
            "laplace": round(float(laplace), 6),
            "wra": round(float(wra_value), 6),
        })

    rules, removed_same, logs_same = dedupe_by_values_and_class_max_confidence(rules)

    if enable_same_class_subset_dedup:
        rules, removed_subset, logs_subset = dedupe_by_subset_short_wins_if_higher_conf(rules)
    else:
        removed_subset, logs_subset = 0, []

    rules_sorted = sorted(rules, key=lambda x: x.get("confidence", 0.0), reverse=True)
    for i, r in enumerate(rules_sorted, start=1):
        r["id"] = i

    output_rules_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_rules_json, "w", encoding="utf-8") as f:
        json.dump({"rules": rules_sorted}, f, ensure_ascii=False, indent=2)

    removed_logs = logs_same + logs_subset
    removed_json_path = output_rules_json.with_name("rules_removed.json")
    with open(removed_json_path, "w", encoding="utf-8") as f:
        json.dump({"removed": removed_logs}, f, ensure_ascii=False, indent=2)

    print(
        f"\n[{target_col}] Generated {output_rules_json} "
        f"(final {len(rules_sorted)} rules, IDs assigned);"
        f"\nRemoval details saved to: {removed_json_path}\n"
    )


def main():
    for t in TASKS:
        compute_rules_for_task(
            data_csv=t["data_csv"],
            target_col=t["target_col"],
            rule_values_json=t["rule_values_json"],
            output_rules_json=t["output_rules_json"],
            lift_threshold=t.get("lift_threshold", GLOBAL_LIFT_THRESHOLD),
            chi2_threshold=t.get("chi2_threshold", GLOBAL_CHI2_THRESHOLD),
            skip_filter_classes=t.get("skip_filter_classes", []),
            enable_same_class_subset_dedup=t.get(
                "enable_same_class_subset_dedup",
                GLOBAL_ENABLE_SAME_CLASS_SUBSET_DEDUP
            ),
        )


if __name__ == "__main__":
    main()
