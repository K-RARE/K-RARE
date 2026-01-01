import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from collections import defaultdict
import math

NA_TOKEN = "<NA>"


RULE_LLM_KNN_DIR = Path("results/Predict_By_Confidence")

# Options: "confidence" or "conviction"
PREDICT_SCORE_METRIC = "confidence"  # set to "conviction" to rank/predict by conviction

TASKS = [
    {
        "name": "pressure_low",
        "test_csv": "datasets/pressureLow_test.csv",
        "target_col": "pressure_low",
        "rules_json": Path("results/rule/pressure_low/rule/rules.json"),
        "out_txt": Path("results/Predict_By_Confidence/pressure_low/accuracy/accuracy.txt"),
        "out_csv": Path("results/Predict_By_Confidence/pressure_low/accuracy/accuracy_details.csv"),
    },
    {
        "name": "pressure_high",
        "test_csv": "datasets/pressureHigh_test.csv",
        "target_col": "pressure_high",
        "rules_json": Path("results/rule/pressure_high/rule/rules.json"),
        "out_txt": Path("results/Predict_By_Confidence/pressure_high/accuracy/accuracy.txt"),
        "out_csv": Path("results/Predict_By_Confidence/pressure_high/accuracy/accuracy_details.csv"),
    },
]


def clean_series(s: pd.Series) -> pd.Series:
    s = s.where(s.notna(), NA_TOKEN)
    mask_zero = (s == 0) | (s.astype(str).str.strip() == "0")
    s = s.mask(mask_zero, NA_TOKEN)
    return s.astype(str)


def build_row_value_sets(df: pd.DataFrame, feature_cols):
    sets = []
    for _, row in df[feature_cols].iterrows():
        vals = set(v for v in row.astype(str).tolist() if v != NA_TOKEN and v != "")
        sets.append(vals)
    return sets


def _to_float(x, default):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def _rule_sort_tuple(rule, metric: str):
    if metric not in ("confidence", "conviction"):
        raise ValueError(f"Unsupported PREDICT_SCORE_METRIC={metric}. Use 'confidence' or 'conviction'.")

    if metric == "conviction":
        primary = _to_float(rule.get("conviction", None), float("-inf"))
    else:
        primary = _to_float(rule.get("confidence", 0.0), 0.0)

    conf = _to_float(rule.get("confidence", 0.0), 0.0)

    lift = _to_float(rule.get("lift", 0.0), 0.0)

    return (primary, conf, lift)


def run_task(test_csv: str, target_col: str, rules_json: Path, out_txt: Path, out_csv: Path):
    if not rules_json.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_json}")

    df = pd.read_csv(test_csv, encoding="utf-8", low_memory=False)
    if target_col not in df.columns:
        raise ValueError(f"Test set is missing column '{target_col}': {test_csv}")

    feature_cols = [c for c in df.columns if c != target_col]
    df_clean = df.copy()
    for col in feature_cols:
        df_clean[col] = clean_series(df_clean[col])
    df_clean[target_col] = df[target_col].astype(str)

    row_sets = build_row_value_sets(df_clean, feature_cols)

    with open(rules_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rules = payload.get("rules", [])
    rules = sorted(
        rules,
        key=lambda r: _rule_sort_tuple(r, PREDICT_SCORE_METRIC),
        reverse=True
    )

    details = []
    total = len(df_clean)
    correct = 0
    matched_cnt = 0

    y_true = []
    y_pred = []

    for i, vals in enumerate(row_sets):
        true_class = df_clean.iloc[i][target_col]
        y_true.append(true_class)

        matched = False
        pred_class = None
        matched_rule_values = None
        matched_rule_class = None
        matched_rule_conf = None
        matched_rule_score = None

        for r in rules:
            cls = str(r.get("class", ""))
            rvals = [str(v) for v in r.get("values", []) if str(v) != ""]
            if not cls or not rvals:
                continue
            if set(rvals).issubset(vals):
                matched = True
                pred_class = cls
                matched_rule_values = rvals
                matched_rule_class = cls
                matched_rule_conf = r.get("confidence", None)

                if PREDICT_SCORE_METRIC == "conviction":
                    matched_rule_score = _to_float(r.get("conviction", None), float("-inf"))
                else:
                    matched_rule_score = _to_float(r.get("confidence", 0.0), 0.0)
                break

        y_pred.append(pred_class if pred_class is not None else None)

        if matched:
            matched_cnt += 1
            is_correct = (pred_class == true_class)
            if is_correct:
                correct += 1
        else:
            is_correct = False

        details.append({
            "row_index": i,
            "pred_class": pred_class if pred_class is not None else "",
            "true_class": true_class,
            "matched": matched,
            "correct": is_correct,
            "matched_rule_values": json.dumps(matched_rule_values, ensure_ascii=False) if matched_rule_values else "",
            "matched_rule_class": matched_rule_class if matched_rule_class is not None else "",
            "matched_rule_confidence": matched_rule_conf if matched_rule_conf is not None else "",
            "matched_rule_metric": PREDICT_SCORE_METRIC,
            "matched_rule_score": matched_rule_score if matched_rule_score is not None else ""
        })

    accuracy = correct / total if total > 0 else 0.0
    matched_accuracy = (correct / matched_cnt) if matched_cnt > 0 else 0.0

    classes_true = set(str(x) for x in y_true)
    classes_pred = set(str(x) for x in y_pred if x not in (None, ""))
    classes = sorted(classes_true | classes_pred)

    TP = defaultdict(int)
    predicted_support = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        t = str(t)
        if p is None or p == "":
            continue
        p = str(p)
        predicted_support[p] += 1
        if p == t:
            TP[p] += 1

    per_class_precision = []
    for c in classes:
        denom = predicted_support[c]
        prec = (TP[c] / denom) if denom > 0 else float('nan')
        per_class_precision.append({
            "class": c,
            "predicted_support": denom,
            "tp": TP[c],
            "fp": denom - TP[c],
            "precision": prec,
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(details).to_csv(out_csv, index=False, encoding="utf-8-sig")

    lines = []
    lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Test file: {test_csv}")
    lines.append(f"Rules file: {rules_json}")
    lines.append(f"Target column: {target_col}")
    lines.append(f"Scoring metric (rule ranking): {PREDICT_SCORE_METRIC}")
    lines.append(f"Total samples: {total}")
    lines.append(f"Matched by rules: {matched_cnt}")
    lines.append(f"Correct predictions: {correct}")
    lines.append(f"Accuracy (overall): {accuracy:.6f}")
    lines.append(f"Accuracy (on matched subset): {matched_accuracy:.6f}")
    lines.append("")
    lines.append("Per-class precision (prediction-based accuracy):")
    lines.append("class\tpredicted_support\ttp\tfp\tprecision")
    for s in per_class_precision:
        lines.append(
            f"{s['class']}\t{s['predicted_support']}\t{s['tp']}\t{s['fp']}\t"
            f"{(s['precision'] if pd.notna(s['precision']) else 'nan')}"
        )

    lines.append("")
    lines.append("DETAILS (tab-separated):")
    header = ["row_index", "pred_class", "true_class", "matched", "correct",
              "matched_rule_values", "matched_rule_class", "matched_rule_confidence",
              "matched_rule_metric", "matched_rule_score"]
    lines.append("\t".join(header))
    for d in details:
        row = [
            str(d["row_index"]),
            str(d["pred_class"]),
            str(d["true_class"]),
            str(d["matched"]),
            str(d["correct"]),
            str(d["matched_rule_values"]),
            str(d["matched_rule_class"]),
            str(d["matched_rule_confidence"]),
            str(d["matched_rule_metric"]),
            str(d["matched_rule_score"]),
        ]
        lines.append("\t".join(row))

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    print(f"[{target_col}] Summary:")
    print("\n".join(lines[:13]))

    print("\nPer-class precision:")
    for s in per_class_precision:
        precision_str = f"{s['precision']:.6f}" if pd.notna(s['precision']) else "nan"
        print(
            f"  Class={s['class']:<20} predicted_support={s['predicted_support']:<6} "
            f"TP={s['tp']:<6} FP={s['fp']:<6} precision={precision_str}"
        )

    print(f"\nSaved summary+details to: {out_txt}")
    print(f"Saved detailed CSV to: {out_csv}")
    print("-" * 80)

    RULE_LLM_KNN_DIR.mkdir(parents=True, exist_ok=True)
    small_rows = []
    for d in details:
        row_idx = int(d["row_index"])
        patient_index = row_idx + 1
        small_rows.append({
            "patient_index": patient_index,
            "gold_label": d["true_class"],
            "majority_pred": d["pred_class"],
        })

    if target_col == "pressure_low":
        out_rule_csv = RULE_LLM_KNN_DIR / "rule_confidence_low.csv"
    elif target_col == "pressure_high":
        out_rule_csv = RULE_LLM_KNN_DIR / "rule_confidence_high.csv"
    else:
        out_rule_csv = RULE_LLM_KNN_DIR / f"rule_confidence_{target_col}.csv"

    pd.DataFrame(small_rows).to_csv(out_rule_csv, index=False, encoding="utf-8-sig")
    print(f"Saved rule confidence CSV to: {out_rule_csv}")
    print("-" * 80)


def main():
    for t in TASKS:
        run_task(
            test_csv=t["test_csv"],
            target_col=t["target_col"],
            rules_json=t["rules_json"],
            out_txt=t["out_txt"],
            out_csv=t["out_csv"],
        )


if __name__ == "__main__":
    main()
