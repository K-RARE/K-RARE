# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

BASE_RESULTS = Path("results")

OUT_DIR = BASE_RESULTS / "summary_four_predict"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COL_PATIENT = "patient_index"
COL_GOLD = "gold_label"

COL_MAJORITY = "majority_pred"
COL_BIO_011 = "biomistral_7b_011"
COL_BIO_111 = "biomistral_7b_111"
COL_EXAMPLE_TOP1 = "biomistral_7B_example_010_top1"

TASKS = [
    {
        "name": "pressure_low",

        "rule_conf_path": BASE_RESULTS / "Predict_By_Confidence" / "rule_confidence_low.csv",

        "with_rules_011_path": BASE_RESULTS / "LLM_With_Rule_y2" / "pressure_low" / "pressure_low_llm_outputs.csv",

        "with_rules_111_path": BASE_RESULTS / "LLM_With_Rule_y3" / "pressure_low" / "pressure_low_llm_outputs.csv",

        "with_examples_path": BASE_RESULTS / "LLM_With_Example_y1" / "pressure_low" / "pressure_low_llm_outputs.csv",

        "out_csv": OUT_DIR / "summary_pressure_low.csv",
    },
    {
        "name": "pressure_high",

        "rule_conf_path": BASE_RESULTS / "Predict_By_Confidence" / "rule_confidence_high.csv",

        "with_rules_011_path": BASE_RESULTS / "LLM_With_Rule_y2" / "pressure_high" / "pressure_high_llm_outputs.csv",

        "with_rules_111_path": BASE_RESULTS / "LLM_With_Rule_y3" / "pressure_high" / "pressure_high_llm_outputs.csv",

        "with_examples_path": BASE_RESULTS / "LLM_With_Example_y1" / "pressure_high" / "pressure_high_llm_outputs.csv",

        "out_csv": OUT_DIR / "summary_pressure_high.csv",
    },
]


def read_csv_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8")


def ensure_patient_index(df: pd.DataFrame, who: str) -> pd.DataFrame:
    if COL_PATIENT not in df.columns:
        raise KeyError(f"[{who}] Missing primary key column '{COL_PATIENT}'. Existing columns: {list(df.columns)}")
    df = df.copy()
    df[COL_PATIENT] = pd.to_numeric(df[COL_PATIENT], errors="coerce")
    df = df.dropna(subset=[COL_PATIENT])
    df[COL_PATIENT] = df[COL_PATIENT].astype(int)
    return df


def merge_one_task(task_cfg: dict):
    name = task_cfg["name"]
    print("\n" + "=" * 90)
    print(f"Start merging task: {name}")
    print("=" * 90)

    rule_conf_path = Path(task_cfg["rule_conf_path"])
    df_rule = ensure_patient_index(read_csv_any(rule_conf_path), f"{name} / rule_confidence")

    if COL_GOLD not in df_rule.columns:
        raise KeyError(f"[{name} / rule_confidence] Missing column '{COL_GOLD}'. Existing columns: {list(df_rule.columns)}")

    if COL_MAJORITY in df_rule.columns:
        pass
    elif "predict_top1_confidence" in df_rule.columns:
        df_rule[COL_MAJORITY] = df_rule["predict_top1_confidence"]
    else:
        raise KeyError(
            f"[{name} / rule_confidence] Missing '{COL_MAJORITY}', and no fallback column "
            f"'predict_top1_confidence' found. Existing columns: {list(df_rule.columns)}"
        )

    df_rule_small = df_rule[[COL_PATIENT, COL_GOLD, COL_MAJORITY]].copy()

    wr011_path = Path(task_cfg["with_rules_011_path"])
    df_011 = ensure_patient_index(read_csv_any(wr011_path), f"{name} / with_rules_y2 (011)")
    if COL_BIO_011 not in df_011.columns:
        raise KeyError(
            f"[{name} / with_rules_y2 (011)] Missing column '{COL_BIO_011}'. Existing columns: {list(df_011.columns)}"
        )

    cols_011 = [COL_PATIENT, COL_BIO_011]
    if COL_GOLD in df_011.columns:
        cols_011.insert(1, COL_GOLD)
    df_011_small = df_011[cols_011].copy()

    wr111_path = Path(task_cfg["with_rules_111_path"])
    df_111 = ensure_patient_index(read_csv_any(wr111_path), f"{name} / with_rules_y3 (111)")
    if COL_BIO_111 not in df_111.columns:
        raise KeyError(
            f"[{name} / with_rules_y3 (111)] Missing column '{COL_BIO_111}'. Existing columns: {list(df_111.columns)}"
        )

    cols_111 = [COL_PATIENT, COL_BIO_111]
    if COL_GOLD in df_111.columns:
        cols_111.insert(1, COL_GOLD)
    df_111_small = df_111[cols_111].copy()

    ex_path = Path(task_cfg["with_examples_path"])
    df_ex = ensure_patient_index(read_csv_any(ex_path), f"{name} / with_examples_y1")
    if COL_EXAMPLE_TOP1 not in df_ex.columns:
        raise KeyError(
            f"[{name} / with_examples_y1] Missing column '{COL_EXAMPLE_TOP1}'. Existing columns: {list(df_ex.columns)}"
        )

    cols_ex = [COL_PATIENT, COL_EXAMPLE_TOP1]
    if COL_GOLD in df_ex.columns:
        cols_ex.insert(1, COL_GOLD)
    df_ex_small = df_ex[cols_ex].copy()

    merged = df_rule_small.merge(df_011_small, on=COL_PATIENT, how="outer", suffixes=("", "__011"))
    merged = merged.merge(df_111_small, on=COL_PATIENT, how="outer", suffixes=("", "__111"))
    merged = merged.merge(df_ex_small, on=COL_PATIENT, how="outer", suffixes=("", "__ex"))

    if f"{COL_GOLD}__011" in merged.columns:
        merged[COL_GOLD] = merged[COL_GOLD].combine_first(merged[f"{COL_GOLD}__011"])
    if f"{COL_GOLD}__111" in merged.columns:
        merged[COL_GOLD] = merged[COL_GOLD].combine_first(merged[f"{COL_GOLD}__111"])
    if f"{COL_GOLD}__ex" in merged.columns:
        merged[COL_GOLD] = merged[COL_GOLD].combine_first(merged[f"{COL_GOLD}__ex"])

    drop_gold_cols = [c for c in [f"{COL_GOLD}__011", f"{COL_GOLD}__111", f"{COL_GOLD}__ex"] if c in merged.columns]
    if drop_gold_cols:
        merged = merged.drop(columns=drop_gold_cols)

    final_cols = [
        COL_PATIENT,
        COL_GOLD,
        COL_MAJORITY,
        COL_BIO_011,
        COL_BIO_111,
        COL_EXAMPLE_TOP1,
    ]
    for c in final_cols:
        if c not in merged.columns:
            merged[c] = None

    merged = merged[final_cols].sort_values(COL_PATIENT).reset_index(drop=True)

    out_path = Path(task_cfg["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"✅ Merge done ({name}).")
    print(f"- rule_confidence : {rule_conf_path}")
    print(f"- with_rules_011  : {wr011_path}")
    print(f"- with_rules_111  : {wr111_path}")
    print(f"- with_examples   : {ex_path}")
    print(f"✅ Saved: {out_path} | rows={len(merged)}")


def main():
    for task in TASKS:
        merge_one_task(task)


if __name__ == "__main__":
    main()
