import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("results/summary_four_predict")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "patient_index"
GOLD_COL = "gold_label"

COL_EXAMPLE_TOP1 = "biomistral_7B_example_010_top1"
COL_BIO_011      = "biomistral_7b_011"
COL_BIO_111      = "biomistral_7b_111"
COL_MAJOR        = "majority_pred"
PRED_COLS_IN_ORDER = [COL_EXAMPLE_TOP1, COL_BIO_011, COL_BIO_111, COL_MAJOR]

WEIGHTS = [0.175, 0.200, 0.275, 0.350]

KNN_ID_COL = "row_index"
KNN_TOP1 = "top1_pred"
KNN_TOP2 = "top2_pred"
KNN_TOP3 = "top3_pred"

TOPK = 3

TASKS = [
    {
        "name": "pressure_low",
        "summary_csv": OUT_DIR / "summary_pressure_low.csv",
        "knn_csv": Path("results/KNN/knn_topk_low.csv"),
        "out_csv": OUT_DIR / "ours_hybrid_pressure_low.csv",
    },
    {
        "name": "pressure_high",
        "summary_csv": OUT_DIR / "summary_pressure_high.csv",
        "knn_csv": Path("results/KNN/knn_topk_high.csv"),
        "out_csv": OUT_DIR / "ours_hybrid_pressure_high.csv",
    },
]


def read_csv_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str)
        except Exception as e:
            last_err = e
    raise last_err


def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


def require_cols(df: pd.DataFrame, cols, who: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{who}] Missing columns: {missing}\nExisting columns: {list(df.columns)}")


def ensure_int_id(df: pd.DataFrame, col: str, who: str) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"[{who}] Missing column '{col}'. Existing columns: {list(df.columns)}")
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col]).copy()
    df[col] = df[col].astype(int)
    return df


def weighted_vote_top1(labels, weights):
    s = sum(weights)
    w_use = [w / s for w in weights] if s > 0 else list(weights)

    sums = {}
    for lab, w in zip(labels, w_use):
        lab = safe_str(lab)
        if not lab:
            continue
        sums[lab] = sums.get(lab, 0.0) + float(w)

    if not sums:
        return ""
    return max(sums.items(), key=lambda kv: (kv[1], kv[0]))[0]


def pick_new_top123(rule_top1, knn_top1, knn_top2, knn_top3):
    r1 = safe_str(rule_top1)
    k1 = safe_str(knn_top1)
    k2 = safe_str(knn_top2)
    k3 = safe_str(knn_top3)

    new_tops = []
    if r1:
        new_tops.append(r1)

    for lab in [k1, k2, k3]:
        lab = safe_str(lab)
        if lab and (lab not in new_tops):
            new_tops.append(lab)
        if len(new_tops) >= 3:
            break

    while len(new_tops) < 3:
        new_tops.append("")

    return new_tops[0], new_tops[1], new_tops[2]


def run_one_task(cfg: dict):
    name = cfg["name"]
    summary_csv = cfg["summary_csv"]
    knn_path = cfg["knn_csv"]
    out_csv = cfg["out_csv"]

    df = read_csv_any(summary_csv)
    df.columns = [str(c).strip() for c in df.columns]
    require_cols(df, [ID_COL, GOLD_COL] + PRED_COLS_IN_ORDER, f"summary_{name}")
    df = ensure_int_id(df, ID_COL, f"summary_{name}").copy()

    df["rule_top1"] = [
        weighted_vote_top1([row[c] for c in PRED_COLS_IN_ORDER], WEIGHTS)
        for _, row in df.iterrows()
    ]

    knn_df = read_csv_any(knn_path)
    knn_df.columns = [str(c).strip() for c in knn_df.columns]

    if KNN_ID_COL in knn_df.columns:
        knn_df = knn_df[knn_df[KNN_ID_COL] != "SUMMARY"].copy()

    if KNN_ID_COL not in knn_df.columns:
        if ID_COL in knn_df.columns:
            knn_df[KNN_ID_COL] = knn_df[ID_COL]
        else:
            raise KeyError(
                f"[{name} / KNN] Missing '{KNN_ID_COL}' and no fallback '{ID_COL}'. Existing columns: {list(knn_df.columns)}"
            )

    require_cols(knn_df, [KNN_ID_COL, KNN_TOP1, KNN_TOP2, KNN_TOP3], f"{name} / KNN")
    knn_df = ensure_int_id(knn_df, KNN_ID_COL, f"{name} / KNN").copy()

    merged = pd.merge(
        df[[ID_COL, GOLD_COL, "rule_top1"]],
        knn_df[[KNN_ID_COL, KNN_TOP1, KNN_TOP2, KNN_TOP3]],
        left_on=ID_COL,
        right_on=KNN_ID_COL,
        how="inner"
    ).sort_values(ID_COL)

    if merged.shape[0] == 0:
        raise RuntimeError(
            f"[{name}] The merged row count is 0. Please check whether {ID_COL} and {KNN_ID_COL} are aligned."
        )

    new_top1_list, new_top2_list, new_top3_list, rank_list = [], [], [], []
    for _, row in merged.iterrows():
        gold = safe_str(row[GOLD_COL])
        rule_top1 = safe_str(row["rule_top1"])

        nt1, nt2, nt3 = pick_new_top123(
            rule_top1=rule_top1,
            knn_top1=row[KNN_TOP1],
            knn_top2=row[KNN_TOP2],
            knn_top3=row[KNN_TOP3],
        )

        rank = None
        for rpos, lab in enumerate([nt1, nt2, nt3], start=1):
            if lab and gold and lab == gold:
                rank = rpos
                break

        new_top1_list.append(nt1)
        new_top2_list.append(nt2)
        new_top3_list.append(nt3)
        rank_list.append("" if rank is None else int(rank))

    out_df = merged.copy()
    out_df["new_top1"] = new_top1_list
    out_df["new_top2"] = new_top2_list
    out_df["new_top3"] = new_top3_list
    out_df["rank_within_top3"] = rank_list

    if KNN_ID_COL in out_df.columns:
        out_df = out_df.drop(columns=[KNN_ID_COL])

    ranks = [int(x) for x in out_df["rank_within_top3"].tolist() if str(x).strip() != ""]
    N = len(out_df)

    print("=" * 80)
    print(f"Ours Hybrid ({name}) | rule_top1 (weighted vote) + KNN top1~3")
    print(f"Summary CSV: {summary_csv}")
    print(f"KNN CSV    : {knn_path}")
    print(f"Used rows  : {N}")
    print(f"Vote cols  : {', '.join(PRED_COLS_IN_ORDER)}")
    print(f"Weights    : {WEIGHTS}")
    print("")

    for k in range(1, TOPK + 1):
        covered = [r for r in ranks if r <= k]
        num_covered = len(covered)
        coverage_k = num_covered / N if N > 0 else 0.0
        avg_rank_k = float(np.mean(covered)) if covered else float("nan")
        print(f"Top-{k} coverage       : {coverage_k:.6f}  ({num_covered}/{N})")
        print(f"Avg rank within Top-{k}: {avg_rank_k:.4f}")
        print("")

    final_cols = [
        ID_COL, GOLD_COL,
        "rule_top1",
        KNN_TOP1, KNN_TOP2, KNN_TOP3,
        "new_top1", "new_top2", "new_top3",
        "rank_within_top3",
    ]
    for c in final_cols:
        if c not in out_df.columns:
            out_df[c] = ""

    out_df = out_df[final_cols].sort_values(ID_COL).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"✅ Saved one CSV: {out_csv} | rows={len(out_df)}")
    print("=" * 80 + "\n")


def main():
    for cfg in TASKS:
        run_one_task(cfg)


if __name__ == "__main__":
    main()
