import argparse
import pandas as pd
from pathlib import Path
from itertools import combinations
import heapq
import math

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

NA_TOKEN = "<NA>"


def sanitize_filename_part(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s)


def clean_series(s: pd.Series) -> pd.Series:
    s = s.where(s.notna(), NA_TOKEN)
    mask_zero = (s == 0) | (s.astype(str).str.strip() == "0")
    s = s.mask(mask_zero, NA_TOKEN)
    return s.astype(str)


def per_column_topk(sub: pd.DataFrame, feature_cols, topk: int) -> pd.DataFrame:
    rows, total = [], len(sub)
    for col in feature_cols:
        series = clean_series(sub[col])
        vc = series[series != NA_TOKEN].value_counts().head(topk)
        for rank, (val, cnt) in enumerate(vc.items(), start=1):
            rows.append({
                "feature": col,
                "rank": rank,
                "value": val,
                "count": int(cnt),
                "total_in_class": int(total),
                "proportion": (int(cnt) / total) if total else 0.0
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["feature", "rank"]).reset_index(drop=True)


def k_hop_topk(sub: pd.DataFrame, feature_cols, k: int, topk: int, progress_desc: str = "") -> pd.DataFrame:
    if len(feature_cols) < k or len(sub) == 0:
        return pd.DataFrame()

    X = sub[feature_cols].copy()
    for col in feature_cols:
        X[col] = clean_series(X[col])

    total = len(sub)

    heap = []
    seq = 0

    def push_record(cnt: int, record: dict, seq_id: int):
        item = (cnt, seq_id, record)
        if len(heap) < topk:
            heapq.heappush(heap, item)
        else:
            if item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    combo_iter = combinations(feature_cols, k)

    if tqdm is not None:
        try:
            total_combos = math.comb(len(feature_cols), k)
        except Exception:
            total_combos = None

        combo_iter = tqdm(
            combo_iter,
            total=total_combos,
            desc=progress_desc or f"{k}-hop",
            unit="combo",
            dynamic_ncols=True,
            mininterval=0.5,
        )

    for cols in combo_iter:
        cols = list(cols)

        mask = (X[cols] != NA_TOKEN).all(axis=1)
        if not mask.any():
            continue

        subset = X.loc[mask, cols]

        gb = (
            subset.groupby(cols, dropna=False)
                  .size()
                  .reset_index(name="count")
                  .sort_values("count", ascending=False)
                  .head(topk) 
        )

        for row in gb.itertuples(index=False, name=None):
            *vals, cnt = row
            cnt = int(cnt)

            record = {}
            for idx, (c, v) in enumerate(zip(cols, vals)):
                record[f"feature{chr(65 + idx)}"] = c
                record[f"value{chr(65 + idx)}"] = v

            record["count"] = cnt
            record["total_in_class"] = int(total)
            record["proportion"] = cnt / total if total else 0.0

            push_record(cnt, record, seq)
            seq += 1


    if not heap:
        return pd.DataFrame()

    out = pd.DataFrame([item[2] for item in heap])
    return out.sort_values("count", ascending=False).head(topk).reset_index(drop=True)


def run_task(
    task_name: str,
    input_path: Path,
    output_dir: Path,
    target_col: str,
    exclude_columns: set,
    exclude_class_values: list,
    topk_map: dict,
    enable_map: dict,
    whitelist_map: dict,
):


    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path, encoding="utf-8", low_memory=False)


    exclude_lower = {c.lower() for c in exclude_columns}
    feature_cols = [c for c in df.columns if c.lower() not in exclude_lower]

    class_values = df[target_col].dropna().unique().tolist()
    if exclude_class_values:
        class_values = [v for v in class_values if v not in exclude_class_values]

    print(f"\n===== Starting task: {task_name} =====")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Target column: {target_col}")
    print(f"Number of feature columns: {len(feature_cols)}")
    print(f"Number of classes: {len(class_values)}")


    for cls in class_values:
        sub = df[df[target_col] == cls]
        cls_safe = sanitize_filename_part(str(cls))

        # 1-hop
        if enable_map.get(1, False):
            topk = topk_map.get(1, 100)
            out1 = output_dir / f"top{topk}_by_1_column_for_{cls_safe}.csv"
            per_column_topk(sub, feature_cols, topk).to_csv(out1, index=False, encoding="utf-8-sig")
            print(f"[{task_name} | {cls}] 1hop: {out1}")

        # 2~10-hop
        for k in range(2, 11):
            if not enable_map.get(k, False):
                continue

            topk = topk_map.get(k, 100)
            whitelist = whitelist_map.get(k, [])
            cols_k = feature_cols if not whitelist else [c for c in feature_cols if c in whitelist]

            outk = output_dir / f"top{topk}_by_{k}_column_for_{cls_safe}.csv"
            desc = f"{task_name} | class={cls_safe} | {k}-hop"

            k_hop_topk(sub, cols_k, k, topk, progress_desc=desc).to_csv(outk, index=False, encoding="utf-8-sig")
            print(f"[{task_name} | {cls}] {k}hop: {outk}")

    print(f"===== Finish: {task_name} =====\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search top frequent feature combinations: run pressure_low then pressure_high (cwd-relative paths)."
    )

    parser.add_argument(
        "--low_input",
        type=str,
        default="datasets/pressureLow_train.csv",
        help="pressure_low csv path (relative to project root by default)"
    )
    parser.add_argument(
        "--high_input",
        type=str,
        default="datasets/pressureHigh_train.csv",
        help="pressure_high csv path (relative to project root by default)"
    )

    parser.add_argument(
        "--low_output",
        type=str,
        default="results/rule/pressure_low/top100_by_N_column",
        help="pressure_low output dir (relative to project root by default)"
    )
    parser.add_argument(
        "--high_output",
        type=str,
        default="results/rule/pressure_high/top100_by_N_column",
        help="pressure_high output dir (relative to project root by default)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    topk_map = {k: 100 for k in range(1, 9)}        # 1..8
    enable_map = {k: True for k in range(1, 9)}     # 1..8 all enabled
    whitelist_map = {k: [] for k in range(1, 9)}  

    run_task(
        task_name="pressure_low",
        input_path=Path(args.low_input),
        output_dir=Path(args.low_output),
        target_col="pressure_low",
        exclude_columns={"pressure_low"},
        exclude_class_values=[""], 
        topk_map=topk_map,
        enable_map=enable_map,
        whitelist_map=whitelist_map,
    )

    run_task(
        task_name="pressure_high",
        input_path=Path(args.high_input),
        output_dir=Path(args.high_output),
        target_col="pressure_high",
        exclude_columns={"pressure_high"},
        exclude_class_values=[],  
        topk_map=topk_map,
        enable_map=enable_map,
        whitelist_map=whitelist_map,
    )

    print("All done!")


if __name__ == "__main__":
    main()
