import pandas as pd
from pathlib import Path
import json

HOPS_DEFAULT = [1, 2, 3, 4, 5, 6, 7, 8]   # top100_by_{k}_column_for_<class>.csv
TOP_DEFAULT  = 30

TASKS = [
    {
        "name": "pressure_low",
        "input_dir": Path("results/rule/pressure_low/top100_by_N_column"),
        "output_json": Path("results/rule/pressure_low/rule/rule_values.json"),
        "classes": ["Low_Pressure_Low", "Low_Pressure_Mid", "Low_Pressure_High"],
        "top": 30,                         # how many rows to read from each CSV; None = read all
        "hops": [1, 2, 3, 4, 5, 6, 7, 8],  # hops used in this task
    },
    {
        "name": "pressure_high",
        "input_dir": Path("results/rule/pressure_high/top100_by_N_column"),
        "output_json": Path("results/rule/pressure_high/rule/rule_values.json"),
        "classes": ["High_Pressure_Low", "High_Pressure_Mid", "High_Pressure_High"],
        "top": 60,
        "hops": [1, 2, 3, 4, 5, 6, 7, 8],  # hops used in this task
    }
]


def order_value_columns(columns):
    cols = list(columns)
    if "value" in cols:  
        base = ["value"]
        others = [c for c in cols if c != "value"]
        others_sorted = sorted(others, key=lambda x: (len(x), x))
        return base + others_sorted

    def value_key(c):
        if c.lower().startswith("value") and len(c) > 5:
            return ord(c[-1].upper()) 
        return 10**9

    return sorted(cols, key=value_key)


def process_task(task):
    input_dir   = task["input_dir"]
    output_json = task["output_json"]
    classes     = task["classes"]
    top         = task.get("top", TOP_DEFAULT)
    hops        = task.get("hops", HOPS_DEFAULT)

    all_records = []

    for cls in classes:
        for k in hops:
            file_path = input_dir / f"top100_by_{k}_column_for_{cls}.csv"
            if not file_path.exists():
                continue

            if top is not None:
                df = pd.read_csv(file_path, encoding="utf-8", low_memory=False, nrows=top)
            else:
                df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)

            value_cols = [c for c in df.columns if "value" in c.lower()]
            if not value_cols:
                continue

            value_cols = order_value_columns(value_cols)

            for _, row in df.iterrows():
                values = []
                for c in value_cols:
                    v = row.get(c, None)
                    if pd.isna(v):
                        continue
                    v_str = str(v).strip()
                    if not v_str:
                        continue
                    values.append(v_str)

                if values:
                    all_records.append({
                        "class": cls,
                        "hop": k,
                        "values": values
                    })

    result = {"values": all_records}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"[{task['name']}] Generated: {output_json} "
        f"(total {len(all_records)} records, top={top if top is not None else 'ALL'}, hops={hops})"
    )


def main():
    for task in TASKS:
        process_task(task)


if __name__ == "__main__":
    main()
