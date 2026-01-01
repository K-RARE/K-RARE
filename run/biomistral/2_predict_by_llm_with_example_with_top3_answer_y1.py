import os
import re
import json
import glob
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

LIMIT_SAMPLES = None  # e.g., set to 5 for quick testing

# ========= Model path (local BioMistral snapshot) =========
MODEL_PATH = ""

# ========= Generation params (keep consistent with your original script) =========
DO_SAMPLE = False
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.6
TOP_P = 0.9

TOP1_COL = "biomistral_7B_example_010_top1"
TOP2_COL = "biomistral_7B_example_010_top2"
TOP3_COL = "biomistral_7B_example_010_top3"

METRIC_TXT_BASENAME = "biomistral_7B_example_010"

TASKS = [
    {
        "name": "pressure_low",
        "json_key": "pressure_low",
        "gold_col": "pressure_low",

        "test_csv": Path("datasets/pressureLow_test.csv"),

        "prompt_dir": Path("llm_prompt/prompts/pressure_low"),

        "output_root": Path("results/LLM_With_Example_y1/pressure_low"),
        "output_csv_name": "pressure_low_llm_outputs.csv",
    },
    {
        "name": "pressure_high",
        "json_key": "pressure_high",
        "gold_col": "pressure_high",

        "test_csv": Path("datasets/pressureHigh_test.csv"),
        "prompt_dir": Path("llm_prompt/prompts/pressure_high"),

        "output_root": Path("results/LLM_With_Example_y1/pressure_high"),
        "output_csv_name": "pressure_high_llm_outputs.csv",
    },
]

# ========= Utility functions =========
def list_prompt_files(prompt_dir: Path):
    files = glob.glob(str(prompt_dir / "prompt_*.txt"))

    def key_fn(p):
        m = re.search(r"prompt_(\d+)\.txt$", os.path.basename(p))
        return int(m.group(1)) if m else 10**9

    return sorted(files, key=key_fn)


def read_text(fp: str) -> str:
    p = Path(fp)
    for enc in ("utf-8", "utf-8-sig", "gbk", "cp936"):
        try:
            return p.read_text(encoding=enc)
        except Exception:
            continue
    return p.read_text(errors="ignore")


def normalize_to_ascii(s: str) -> str:
    if not isinstance(s, str):
        return s
    mapping = {
        "\u201c": '"',  "\u201d": '"',
        "\u2018": "'",  "\u2019": "'",
        "\uFF02": '"',  "\uFF07": "'",
        "\uFF0C": ",",  "\uFF1A": ":",
        "\uFF1B": ";",  "\u3001": ",",
        "\u3002": ".",  "\uFF08": "(",
        "\uFF09": ")",  "\u3010": "[",
        "\u3011": "]",  "\u2013": "-",
        "\u2014": "-",  "\u2026": "...",
        "\u00A0": " ",
    }
    trans = str.maketrans(mapping)
    s = str(s).translate(trans)
    return s.replace("\r\n", "\n").replace("\r", "\n")


def repair_json_text(s: str) -> str:
    if not s:
        return s
    s = normalize_to_ascii(s).strip()
    start = s.find("{")
    if start >= 0:
        s = s[start:]
    s = re.sub(r",\s*(\]|\})", r"\1", s)
    opens = s.count("{")
    closes = s.count("}")
    if closes < opens:
        s = s + ("}" * (opens - closes))
    return s


def _json_soft_clean(s: str) -> str:
    s = normalize_to_ascii(s).strip()
    s = s.strip("` \n\t")
    if s.startswith("{{") and s.endswith("}}"):
        s = s[1:-1]

    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        s = s[start:end + 1]

    s = re.sub(r"(?<!\\)'([^']*?)'(?!\s*:)", r'"\1"', s)
    s = re.sub(r"(?<!\\)'([^']*?)'\s*:", r'"\1":', s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _split_list_items(inner: str):
    items, buf = [], []
    in_str, esc, quote = False, False, None
    depth = 0

    for ch in inner:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str, quote = False, None
        else:
            if ch in ("'", '"'):
                in_str, quote = True, ch
                buf.append(ch)
            elif ch == "," and depth == 0:
                token = "".join(buf).strip()
                if token:
                    items.append(token)
                buf = []
            else:
                if ch in "[{(":
                    depth += 1
                elif ch in "]})" and depth > 0:
                    depth -= 1
                buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        items.append(tail)

    cleaned = [re.sub(r'^(["\'])(.*)\1$', r"\2", it.strip()) for it in items]
    return [it.strip() for it in cleaned if it.strip()]


def _loose_grab_after_key_array(text: str, key: str):
    patt = re.compile(rf'"{re.escape(key)}"\s*:\s*\[', re.S)
    m = patt.search(text)
    if not m:
        return []
    tail = text[m.end():]
    close_pos = tail.find("]")
    seg = tail[:2000] if close_pos == -1 else tail[:close_pos]
    vals = re.findall(r'"([^"]+)"|\'([^\']+)\'', seg)
    out = []
    for a, b in vals:
        out.append((a or b).strip())
    return [x for x in out if x]


def extract_json_or_list(text: str, key: str):
    if not text:
        return []

    text = normalize_to_ascii(text)
    t = text.strip()

    if t.startswith("{{") and t.endswith("}}"):
        t = t[1:-1]

    try:
        data = json.loads(t)
        if isinstance(data, dict) and key in data:
            val = data[key]
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
            return [str(val).strip()] if val is not None else []
    except Exception:
        pass

    t2 = _json_soft_clean(text)
    try:
        data = json.loads(t2)
        if isinstance(data, dict) and key in data:
            val = data[key]
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
            return [str(val).strip()] if val is not None else []
    except Exception:
        pass

    pat = re.compile(rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]', re.S)
    m = pat.search(t2) or pat.search(text)
    if m:
        inner = m.group(1)
        return _split_list_items(inner)

    repaired = repair_json_text(text)
    try:
        data = json.loads(repaired)
        if isinstance(data, dict) and key in data:
            val = data[key]
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
            return [str(val).strip()] if val is not None else []
    except Exception:
        pass

    loose = _loose_grab_after_key_array(text, key)
    if loose:
        return loose

    return []


def get_top123(items, k=3, unique=True, pad=""):
    if not items:
        return [pad] * k

    if not unique:
        out = list(items[:k])
        while len(out) < k:
            out.append(pad)
        return out

    seen, out = set(), []
    for x in items:
        sx = str(x).strip()
        if not sx:
            continue
        if sx not in seen:
            seen.add(sx)
            out.append(sx)
        if len(out) >= k:
            break
    while len(out) < k:
        out.append(pad)
    return out[:k]


def save_topk_csv_preserve_old(rows, csv_path, top_cols, overwrite_gold=True):
    csv_path = Path(csv_path)
    new_df = pd.DataFrame(rows)
    keep_cols = ["patient_index", "gold_label"] + top_cols
    new_df = new_df[keep_cols].copy()
    new_df["patient_index"] = new_df["patient_index"].astype(int)
    new_df = new_df.set_index("patient_index")

    if csv_path.exists():
        old_df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "patient_index" in old_df.columns:
            old_df["patient_index"] = old_df["patient_index"].astype(int)
            old_df = old_df.set_index("patient_index")
        else:
            old_df.index = range(1, len(old_df) + 1)

        out_df = old_df.reindex(old_df.index.union(new_df.index)).copy()

        if overwrite_gold:
            out_df.loc[new_df.index, "gold_label"] = new_df["gold_label"]
        else:
            if "gold_label" not in out_df.columns:
                out_df["gold_label"] = ""
            need_fill = out_df["gold_label"].isna() | (out_df["gold_label"].astype(str).str.strip() == "")
            out_df.loc[need_fill & out_df.index.isin(new_df.index), "gold_label"] = new_df["gold_label"]

        for c in top_cols:
            out_df.loc[new_df.index, c] = new_df[c]

        out_df.reset_index().to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        new_df.reset_index().to_csv(csv_path, index=False, encoding="utf-8-sig")


def build_prompt(tokenizer, user_prompt: str, json_key: str):
    system = (
        f"Return ONLY one valid JSON object with a single key '{json_key}'. "
        f"The JSON MUST be ASCII-only and closed properly. "
        f"Use straight double quotes (\") for all keys/values. No smart quotes. "
        f"No markdown, no code fences, no extra text. "
        f"Also, when deciding the ranked answers: first look at the labels that already appear in the examples, "
        f"reorder those labels by how confident you are, and put them at the beginning of the answer list; "
        f"if the list still has fewer than 3 items, then add your own predicted labels to fill it to exactly 3, "
        f"without duplicates."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
        try:
            messages_user_only = [{"role": "user", "content": system + "\n\n" + user_prompt}]
            return tokenizer.apply_chat_template(messages_user_only, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    return f"<s>[INST] {system}\n\n{user_prompt} [/INST]"


def run_task(task: dict, model, tokenizer):
    name = task["name"]
    json_key = task["json_key"]
    gold_col = task["gold_col"]
    test_csv = Path(task["test_csv"])
    prompt_dir = Path(task["prompt_dir"])
    output_root = Path(task["output_root"])
    output_csv_path = output_root / task["output_csv_name"]
    accuracy_dir = output_root / "accuracy"
    metric_txt_path = accuracy_dir / f"{METRIC_TXT_BASENAME}_{name}.txt"

    print("\n" + "=" * 80)
    print(f"Task: {name} (BioMistral-7B, with examples)")
    print(f"  json_key      : {json_key}")
    print(f"  gold_col      : {gold_col}")
    print(f"  test_csv      : {test_csv}")
    print(f"  prompt_dir    : {prompt_dir}")
    print(f"  output_root   : {output_root}")
    print(f"  output_csv    : {output_csv_path}")
    print(f"  metric_txt    : {metric_txt_path}")
    print("=" * 80)

    if not test_csv.exists():
        raise FileNotFoundError(f"[{name}] Test set not found: {test_csv}")
    if not prompt_dir.exists():
        raise FileNotFoundError(f"[{name}] Prompt directory not found: {prompt_dir}")

    df = pd.read_csv(str(test_csv), encoding="utf-8", low_memory=False)
    if gold_col not in df.columns:
        raise KeyError(f"[{name}] Test set is missing column '{gold_col}'. Current columns: {list(df.columns)}")

    if "patient_index" in df.columns:
        patient_indices = list(df["patient_index"].astype(int).values)
    else:
        patient_indices = list(range(1, len(df) + 1))

    gold_series = df[gold_col].astype(str).fillna("")
    total_gold = len(gold_series)

    files = list_prompt_files(prompt_dir)
    if LIMIT_SAMPLES is not None:
        files = files[:int(LIMIT_SAMPLES)]
    total = len(files)
    if total == 0:
        print(f"[{name}] No prompt_N.txt found in: {prompt_dir}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    accuracy_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    ranks = []

    with tqdm(total=total, desc=f"[{name}] Processing prompts", unit="file", ncols=100) as pbar:
        for idx, fp in enumerate(files, start=1):
            base = os.path.basename(fp)
            user_prompt = read_text(fp)

            chat_text = build_prompt(tokenizer, user_prompt, json_key)

            inputs = tokenizer(chat_text, return_tensors="pt")
            input_len = inputs["input_ids"].shape[1]
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            gen_kwargs = dict(
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if DO_SAMPLE:
                gen_kwargs.update(dict(temperature=TEMPERATURE, top_p=TOP_P))

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            generated_ids = outputs[0]
            new_tokens = generated_ids.shape[0] - input_len

            gen_text = tokenizer.decode(generated_ids[input_len:], skip_special_tokens=True).strip()

            items = extract_json_or_list(gen_text, json_key)
            top1, top2, top3 = get_top123(items, k=3, unique=True, pad="")

            gold = gold_series.iloc[idx - 1] if idx - 1 < total_gold else ""
            patient_idx = patient_indices[idx - 1] if idx - 1 < len(patient_indices) else idx

            rank = None
            for rpos, lab in enumerate([top1, top2, top3], start=1):
                if lab and lab == gold:
                    rank = rpos
                    break
            ranks.append(rank)

            print("\n" + "=" * 80)
            print(f"[{name}] [{idx}/{total}] File: {base}")
            print(f"Target key: {json_key}")
            print(f"Prompt tokens: {input_len} | Output tokens: {new_tokens}")
            print("\nRaw LLM output:")
            print(gen_text if gen_text else "<EMPTY>")
            print("\nParsed Top1~Top3:")
            print({TOP1_COL: top1, TOP2_COL: top2, TOP3_COL: top3})
            print(f"\nGOLD {json_key}: {gold}")
            print(f"RANK within Top3: {rank if rank is not None else 'Not in Top3'}")
            print("=" * 80 + "\n")

            rows.append({
                "patient_index": int(patient_idx),
                "gold_label": str(gold),
                TOP1_COL: top1,
                TOP2_COL: top2,
                TOP3_COL: top3,
            })

            pbar.update(1)

    save_topk_csv_preserve_old(
        rows=rows,
        csv_path=output_csv_path,
        top_cols=[TOP1_COL, TOP2_COL, TOP3_COL],
        overwrite_gold=True,
    )
    print(f"\n[{name}] CSV saved/updated (merged with history): {output_csv_path}")

    N = len(ranks)
    lines = []
    lines.append(f"Task                  : {name}")
    lines.append(f"JSON key              : {json_key}")
    lines.append(f"Total samples         : {N}")
    lines.append(f"CSV path              : {output_csv_path}")
    lines.append(f"Prompt dir            : {prompt_dir}")
    lines.append("")

    for k in range(1, 4):
        covered = [r for r in ranks if (r is not None and r <= k)]
        num_covered = len(covered)
        coverage_k = num_covered / N if N > 0 else 0.0
        avg_rank_k = float(np.mean(covered)) if covered else float("nan")
        lines.append(f"Top-{k} coverage       : {coverage_k:.6f}")
        lines.append(f"Avg rank within Top-{k}: {avg_rank_k:.4f}")
        lines.append("")

    metric_txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[{name}] TXT metrics saved: {metric_txt_path}")

    print("\n" + "=" * 80)
    print(f"[{name}] FINAL SUMMARY")
    print("=" * 80)
    print("\n".join(lines))


def main():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("==== Environment ====")
    print("Torch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU 0:", torch.cuda.get_device_name(0))

    print(f"\nLoading BioMistral from local snapshot:\n{MODEL_PATH}\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    ).eval()

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = "unknown"
    print("\n==== Runtime Info ====")
    print(f"Model device: {model_device}")
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"GPU memory used (after load): {mem:.2f} GB")

    for task in TASKS:
        run_task(task, model, tokenizer)


if __name__ == "__main__":
    main()
