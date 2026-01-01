import os
import re
import json
import glob
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import pandas as pd

# ========= Model path =========
MODEL_PATH = r""

LLM_OUTPUT_COL_NAME = "biomistral_7b_011"

DO_SAMPLE = True
MAX_NEW_TOKENS = 1024
MIN_NEW_TOKENS = 5
TEMPERATURE = 0.6
TOP_P = 0.9

TASKS = [
    {
        "name": "pressure_low",
        "json_key": "pressure_low",
        "gold_col": "pressure_low",
        "test_csv": Path("datasets/pressureLow_test.csv"),
        "prompt_dir": Path("llm_prompt/prompts/pressure_low"),
        "output_root": Path("results/LLM_With_Rule_y2/pressure_low"),
        "output_csv_name": "pressure_low_llm_outputs.csv",
        "accuracy_txt_name": "pressure_low_accuracy_results_biomistral_011.txt",
    },
    {
        "name": "pressure_high",
        "json_key": "pressure_high",
        "gold_col": "pressure_high",
        "test_csv": Path("datasets/pressureHigh_test.csv"),
        "prompt_dir": Path("llm_prompt/prompts/pressure_high"),
        "output_root": Path("results/LLM_With_Rule_y2/pressure_high"),
        "output_csv_name": "pressure_high_llm_outputs.csv",
        "accuracy_txt_name": "pressure_high_accuracy_results_biomistral_011.txt",
    },
]


def list_prompt_files(prompt_dir: str):
    files = glob.glob(os.path.join(prompt_dir, "prompt_*.txt"))

    def key_fn(p):
        m = re.search(r"prompt_(\d+)\.txt$", os.path.basename(p))
        return int(m.group(1)) if m else 10**9

    return sorted(files, key=key_fn)


def read_text(fp: str):
    return Path(fp).read_text(encoding="utf-8")


def extract_json(text: str):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        return None

    stack = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0:
                    cand = text[start:i + 1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        break
    return None


def repair_json_text(text: str):
    if not text:
        return text
    s = text.strip()

    start = s.find("{")
    if start == -1:
        return s
    s = s[start:]

    s = re.sub(r",\s*(\}|\Z)", r"\1", s)

    opens = 0
    closes = 0
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                opens += 1
            elif ch == "}":
                closes += 1
    if closes < opens:
        s = s + ("}" * (opens - closes))

    return s


def pick_pred_from_parsed(parsed: dict, json_key: str, idx: int, base: str):
    if not isinstance(parsed, dict):
        return None, None

    expected = json_key.strip().lower()
    norm_map = {str(k).strip().lower(): v for k, v in parsed.items()}

    if expected in norm_map:
        return norm_map[expected], None

    if len(norm_map) == 1:
        only_key = next(iter(norm_map.keys()))
        fixed = {json_key: norm_map[only_key]}
        return norm_map[only_key], fixed

    def deul(s): return s.replace("_", "")
    candidates = [
        k for k in norm_map.keys()
        if expected in k or k in expected or deul(k) == deul(expected)
    ]
    if candidates:
        best = sorted(candidates, key=lambda k: abs(len(k) - len(expected)))[0]
        return norm_map[best], None

    return None, None


def build_prompt(tokenizer, user_prompt: str, json_key: str):
    system = (
        f"Return ONLY one valid JSON object with a single key '{json_key}'. "
        f"The JSON MUST be syntactically valid and CLOSED with '}}'. "
        f"Use straight double quotes (\") only. "
        f"No markdown, no code fences, no extra text."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
        try:
            messages_user_only = [{"role": "user", "content": system + "\n\n" + user_prompt}]
            return tokenizer.apply_chat_template(
                messages_user_only, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    return f"<s>[INST] {system}\n\n{user_prompt} [/INST]"


def save_llm_outputs_to_csv(predictions, csv_path: Path, json_key: str):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_pred = pd.DataFrame({
        "patient_index": [p["patient_index"] for p in predictions],
        "gold_label": [p.get(f"gold_{json_key}") for p in predictions],
        LLM_OUTPUT_COL_NAME: [p.get(f"pred_{json_key}") for p in predictions],
    })

    if csv_path.exists():
        df_exist = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "patient_index" in df_exist.columns:
            df_exist = df_exist.set_index("patient_index")
            df_pred = df_pred.set_index("patient_index")

            df_exist["gold_label"] = df_pred["gold_label"]
            df_exist[LLM_OUTPUT_COL_NAME] = df_pred[LLM_OUTPUT_COL_NAME]
            df_out = df_exist.reset_index()
        else:
            if len(df_exist) != len(df_pred):
                df_out = df_pred
            else:
                df_exist["patient_index"] = df_pred["patient_index"].values
                df_exist["gold_label"] = df_pred["gold_label"].values
                df_exist[LLM_OUTPUT_COL_NAME] = df_pred[LLM_OUTPUT_COL_NAME].values
                df_out = df_exist
    else:
        df_out = df_pred

    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")


def update_accuracy_txt(json_key: str, matches: int, total_used: int, acc: float, txt_path: Path):
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    lines_map = {}
    if txt_path.exists():
        content = txt_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                lines_map[k.strip()] = v.strip()

    lines_map[json_key] = f"accuracy={acc:.4f} ({matches}/{total_used})"

    with txt_path.open("w", encoding="utf-8") as f:
        for k, v in lines_map.items():
            f.write(f"{k}: {v}\n")


def run_task(task, model, tokenizer):
    name = task["name"]
    json_key = task["json_key"]
    gold_col = task["gold_col"]
    test_csv = Path(task["test_csv"])
    prompt_dir = Path(task["prompt_dir"])
    output_root = Path(task["output_root"])
    output_csv_path = output_root / task["output_csv_name"]
    accuracy_dir = output_root / "accuracy"
    accuracy_txt_path = accuracy_dir / task["accuracy_txt_name"]

    print("\n" + "=" * 80)
    print(f"Task: {name} (BioMistral-7B, use rules, single-label)")
    print(f"  json_key      : {json_key}")
    print(f"  gold_col      : {gold_col}")
    print(f"  test_csv      : {test_csv}")
    print(f"  prompt_dir    : {prompt_dir}")
    print(f"  output_root   : {output_root}")
    print(f"  output_csv    : {output_csv_path}")
    print(f"  accuracy_txt  : {accuracy_txt_path}")
    print("=" * 80)

    if not test_csv.exists():
        raise FileNotFoundError(f"[{name}] Test set not found: {test_csv}")
    if not prompt_dir.exists():
        raise FileNotFoundError(f"[{name}] Prompt directory not found: {prompt_dir}")

    df = pd.read_csv(test_csv, encoding="utf-8")
    if gold_col not in df.columns:
        raise KeyError(f"[{name}] Cannot find column '{gold_col}' in test set. Current columns: {list(df.columns)}")

    if "patient_index" in df.columns:
        patient_indices = list(df["patient_index"].astype(int).values)
    else:
        patient_indices = list(range(1, len(df) + 1))

    gold_series = df[gold_col].astype(str).fillna("")
    total_gold = len(gold_series)

    output_root.mkdir(parents=True, exist_ok=True)
    accuracy_dir.mkdir(parents=True, exist_ok=True)

    files = list_prompt_files(str(prompt_dir))
    total = len(files)
    if total == 0:
        print(f"[{name}] No prompt_N.txt found in: {prompt_dir}")
        return

    predictions = []
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
                min_new_tokens=MIN_NEW_TOKENS,
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

            gen_text = tokenizer.decode(
                generated_ids[input_len:], skip_special_tokens=True
            ).strip()

            parsed = extract_json(gen_text)
            if parsed is None:
                fixed_text = repair_json_text(gen_text)
                parsed = extract_json(fixed_text)

            pred_val, fixed_dict = pick_pred_from_parsed(parsed, json_key, idx, base)
            if fixed_dict is not None:
                parsed = fixed_dict

            gold = gold_series.iloc[idx - 1] if idx - 1 < total_gold else ""
            patient_idx = patient_indices[idx - 1] if idx - 1 < len(patient_indices) else idx
            ok = (str(pred_val) == str(gold))

            print("\n" + "=" * 80)
            print(f"[{name}] [{idx}/{total}] File: {base}")
            print(f"Target key: {json_key}")
            print(f"Prompt tokens: {input_len} | Output tokens: {new_tokens}")
            print("\nRaw LLM output:")
            print(gen_text if gen_text else "<EMPTY>")
            print("\nParsed JSON:")
            if parsed is not None:
                print(json.dumps(parsed, ensure_ascii=False, indent=2))
            else:
                print("<FAILED TO PARSE JSON>")
            print(f"\nGOLD {json_key}: {gold}")
            print(f"PRED {json_key}: {pred_val}")
            print(f"MATCH: {ok}")
            print("=" * 80 + "\n")

            predictions.append({
                "index": idx,
                "patient_index": int(patient_idx),
                "file": base,
                "prompt_tokens": int(input_len),
                "output_tokens": int(new_tokens),
                "raw": gen_text,
                "json": parsed,
                f"pred_{json_key}": pred_val,
                f"gold_{json_key}": gold,
                "match": bool(ok),
                "target": name,
            })

            pbar.update(1)

    total_used = min(total, total_gold)
    matches = sum(1 for p in predictions[:total_used] if p["match"])
    acc = matches / total_used if total_used > 0 else 0.0

    save_llm_outputs_to_csv(predictions, output_csv_path, json_key=json_key)
    update_accuracy_txt(json_key, matches, total_used, acc, txt_path=accuracy_txt_path)

    summary = {
        "task": name,
        "json_key": json_key,
        "total_prompts": total,
        "total_gold": int(total_gold),
        "paired_for_eval": int(total_used),
        "correct": int(matches),
        "accuracy": acc,
        "prompt_dir": str(prompt_dir),
        "output_root": str(output_root),
        "llm_output_csv": str(output_csv_path),
        "accuracy_txt": str(accuracy_txt_path),
    }

    print("\n" + "=" * 80)
    print(f"[{name}] FINAL EVALUATION")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


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
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, use_fast=True, local_files_only=True
    )
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
