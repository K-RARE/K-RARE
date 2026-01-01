import json
import os
from pathlib import Path

LOW_IN  = "results/rule/pressure_low/rule/rules.json"
HIGH_IN = "results/rule/pressure_high/rule/rules.json"

LOW_OUT  = "results/rule/pressure_low/rule/llm_rules_low.json"
HIGH_OUT = "results/rule/pressure_high/rule/llm_rules_high.json"


def convert_rules(input_file: str, output_file: str, pressure_word: str):
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    new_rules = {}
    with_example_cnt = 0

    for rule in data.get("rules", []):
        rule_id     = str(rule.get("id"))
        confidence  = rule.get("confidence")
        rule_class  = rule.get("class")
        values      = rule.get("values", []) or []
        example     = rule.get("example", None)

        values_text = "、".join(str(v) for v in values)
        text = (
            f"If the characteristics of a patient include {values_text}, "
            f"then the {pressure_word} pressure might need to be set to {rule_class}."
        )

        item = {
            "text": text,
            "confidence": confidence,
            "class": rule_class
        }

        if example:
            example_list = [str(x) for x in example]
            item["example"] = example_list

            rc_str = "" if rule_class is None else str(rule_class)
            rc_prefix = rc_str.lower()

            if rc_prefix.startswith("low_pressure"):
                tail = f"The low-pressure parameters of the ventilator we chose for it are {rc_str}."
            elif rc_prefix.startswith("high_pressure"):
                tail = f"The high-pressure parameters of the ventilator we chose for it are {rc_str}."
            else:
                tail = f"The ventilator parameters we chose for it are {rc_str}."

            joined = ", ".join(example_list)
            item["example_text"] = f"The characteristic of a patient is {joined}. {tail}"

            with_example_cnt += 1

        new_rules[rule_id] = item

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_rules, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Done! Converted {len(new_rules)} rules "
        f"(with example: {with_example_cnt}) and saved to:\n{output_file}"
    )


if __name__ == "__main__":
    convert_rules(LOW_IN,  LOW_OUT,  pressure_word="low")
    convert_rules(HIGH_IN, HIGH_OUT, pressure_word="high")
