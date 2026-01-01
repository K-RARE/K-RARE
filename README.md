## Description

K-RARE (Knowledge-prior Rule-Augmented LLM Re-ranking) is an explainable parameter recommendation framework designed to assist initial pressure setting (IPAP/EPAP) for PAP therapy in patients with Obstructive Sleep Apnea (OSA). In real-world settings where unified expert thresholds are unavailable, K-RARE uses equal-frequency binning to construct a reproducible discretized feature representation, mines class-wise (label-to-feature) combinations of 1–8 features from the training data, and builds a compact rule library with representative instances via statistical metrics (e.g., confidence, lift, χ²) and subset pruning. During inference, KNN first generates a Top-3 candidate sequence in the discretized space; then matched rules and their confidence/instance evidence are provided as structured priors to an LLM for consistency checking and re-ranking, producing an auditable Top-3 recommendation list that reduces trial-and-error and improves reproducibility (this system is intended to assist—not replace—clinical judgment; all recommendations should be interpreted together with patient status, monitoring feedback, and clinician experience). Here we provide instructions for running the K-RARE scripts; this project and related code are for research and non-commercial use only.

## How to use

### Data preparation

K-RARE takes as input a discretized feature dataset. Each continuous numerical feature is mapped to a discrete token according to predefined threshold intervals (e.g., `BMI_VeryLow`, `Age_Mid`), converting continuous values into interval-based representations. Each row corresponds to one sample, columns are discretized feature intervals, and the final label column is the target pressure level to recommend (e.g., `Low_Pressure_Mid` or `High_Pressure_Mid`).

### Example

Below is an example of a discretized feature table (5 rows). Each row represents one sample, and the last column is the label (pressure level):

| neck_circumference | WHR         | BMI          | age         | ESS          | Friedman      | tonsil     | loudness_score     | pharyngeal_stenosis_score | AHI          | pressure_low / pressure_high |
| ------------------ | ----------- | ------------ | ----------- | ------------ | ------------- | ---------- | ------------------ | ------------------------- | ------------ | ---------------------------- |
| Neck_Low           | WHR_VeryLow | BMI_VeryLow  | Age_Mid     | ESS_Mid      | Friedman_Mid  | Tonsil_Low | Loundness_Mid      | Pharyngeal_Mid            | AHI_Low      | Low_Pressure_Low             |
| Neck_High          | WHR_High    | BMI_High     | Age_Extreme | ESS_VeryHigh | Friedman_High | Tonsil_Mid | Loundness_High     | Pharyngeal_Mid            | AHI_Mid      | Low_Pressure_Mid             |
| Neck_Mid           | WHR_Low     | BMI_VeryLow  | Age_High    | ESS_VeryHigh | Friedman_High | Tonsil_Low | Loundness_VeryHigh | Pharyngeal_Mid            | AHI_VeryHigh | Low_Pressure_Mid             |
| Neck_Low           | WHR_Mid     | BMI_VeryLow  | Age_Mid     | ESS_Mid      | Friedman_High | Tonsil_Low | Loundness_VeryHigh | Pharyngeal_Mid            | AHI_VeryHigh | High_Pressure_Mid            |
| Neck_High          | WHR_High    | BMI_VeryHigh | Age_Mid     | ESS_Mid      | Friedman_High | Tonsil_Low | Loundness_Low      | Pharyngeal_High           | AHI_High     | High_Pressure_Mid            |

The following provides brief explanations of some discretized intervals. Each string token (e.g., `BMI_VeryLow`) indicates which interval a continuous feature value falls into after binning, i.e., every continuous value is mapped to one and only one corresponding interval token, converting continuous variables into discrete representations.

- **BMI_VeryLow**: The meaning of `BMI_VeryLow` is that **BMI ≤ 24.39**.
- **BMI_High**: The meaning of `BMI_High` is that **27.69 < BMI ≤ 29.75**.
- **Neck_Low**: The meaning of `Neck_Low` is that **36 < neck circumference ≤ 38**.
- **Neck_High**: The meaning of `Neck_High` is that **39 < neck circumference ≤ 41**.
- **Age_Mid**: The meaning of `Age_Mid` is that **33 < age ≤ 39**.
- **Age_Extreme**: The meaning of `Age_Extreme` is that **age > 48**.

> Note: The exact bin boundaries should be determined according to your own dataset and applied consistently.

### Running Steps

This repository is organized into several folders corresponding to each module/stage of the K-RARE pipeline:

- `get_rules/`: rule mining and construction (rule search, filtering/pruning, generating rule libraries / representative instances)
- `biomistral/`: predictions for Methods 1–3 (LLM inference under rule priors / representative-instance constraints)
- `predict_by_confidence/`: predictions for Method 4 (pure rule-based branch: selecting the best rule by confidence)
- `KNN/`: KNN candidate sequence generation (Top-3 recommendation candidates for each sample)
- `summary_four_predict/`: weighted ensemble and re-ranking (combine Methods 1–4 Top-1 and rerank KNN candidates to obtain the final Top-3)

Please run the pipeline in the following order:

1. **Rule mining & rule library construction (`get_rules/`)**First run the scripts in `get_rules/` to search and extract rules, apply filtering/pruning, and build the rule libraries (including representative instances).**Output:** rule files/libraries required by downstream modules.
2. **Methods 1–3: LLM-based inference (`biomistral/`)**Run the scripts in `biomistral/` to generate prediction results for Methods 1–3 (LLM inference guided by rule/instance evidence).**Output:** Top-1 predictions from Methods 1–3 (one per sample).
3. **Method 4: confidence-based rule prediction (`predict_by_confidence/`)**Run the scripts in `predict_by_confidence/` to generate Method 4 predictions (selecting the best matched rule by confidence).**Output:** Top-1 predictions from Method 4.
4. **KNN candidate sequences (`KNN/`)**Run the scripts in `KNN/` to generate KNN-based Top-3 candidate recommendation sequences for each sample.**Output:** KNN Top-3 candidate lists.
5. **Weighted ensemble + re-ranking (`summary_four_predict/`)**
   Finally, run the scripts in `summary_four_predict/` to combine Methods 1–4 using fixed weights to obtain the final Top-1, then re-rank the KNN Top-3 candidates accordingly to produce the final Top-3 recommendation sequence.
   **Output:** final Top-3 recommendations (after weighted fusion and re-ranking).

> Note: Please make sure to run all commands from the repository root directory.
