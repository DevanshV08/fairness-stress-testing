# Fairness Stress Testing for Tabular Machine Learning in High-Stakes Decision Systems

**Devansh Verma** · Arizona State University · dverma26@asu.edu  
**Naman Dewan** · Arizona State University · ndewan@asu.edu  
**Manan Batra** · Arizona State University · mbatra@asu.edu  
**Khushi Thakare** · Arizona State University · kthakare@asu.edu  
**Anushka Mote** · Arizona State University · amote@asu.edu

---

## Abstract

Fairness evaluations in machine learning are routinely reported as a single metric at a single operating point, obscuring the variance that practitioners will encounter under realistic deployment conditions. This project proposes a fairness stress-testing framework that evaluates three standard debiasing methods — Reweighing, Exponentiated Gradient, and Threshold Optimization — across five random seeds, nine classification thresholds, and intersectional subgroups on three benchmark datasets: Adult Income, German Credit, and COMPAS. We evaluate Logistic Regression, Random Forest, and XGBoost as base models. Our central finding confirms the project hypothesis: fairness interventions are real but fragile. On Adult Income, Exponentiated Gradient reduces the Logistic Regression demographic parity difference from 0.174 ± 0.007 to 0.019 ± 0.008 — an 89% reduction — with only a 1.9 percentage-point accuracy cost, but the same method applied to Random Forest costs 6.0 points of accuracy for a 64% fairness gain. Threshold choice alone produces a 63% swing in demographic parity difference on the LR model without any debiasing at all. Intersectional analysis reveals that aggregate sex-based DP difference of 0.183 hides a four-way subgroup spread from 6.4% selection rate (non-white females) to 26.6% (white males). The German Credit results exhibit extreme seed variance — DP difference ranging from −0.036 to +0.182 across seeds for LR — demonstrating that the small-dataset regime amplifies fragility dramatically. Taken together, these results make the case that fairness should be reported as a distribution over seeds, thresholds, and subgroups, not as a scalar benchmark number.

**Keywords:** algorithmic fairness, fairness stress testing, demographic parity, threshold sensitivity, intersectional fairness, Reweighing, Exponentiated Gradient, Threshold Optimization, tabular machine learning, Adult Income, German Credit, COMPAS

---

## 1. Introduction

Machine learning systems are increasingly deployed in high-stakes domains: credit underwriting, hiring pipelines, and criminal justice risk scoring. In all three domains, a single prediction can determine whether a person receives a loan, advances past a screening round, or is recommended for release. The growing use of automated decision systems in these settings has elevated fairness from an ethical aspiration to an engineering requirement.

The standard response to this requirement is to apply a fairness intervention and report the resulting metric improvement. A researcher trains a model, applies Reweighing or Threshold Optimization, and reports — for example — that demographic parity difference falls from 0.19 to 0.09. This report appears in a table. The table is cited. The method is deemed effective.

The problem with this reporting convention is not that the result is wrong. It is that the result is a point estimate for one seed, one split, and one threshold, and point estimates can be misleading when the underlying function is high-variance. Real deployed systems face exactly the conditions that single-point evaluation ignores: the training set changes over time, the decision threshold is tuned to business constraints, and the population contains subgroups that aggregate metrics erase.

This project proposes a systematic stress-testing framework designed to expose this fragility. Rather than asking whether a mitigation method improves a fairness metric, we ask whether the improvement is stable. Specifically, we evaluate every (model × method × dataset) combination across five random seeds, across nine classification thresholds from 0.30 to 0.70, and across intersectional subgroups defined by the joint distribution of two sensitive attributes. We use three benchmark datasets specified by the course project: Adult Income [8], German Credit [8], and COMPAS [9]. We evaluate three model families — Logistic Regression, Random Forest, and XGBoost — against three debiasing categories: pre-processing (Reweighing [3]), in-processing (Exponentiated Gradient [4]), and post-processing (Threshold Optimization [5]).

Our headline finding, already visible in the executed pipeline, is that the effectiveness of a fairness intervention depends on the model, the dataset size, and the operating point — not just on the method itself. A practitioner who reads a benchmark table and selects a method based on a reported DP difference number is selecting on one sample from a distribution that may have substantial variance.

---

## 2. Related Work

### 2.1 Fairness in Machine Learning

The literature on group fairness in supervised learning is extensive and well-surveyed [6, 7]. The three primary fairness criteria most relevant to tabular classification are demographic parity, which requires equal positive selection rates across groups; equal opportunity, which requires equal true positive rates; and equalized odds, which requires both TPR and FPR to be equal across groups. These criteria are mutually incompatible in the general case [5], so any evaluation must report tradeoffs rather than expecting joint satisfaction.

Barocas, Hardt, and Narayanan [6] provide the canonical treatment of these impossibility results and their practical implications. The central lesson is that fairness interventions do not eliminate the underlying tradeoff — they choose a different point on it.

### 2.2 Debiasing Methods

Three categories of intervention dominate the fairness mitigation literature. Pre-processing methods modify the training data before model fitting; Kamiran and Calders' Reweighing [3] is the most widely cited, assigning sample weights that equalize group representation in the training loss. In-processing methods add fairness constraints during training; Agarwal et al.'s Exponentiated Gradient [4] is a general reduction that converts a fairness-constrained classification problem into a sequence of weighted standard classification problems. Post-processing methods adjust the classifier's decision boundary after training; Hardt, Price, and Srebro's equalized odds post-processing [5] is the canonical method, and Fairlearn's ThresholdOptimizer implements it as group-specific threshold selection.

### 2.3 The Single-Point Evaluation Problem

Despite active work on both new metrics and new methods, published fairness benchmarks almost universally report a single seed, a single threshold (typically the default 0.5), and an aggregate sensitive attribute rather than intersectional subgroups. Uddin et al. [1] and Chen [2] both note this limitation in recent surveys of robustness in fairness-aware ML. The triangular tradeoff survey [10] highlights that accuracy, robustness, and fairness interact in ways that single-point evaluation cannot capture. Our framework is designed to make this interaction visible.

### 2.4 Intersectional Fairness

Intersectional fairness, originating from Crenshaw's [11] sociological concept of intersectionality, recognizes that individuals belong to multiple protected groups simultaneously and that aggregate metrics over a single sensitive attribute can mask severe harm to multiply-marginalized subgroups. Recent work [1, 2] confirms that intersectional analysis is rarely performed in benchmark evaluations despite being straightforward to implement. We include it as a required component of our evaluation protocol.

### 2.5 Dataset Validity

All three datasets we use have known limitations. The Adult Income dataset is derived from 1994 Census data and reflects historical patterns of discrimination; the labels are not causal ground truth. German Credit has a sample size of 1,000, which produces wide confidence intervals for any metric estimate — a fact we treat as a feature of the stress-test story rather than a bug. COMPAS has been subject to extensive methodological critique, most comprehensively by Bao et al. [9] ("It's COMPASlicated"), who show that the recidivism labels themselves embed contested assumptions. We report COMPAS results with explicit acknowledgment that the labels should not be treated as ground truth.

---

## 3. Methodology

### 3.1 Overall Framework

Our pipeline has four phases:

- **Phase 1 — Data Loading and Preprocessing:** Each dataset is loaded via a dedicated function that returns (X, y, sensitive_attrs) with a consistent schema. Sensitive attributes are carried as metadata and are never included as model features.
- **Phase 2 — Model Training and Baseline Evaluation:** Each (model × method × seed) combination is trained on a stratified 70/30 split and evaluated on the held-out test set. Metrics are recorded to JSON.
- **Phase 3 — Stress Testing:** The multi-seed loop, threshold sweep, and intersectional breakdown are run from the saved model outputs.
- **Phase 4 — Visualization:** All figures are generated from the results CSVs by a dedicated script.

### 3.2 Datasets

**Table 1: Dataset statistics.**

| Dataset | N (train) | N (test) | Features | Sensitive attributes | Positive base rate |
|---|---|---|---|---|---|
| Adult Income | ~34,000 | ~9,769 | 13 | sex, race | ~24% |
| German Credit | 700 | 300 | 20 | sex, age | 70% |
| COMPAS | ~4,300 | ~1,850 | 10 | race | ~46% |

The favorable outcome is encoded as `1` in every dataset. For Adult Income, we use sex (Female=0, Male=1) as the primary sensitive attribute and race (Non-White=0, White=1) for intersectional analysis. For German Credit, sex is derived from the personal_status feature; age is binarized at 25. For COMPAS, race is binarized as African-American versus Caucasian, with African-American as group 0 (the historically disadvantaged group).

### 3.3 Model Suite

We evaluate three model families.

**Logistic Regression (LR):** A linear classifier with L2 regularization (C=1.0). This is the canonical fairness benchmark model; results are directly comparable to published numbers. It is also the only model for which all three debiasing methods have well-understood behavior.

**Random Forest (RF):** An ensemble of 100 decision trees. RF is more expressive than LR and is often used in credit scoring applications. Its lack of a native probability calibration makes threshold optimization more sensitive.

**XGBoost:** A gradient-boosted tree ensemble. XGBoost consistently achieves higher accuracy than LR and RF on tabular data and is the most common model in production tabular ML pipelines.

All models accept a `seed` argument that controls all internal random state. Models are wrapped behind a common `build_model(name, seed, **kwargs)` interface defined in `code/models.py`.

### 3.4 Fairness Metrics

We report the following fairness metrics from `code/fairness_metrics.py`:

**Demographic Parity Difference (DP diff):** The absolute difference in positive prediction rates between the advantaged and disadvantaged group:

$$\text{DP diff} = P(\hat{Y}=1 \mid A=1) - P(\hat{Y}=1 \mid A=0)$$

A value of 0 indicates demographic parity. Positive values indicate the advantaged group (A=1) receives more favorable predictions.

**Equal Opportunity Difference (EO diff):** The difference in true positive rates:

$$\text{EO diff} = \text{TPR}_{A=1} - \text{TPR}_{A=0}$$

**Equalized Odds Difference (EOd diff):** The maximum of the TPR difference and FPR difference across groups.

We also report per-group selection rate, TPR, FPR, and Expected Calibration Error (ECE) for baseline models. Accuracy and AUC (where available) are the utility metrics.

### 3.5 Debiasing Methods

**Reweighing [3]:** A pre-processing method that assigns each training sample a weight inversely proportional to the joint frequency of its (group, label) cell. Samples from underrepresented (group, label) combinations receive higher weight. Because Reweighing only changes the sample weights passed to `fit()`, it is compatible with any sklearn-compatible estimator.

**Exponentiated Gradient [4]:** An in-processing method that solves a constrained fair classification problem by iteratively reweighting a standard classifier. We use Fairlearn's `ExponentiatedGradient` with a `DemographicParity` constraint and a tolerance of 0.01. This method produces a randomized classifier internally, so its predictions are not directly comparable to threshold-based methods for equalized odds analysis.

**Threshold Optimization [5]:** A post-processing method that selects group-specific decision thresholds on the validation set to satisfy a fairness constraint. We use Fairlearn's `ThresholdOptimizer` with `demographic_parity` as the objective constraint. This method requires probability outputs, so it is applied only to LR and RF baselines (XGBoost threshold optimization is included where calibrated probabilities are available).

### 3.6 Stress-Testing Framework

The stress-testing layer is the central contribution of the project. It has three components:

**Seed sweep:** Every (model × method × dataset) combination is run across seeds {0, 1, 2, 3, 4}. We report mean ± standard deviation across seeds for every metric. A method that shows high standard deviation in DP diff is fragile by definition — the reported improvement at seed 0 does not generalize across initializations.

**Threshold sweep:** For probabilistic classifiers (LR, RF, XGBoost), we sweep the classification threshold from 0.30 to 0.70 in steps of 0.05 and record accuracy and all fairness metrics at each point. This reveals the sensitivity of fairness conclusions to the operating point choice, independent of any debiasing method.

**Intersectional breakdown:** We compute per-subgroup statistics for all (sex × race) cells on Adult, (sex × age) cells on German, and report selection rate, accuracy, TPR, and FPR for each cell. Cells with n < 100 are flagged; cells with n < 30 are dropped.

---

## 4. Evaluation and Analysis

### 4.1 Evaluation Goals

We evaluate the pipeline at two levels:

**Method-level evaluation:** Does the debiasing method reduce the fairness gap? By how much, and at what accuracy cost? Is the reduction consistent across seeds?

**Stress-test evaluation:** How much does fairness depend on seed choice, threshold choice, and subgroup granularity? Can a method that looks effective at one operating point look ineffective — or harmful — at another?

The combination of these two evaluations is the project's contribution. Most published benchmarks address only the first question.

### 4.2 Baseline Results

**Table 2: Baseline accuracy and DP difference across seeds (mean ± std). Adult Income dataset, sex attribute.**

| Model | Accuracy | DP Difference | AUC |
|---|---|---|---|
| Logistic Regression | 0.8541 ± 0.0019 | 0.174 ± 0.007 | 0.906 ± 0.004 |
| Random Forest | 0.8556 ± 0.0014 | 0.185 ± 0.006 | 0.906 ± 0.004 |
| XGBoost | 0.8713 ± 0.0011 | 0.179 ± 0.006 | 0.927 ± 0.002 |

The LR baseline aligns with the expected range from prior published results (84–86% accuracy, DP diff 0.17–0.19 [6]), confirming that our preprocessing and evaluation pipeline are behaving correctly. XGBoost achieves 1.72 percentage points higher accuracy than LR, consistent with its known advantage on tabular data. All three models produce similar DP differences, suggesting that model family does not substantially change the baseline discrimination on this dataset at the aggregate level.

Seed variance for accuracy is low for all three models (std ≤ 0.002), reflecting the large size of the Adult dataset. Seed variance for DP difference is larger (std ≈ 0.006–0.007), which has a meaningful practical consequence: a practitioner who reports a single DP difference at one seed is reporting a value that could differ by ±0.014 (two standard deviations) across re-runs.

**Table 3: Baseline accuracy and DP difference. German Credit dataset, sex attribute.**

| Model | Accuracy | DP Difference | AUC |
|---|---|---|---|
| Logistic Regression | 0.756 ± 0.038 | 0.060 ± 0.085 | 0.784 ± 0.048 |
| Random Forest | 0.768 ± 0.014 | 0.047 ± 0.073 | 0.795 ± 0.036 |
| XGBoost | 0.757 ± 0.036 | 0.036 ± 0.056 | 0.775 ± 0.047 |

The German Credit results immediately illustrate the small-sample fragility that the stress-testing framework is designed to expose. LR accuracy varies from 0.690 to 0.800 across five seeds — a 11-point swing on a metric that is treated as a fixed property of the model in standard benchmark reporting. The DP difference ranges from −0.036 to +0.182, meaning that the sign of the fairness gap itself flips across seeds. This is not a bug in the pipeline; it is the correct empirical result for a dataset with n=1,000. It is precisely the kind of finding that should appear in published benchmarks but almost never does.

**Table 4: Baseline results. COMPAS dataset, race attribute (seed 0).**

| Model | Accuracy | DP Difference | EO Difference | AUC |
|---|---|---|---|---|
| Logistic Regression | 0.6619 | 0.230 | 0.134 | 0.702 |
| Random Forest | 0.6155 | 0.149 | 0.089 | 0.657 |
| XGBoost | 0.6468 | 0.199 | 0.134 | 0.670 |

COMPAS results are reported with the standard caveat that the recidivism labels encode contested assumptions and should not be treated as ground truth [9]. The DP differences (0.149–0.230) are substantially larger than on Adult, reflecting the well-documented racial disparity in this dataset. The accuracy ceiling (~66%) reflects both label noise and the difficulty of the prediction task, not a modeling failure.

### 4.3 Debiasing Results

**Table 5: Effect of debiasing on DP difference and accuracy. Adult Income, LR model, mean across seeds 0–4.**

| Method | Accuracy | DP Difference | DP Reduction | Accuracy Cost |
|---|---|---|---|---|
| None (baseline) | 0.8541 ± 0.0019 | 0.174 ± 0.007 | — | — |
| Reweighing | 0.8483 ± 0.0016 | 0.095 ± 0.004 | 45% | 0.6 pp |
| Exponentiated Gradient | 0.8350 ± 0.0024 | 0.019 ± 0.008 | 89% | 1.9 pp |
| Threshold Optimizer | 0.8330 ± 0.0027 | 0.003 ± 0.011 | 98% | 2.1 pp |

All three methods reduce DP difference meaningfully. Exponentiated Gradient and Threshold Optimizer both achieve near-zero aggregate DP difference with less than 2.5 percentage points of accuracy cost, satisfying the project's success criterion (>50% DP reduction, <3% accuracy loss). Reweighing produces a smaller fairness gain but is nearly lossless in accuracy (0.6 pp cost).

The standard deviation of DP difference increases substantially after debiasing: from 0.007 (baseline) to 0.011 (ThresholdOptimizer). This is a key finding. The method that produces the largest fairness improvement at the mean also produces the largest variance across seeds, meaning its effectiveness is most sensitive to random initialization.

**Table 6: Effect of debiasing on DP difference and accuracy. Adult Income, RF model, mean across seeds 0–4.**

| Method | Accuracy | DP Difference | DP Reduction | Accuracy Cost |
|---|---|---|---|---|
| None (baseline) | 0.8556 ± 0.0014 | 0.185 ± 0.006 | — | — |
| Reweighing | 0.8553 ± 0.0016 | 0.185 ± 0.008 | ~0% | 0.0 pp |
| Exponentiated Gradient | 0.7953 ± 0.0028 | 0.067 ± 0.007 | 64% | 6.0 pp |
| Threshold Optimizer | 0.7870 ± 0.0062 | −0.064 ± 0.046 | — | 6.9 pp |

The RF results tell a qualitatively different story. Reweighing has essentially no effect on RF's DP difference (185 → 185), while Exponentiated Gradient achieves a 64% reduction at a cost of 6 percentage points of accuracy — three times the LR cost for a smaller fairness gain. Threshold Optimizer for RF produces a negative mean DP difference (−0.064), meaning the method has over-corrected and reversed the direction of the disparity. Its standard deviation (0.046) is more than four times that of the LR equivalent, indicating that the post-processing correction is highly unstable on RF. This result demonstrates that method effectiveness is strongly model-dependent, and that a practitioner who selects a debiasing method based on its published LR results and applies it to an RF model may obtain a qualitatively different outcome.

**Table 7: Effect of debiasing on DP difference and accuracy. Adult Income, XGBoost model, mean across seeds 0–4.**

| Method | Accuracy | DP Difference | DP Reduction | Accuracy Cost |
|---|---|---|---|---|
| None (baseline) | 0.8713 ± 0.0011 | 0.179 ± 0.006 | — | — |
| Reweighing | 0.8683 ± 0.0022 | 0.118 ± 0.008 | 34% | 0.3 pp |
| Exponentiated Gradient | 0.8519 ± 0.0031 | 0.027 ± 0.009 | 85% | 1.9 pp |

XGBoost with Exponentiated Gradient achieves an 85% DP reduction with only 1.9 pp accuracy cost — comparable to LR's result — while retaining a 1.7 pp accuracy advantage over LR at the mitigated operating point (0.8519 vs 0.8350). This suggests that XGBoost + Exponentiated Gradient is the best overall tradeoff on the Adult dataset when fairness and utility are both valued.

### 4.4 Seed Variance Analysis

The seed sweep results quantify the stability of each (model × method) configuration. The key comparison is the coefficient of variation (CV = std / |mean|) of DP difference across seeds.

**Table 8: Coefficient of variation of DP difference across 5 seeds. Adult Income.**

| Model | None | Reweighing | Exp. Gradient | Threshold Opt. |
|---|---|---|---|---|
| LR | 0.04 | 0.04 | 0.39 | 3.44 |
| RF | 0.03 | 0.04 | 0.10 | — (sign flip) |
| XGBoost | 0.03 | 0.07 | 0.34 | — |

Three patterns emerge. First, baseline models are relatively stable in DP difference across seeds on the Adult dataset (CV ≈ 0.03–0.04). Second, Reweighing preserves that stability (CV ≈ 0.04–0.07). Third, the more aggressive methods — Exponentiated Gradient and Threshold Optimizer — have dramatically higher CV values (0.34–3.44), with the ThresholdOptimizer's near-zero mean DP difference making even small absolute fluctuations appear as large relative fluctuations.

The practical implication is that a fairness gain of 98% at seed 0 (ThresholdOptimizer on LR) may represent genuine progress, but it should be reported alongside the standard deviation (±0.011) to make clear that a replicated run might yield a 0.003 or a 0.017 DP difference — a factor-of-six range.

On German Credit, the seed variance story is even more striking. LR DP difference spans −0.036 to +0.182 across five seeds, a range of 0.218 units — larger than the entire DP difference range observed on Adult across all seeds and methods. This is a direct consequence of small sample size (n=1,000): the test set contains only ~300 examples, and a difference of a few predictions in a subgroup produces a large metric swing.

### 4.5 Threshold Sensitivity Analysis

**Table 9: Adult Income, LR baseline. Fairness and accuracy across classification thresholds (seed 0).**

| Threshold | Accuracy | DP Difference | EO Difference |
|---|---|---|---|
| 0.30 | 0.8306 | 0.282 | 0.122 |
| 0.35 | 0.8436 | 0.251 | 0.113 |
| 0.40 | 0.8498 | 0.227 | 0.100 |
| 0.45 | 0.8528 | 0.200 | 0.090 |
| **0.50** | **0.8541** | **0.183** | **0.103** |
| 0.55 | 0.8533 | 0.164 | 0.100 |
| 0.60 | 0.8499 | 0.144 | 0.090 |
| 0.65 | 0.8457 | 0.120 | 0.065 |
| 0.70 | 0.8353 | 0.103 | 0.079 |

Moving the threshold from 0.30 to 0.70 reduces DP difference from 0.282 to 0.103 — a 63% reduction — without applying any debiasing method at all. The accuracy cost of this shift is only 1.88 percentage points (0.8541 → 0.8353). This means that a practitioner who increases the classification threshold from the default 0.50 to 0.70 obtains more than half the fairness improvement that Reweighing delivers, at a comparable accuracy cost, with no modification to the training procedure.

This finding has a direct consequence for how published results should be interpreted. When a paper reports that Method X reduces DP difference by 50% at a default threshold, the reader cannot distinguish between the fairness improvement that comes from the method and the fairness improvement that would come from simply raising the threshold. A threshold-fairness curve (plotting DP difference vs. accuracy across thresholds) is a prerequisite for interpreting debiasing claims honestly.

The threshold sweep also reveals that fairness and accuracy are not monotonically opposed on this dataset. Accuracy peaks at threshold 0.45–0.50 and declines on both sides, while DP difference decreases monotonically as the threshold increases. This means there exists a range of thresholds (0.55–0.65) where both accuracy and fairness are simultaneously better than at the default 0.50 — a Pareto improvement that a practitioner would find if they consulted a threshold curve but would miss if they evaluated only at the default.

### 4.6 Intersectional Analysis

The aggregate sex-based DP difference of 0.183 (LR, seed 0) compares a 7.2% positive selection rate for the female group to a 25.5% positive selection rate for the male group. The intersectional breakdown by sex × race tells a more detailed story.

**Table 10: Intersectional selection rates. Adult Income, LR baseline, seed 0.**

| Subgroup | N (test) | Selection Rate | Accuracy | TPR |
|---|---|---|---|---|
| Female, Non-White | 642 | 6.4% | 93.6% | 51.3% |
| Female, White | 2,589 | 7.4% | 92.9% | 51.6% |
| Male, Non-White | 804 | 17.9% | 85.6% | 58.2% |
| Male, White | 5,734 | 26.6% | 81.1% | 61.7% |

The four-subgroup view reveals structure that the aggregate metric hides. Non-white females have the lowest selection rate (6.4%), while white males have the highest (26.6%) — a 4.2-fold ratio. The aggregate sex-based DP difference (0.183) reflects the average gap between male and female groups, but it does not surface the race × sex interaction: even within the female group, white females have a 1.0 percentage-point higher selection rate than non-white females; within the male group, white males have an 8.7 percentage-point advantage over non-white males.

The accuracy pattern runs in the opposite direction: non-white females have the highest per-group accuracy (93.6%), and white males have the lowest (81.1%). This apparent paradox reflects the base rate. The model achieves high accuracy on non-white females partly because it predicts negative for nearly all of them — and the ground-truth positive rate in that subgroup is genuinely low. High per-group accuracy does not imply fairness.

The TPR column further illustrates the problem. All four subgroups have true positive rates in the 51–62% range, suggesting that the model is not dramatically worse at recognizing positive cases in any single subgroup. The disparity in selection rate is driven primarily by the difference in how often the model predicts positive for different groups relative to their base rates — the demographic parity issue — rather than by a failure to identify positives when they are present. This distinction matters for which fairness criterion is most appropriate for this application.

### 4.7 Summary of Evaluation Findings

The main findings are:

- Debiasing methods reduce DP difference consistently on Adult Income, but effectiveness is model-dependent. Exponentiated Gradient achieves 89% DP reduction on LR at 1.9 pp accuracy cost, but only 64% reduction on RF at 6.0 pp cost.
- Reweighing has essentially no effect on Random Forest's fairness gap, despite being the most commonly recommended pre-processing method.
- Seed variance for DP difference is low at baseline but increases substantially with aggressive debiasing, particularly Threshold Optimization (CV = 3.44 on LR due to near-zero mean).
- Threshold choice alone produces a 63% swing in DP difference without any debiasing, suggesting that threshold calibration is an underappreciated lever in fairness-aware deployment.
- Intersectional analysis reveals a 4.2-fold selection rate ratio (white males vs. non-white females) hidden behind an aggregate DP difference of 0.183.
- German Credit exhibits extreme seed variance (DP diff ranging from −0.036 to +0.182 across seeds for LR), illustrating that small-dataset fairness benchmarks are unreliable without distributional reporting.
- COMPAS shows the highest baseline DP differences (0.149–0.230), but results are interpreted with caution given contested label validity [9].

---

## 5. Team Contributions

1. **Devansh Verma** (Project Lead / Fairness Modeling Architect): Designed the end-to-end pipeline architecture, implemented the fairness metrics module (`fairness_metrics.py`), integrated all three debiasing methods with the common model interface, ran the full experiment grid, and consolidated the report and results analysis.

2. **Naman Dewan** (Data Engineering Lead): Implemented the dataset loaders (`data_loaders.py`) for Adult Income, German Credit, and COMPAS. Designed the consistent (X, y, sensitive_attrs) schema. Implemented stratified splitting, sensitive attribute binarization, and train/test consistency guarantees.

3. **Manan Batra** (Model Development Lead): Implemented the model builders (`models.py`) for Logistic Regression, Random Forest, and XGBoost behind the common `build_model` interface. Managed the FT-Transformer wrapper for the rtdl_revisiting_models API.

4. **Khushi Thakare** (Debiasing Strategy Lead): Implemented Reweighing, ExponentiatedGradient, and ThresholdOptimizer in `debiasing.py`. Designed the stress-testing module (`stress_tests.py`) including the threshold sweep, seed sweep loop, and intersectional breakdown.

5. **Anushka Mote** (Documentation & Viz Lead): Implemented the visualization module (`visualize.py`), produced all figures, conducted the literature review, and assembled the final presentation deck.

---

## 6. Conclusion

This project demonstrates that standard fairness evaluation practice — a single metric at a single seed and threshold — provides an incomplete and often misleading view of a method's practical effectiveness. Across three benchmark datasets and three model families, we show that:

Debiasing works in expectation. Exponentiated Gradient and Threshold Optimization reduce demographic parity difference by 85–98% on Logistic Regression and XGBoost with less than 2 percentage points of accuracy cost. These gains are real.

But debiasing is fragile in detail. The same methods applied to Random Forest produce qualitatively different outcomes: Reweighing fails entirely, and Threshold Optimization reverses the direction of the disparity with high variance across seeds. A practitioner who selects a method based on a published benchmark table will not encounter this failure mode unless they evaluate on their specific model.

Threshold choice rivals method choice in impact. Moving the LR threshold from 0.30 to 0.70 reduces DP difference by 63% — more than Reweighing achieves — without any modification to the training procedure. This threshold effect is almost never reported in fairness benchmarks.

Intersectional analysis is necessary. An aggregate DP difference of 0.183 (sex-based, Adult Income) masks a 4.2-fold selection rate disparity between white males and non-white females. Aggregate reporting is not wrong, but it is incomplete.

Small datasets amplify all of the above. German Credit's DP difference ranges from −0.036 to +0.182 across five seeds for LR, demonstrating that fairness metrics for small datasets are effectively meaningless as point estimates.

The recommendation that follows from these findings is methodological: fairness evaluations should report distributions over seeds, threshold curves over operating points, and intersectional breakdowns over subgroups — not single numbers. This is not a large additional burden; our stress-testing framework implements all three in a single CLI call.

---

## 7. Future Work

Several directions extend this work:

- **FT-Transformer integration:** The codebase includes an sklearn-compatible FT-Transformer wrapper built on `rtdl_revisiting_models`. Completing the GPU-enabled evaluation would add a tabular transformer to the model comparison, which is largely absent from the fairness benchmarking literature [1, 2].
- **Distribution shift stress testing:** Using Folktables [8] to construct time-series splits of the ACS data would test whether fairness improvements hold under covariate shift, extending the stress-test framework from initialization variance to distributional variance.
- **Fragility index formalization:** The seed × threshold × intersectional variance could be aggregated into a scalar "fragility index" per (model × method) configuration, providing a single number that summarizes stability rather than a point estimate of fairness.
- **ExpGradient for non-LR models:** Our current evaluation applies Exponentiated Gradient to RF and XGBoost via Fairlearn's base estimator interface. A deeper analysis of why RF + ExpGradient incurs a 6 pp accuracy cost while LR + ExpGradient incurs only 1.9 pp would illuminate the method's dependence on the model's hypothesis class.
- **Extended seed grid:** Moving from 5 seeds to 10 would narrow the standard error estimates and make the distributional claims in the stress-test analysis statistically tighter.

---

## References

[1] Uddin, S., Lu, H., & Hajati, F. (2026). Fairness-preserving framework for machine learning: data bias quantification, model evaluation, and robustness across multiple datasets. *AI and Ethics*, 6, 25.

[2] Chen, N.L. (2025). Robustness and fairness evaluation of machine learning models in high-stakes decision systems. *International Journal of Machine Learning in Communications and Networking*, 2(1), 8–12.

[3] Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. *Knowledge and Information Systems*, 33(1), 1–33.

[4] Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., & Wallach, H. (2018). A reductions approach to fair classification. *Proceedings of the 35th International Conference on Machine Learning (ICML)*.

[5] Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 29.

[6] Barocas, S., Hardt, M., & Narayanan, A. (2024). *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press.

[7] Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2024). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 57(1), 1–38.

[8] Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). Retiring adult: New datasets for fair machine learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 34.

[9] Bao, M., Zhou, A., Zottola, S., Brubach, B., Desmarais, S., Horowitz, A., Lum, K., & Venkatasubramanian, S. (2021). It's COMPASlicated: The messy relationship between RAI datasets and algorithmic fairness benchmarks. *NeurIPS Datasets and Benchmarks Track*.

[10] Triangular trade-off between robustness, accuracy, and fairness in deep neural networks. (2024). *ACM Computing Surveys*.

[11] Crenshaw, K. (1989). Demarginalizing the intersection of race and sex: A black feminist critique of antidiscrimination doctrine, feminist theory and antiracist politics. *University of Chicago Legal Forum*, 1989(1), Article 8.
