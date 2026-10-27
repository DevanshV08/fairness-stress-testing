# Presentation Script — CSE 575 Project P17: Fairness Under Stress

**Team 17:** Devansh Verma · Naman Dewan · Manan Batra · Anushka Mote · Khushi Thakare
**Total slides:** 13 | **Target time:** ~13–15 minutes + Q&A

**Block structure — no one speaks twice:**
- Devansh: Slides 1–2
- Naman: Slides 3–4
- Manan: Slides 5–6
- Khushi: Slides 7–9
- Anushka: Slides 10–13

---

## BLOCK 1 — DEVANSH (Slides 1–2)

---

### SLIDE 1 — Title: Fairness Under Stress

> "Hi everyone — we're Team 17, and our project is called Fairness Under Stress.
>
> The title is intentional. Fairness-aware machine learning has a whole toolkit of
> methods designed to reduce bias in automated decisions. Our question is not
> whether those methods work in a lab setting. Our question is whether they hold
> up when you stress them — change the random seed, move the decision threshold,
> or look at a subgroup the aggregate metric was quietly hiding.
>
> I'll set up the problem and hand off to Naman who will walk through how
> we structured our experiments."

---

### SLIDE 2 — Automated Decisions in High-Stakes Domains

> "The stakes here are real. Automated systems are now making or heavily
> influencing decisions in hiring — resume screening, offer scoring.
> In credit — loan approvals, risk scoring.
> And in criminal justice — bail, sentencing, recidivism prediction.
>
> The core problem: ML models learn from historical data, and historical data
> encodes historical discrimination. The model does not just reproduce that bias —
> it can amplify it at scale, across millions of decisions, faster than any human
> reviewer could catch.
>
> Fairness-aware ML was built to address this. There are pre-processing methods
> that adjust the training data, in-processing methods that constrain the
> training objective, and post-processing methods that adjust the output thresholds.
> These methods exist. They are published. They show fairness improvements.
>
> But how reliable are those improvements when you look past a single reported number?
> That is what we tested."

---

## BLOCK 2 — NAMAN (Slides 3–4)

---

### SLIDE 3 — Our Approach: Stress-Testing Fairness Mitigation

> "Here is the full scope of what we built.
>
> Three datasets: Adult Income, German Credit, and COMPAS — the standard
> fairness benchmarks in the literature.
>
> Three models: Logistic Regression as our interpretable baseline,
> Random Forest, and XGBoost.
>
> Four configurations per model: baseline with no debiasing, plus three
> debiasing methods — Reweighing, Exponentiated Gradient, and ThresholdOptimizer.
>
> And five random seeds for every single combination.
>
> That gives us 3 datasets times 3 models times 4 methods times 5 seeds —
> 180 experiment configurations in total, all run from a single reproducible pipeline.
>
> Every sensitive attribute is held out of the feature matrix and passed
> as a separate argument. Splits are stratified on the label. Seeds are set
> across numpy, Python's random module, and PyTorch. Nothing is left to chance
> except the seed itself."

---

### SLIDE 4 — Data Snapshot: What We're Working With

> "Let me give you the concrete numbers on each dataset.
>
> Adult Income: 48,842 records. Only 24.1% earn more than 50,000 dollars a year —
> so this is a class-imbalanced task. Sensitive attributes are sex and race.
> This is our primary dataset and the one we analyze most deeply.
>
> German Credit: 1,000 records. 70% are labeled good credit.
> Sensitive attributes are sex, derived from personal status codes in the original
> data, and age binarized at 25. The small sample size means we expect wide
> confidence intervals on all group-level metrics — and that is actually part
> of the story. Wide error bars on German show exactly why reporting a single seed
> is dangerous on small datasets.
>
> COMPAS: 7,214 defendants after applying the standard ProPublica filters.
> 45.1% are labeled non-recidivist. Sensitive attribute is race.
> One important caveat that we carry throughout: Bao et al. 2021 showed that
> the recidivism labels themselves encode systemic bias. We report COMPAS results
> but we do not treat them as ground truth.
>
> On Adult, our intersectional cells — race crossed with sex — range from
> n equals 215 for non-white females up to n equals 27,816 for white males.
> We flag any subgroup below 100 and drop below 30."

---

## BLOCK 3 — MANAN (Slides 5–6)

---

### SLIDE 5 — Finding 1: Baseline Bias Is Real and Dataset-Dependent

> "Finding one: before we apply any debiasing, how biased are the models?
>
> On Adult Income, all three models land around 85% accuracy,
> and all three show a demographic parity difference of approximately 0.18 —
> meaning men are selected at a rate about 18 percentage points higher than women.
> That is consistent with published benchmarks, which confirms our pipeline is correct.
>
> On COMPAS, accuracy drops to around 65%, and the DP difference is 0.18 on average,
> with Logistic Regression reaching 0.22 — Caucasian defendants are flagged as
> low-risk at a substantially higher rate than African-American defendants.
>
> On German Credit, accuracy is around 76%, and the DP difference is only 0.05 —
> much smaller. But German has only 1,000 records, so those error bars are wide.
>
> The key takeaway: the level of bias is dataset-dependent, not model-dependent.
> Within each dataset, Logistic Regression, Random Forest, and XGBoost all show
> comparable DP differences. Upgrading from LR to XGBoost gains you accuracy
> but does not make your system meaningfully fairer. The bias lives in the data,
> not in the model family."

*[Charts on slide show Baseline Accuracy by Dataset and Baseline DP Difference by Dataset]*

---

### SLIDE 6 — Finding 2: Debiasing Effect Depends on the Model

> "Finding two: when we apply the debiasing methods, they work — but not uniformly
> across models.
>
> On Logistic Regression with Adult Income, here are the DP differences:
> Baseline sits at 0.173.
> Reweighing brings it down to 0.095.
> Exponentiated Gradient brings it down to 0.019.
> ThresholdOptimizer brings it nearly to zero — 0.003.
> And across all three methods, the accuracy cost is less than 2 percentage points.
>
> That sounds like a clean success story. But look at the bar chart on the right —
> Random Forest tells a completely different story.
> Reweighing on RF barely moves the DP difference at all.
> ThresholdOptimizer on RF achieves a large reduction but at a much steeper
> accuracy cost than on LR.
>
> The implication: a debiasing method validated on Logistic Regression
> cannot be assumed to transfer to Random Forest. Methods are not portable
> across model families. The fairness improvement you publish depends on
> which model you happened to choose as your base learner.
> I'll hand it to Khushi for the stress-test findings."

*[Charts show Adult Accuracy by Debiasing Method and Adult DP Difference by Debiasing Method, with LR, RF, XGBoost as colored bars]*

---

## BLOCK 4 — KHUSHI (Slides 7–9)

---

### SLIDE 7 — Finding 3: Seed Stability — The Honest View

> "Finding three is about what happens when we run each configuration
> across five random seeds instead of just one.
>
> What we expected going in: wide seed-to-seed variance large enough to erase
> reported fairness gains — the intervention works on seed zero but not on seed three.
>
> What we actually found: the means are stable. Across seeds zero through four,
> accuracy and DP difference are consistent for the baseline and for in-processing methods.
>
> But here is the result that changes the story: ThresholdOptimizer on Random Forest
> swings from a DP difference of 0.00 at seed zero to 0.14 at seed one —
> a jump from near-perfect fairness to nearly the same level as the unmitigated baseline,
> from one random seed to the next.
>
> Post-processing is the seed-fragile method. The threshold calibration is
> sensitive to the specific random draw that determined the training split.
> A paper reporting ThresholdOptimizer results on a single seed could show
> either of those outcomes depending on luck.
> Baselines and in-processing stay flat — post-processing is the one to watch."

*[Charts show Seed Sweep Accuracy and Seed Sweep DP Difference, with lines per method across seeds 0–4]*

---

### SLIDE 8 — Finding 4: Threshold Rivals Method Choice

> "Finding four is the one that surprised us most.
>
> From a single trained model — same weights, same parameters, nothing changed —
> the demographic parity difference ranges from 0.04 to 0.28 just by moving
> the decision threshold. That is a 7 times swing on a single dial.
>
> To put that in context: the difference between the best debiasing method
> and no debiasing at the default threshold of 0.5 is roughly 0.17.
> The difference caused by shifting the threshold is larger than the debiasing
> method itself.
>
> And here is the headline result: Reweighing at threshold 0.7 achieves a DP difference
> of 0.04 — lower than the unmitigated baseline at threshold 0.5.
> You can reach a better fairness number by adjusting the threshold on a
> non-debiased model than by applying a certified debiasing method at the default threshold.
>
> Fairness is a dial, not a binary. Any paper that reports a single fairness number
> at a single threshold is reporting one point on a curve that spans a 7x range."

*[Charts show Threshold Sweep Accuracy and Threshold Sweep DP Difference, lines for none and reweighing across thresholds 0.3–0.7]*

---

### SLIDE 9 — Finding 5: Aggregate Metrics Can Hide Subgroup Harm

> "Finding five is about what happens when we look beyond the aggregate metric
> and break the data into intersectional subgroups — race crossed with sex.
>
> The four cells on Adult Income with Reweighing applied:
> White males are selected at 17.2%.
> White females at 15.0%.
> Non-white females at 12.5%.
> Non-white males at 11.9%.
>
> White men are selected at 17.2% versus 11.9% for non-white men.
> That is a 45% relative gap between those two subgroups.
>
> Now here is the critical point: the aggregate demographic parity metric
> computes a single number by collapsing across all men versus all women.
> That number passes. The aggregate metric looks acceptable.
> But inside that aggregate, the most disadvantaged subgroup — non-white men —
> is still being selected at a rate 45% lower than the most privileged subgroup.
>
> Aggregate fairness metrics can silently miss this. A system that satisfies
> the demographic parity threshold at the aggregate level can still be
> systematically harming a specific intersectional group.
> Intersectional breakdown is not optional — it is the check that the aggregate metric
> cannot do."

*[Slide shows 2x2 heatmap of intersectional selection rates and the four percentage boxes]*

---

## BLOCK 5 — ANUSHKA (Slides 10–13)

---

### SLIDE 10 — The Accuracy–Fairness Tradeoff Space

> "This is the Pareto front — every dot is one experiment configuration:
> one model, one debiasing method, one seed. The x-axis is DP difference,
> the y-axis is accuracy.
>
> Three regions emerge clearly.
>
> The fair region — DP difference below 0.05. ThresholdOptimizer and
> ExponentiatedGradient cluster here, with accuracy between 83 and 86%.
> These methods achieve the strongest fairness but pay a cost in accuracy.
>
> The middle region — DP difference between 0.05 and 0.15. Reweighing lives here.
> Modest fairness gains at modest accuracy cost. A reasonable operating point
> if you need to preserve accuracy.
>
> The biased region — DP difference above 0.15. All the baselines fall here.
> XGBoost reaches 87% accuracy — the highest of any configuration —
> but with a DP difference around 0.18.
>
> The key message: there is no single dominant point. No method simultaneously
> maximizes accuracy and minimizes DP difference. Which point is right
> depends on how much accuracy the application can afford to trade for fairness —
> and that is a decision that belongs to the people deploying the system,
> not the model."

*[Pareto front scatter plot: accuracy vs DP difference, dots colored by model, shaped by debiasing method]*

---

### SLIDE 11 — Key Findings

> "Pulling it together: three findings that hold across datasets and seeds.
>
> First, debiasing is model-dependent. ThresholdOptimizer drops DP difference
> to near zero on LR and XGBoost, but stalls at 0.06 on Random Forest.
> A method certified on one base learner does not transfer.
>
> Second, threshold rivals method choice. One model, one set of trained weights —
> DP difference ranges from 0.04 to 0.28 depending on where you set the threshold.
> A 7 times swing on a single operational parameter that practitioners change
> routinely for business reasons, often without realizing it affects their
> certified fairness properties.
>
> Third, intersectional gaps persist even after debiasing. After Reweighing,
> white men are still selected at 17.2% versus 11.9% for non-white men.
> The aggregate metric improved. The subgroup gap remained.
>
> And the COMPAS note: we report those results but do not treat them as ground truth.
> Per Bao et al. 2021, the recidivism labels themselves encode systemic bias.
> Data quality is the first fairness decision — before any method is chosen.
> Fairness is a system-level property, not a model property."

---

### SLIDE 12 — What We Didn't Do, and What to Read Carefully

> "We want to be explicit about the limits of this work — this is part of
> the methodology, not an afterthought.
>
> On the data side:
> COMPAS labels are contested, as we noted, and we carry that caveat throughout.
> German Credit is statistically thin — 1,000 records yield wide confidence intervals
> on group-level metrics. The patterns there are suggestive, not conclusive.
> And we binarized sensitive attributes: race as white versus non-white,
> sex as male or female. Real-world identity categories are richer than this,
> and our intersectional resolution is limited by these binarizations.
>
> On the methodology side:
> FT-Transformer is implemented in the codebase but was not benchmarked at scale —
> the environment was CPU-only during the sprint window.
> Results are for LR, RF, and XGBoost only.
> We ran 5 seeds per configuration — enough to see the pattern, but tighter
> confidence intervals on the fragility claims would require 20 to 30.
> And our main slides lead with demographic parity. Equal opportunity and
> equalized odds are computed and in the results files, but under-discussed
> in this presentation.
>
> Stating these limits explicitly is part of rigorous science —
> we would rather you know where the boundaries are."

---

### SLIDE 13 — Contributions and Next Steps

> "Here is what we shipped this term.
>
> A fully reproducible pipeline: 3 datasets, 3 models, 3 debiasing methods,
> 180 configurations, all from a single CLI entry point.
> A 5-seed stress-test framework that runs seed sweep, threshold sweep,
> and intersectional breakdown automatically.
> The threshold sensitivity finding: a 7 times DP-difference range from a single model.
> An intersectional subgroup analysis on Adult Income revealing a 45% relative
> selection rate gap that the aggregate metric does not surface.
> And the Pareto front showing the full accuracy-fairness tradeoff space
> across all configurations.
>
> For future work:
> Benchmark FT-Transformer on Adult — needs a proper GPU environment.
> Distribution-shift experiments using Folktables — the same income prediction task
> evaluated across different US states and census years, which tests whether
> a fairness intervention that works in 2018 data still works in 2022 data.
> Formalize a scalar fragility index that combines seed variance, threshold sensitivity,
> and intersectional dispersion into a single auditable number.
> And submit this as a paper — target venues are the NeurIPS Algorithmic Fairness
> Workshop and FAccT 2026.
>
> Thank you — we're happy to take questions."

---

## Q&A Prep — Likely Questions

**Q: Why not AIF360?**
> "Fairlearn covers what we need and integrates cleanly with scikit-learn.
> AIF360 would add complexity without strengthening the specific claims we're making."

**Q: FT-Transformer is missing from the results — is that a gap?**
> "Yes, and we flag it explicitly on slide 12. The pipeline supports it and the
> wrapper is written — we ran out of GPU time. The three-model results are still
> sufficient to demonstrate the core fragility findings."

**Q: Is 5 seeds enough to make fragility claims?**
> "Enough to see the pattern clearly — ThresholdOptimizer on RF swings from 0.00 to 0.14.
> For tight confidence intervals on the fragility index we would want 20-30 seeds.
> We flag this as a limitation."

**Q: The COMPAS results — how do you interpret them given label bias?**
> "We follow Bao et al. 2021: report the numbers, cite the controversy,
> and do not present them as fairness ground truth. Slide 11 and slide 12
> both carry this caveat explicitly."

**Q: Why binary sensitive attributes?**
> "DP difference and EO difference require a two-group comparison.
> We acknowledge the binarization as a limitation on slide 12.
> Non-binary breakdown is on the future work list."

---

## Speaker Assignments Summary

| Block | Speaker | Slides | Content |
|---|---|---|---|
| 1 | Devansh | 1–2 | Title + motivation |
| 2 | Naman | 3–4 | Approach overview + data snapshot |
| 3 | Manan | 5–6 | Finding 1 (baseline) + Finding 2 (debiasing model-dependent) |
| 4 | Khushi | 7–9 | Finding 3 (seed) + Finding 4 (threshold) + Finding 5 (intersectional) |
| 5 | Anushka | 10–13 | Pareto front + key findings + limitations + next steps |

---

## Timing Guide

| Block | Speaker | Slides | Target |
|---|---|---|---|
| 1 | Devansh | 1–2 | 2.0 min |
| 2 | Naman | 3–4 | 2.5 min |
| 3 | Manan | 5–6 | 3.0 min |
| 4 | Khushi | 7–9 | 4.0 min |
| 5 | Anushka | 10–13 | 3.5 min |
| **Total** | | **13** | **~15 min** |
