# Statistical Analysis Results

> **N = 30 participants** | Primary test: Paired t-test | α = 0.05, two-tailed

---

## 1. Descriptive Statistics

| Measure | Condition | Mean | SD | Median |
|---------|-----------|------|----|--------|
| PETS (0–100) | Semantic | 70.83 | 17.45 | 71.75 |
| | Prosodic | 75.45 | 14.81 | 76.25 |
| SUS (0–100) | Semantic | 81.08 | 10.62 | 82.50 |
| | Prosodic | 81.50 | 11.02 | 83.75 |

**Mean PETS difference (Prosodic − Semantic):** 4.62 points

**Mean SUS difference (Prosodic − Semantic):** 0.42 points

---

## 2. Assumption Checks — Shapiro-Wilk Test

> **What this test does:** Before running the main analysis, we check whether the distribution of individual difference scores (each participant's Prosodic score minus their Semantic score) looks roughly like a bell curve. This is required for the paired t-test to be valid. If the data passes (p ≥ .05), we use the paired t-test. If it fails (p < .05), we switch to the Wilcoxon signed-rank test, which does not require this assumption.

| Measure | W statistic | p-value | Verdict |
|---------|-------------|---------|---------|
| PETS difference | 0.9588 | 0.2881 | ✅ Normal → use paired t-test |
| SUS difference  | 0.9665 | 0.4496 | ✅ Normal → use paired t-test |

---

## 3. Primary Analysis — PETS Scores

> **What this test does:** The paired t-test compares each participant's PETS score under both conditions and checks whether the average difference is statistically distinguishable from zero. Because every participant experienced both agents, we directly compare their two scores rather than comparing group averages — this makes the test more sensitive to real differences.

| Statistic | Value |
|-----------|-------|
| Semantic mean (SD) | 70.83 (17.45) |
| Prosodic mean (SD) | 75.45 (14.81) |
| Mean difference | +4.62 |
| Test statistic | t(29) = 1.5386 |
| p-value | 0.1347 (n.s.) |
| Cohen's d | 0.2809 (small effect) |
| 95% CI on difference | [-1.52, 10.77] |

> **What Cohen's d means:** The p-value tells you *whether* a difference exists; Cohen's d tells you *how big* it is. It is the mean difference divided by the standard deviation of the differences — a standardised, unit-free measure. By convention: d < 0.2 = negligible, d = 0.2–0.5 = small, d = 0.5–0.8 = medium, d > 0.8 = large.

> **What the 95% CI means:** We are 95% confident that the true mean difference between conditions lies within this range. If the interval contains zero, the result is non-significant — zero (no difference) is a plausible value.

**Interpretation:** The prosodic agent scored 4.62 points higher on perceived empathy on average, but this difference did not reach statistical significance (p = 0.135). The effect size of d = 0.28 is small, and the confidence interval [-1.52, 10.77] crosses zero, meaning we cannot rule out that the true difference is zero. With N = 30, this study was powered to detect medium-to-large effects (d ≥ 0.5); the observed trend warrants replication with a larger sample.

---

## 4. Usability Confound Check — SUS Scores

> **What this check does:** If the prosodic agent scored higher on PETS, we need to rule out the possibility that participants simply rated it higher because it was easier or more pleasant to use overall — not because it actually felt more empathetic. By running the same test on SUS scores, we check whether usability was equivalent across conditions. If it is (p ≥ .05), any PETS difference can be confidently attributed to perceived empathy rather than usability.

| Statistic | Value |
|-----------|-------|
| Semantic SUS mean (SD) | 81.08 (10.62) — Good |
| Prosodic SUS mean (SD) | 81.50 (11.02) — Good |
| Mean difference | +0.42 |
| Test statistic | t(29) = 0.2714 |
| p-value | 0.7880 (n.s.) |
| Cohen's d | 0.0495 (negligible effect) |

**Interpretation:** SUS scores were virtually identical across conditions (Δ = 0.42, p = 0.788, d = 0.05). Usability was equivalent, confirming it is not a confounding factor in interpreting the PETS results. Both agents fall in the **Good** usability range.

---

## 5. Sensitivity Check — Presentation Order

> **What this check does:** Because participants experienced both agents in sequence, there is a risk that the *order* they encountered them in influenced their ratings — for example, participants might rate the second agent higher simply because they were more comfortable with the system by then. We split participants by which agent they saw first and compare the PETS gain for each subgroup. If the effect is consistent across both orders, it is more likely to be a genuine modality effect rather than an order artefact.

| First condition | N | Mean PETS gain (Prosodic − Semantic) | SD |
|----------------|---|--------------------------------------|----|
| Prosodic | 10 | -0.27 | 20.87 |
| Semantic Only | 20 | +7.07 | 13.72 |

**Interpretation:** Participants who interacted with the semantic agent first showed a larger average PETS gain for the prosodic agent (Δ = +7.07) compared to those who started with the prosodic agent (Δ = -0.27). This pattern is consistent with a mild contrast effect — experiencing the less empathetic agent first may make the prosodic agent feel comparatively better. Given the small subgroup sizes, this finding is reported as a transparency check and should be interpreted cautiously.

---

## 6. Summary

| | Semantic | Prosodic | Δ | p | d | Significant? |
|--|----------|----------|---|---|---|--------------|
| PETS | 70.8 ± 17.5 | 75.5 ± 14.8 | +4.6 | 0.135 | 0.28 | No ❌ |
| SUS  | 81.1 ± 10.6 | 81.5 ± 11.0 | +0.4 | 0.788 | 0.05 | No ❌ |

**Key takeaways:**
- The prosodic agent showed a trend toward higher perceived empathy (PETS Δ = +4.62), but this did not reach significance at N = 30.
- Effect size (d = 0.28) is small — the study was powered for d ≥ 0.5 only.
- Usability was equivalent across conditions (SUS Δ = +0.42, p = 0.788), ruling out usability as a confound.
- A mild presentation order effect was observed and should be noted as a limitation.
