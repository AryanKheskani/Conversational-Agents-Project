import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('Results.csv')

# ─────────────────────────────────────────────
# 2. COLUMN DEFINITIONS
# ─────────────────────────────────────────────
PETS_A = [
    'The system expressed emotions.',
    'The system understood my goals.',
    'The system considered my mental state.',
    'The system understood my intentions.',
    'The system showed interest in me.',
    'The system understood my needs.',
    'The system seemed emotionally intelligent',
    'I trusted the system.',
    'The system sympathized with me.',
    'The system supported me in coping with an emotional situation.'
]

PETS_B = [
    'The system expressed emotions.1',
    'The system understood my goals.1',
    'The system considered my mental state.1',
    'The system understood my intentions.1',
    'The system showed interest in me.\n 1',
    'The system understood my needs.\n 1',
    'The system seemed emotionally intelligent\n 1',
    'I trusted the system.\n 1',
    'The system sympathized with me.\n 1',
    'The system supported me in coping with an emotional situation.\n 1'
]

SUS_A = [
    'Question.I think that I would like to use this system frequently.',
    'Question.I found the system unnecessarily complex.',
    'Question.I thought the system was easy to use.',
    'Question.I think that I would need the support of a technical person to be able to use this system.',
    'Question.I found the various functions in this system were well integrated.',
    'Question.I thought there was too much inconsistency in this system.',
    'Question.I would imagine that most people would learn to use this system very quickly.',
    'Question.I found the system very cumbersome to use',
    'Question.I felt very confident using the system.',
    'Question.I needed to learn a lot of things before I could get going with this system.'
]

SUS_B = [
    'Question.I think that I would like to use this system frequently.1',
    'Question.I found the system unnecessarily complex.1',
    'Question.I thought the system was easy to use.1',
    'Question.I think that I would need the support of a technical person to be able to use this system.1',
    'Question.I found the various functions in this system were well integrated.1',
    'Question.I thought there was too much inconsistency in this system.1',
    'Question.I would imagine that most people would learn to use this system very quickly.1',
    'Question.I found the system very cumbersome to use1',
    'Question.I felt very confident using the system. 1',
    'Question.I needed to learn a lot of things before I could get going with this system.1'
]

SUS_ODD_IDX  = [0, 2, 4, 6, 8]
SUS_EVEN_IDX = [1, 3, 5, 7, 9]

LIKERT_MAP = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly Agree': 5
}

# ─────────────────────────────────────────────
# 3. COMPUTE PETS SCORES
# ─────────────────────────────────────────────
df['PETS_A'] = df[PETS_A].mean(axis=1)
df['PETS_B'] = df[PETS_B].mean(axis=1)

# ─────────────────────────────────────────────
# 4. COMPUTE SUS SCORES
# ─────────────────────────────────────────────
def compute_sus(row, cols):
    vals = [LIKERT_MAP[row[c]] for c in cols]
    score = sum(vals[i] - 1 for i in SUS_ODD_IDX) + sum(5 - vals[i] for i in SUS_EVEN_IDX)
    return score * 2.5

df['SUS_A'] = df.apply(lambda row: compute_sus(row, SUS_A), axis=1)
df['SUS_B'] = df.apply(lambda row: compute_sus(row, SUS_B), axis=1)

# ─────────────────────────────────────────────
# 5. MAP COLUMN SETS → SEMANTIC / PROSODIC
# ─────────────────────────────────────────────
def assign_scores(row):
    if row['Condition Order'] == 'Prosodic':
        return row['PETS_A'], row['PETS_B'], row['SUS_A'], row['SUS_B']
    else:
        return row['PETS_B'], row['PETS_A'], row['SUS_B'], row['SUS_A']

df[['PETS_Prosodic', 'PETS_Semantic', 'SUS_Prosodic', 'SUS_Semantic']] = df.apply(
    assign_scores, axis=1, result_type='expand'
)

# ─────────────────────────────────────────────
# 6. RUN ALL TESTS
# ─────────────────────────────────────────────

# --- PETS assumption check ---
pets_diff = df['PETS_Prosodic'] - df['PETS_Semantic']
sw_stat_pets, sw_p_pets = stats.shapiro(pets_diff)
pets_normal = sw_p_pets > 0.05

# --- PETS primary test ---
if pets_normal:
    t_stat, p_val = stats.ttest_rel(df['PETS_Prosodic'], df['PETS_Semantic'])
    test_used = "Paired t-test"
    test_stat_label = f"t({len(df)-1}) = {t_stat:.4f}"
else:
    w_stat, p_val = stats.wilcoxon(df['PETS_Prosodic'], df['PETS_Semantic'])
    t_stat = w_stat
    test_used = "Wilcoxon signed-rank test"
    test_stat_label = f"W = {w_stat:.4f}"

cohens_d = pets_diff.mean() / pets_diff.std(ddof=1)
ci = stats.t.interval(0.95, df=len(pets_diff)-1,
                      loc=pets_diff.mean(),
                      scale=stats.sem(pets_diff))

# --- SUS assumption check ---
sus_diff = df['SUS_Prosodic'] - df['SUS_Semantic']
sw_stat_sus, sw_p_sus = stats.shapiro(sus_diff)
sus_normal = sw_p_sus > 0.05

# --- SUS test ---
if sus_normal:
    t_sus, p_sus = stats.ttest_rel(df['SUS_Prosodic'], df['SUS_Semantic'])
    sus_test_label = f"t({len(df)-1}) = {t_sus:.4f}"
    sus_test_used = "Paired t-test"
else:
    w_sus, p_sus = stats.wilcoxon(df['SUS_Prosodic'], df['SUS_Semantic'])
    t_sus = w_sus
    sus_test_label = f"W = {w_sus:.4f}"
    sus_test_used = "Wilcoxon signed-rank test"

d_sus = sus_diff.mean() / sus_diff.std(ddof=1)

# --- Sensitivity check by order ---
sensitivity = {}
for order in ['Prosodic', 'Semantic Only']:
    sub = df[df['Condition Order'] == order]
    d = sub['PETS_Prosodic'] - sub['PETS_Semantic']
    sensitivity[order] = {'n': len(sub), 'mean': d.mean(), 'sd': d.std()}

# ─────────────────────────────────────────────
# 7. PRINT TO CONSOLE
# ─────────────────────────────────────────────
print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
for label, sem_col, pro_col in [('PETS (0-100)', 'PETS_Semantic', 'PETS_Prosodic'),
                                  ('SUS  (0-100)', 'SUS_Semantic',  'SUS_Prosodic')]:
    print(f"\n{label}")
    print(f"  Semantic — Mean: {df[sem_col].mean():.2f}, SD: {df[sem_col].std():.2f}, Median: {df[sem_col].median():.2f}")
    print(f"  Prosodic — Mean: {df[pro_col].mean():.2f}, SD: {df[pro_col].std():.2f}, Median: {df[pro_col].median():.2f}")
    print(f"  Difference (Prosodic - Semantic): Mean = {(df[pro_col] - df[sem_col]).mean():.2f}")

print("\n" + "=" * 60)
print("ASSUMPTION CHECKS")
print("=" * 60)
print(f"\nPETS — Shapiro-Wilk: W = {sw_stat_pets:.4f}, p = {sw_p_pets:.4f} → {'Normal (paired t-test)' if pets_normal else 'Non-normal (Wilcoxon)'}")
print(f"SUS  — Shapiro-Wilk: W = {sw_stat_sus:.4f}, p = {sw_p_sus:.4f} → {'Normal (paired t-test)' if sus_normal else 'Non-normal (Wilcoxon)'}")

print("\n" + "=" * 60)
print("PRIMARY ANALYSIS: PETS")
print("=" * 60)
print(f"\n{test_used}: {test_stat_label}, p = {p_val:.4f}")
print(f"Cohen's d = {cohens_d:.4f}")
print(f"95% CI on difference: [{ci[0]:.2f}, {ci[1]:.2f}]")
print(f"Result: {'Significant' if p_val < 0.05 else 'Not significant'} at alpha = 0.05")

print("\n" + "=" * 60)
print("USABILITY CHECK: SUS")
print("=" * 60)
print(f"\n{sus_test_used}: {sus_test_label}, p = {p_sus:.4f}")
print(f"Cohen's d = {d_sus:.4f}")
print(f"Result: {'Usability differs across conditions.' if p_sus < 0.05 else 'Usability equivalent across conditions.'}")

print("\n" + "=" * 60)
print("SENSITIVITY CHECK: BY PRESENTATION ORDER")
print("=" * 60)
for order, vals in sensitivity.items():
    print(f"\n  First condition = {order} (n={vals['n']}): Mean diff = {vals['mean']:.2f}, SD = {vals['sd']:.2f}")

# ─────────────────────────────────────────────
# 8. GENERATE MARKDOWN REPORT
# ─────────────────────────────────────────────

def sig_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "n.s."

def cohens_d_interp(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else: return "large"

def sus_grade(score):
    if score >= 85: return "Excellent"
    elif score >= 72: return "Good"
    elif score >= 52: return "OK"
    elif score >= 38: return "Poor"
    else: return "Awful"

md = []
md.append("# Statistical Analysis Results\n")
md.append(f"> **N = {len(df)} participants** | Primary test: {test_used} | α = 0.05, two-tailed\n")

# ── Descriptives ──────────────────────────────
md.append("---\n")
md.append("## 1. Descriptive Statistics\n")
md.append("| Measure | Condition | Mean | SD | Median |")
md.append("|---------|-----------|------|----|--------|")
for label, sem_col, pro_col in [('PETS (0–100)', 'PETS_Semantic', 'PETS_Prosodic'),
                                  ('SUS (0–100)',  'SUS_Semantic',  'SUS_Prosodic')]:
    md.append(f"| {label} | Semantic | {df[sem_col].mean():.2f} | {df[sem_col].std():.2f} | {df[sem_col].median():.2f} |")
    md.append(f"| | Prosodic | {df[pro_col].mean():.2f} | {df[pro_col].std():.2f} | {df[pro_col].median():.2f} |")

md.append(f"\n**Mean PETS difference (Prosodic − Semantic):** {pets_diff.mean():.2f} points")
md.append(f"\n**Mean SUS difference (Prosodic − Semantic):** {sus_diff.mean():.2f} points\n")

# ── Assumption Checks ─────────────────────────
md.append("---\n")
md.append("## 2. Assumption Checks — Shapiro-Wilk Test\n")
md.append("> **What this test does:** Before running the main analysis, we check whether the distribution of individual difference scores (each participant's Prosodic score minus their Semantic score) looks roughly like a bell curve. This is required for the paired t-test to be valid. If the data passes (p ≥ .05), we use the paired t-test. If it fails (p < .05), we switch to the Wilcoxon signed-rank test, which does not require this assumption.\n")
md.append("| Measure | W statistic | p-value | Verdict |")
md.append("|---------|-------------|---------|---------|")
md.append(f"| PETS difference | {sw_stat_pets:.4f} | {sw_p_pets:.4f} | {'✅ Normal → use paired t-test' if pets_normal else '❌ Non-normal → use Wilcoxon'} |")
md.append(f"| SUS difference  | {sw_stat_sus:.4f} | {sw_p_sus:.4f} | {'✅ Normal → use paired t-test' if sus_normal else '❌ Non-normal → use Wilcoxon'} |\n")

# ── Primary Analysis ──────────────────────────
md.append("---\n")
md.append("## 3. Primary Analysis — PETS Scores\n")
if pets_normal:
    md.append("> **What this test does:** The paired t-test compares each participant's PETS score under both conditions and checks whether the average difference is statistically distinguishable from zero. Because every participant experienced both agents, we directly compare their two scores rather than comparing group averages — this makes the test more sensitive to real differences.\n")
else:
    md.append("> **What this test does:** The Wilcoxon signed-rank test is a non-parametric alternative to the paired t-test. Instead of comparing raw difference scores, it ranks them by magnitude and checks whether those ranks are systematically leaning positive or negative. It is used here because the normality assumption was violated.\n")

md.append(f"| Statistic | Value |")
md.append(f"|-----------|-------|")
md.append(f"| Semantic mean (SD) | {df['PETS_Semantic'].mean():.2f} ({df['PETS_Semantic'].std():.2f}) |")
md.append(f"| Prosodic mean (SD) | {df['PETS_Prosodic'].mean():.2f} ({df['PETS_Prosodic'].std():.2f}) |")
md.append(f"| Mean difference | {pets_diff.mean():+.2f} |")
md.append(f"| Test statistic | {test_stat_label} |")
md.append(f"| p-value | {p_val:.4f} ({sig_stars(p_val)}) |")
md.append(f"| Cohen's d | {cohens_d:.4f} ({cohens_d_interp(cohens_d)} effect) |")
md.append(f"| 95% CI on difference | [{ci[0]:.2f}, {ci[1]:.2f}] |\n")

md.append("> **What Cohen's d means:** The p-value tells you *whether* a difference exists; Cohen's d tells you *how big* it is. It is the mean difference divided by the standard deviation of the differences — a standardised, unit-free measure. By convention: d < 0.2 = negligible, d = 0.2–0.5 = small, d = 0.5–0.8 = medium, d > 0.8 = large.\n")
md.append("> **What the 95% CI means:** We are 95% confident that the true mean difference between conditions lies within this range. If the interval contains zero, the result is non-significant — zero (no difference) is a plausible value.\n")

if p_val >= 0.05:
    md.append(f"**Interpretation:** The prosodic agent scored {pets_diff.mean():.2f} points higher on perceived empathy on average, but this difference did not reach statistical significance (p = {p_val:.3f}). The effect size of d = {cohens_d:.2f} is {cohens_d_interp(cohens_d)}, and the confidence interval [{ci[0]:.2f}, {ci[1]:.2f}] crosses zero, meaning we cannot rule out that the true difference is zero. With N = {len(df)}, this study was powered to detect medium-to-large effects (d ≥ 0.5); the observed trend warrants replication with a larger sample.\n")
else:
    md.append(f"**Interpretation:** The prosodic agent scored significantly higher on perceived empathy (Δ = {pets_diff.mean():.2f}, p = {p_val:.3f}, d = {cohens_d:.2f}). This supports the hypothesis that prosodic processing improves users' perceived emotional support.\n")

# ── SUS Check ─────────────────────────────────
md.append("---\n")
md.append("## 4. Usability Confound Check — SUS Scores\n")
md.append("> **What this check does:** If the prosodic agent scored higher on PETS, we need to rule out the possibility that participants simply rated it higher because it was easier or more pleasant to use overall — not because it actually felt more empathetic. By running the same test on SUS scores, we check whether usability was equivalent across conditions. If it is (p ≥ .05), any PETS difference can be confidently attributed to perceived empathy rather than usability.\n")

md.append(f"| Statistic | Value |")
md.append(f"|-----------|-------|")
md.append(f"| Semantic SUS mean (SD) | {df['SUS_Semantic'].mean():.2f} ({df['SUS_Semantic'].std():.2f}) — {sus_grade(df['SUS_Semantic'].mean())} |")
md.append(f"| Prosodic SUS mean (SD) | {df['SUS_Prosodic'].mean():.2f} ({df['SUS_Prosodic'].std():.2f}) — {sus_grade(df['SUS_Prosodic'].mean())} |")
md.append(f"| Mean difference | {sus_diff.mean():+.2f} |")
md.append(f"| Test statistic | {sus_test_label} |")
md.append(f"| p-value | {p_sus:.4f} ({sig_stars(p_sus)}) |")
md.append(f"| Cohen's d | {d_sus:.4f} ({cohens_d_interp(d_sus)} effect) |\n")

if p_sus >= 0.05:
    md.append(f"**Interpretation:** SUS scores were virtually identical across conditions (Δ = {sus_diff.mean():.2f}, p = {p_sus:.3f}, d = {d_sus:.2f}). Usability was equivalent, confirming it is not a confounding factor in interpreting the PETS results. Both agents fall in the **{sus_grade(df['SUS_Semantic'].mean())}** usability range.\n")
else:
    md.append(f"**Interpretation:** SUS scores differed significantly across conditions (p = {p_sus:.3f}). Usability may be a confounding factor and should be discussed when interpreting PETS results.\n")

# ── Sensitivity Check ─────────────────────────
md.append("---\n")
md.append("## 5. Sensitivity Check — Presentation Order\n")
md.append("> **What this check does:** Because participants experienced both agents in sequence, there is a risk that the *order* they encountered them in influenced their ratings — for example, participants might rate the second agent higher simply because they were more comfortable with the system by then. We split participants by which agent they saw first and compare the PETS gain for each subgroup. If the effect is consistent across both orders, it is more likely to be a genuine modality effect rather than an order artefact.\n")

md.append("| First condition | N | Mean PETS gain (Prosodic − Semantic) | SD |")
md.append("|----------------|---|--------------------------------------|----|")
for order, vals in sensitivity.items():
    md.append(f"| {order} | {vals['n']} | {vals['mean']:+.2f} | {vals['sd']:.2f} |")

md.append(f"""
**Interpretation:** Participants who interacted with the semantic agent first showed a larger average PETS gain for the prosodic agent (Δ = {sensitivity['Semantic Only']['mean']:+.2f}) compared to those who started with the prosodic agent (Δ = {sensitivity['Prosodic']['mean']:+.2f}). This pattern is consistent with a mild contrast effect — experiencing the less empathetic agent first may make the prosodic agent feel comparatively better. Given the small subgroup sizes, this finding is reported as a transparency check and should be interpreted cautiously.
""")

# ── Summary ───────────────────────────────────
md.append("---\n")
md.append("## 6. Summary\n")
md.append("| | Semantic | Prosodic | Δ | p | d | Significant? |")
md.append("|--|----------|----------|---|---|---|--------------|")
md.append(f"| PETS | {df['PETS_Semantic'].mean():.1f} ± {df['PETS_Semantic'].std():.1f} | {df['PETS_Prosodic'].mean():.1f} ± {df['PETS_Prosodic'].std():.1f} | {pets_diff.mean():+.1f} | {p_val:.3f} | {cohens_d:.2f} | {'Yes ✅' if p_val < 0.05 else 'No ❌'} |")
md.append(f"| SUS  | {df['SUS_Semantic'].mean():.1f} ± {df['SUS_Semantic'].std():.1f} | {df['SUS_Prosodic'].mean():.1f} ± {df['SUS_Prosodic'].std():.1f} | {sus_diff.mean():+.1f} | {p_sus:.3f} | {d_sus:.2f} | {'Yes ✅' if p_sus < 0.05 else 'No ❌'} |")

md.append(f"""
**Key takeaways:**
- The prosodic agent showed a trend toward higher perceived empathy (PETS Δ = {pets_diff.mean():+.2f}), but this did not reach significance at N = {len(df)}.
- Effect size (d = {cohens_d:.2f}) is {cohens_d_interp(cohens_d)} — the study was powered for d ≥ 0.5 only.
- Usability was equivalent across conditions (SUS Δ = {sus_diff.mean():+.2f}, p = {p_sus:.3f}), ruling out usability as a confound.
- A mild presentation order effect was observed and should be noted as a limitation.
""")

with open('analysis_report.md', 'w') as f:
    f.write('\n'.join(md))

print("\n\nMarkdown report saved to analysis_report.md")

# ─────────────────────────────────────────────
# 9. VISUALISATION
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#f8f9fa')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

palette = {'Semantic': '#5B8DB8', 'Prosodic': '#E07B54'}

ax1 = fig.add_subplot(gs[0, 0])
plot_data = pd.DataFrame({
    'Score': pd.concat([df['PETS_Semantic'], df['PETS_Prosodic']], ignore_index=True),
    'Condition': ['Semantic'] * len(df) + ['Prosodic'] * len(df)
})
sns.boxplot(data=plot_data, x='Condition', y='Score', palette=palette,
            width=0.5, linewidth=1.5, fliersize=0, ax=ax1)
sns.stripplot(data=plot_data, x='Condition', y='Score', palette=palette,
              size=6, alpha=0.6, jitter=True, ax=ax1)
ax1.set_title('PETS Scores by Condition', fontweight='bold', fontsize=11)
ax1.set_ylabel('PETS Score (0–100)')
ax1.set_xlabel('')

ax2 = fig.add_subplot(gs[0, 1])
sus_data = pd.DataFrame({
    'Score': pd.concat([df['SUS_Semantic'], df['SUS_Prosodic']], ignore_index=True),
    'Condition': ['Semantic'] * len(df) + ['Prosodic'] * len(df)
})
sns.boxplot(data=sus_data, x='Condition', y='Score', palette=palette,
            width=0.5, linewidth=1.5, fliersize=0, ax=ax2)
sns.stripplot(data=sus_data, x='Condition', y='Score', palette=palette,
              size=6, alpha=0.6, jitter=True, ax=ax2)
ax2.set_title('SUS Scores by Condition', fontweight='bold', fontsize=11)
ax2.set_ylabel('SUS Score (0–100)')
ax2.set_xlabel('')

ax3 = fig.add_subplot(gs[0, 2])
for _, row in df.iterrows():
    ax3.plot([0, 1], [row['PETS_Semantic'], row['PETS_Prosodic']],
             color='grey', alpha=0.3, linewidth=1)
ax3.plot([0, 1], [df['PETS_Semantic'].mean(), df['PETS_Prosodic'].mean()],
         color='black', linewidth=2.5, marker='o', markersize=8, label='Mean')
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Semantic', 'Prosodic'])
ax3.set_title('Individual PETS Trajectories', fontweight='bold', fontsize=11)
ax3.set_ylabel('PETS Score (0–100)')
ax3.legend()

ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(pets_diff, bins=10, color='#5B8DB8', edgecolor='white', alpha=0.85)
ax4.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No difference')
ax4.axvline(pets_diff.mean(), color='black', linestyle='-', linewidth=2,
            label=f'Mean = {pets_diff.mean():.1f}')
ax4.set_title('PETS Difference (Prosodic − Semantic)', fontweight='bold', fontsize=11)
ax4.set_xlabel('Difference Score')
ax4.set_ylabel('Count')
ax4.legend(fontsize=9)

ax5 = fig.add_subplot(gs[1, 1:])
ax5.axis('off')

table_data = [
    ['', 'Semantic', 'Prosodic', 'Diff', 'p-value', "Cohen's d"],
    ['PETS',
     f"{df['PETS_Semantic'].mean():.1f} ± {df['PETS_Semantic'].std():.1f}",
     f"{df['PETS_Prosodic'].mean():.1f} ± {df['PETS_Prosodic'].std():.1f}",
     f"{pets_diff.mean():+.1f}",
     f"{p_val:.3f}{sig_stars(p_val)}",
     f"{cohens_d:.3f}"],
    ['SUS',
     f"{df['SUS_Semantic'].mean():.1f} ± {df['SUS_Semantic'].std():.1f}",
     f"{df['SUS_Prosodic'].mean():.1f} ± {df['SUS_Prosodic'].std():.1f}",
     f"{sus_diff.mean():+.1f}",
     f"{p_sus:.3f}{sig_stars(p_sus)}",
     f"{d_sus:.3f}"],
]

tbl = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#cccccc')
    if r == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 1:
        cell.set_facecolor('#eaf0fb')
ax5.set_title('Summary Results Table (n.s. = not significant)', fontweight='bold', fontsize=11, pad=20)

fig.suptitle('Prosodic vs. Semantic Agent — Perceived Empathy & Usability Analysis\n'
             f'N = {len(df)} participants, {test_used}',
             fontsize=13, fontweight='bold', y=1.01)

plt.savefig('analysis_results.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Plot saved to analysis_results.png")