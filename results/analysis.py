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

# SUS: odd items (1-indexed: 1,3,5,7,9) are positive, even (2,4,6,8,10) are negative
SUS_ODD_IDX  = [0, 2, 4, 6, 8]   # 0-indexed
SUS_EVEN_IDX = [1, 3, 5, 7, 9]

LIKERT_MAP = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly Agree': 5
}

# ─────────────────────────────────────────────
# 3. COMPUTE PETS SCORES (0–100 scale, mean of 10 items)
# ─────────────────────────────────────────────
df['PETS_A'] = df[PETS_A].mean(axis=1)
df['PETS_B'] = df[PETS_B].mean(axis=1)

# ─────────────────────────────────────────────
# 4. COMPUTE SUS SCORES (standard 0–100 formula)
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
# Condition Order = what was done FIRST (column set A)
# Condition Order1 = what was done SECOND (column set B)

def assign_scores(row):
    if row['Condition Order'] == 'Prosodic':
        return row['PETS_A'], row['PETS_B'], row['SUS_A'], row['SUS_B']
    else:  # Semantic Only first
        return row['PETS_B'], row['PETS_A'], row['SUS_B'], row['SUS_A']

df[['PETS_Prosodic', 'PETS_Semantic', 'SUS_Prosodic', 'SUS_Semantic']] = df.apply(
    assign_scores, axis=1, result_type='expand'
)

# ─────────────────────────────────────────────
# 6. PRINT DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────
print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

for label, sem_col, pro_col in [('PETS (0–100)', 'PETS_Semantic', 'PETS_Prosodic'),
                                  ('SUS  (0–100)', 'SUS_Semantic',  'SUS_Prosodic')]:
    print(f"\n{label}")
    print(f"  Semantic  — Mean: {df[sem_col].mean():.2f}, SD: {df[sem_col].std():.2f}, "
          f"Median: {df[sem_col].median():.2f}")
    print(f"  Prosodic  — Mean: {df[pro_col].mean():.2f}, SD: {df[pro_col].std():.2f}, "
          f"Median: {df[pro_col].median():.2f}")
    print(f"  Difference (Prosodic − Semantic): Mean = {(df[pro_col] - df[sem_col]).mean():.2f}")

# ─────────────────────────────────────────────
# 7. ASSUMPTION CHECKS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ASSUMPTION CHECKS")
print("=" * 60)

for label, sem_col, pro_col in [('PETS', 'PETS_Semantic', 'PETS_Prosodic'),
                                  ('SUS',  'SUS_Semantic',  'SUS_Prosodic')]:
    diff = df[pro_col] - df[sem_col]
    stat, p = stats.shapiro(diff)
    print(f"\n{label} difference score — Shapiro-Wilk: W = {stat:.4f}, p = {p:.4f}", end="")
    print(f"  →  {'✓ Normal (use paired t-test)' if p > 0.05 else '✗ Non-normal (use Wilcoxon)'}")

# ─────────────────────────────────────────────
# 8. PRIMARY ANALYSIS — PETS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PRIMARY ANALYSIS: PETS SCORES")
print("=" * 60)

pets_diff = df['PETS_Prosodic'] - df['PETS_Semantic']
sw_stat, sw_p = stats.shapiro(pets_diff)

if sw_p > 0.05:
    t_stat, p_val = stats.ttest_rel(df['PETS_Prosodic'], df['PETS_Semantic'])
    test_used = "Paired t-test"
    print(f"\nPaired t-test: t({len(df)-1}) = {t_stat:.4f}, p = {p_val:.4f}")
else:
    w_stat, p_val = stats.wilcoxon(df['PETS_Prosodic'], df['PETS_Semantic'])
    t_stat = w_stat
    test_used = "Wilcoxon signed-rank"
    print(f"\nWilcoxon signed-rank: W = {w_stat:.4f}, p = {p_val:.4f}")

# Cohen's d
cohens_d = pets_diff.mean() / pets_diff.std(ddof=1)
print(f"Cohen's d = {cohens_d:.4f}")
print(f"Interpretation: p {'< 0.05 → statistically significant difference' if p_val < 0.05 else '≥ 0.05 → no significant difference'}")

# 95% CI on the difference
ci = stats.t.interval(0.95, df=len(pets_diff)-1,
                      loc=pets_diff.mean(),
                      scale=stats.sem(pets_diff))
print(f"95% CI on difference: [{ci[0]:.2f}, {ci[1]:.2f}]")

# ─────────────────────────────────────────────
# 9. USABILITY CHECK — SUS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("USABILITY CHECK: SUS SCORES")
print("=" * 60)

sus_diff = df['SUS_Prosodic'] - df['SUS_Semantic']
sw_s, sw_p_sus = stats.shapiro(sus_diff)

if sw_p_sus > 0.05:
    t_sus, p_sus = stats.ttest_rel(df['SUS_Prosodic'], df['SUS_Semantic'])
    print(f"\nPaired t-test: t({len(df)-1}) = {t_sus:.4f}, p = {p_sus:.4f}")
else:
    w_sus, p_sus = stats.wilcoxon(df['SUS_Prosodic'], df['SUS_Semantic'])
    t_sus = w_sus
    print(f"\nWilcoxon signed-rank: W = {w_sus:.4f}, p = {p_sus:.4f}")

d_sus = sus_diff.mean() / sus_diff.std(ddof=1)
print(f"Cohen's d = {d_sus:.4f}")
print(f"Interpretation: {'SUS scores differ — usability is NOT equivalent across conditions.' if p_sus < 0.05 else 'SUS scores are similar → usability is equivalent across conditions. ✓'}")

# ─────────────────────────────────────────────
# 10. SENSITIVITY CHECK — by Condition Order
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SENSITIVITY CHECK: BY PRESENTATION ORDER")
print("=" * 60)

for order in ['Prosodic', 'Semantic Only']:
    sub = df[df['Condition Order'] == order]
    d = sub['PETS_Prosodic'] - sub['PETS_Semantic']
    print(f"\n  Order first = {order} (n={len(sub)}): "
          f"Mean diff = {d.mean():.2f}, SD = {d.std():.2f}")

# ─────────────────────────────────────────────
# 11. VISUALISATION
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#f8f9fa')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

palette = {'Semantic': '#5B8DB8', 'Prosodic': '#E07B54'}

# Panel A — PETS box + strip
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

# Panel B — SUS box + strip
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

# Panel C — paired lines for PETS
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

# Panel D — difference score histogram
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(pets_diff, bins=10, color='#5B8DB8', edgecolor='white', alpha=0.85)
ax4.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No difference')
ax4.axvline(pets_diff.mean(), color='black', linestyle='-', linewidth=2,
            label=f'Mean = {pets_diff.mean():.1f}')
ax4.set_title('PETS Difference (Prosodic − Semantic)', fontweight='bold', fontsize=11)
ax4.set_xlabel('Difference Score')
ax4.set_ylabel('Count')
ax4.legend(fontsize=9)

# Panel E — summary stats table
ax5 = fig.add_subplot(gs[1, 1:])
ax5.axis('off')

pets_sem_mean = df['PETS_Semantic'].mean()
pets_pro_mean = df['PETS_Prosodic'].mean()
sus_sem_mean  = df['SUS_Semantic'].mean()
sus_pro_mean  = df['SUS_Prosodic'].mean()

table_data = [
    ['', 'Semantic', 'Prosodic', 'Diff', 'p-value', "Cohen's d"],
    ['PETS',
     f"{pets_sem_mean:.1f} ± {df['PETS_Semantic'].std():.1f}",
     f"{pets_pro_mean:.1f} ± {df['PETS_Prosodic'].std():.1f}",
     f"{pets_pro_mean - pets_sem_mean:+.1f}",
     f"{p_val:.3f}{'*' if p_val < 0.05 else ''}",
     f"{cohens_d:.3f}"],
    ['SUS',
     f"{sus_sem_mean:.1f} ± {df['SUS_Semantic'].std():.1f}",
     f"{sus_pro_mean:.1f} ± {df['SUS_Prosodic'].std():.1f}",
     f"{sus_pro_mean - sus_sem_mean:+.1f}",
     f"{p_sus:.3f}{'*' if p_sus < 0.05 else ''}",
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
ax5.set_title('Summary Results Table (* p < .05)', fontweight='bold', fontsize=11, pad=20)

fig.suptitle('Prosodic vs. Semantic Agent — Perceived Empathy & Usability Analysis\n'
             f'N = {len(df)} participants, {test_used}',
             fontsize=13, fontweight='bold', y=1.01)

plt.savefig('analysis_results.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("\n\nPlot saved.")