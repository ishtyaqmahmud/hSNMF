import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

# --- 1) Read and prepare the SNMF data ---
# Define the path to your data file
file_path = "/project/banerjee/ishtyaq/CPRIT/project2/SNMF/snmf_rank_resolution_sweep_with_spatial_withoutHybrid.csv"

# Read the CSV into a pandas DataFrame
df_snmf = pd.read_csv(file_path)

# Rename the columns for easier access
df_snmf = df_snmf.rename(columns={
    'Silhouette': 'sil',
    'DBI_NMF': 'db',
    'MarkerFraction': 'frac',
    'Enrich': 'enrich',
    'Moran_I_Comp1': 'moran_i'
})


# --- 2) Compute ALL derived metrics for both plots ---
# Metrics for Plot 1
df_snmf['moran_by_chaos_scaled'] = df_snmf['moran_i'] / (df_snmf['CHAOS'] * 100)
df_snmf['cmc_enrich']            = df_snmf['frac'] * df_snmf['enrich']

# Metrics for Plot 2
df_snmf['moran_by_chaos_scaled'] = df_snmf['moran_i'] / (df_snmf['CHAOS'] * 100)
# df_snmf['moran_by_chaos'] = df_snmf['moran_i'] / df_snmf['CHAOS']
df_snmf['sil_by_db']      = df_snmf['sil'] / df_snmf['db']

def is_pareto(pts):
    """
    Identifies the Pareto optimal points for two metrics that should be MAXIMIZED.
    """
    is_opt = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if np.any((pts[:,0] >= p[0]) & (pts[:,1] >= p[1]) &
                  ((pts[:,0] > p[0]) | (pts[:,1] > p[1]))):
            is_opt[i] = False
    return is_opt

# Calculate two separate Pareto frontiers, one for each plot
df_snmf['pareto1'] = is_pareto(df_snmf[['moran_by_chaos_scaled', 'cmc_enrich']].values)
df_snmf['pareto2'] = is_pareto(df_snmf[['moran_by_chaos_scaled', 'sil_by_db']].values)


# --- 3) Create, save, and show PLOT 1 ---
fig1, ax1 = plt.subplots(1, 1, figsize=(14, 10))

colors1 = np.where(df_snmf['pareto1'], 'red', 'blue')
alphas1 = np.where(df_snmf['pareto1'], 1.0, 0.7)

ax1.scatter(df_snmf.moran_by_chaos_scaled, df_snmf.cmc_enrich, c=colors1, alpha=alphas1, s=60)

texts1 = []
for i, row in df_snmf.iterrows():
    label = f"k={row['Rank']},ρ={row['Resolution']}\nMI={row['moran_i']:.3f}, C={row['CHAOS']:.3f}"
    texts1.append(ax1.text(row['moran_by_chaos_scaled'], row['cmc_enrich'], label, fontsize=8))

adjust_text(texts1, ax=ax1, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

ax1.set_title("Pareto Front: snmf_moran_chaos_vs_enrichment", fontsize=16)
ax1.set_xlabel("Moran's I / (CHAOS * 100) (Higher is Better)", fontsize=12)
ax1.set_ylabel("CMC * Enrichment (Higher is Better)", fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)

# Save the first plot to a unique file
plt.savefig('snmf_moran_chaos_vs_enrich_plot.png', dpi=300, bbox_inches='tight')


# --- 4) Create, save, and show PLOT 2 ---
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))

colors2 = np.where(df_snmf['pareto2'], 'red', 'blue')
alphas2 = np.where(df_snmf['pareto2'], 1.0, 0.7)

ax2.scatter(df_snmf.moran_by_chaos_scaled, df_snmf.sil_by_db, c=colors2, alpha=alphas2, s=60)

texts2 = []
for i, row in df_snmf.iterrows():
    label = f"k={row['Rank']},ρ={row['Resolution']}\nMI={row['moran_i']:.3f}, C={row['CHAOS']:.3f}"
    texts2.append(ax2.text(row['moran_by_chaos_scaled'], row['sil_by_db'], label, fontsize=8))

adjust_text(texts2, ax=ax2, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

ax2.set_title("Pareto Front: snmf_moran_chaos_vs_silhouette_DBI", fontsize=16)
ax2.set_xlabel("Moran's I / CHAOS (Higher is Better)", fontsize=12)
ax2.set_ylabel("Silhouette / DBI (Higher is Better)", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)

# Save the second plot to a unique file
plt.savefig('snmf_moran_chaos_vs_silhouette_DBI_plot.png', dpi=300, bbox_inches='tight')

# This command will display both generated figures
plt.show()