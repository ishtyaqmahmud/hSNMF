from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_squared_error
from typing import List, Tuple, Dict, Any
import numpy as np
import scanpy as sc
from scipy.sparse import issparse
import gseapy
import pickle
import os
from requests.exceptions import RequestException
import time
from matplotlib.lines import Line2D
from collections import Counter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import seaborn as sns

# -----------------------------------------------------------------------------
# CONFIGURATION GRID – edit here ------------------------------------------------
# -----------------------------------------------------------------------------
RANDOM_STATE: int = 0
SAMPLE_SIZE: int = 10_000            # subsample for silhouette/DB if > this many
UMAP_SEEDS: int = 0
MIN_DIST: float = 0.4
N_NEIGHBORS: int = 15

# -----------------------------------------------------------------------------
# Helper functions -------------------------------------------------------------
# -----------------------------------------------------------------------------

def _subsample(X, labels, max_n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample rows of *X* and *labels* if necessary (stratified)."""
    n_samples = X.shape[0]
    if n_samples <= max_n:
        return X, labels

    unique, counts = np.unique(labels, return_counts=True)
    idx = []
    for u, c in zip(unique, counts):
        take = max(1, int(round(max_n * c / n_samples)))
        idx.extend(rng.choice(np.where(labels == u)[0], size=take, replace=False))
    idx = np.array(idx)

    X_sub = X[idx] if issparse(X) else X[idx]
    X_sub = X_sub.toarray() if issparse(X_sub) else X_sub
    return X_sub, labels[idx]


def compute_silhouette_score(
    adata: sc.AnnData,
    cluster_key: str,
    use_rep: str,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> float:
    """Return Silhouette score (Euclidean) with optional sub‑sampling."""
    try:
        X = adata.X if use_rep == "X" else adata.obsm[use_rep]
        labels = adata.obs[cluster_key].values
        rng = np.random.default_rng(random_state)
        X, labels = _subsample(X, labels, sample_size, rng)
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(silhouette_score(X, labels, metric="euclidean"))
    except Exception as exc:
        print(f"Silhouette error for {cluster_key}: {exc}")
        return float("nan")


def compute_db_index(
    adata: sc.AnnData,
    cluster_key: str,
    use_rep: str,
    umap_key_base: str,
    neighbors_key: str,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    umap_seeds: int = UMAP_SEEDS,
) -> Tuple[float, float]:
    """Return (DBI in *use_rep*, mean DBI in UMAP space)."""
    try:
        # ---- DBI in representation space ------------------------------------
        X = adata.X if use_rep == "X" else adata.obsm[use_rep]
        labels = adata.obs[cluster_key].values
        rng = np.random.default_rng(random_state)
        X_sub, labels_sub = _subsample(X, labels, sample_size, rng)
        db_rep = float("nan") if len(np.unique(labels_sub)) < 2 else float(
            davies_bouldin_score(X_sub, labels_sub)
        )

        # ---- DBI in UMAP space ----------------------------------------------
        if umap_seeds == 0:
            return db_rep, float("nan")

        db_umap_vals: List[float] = []
        neigh_key = adata.uns_keys()[-1]  # last built neighbours key
        for seed in range(umap_seeds):
            umap_key = f"{umap_key_base}_seed{seed}"
            sc.tl.umap(
                adata,
                neighbors_key=neighbors_key,
                min_dist=MIN_DIST,
                random_state=seed,
                key_added=umap_key,
            )
            emb = adata.obsm[umap_key]
            emb_sub = emb[labels_sub.index] if hasattr(labels_sub, "index") else emb[: len(labels_sub)]
            db_umap_vals.append(
                davies_bouldin_score(emb_sub, labels_sub)
            )
        return db_rep, float(np.mean(db_umap_vals))

    except Exception as exc:
        print(f"DBI error for {cluster_key}: {exc}")
        return float("nan"), float("nan")

def compute_marker_genes(adata, groupby='leiden', method='wilcoxon', n_genes=5, output_csv=None):
    """
    Compute marker genes per cluster using Scanpy's rank_genes_groups.
    Parameters:
    - adata: AnnData object
    - groupby: column in adata.obs to group on (e.g., 'leiden')
    - method: test method ('t-test', 'wilcoxon', etc.)
    - n_genes: number of top marker genes to extract per cluster
    - output_csv: optional filepath to save results
    Returns:
    - DataFrame with columns [cluster, gene, score, logfoldchanges, pvals_adj]
    """
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, n_genes=n_genes)
    clusters = adata.uns['rank_genes_groups']['names'].dtype.names
    
    records = []
    for cl in clusters:
        names = adata.uns['rank_genes_groups']['names'][cl]
        scores = adata.uns['rank_genes_groups']['scores'][cl]
        lfc = adata.uns['rank_genes_groups']['logfoldchanges'][cl]
        pvals_adj = adata.uns['rank_genes_groups']['pvals_adj'][cl]
        for gene, score, lf, pv in zip(names, scores, lfc, pvals_adj):
            records.append({
                'cluster': cl,
                'gene': gene,
                'score': score,
                'logfoldchange': lf,
                'pval_adj': pv
            })
    df = pd.DataFrame.from_records(records)
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df

def compute_enrichment(adata, cluster_key, all_genes, top_genes, gene_set_library='KEGG_2021_Human'):
    try:
        # Try Enrichr first, fall back to local gene set
        gmt_file = 'kegg_2021_human.gmt.pkl'
        gene_sets = None
        if os.path.exists(gmt_file):
            try:
                with open(gmt_file, 'rb') as f:
                    gene_sets = pickle.load(f)
                print(f"Loaded local gene set: {gmt_file}")
            except Exception as e:
                print(f"Failed to load local gene set: {e}")

        scores = []
        for gene_set in top_genes:
            if not gene_set:
                continue
            print(f"Querying gseapy for {cluster_key} with genes: {gene_set[:5]}...")
            if gene_sets is None:  # Try Enrichr API
                for attempt in range(3):  # Retry up to 3 times
                    try:
                        res = gseapy.enrichr(
                            gene_list=gene_set,
                            gene_sets=gene_set_library,
                            organism='Human',
                            cutoff=0.05
                        )
                        if not res.results.empty:
                            significant = res.results[res.results['Adjusted P-value'] < 0.05]
                            if not significant.empty:
                                pval = significant['Adjusted P-value'].min()
                                scores.append(-np.log10(max(pval, 1e-10)))
                            else:
                                print(f"No significant enrichment for {cluster_key} gene set: {gene_set[:5]}")
                        else:
                            print(f"No enrichment for {cluster_key} gene set: {gene_set[:5]}")
                    except (RequestException, Exception) as e:
                        print(f"Gseapy error for {cluster_key} gene set (attempt {attempt+1}): {e}")
                        if attempt < 2:
                            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
                        else:
                            print(f"Failed after 3 attempts for {cluster_key} gene set: {gene_set[:5]}")
                time.sleep(0.1)  # Avoid rate limiting
            else:  # Use local gene set
                try:
                    res = gseapy.enrich(
                        gene_list=gene_set,
                        gene_sets=gene_sets,
                        background=all_genes,
                        cutoff=0.05
                    )
                    if not res.results.empty:
                        pval = res.results['Adjusted P-value'].min()
                        scores.append(-np.log10(max(pval, 1e-10)))
                    else:
                        print(f"No enrichment for {cluster_key} gene set: {gene_set[:5]}")
                except Exception as e:
                    print(f"Local gseapy error for {cluster_key} gene set: {e}")
        return np.median(scores) if scores else np.nan
    except Exception as e:
        print(f"Error computing enrichment for {cluster_key}: {e}")
        return np.nan

def compute_reconstruction_metrics(adata, recon_key, layer='lognorm'):
    X = adata.X
    X_recon = adata.obsm[recon_key]
    try:
        if X_recon is None:
            raise ValueError(f"No reconstructed X found for {recon_key}.")
        X = X.toarray() if issparse(X) else X
        X_recon = X_recon.toarray() if issparse(X_recon) else X_recon
        mse = mean_squared_error(X, X_recon)
        X_centered = X - np.mean(X, axis=0)
        total_var = np.mean(X_centered ** 2)
        evr = 1 - (mse / total_var) if total_var > 0 else 0.0

        # Check overlap with marker genes

        return {
            'reconstruction_error': mse,
            'total_variance': total_var,
            'explained_variance': evr,
        }
    except Exception as e:
        print(f"Error computing reconstruction metrics for {recon_key}: {e}")
        return {
            'reconstruction_error': np.nan,
            'total_variance': np.nan,
            'explained_variance': 0.0,
        }

def get_marker_genes_dict(adata, groupby='leiden', method='wilcoxon', n_genes=5):
    """
    Compute top marker genes per cluster and return as a dict.
    
    Returns:
        markers: dict where keys are cluster labels and values are lists of gene names
    """
    # Run the test
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, n_genes=n_genes)
    
    # Extract cluster names
    clusters = adata.uns['rank_genes_groups']['names'].dtype.names
    
    # Build dict
    markers = {
        cl: [
            adata.uns['rank_genes_groups']['names'][cl][i]
            for i in range(n_genes)
        ]
        for cl in clusters
    }
    return markers

def cluster_marker_fraction(adata, cluster_key, marker_genes):
    """
    marker_genes: dict mapping cluster label -> list of marker-gene names
    """
    X = adata.X.toarray()
    obs = adata.obs[cluster_key]
    fractions = []
    for c, genes in marker_genes.items():
        # get cell indices for this cluster
        idx = np.where(obs == c)[0]
        if len(idx) == 0 or len(genes) == 0:
            continue
        # find gene indices
        gene_idx = [adata.var_names.get_loc(g) for g in genes if g in adata.var_names]
        # boolean matrix: cells × markers
        expr = X[np.ix_(idx, gene_idx)] > 0
        # fraction per cell
        f_i = expr.sum(axis=1) / len(gene_idx)
        # average for this cluster
        fractions.append(f_i.mean())
    # overall mean
    return np.mean(fractions), dict(zip(marker_genes.keys(), fractions))

def weighted_marker_fraction(adata, cluster_key, marker_genes):
    X = adata.X
    if hasattr(X, 'toarray'): X = X.toarray()
    obs = adata.obs[cluster_key].values
    total_cells = len(obs)

    weighted_sum = 0.0
    for c, genes in marker_genes.items():
        idx = np.where(obs == c)[0]
        if len(idx) == 0: continue
        gene_idx = [adata.var_names.get_loc(g) for g in genes if g in adata.var_names]
        expr = (X[np.ix_(idx, gene_idx)] > 0).sum(axis=1) / len(gene_idx)
        F_c = expr.mean()
        weighted_sum += F_c * len(idx)

    return weighted_sum / total_cells

def cluster_marker_expression_score(adata, cluster_key, marker_genes, layer='lognorm'):
    """
    For each cluster c, compute:
      MES_c = (1 / |I_c|) * sum_{i in I_c} sum_{g in M_c} X[i,g] / G_c
    where I_c = indices of cells in cluster c,
          M_c = marker_genes[c],
          G_c = number of markers in M_c (to normalize per gene).
    Returns a dict {cluster_label: MES_c}
    """
    X = adata.X
    clusters = adata.obs[cluster_key].values
    unique = np.unique(clusters)
    mes = {}

    for c in unique:
        idx = np.where(clusters == c)[0]
        genes = marker_genes.get(c, [])
        # get column indices of those genes
        gene_idx = [adata.var_names.get_loc(g) for g in genes if g in adata.var_names]
        if len(gene_idx) == 0 or len(idx) == 0:
            mes[c] = np.nan
            continue
        # extract submatrix: cells × markers
        sub = X[np.ix_(idx, gene_idx)]
        # normalize each marker by its max (optional) or by 1
        # here we just sum raw normalized expression
        per_cell = np.sum(sub, axis=1) / len(gene_idx)
        mes[c] = np.mean(per_cell)

    return mes

def compute_specificity(adata, cluster_key, marker_genes, layer='lognorm', threshold=0):
    clusters = adata.obs[cluster_key].unique()
    specs = {}
    for cluster in clusters:
        in_cl = adata.obs[cluster_key] == cluster
        out_cl = ~in_cl
        genes = marker_genes.get(cluster, [])
        if not genes:
            continue
        spec_scores = []
        for gene in genes:
            if gene not in adata.var_names:
                continue
            expr = adata[:, gene].X.toarray().flatten() if issparse(adata[:, gene].X) else adata[:, gene].X.flatten()
            pct_in = np.mean(expr[in_cl] > threshold)
            pct_out = np.mean(expr[out_cl] > threshold)
            spec_scores.append(pct_in - pct_out)
        if spec_scores:
            specs[cluster] = np.mean(spec_scores)
    return np.mean(list(specs.values())) if specs else 0.0, specs

def compute_marker_exclusion_rate(adata,
                                      cluster_key: str,
                                      marker_genes: dict[str, list[str]],
                                      layer: str | None = None):
    """
    1) For each cell, find the cluster c* whose marker set M_c has the largest
       number of expressed markers in that cell.
    2) Mask to only those cells with at least one marker expressed.
    3) Compute MAE = fraction where predicted cluster ≠ assigned c*, over marker+ cells.
    
    Parameters
    ----------
    adata
      AnnData with .obs[cluster_key] giving the predicted cluster.
    marker_genes
      dict mapping cluster label (same type as adata.obs[cluster_key]) → list of gene names.
    layer
      If given, use adata.layers[layer] for expression; else use adata.X.
    
    Returns
    -------
    mer : float
      Misassignment error over all cells.
    n_no_marker : int
      Number of cells with no marker signals.
    n_marker_pos : int
      Number of marker-positive cells.
    """
    # 1. get a boolean expression matrix: cells × genes
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X
    # convert to dense if needed
    X = X.toarray() if not isinstance(X, np.ndarray) else X

    var_names = list(adata.var_names)

    # 2. precompute marker gene indices
    marker_idx: dict[str, np.ndarray] = {}
    for c, genes in marker_genes.items():
        idx = [var_names.index(g) for g in genes if g in var_names]
        marker_idx[c] = np.array(idx, dtype=int)

    # 3. for each cell count how many markers in each cluster
    cells = X.shape[0]
    assignments = np.empty(cells, dtype=object)
    marker_positive = np.zeros(cells, dtype=bool)
    pred = adata.obs[cluster_key].astype(str).to_numpy()

    for i in range(cells):
        scores = {}
        for c, idx in marker_idx.items():
            # sum *actual* expression for marker genes of cluster c
            # (you could also take the mean if you prefer normalization)
            scores[c] = X[i, idx].sum() if idx.size else 0.0
    
        # pick the cluster with the highest summed expression
        best_c = max(scores, key=scores.get)
        best_score = scores[best_c]
    
        # original cluster and its score
        orig_c    = pred[i]
        orig_score = scores.get(orig_c, 0.0)

        # only reassign if the improvement over the original is > 1e-6
        if best_score - orig_score > 1e-1:
            assignments[i] = best_c
        else:
            assignments[i] = orig_c

        # marker_positive if there was *any* expression (sum > 0)
        marker_positive[i] = (best_score > 0.0)

    # 4. compare to the predicted labels
    gt   = assignments.astype(str)

#    # 1) build a mask of exactly those cells that both are marker-positive and got reassigned
#    flip_mask = (pred != gt) & marker_positive
#    
#    # 2) extract their integer indices
#    flip_idx = np.where(flip_mask)[0]
#    
#    out_path = "flipped_cells.txt"
#    with open(out_path, "a") as fout:
#        fout.write("cell_barcode\tleiden\treassigned\n")
#        for i in flip_idx:
#            cell_id = adata.obs_names[i]
#            fout.write(f"{i}th cell: {cell_id}\t{pred[i]}\t{gt[i]}\n")
#        fout.write("Done.")
#    
#    print(f"Written {len(flip_idx)} flipped cells to {out_path}")

    # mask to only marker-positive cells
    mask = marker_positive
    if mask.sum() == 0:
        raise ValueError("No marker-positive cells found; cannot compute MER.")
    if cells == 0:
        raise ValueError("No cells found; cannot compute MER.")
    
    mer = np.sum(pred[mask] != gt[mask])/cells
    n_marker_pos = mask.sum()
    total_marker_zero = (cells - n_marker_pos)

    return mer, total_marker_zero, n_marker_pos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_marker_dotplot(adata, marker_genes, before_key, after_key, layer=None):
    """
    Creates a dot-plot heatmap showing, for each cluster and marker gene, the fraction of cells
    expressing the gene (dot size) and the average expression level (dot color), before and after reassignment.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with .obs containing before_key and after_key cluster labels.
    marker_genes : dict
        Mapping from cluster labels to list of marker genes.
    before_key : str
        Column name in adata.obs for pre-reassignment cluster labels.
    after_key : str
        Column name in adata.obs for post-reassignment cluster labels.
    layer : str, optional
        If provided, use adata.layers[layer] for expression; otherwise, use adata.X.
    """
    # 1) Extract expression matrix
    X = adata.layers[layer] if layer else adata.X
    X = X.toarray() if not isinstance(X, np.ndarray) else X
    genes = adata.var_names.tolist()
    
    # 2) Prepare data rows
    records = []
    for state_key, label_key in [('before', before_key), ('after', after_key)]:
        for cluster in sorted(adata.obs[label_key].unique()):
            idx = np.where(adata.obs[label_key] == cluster)[0]
            if len(idx) == 0:
                continue
            subX = X[idx]
            for gene in marker_genes.get(str(cluster), []):
                if gene not in genes:
                    continue
                g_idx = genes.index(gene)
                expr_vals = subX[:, g_idx]
                pct_expr = np.mean(expr_vals > 0)
                avg_expr = np.mean(expr_vals[expr_vals > 0]) if pct_expr > 0 else 0
                records.append({
                    'state': state_key,
                    'cluster': str(cluster),
                    'gene': gene,
                    'pct_expr': pct_expr,
                    'avg_expr': avg_expr
                })
    
    df = pd.DataFrame(records)
    
    # 3) Plot
    fig, ax = plt.subplots(figsize=(1.5 * len(marker_genes), len(df['gene'].unique()) * 0.3))
    # Create an ordering for the x-axis: interleave before/after for each cluster
    clusters = sorted(df['cluster'].unique(), key=int)
    x_labels = [f"{c}\n{state}" for c in clusters for state in ['before', 'after']]
    x_map = {label: i for i, label in enumerate(x_labels)}
    
    # Scatter each point
    for _, row in df.iterrows():
        x = x_map[f"{row['cluster']}\n{row['state']}"]
        y = df['gene'].unique().tolist().index(row['gene'])
        size = row['pct_expr'] * 200  # scale for visibility
        color = row['avg_expr']
        ax.scatter(x, y, s=size, c=[color], cmap='viridis', vmin=df['avg_expr'].min(), vmax=df['avg_expr'].max(), edgecolors='grey')
    
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticks(range(len(df['gene'].unique())))
    ax.set_yticklabels(df['gene'].unique())
    ax.set_xlabel('Cluster (state)')
    ax.set_ylabel('Marker Gene')
    ax.set_title('Marker Expression Before vs After MER Reassignment')
    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Avg Expression')
    fig.tight_layout()
    return fig

# Example usage:
# fig = plot_marker_dotplot(adata, marker_genes, 'leiden_pca_before', 'leiden_pca_after', layer='lognorm')
# fig.savefig('marker_dotplot.png')

def plot_marker_dotplot_subset(adata,
                               marker_genes,
                               before_key,
                               after_key,
                               layer=None,
                               gain_threshold=0.10):
    """
    Dot‐plot of marker expression before vs after MER reassignment,
    but only for clusters whose max marker mean‐expression gain > gain_threshold.
    """
    # 1) get expression matrix
    X = adata.layers[layer] if layer else adata.X
    X = X.toarray() if not isinstance(X, np.ndarray) else X
    genes = list(adata.var_names)

    # 2) collect mean expression per cluster×gene for before/after
    def collect(label_key, tag):
        recs = []
        # pull unique cluster labels
        labels = adata.obs[label_key].unique().astype(str)
        try:
            labels = sorted(labels, key=lambda x: int(x))
        except ValueError:
            labels = sorted(labels)
        for cl in labels:
            idx = np.where(adata.obs[label_key].astype(str) == cl)[0]
            sub  = X[idx, :]
            for g in marker_genes.get(cl, []):
                if g not in genes: continue
                gi       = genes.index(g)
                vals     = sub[:, gi]
                pct      = (vals > 0).mean()
                mean_val = vals[vals>0].mean() if pct>0 else 0.0
                recs.append({
                    'cluster': cl,
                    'gene':    g,
                    'pct_'+tag: pct,
                    'mean_'+tag: mean_val
                })
        return pd.DataFrame(recs)

    df_b = collect(before_key, 'b')
    df_a = collect(after_key,  'a')

    # 3) merge and filter for clusters with any gain > threshold
    df = df_b.merge(df_a, on=['cluster','gene'])
    df['gain'] = df['mean_a'] - df['mean_b']
    df['rel_impr'] = np.where(
        df['mean_b']>0,
        df['gain'] / df['mean_b'],
        np.nan
    )
    mean_abs_impr = df['gain'].mean()
    mean_rel_impr = df['rel_impr'].mean() * 100
    good = df.groupby('cluster')['gain'].max().loc[lambda s: s>gain_threshold].index
    df = df[df['cluster'].isin(good)].copy()
    if df.empty:
        raise ValueError("No clusters exceed gain_threshold")

    mean_abs_impr = df['gain'].mean()            # average absolute gain
    mean_rel_impr = df['rel_impr'].mean() * 100  # average % gain

    # 4) prepare plotting ordering
    clusters = sorted(df['cluster'].unique(), key=int)
    genes     = sorted(df['gene'].unique())
    x_labels  = [f"{c}\n{state}" for c in clusters for state in ('before','after')]
    x_map     = {lbl:i for i,lbl in enumerate(x_labels)}

#    # 5) draw
#    fig, ax = plt.subplots(figsize=(1.2*len(x_labels), 0.4*len(genes)))
#    vmin, vmax = df[['mean_b','mean_a']].min().min(), df[['mean_b','mean_a']].max().max()
#    for _, row in df.iterrows():
#        xi = x_map[f"{row['cluster']}\nbefore"]
#        yi = genes.index(row['gene'])
#        ax.scatter(xi, yi,
#                   s=row['pct_b']*200,
#                   c=[row['mean_b']],
#                   cmap='viridis', vmin=vmin, vmax=vmax,
#                   edgecolors='grey', linewidth=0.3)
#        xi = x_map[f"{row['cluster']}\nafter"]
#        ax.scatter(xi, yi,
#                   s=row['pct_a']*200,
#                   c=[row['mean_a']],
#                   cmap='viridis', vmin=vmin, vmax=vmax,
#                   edgecolors='grey', linewidth=0.3)
#
#    # 6) tidy up
#    ax.set_xticks(range(len(x_labels)))
#    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
#    ax.set_yticks(range(len(genes)))
#    ax.set_yticklabels(genes, fontsize=8)
#    ax.set_xlabel('Cluster (state)')
#    ax.set_ylabel('Marker gene')
#    ax.set_title('Markers with > {:.0f}% mean‐expression gain'.format(gain_threshold*100))
#    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin, vmax)),
#                        ax=ax, fraction=0.02, pad=0.04)
#    cbar.set_label('Avg. expression')
#    plt.tight_layout()
    fig, ax = plt.subplots(figsize=(1.2*len(x_labels), 0.4*len(genes)))
    vmin, vmax = df[['mean_b','mean_a']].min().min(), df[['mean_b','mean_a']].max().max()
    
    # Font size parameters
    fontsize_title = 20
    fontsize_xlabel = 20
    fontsize_ylabel = 20
    fontsize_xtick = 16
    fontsize_ytick = 16
    fontsize_legend_title = 20
    fontsize_legend_labels = 14
    
    for _, row in df.iterrows():
        xi = x_map[f"{row['cluster']}\nbefore"]
        yi = genes.index(row['gene'])
        ax.scatter(xi, yi,
                   s=row['pct_b']*200,
                   c=[row['mean_b']],
                   cmap='viridis', vmin=vmin, vmax=vmax,
                   edgecolors='grey', linewidth=0.3)
        xi = x_map[f"{row['cluster']}\nafter"]
        ax.scatter(xi, yi,
                   s=row['pct_a']*200,
                   c=[row['mean_a']],
                   cmap='viridis', vmin=vmin, vmax=vmax,
                   edgecolors='grey', linewidth=0.3)
    
    # Tidy up with custom font sizes
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=fontsize_xtick)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=fontsize_ytick)
    ax.set_xlabel('Cluster (state)', fontsize=fontsize_xlabel)
    ax.set_ylabel('Marker gene', fontsize=fontsize_ylabel)
    ax.set_title('Markers with > {:.0f}% mean‐expression gain'.format(gain_threshold*100), 
                 fontsize=fontsize_title)
    
    # Colorbar with custom font sizes
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin, vmax)),
                        ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Avg. expression', fontsize=fontsize_legend_title)
    cbar.ax.tick_params(labelsize=fontsize_legend_labels)
    
    plt.tight_layout()
    return fig, mean_abs_impr, mean_rel_impr

def load_spatial_coordinates(adata, data_type='xenium', visium_path=None, xenium_cells=None):
    # Load spatial coordinates
    if data_type == 'visium' and visium_path:
        adata = sc.read_visium(visium_path, count_file='filtered_feature_bc_matrix.h5')
        print(f"Loaded Visium data with {adata.n_obs} cells.")
    elif data_type == 'xenium' and xenium_cells:
        cells = pd.read_csv(xenium_cells)
        coords = cells.set_index('cell_id')[['x_centroid', 'y_centroid']]
        common_barcodes = coords.index.intersection(adata.obs.index)
        if len(common_barcodes) != adata.n_obs:
            print(f"Warning: Only {len(common_barcodes)} of {adata.n_obs} barcodes match.")
        coords = coords.loc[common_barcodes, ['x_centroid', 'y_centroid']].reindex(adata.obs.index).values
        adata.obsm['spatial'] = coords
        print(f"Loaded Xenium coordinates for {len(coords)} cells.")
    else:
        raise ValueError("Specify data_type='visium' with visium_path or 'xenium' with xenium_cells.")

    # Validate spatial coordinates
    if adata.obsm['spatial'].shape[0] != adata.n_obs:
        raise ValueError(f"Spatial coordinates mismatch: {adata.obsm['spatial'].shape[0]} vs {adata.n_obs}.")
    if np.any(np.isnan(adata.obsm['spatial'])):
        raise ValueError("NaN values found in spatial coordinates.")

def plot_mtap_neg(adata, marker_genes, key, n_clusters, method, state): 

    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    
    cmap = plt.cm.tab20c
    handles = []
    labels = []
    
    if adata.obsm.get('spatial') is None:
        xenium_cells = '/project/banerjee/MDACC_CholangioData/output-XETG00074_0024837_MTAPneg_A2_20240607_194742/cells.csv'
        cells = pd.read_csv(xenium_cells)
        coords = cells.set_index('cell_id')[['x_centroid', 'y_centroid']]
        common_barcodes = coords.index.intersection(adata.obs.index)
        if len(common_barcodes) != adata.n_obs:
            print(f"Warning: Only {len(common_barcodes)} of {adata.n_obs} barcodes match.")
        coords = coords.loc[common_barcodes, ['x_centroid', 'y_centroid']].reindex(adata.obs.index).values
        adata.obsm['spatial'] = coords
        print(f"Loaded Xenium coordinates for {len(coords)} cells.")
        
    coords = adata.obsm['spatial']  # shape (n_cells, 2)
    x, y = coords[:, 0], coords[:, 1]
    mask = (x >= 0) & (x <= 15000) & (y >= 0) & (y <= 15000) # Sample 34
    coords_filt = coords[mask]

    # Get cluster labels for filtered cells
    clusters = adata.obs[key].values.astype(int)[mask]
    unique_clusters = np.unique(clusters)
    cluster_counts = dict(Counter(clusters))

    # -------- NEW: dump full cluster summary, sorted by size --------
    summary_rows = [
        {
            'cluster_id'  : cid,
            'marker_genes': marker_genes.get(str(cid), ''),  # empty if not in dict
            'n_cells'     : cnt
        }
        for cid, cnt in cluster_counts.items()
    ]

    df_summary = (pd.DataFrame(summary_rows)
                    .sort_values('n_cells', ascending=False)
                    .reset_index(drop=True))

    out_path = f'{method}_{state}_cluster_summary_mtap_neg.csv'       # customise as needed
    df_summary.to_csv(out_path, index=False)
    print(f"Saved full cluster summary → {out_path}")
    # ---------------------------------------------------------------

    
    # Fixed range of cluster IDs you care about
    all_cluster_ids = range(0, n_clusters)  # or max cluster ID + 1
    palette = sns.color_palette("husl", n_clusters)  # or try "hls"
    cluster_color_dict = {cid: palette[i] for i, cid in enumerate(all_cluster_ids)}

    # Use filtered clusters in your current sample
    colors_for_clusters = [cluster_color_dict[int(cid)] for cid in clusters]

    # Sort clusters by count and keep top 20
    top_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    top_cluster_ids = [clust for clust, _ in top_clusters]
    mask_top = np.isin(clusters, top_cluster_ids)

    # Now only plot those
    x = adata.obsm['spatial'][mask][mask_top, 0]
    y = adata.obsm['spatial'][mask][mask_top, 1]
    filtered_clusters = clusters[mask_top]

    sc = ax.scatter(
        coords_filt[:, 0], coords_filt[:, 1],
        c=colors_for_clusters, s=5, alpha=0.8
    )
    ax.set_aspect('equal')
    for idx, clust in enumerate(top_cluster_ids):
        color = cluster_color_dict[clust]
        handles.append(Line2D([0], [0],
                              marker='o',
                              color='w',
                              markerfacecolor=color,
                              markersize=5,
                              linestyle=''))
        genes = marker_genes.get(str(int(clust)))
        count = cluster_counts.get(int(clust))
        labels.append(f"{str(int(clust))}: {{{genes}}}, {count}")
    ax.legend(handles, labels,
              title='Cluster',
              ncol=1,
              bbox_to_anchor=(0.5, 0.08),
              loc='upper center',
              fontsize='x-small',
              frameon=True)
    ax.axis('off')
    #plt.show()
    return fig

def plot_mtap_pos(adata, marker_genes, key, n_clusters, method, state): 

    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    
    cmap = plt.cm.tab20c
    handles = []
    labels = []
    
    if adata.obsm.get('spatial') is None:
        xenium_cells = '/project/banerjee/MDACC_CholangioData/output-XETG00074__0024837__MTAPpos__20240607__194742/cells.csv'
        cells = pd.read_csv(xenium_cells)
        coords = cells.set_index('cell_id')[['x_centroid', 'y_centroid']]
        common_barcodes = coords.index.intersection(adata.obs.index)
        if len(common_barcodes) != adata.n_obs:
            print(f"Warning: Only {len(common_barcodes)} of {adata.n_obs} barcodes match.")
        coords = coords.loc[common_barcodes, ['x_centroid', 'y_centroid']].reindex(adata.obs.index).values
        adata.obsm['spatial'] = coords
        print(f"Loaded Xenium coordinates for {len(coords)} cells.")
        
    coords = adata.obsm['spatial']  # shape (n_cells, 2)
    x, y = coords[:, 0], coords[:, 1]
    mask = (x >= 0) & (x <= 15000) & (y >= 0) & (y <= 15000) # Sample 34
    coords_filt = coords[mask]

    # Get cluster labels for filtered cells
    clusters = adata.obs[key].values.astype(int)[mask]
    unique_clusters = np.unique(clusters)
    cluster_counts = dict(Counter(clusters))

    # -------- NEW: dump full cluster summary, sorted by size --------
    summary_rows = [
        {
            'cluster_id'  : cid,
            'marker_genes': marker_genes.get(str(cid), ''),  # empty if not in dict
            'n_cells'     : cnt
        }
        for cid, cnt in cluster_counts.items()
    ]

    df_summary = (pd.DataFrame(summary_rows)
                    .sort_values('n_cells', ascending=False)
                    .reset_index(drop=True))

    out_path = f'{method}_{state}_cluster_summary_mtap_pos.csv'       # customise as needed
    df_summary.to_csv(out_path, index=False)
    print(f"Saved full cluster summary → {out_path}")
    # ---------------------------------------------------------------

    
    # Fixed range of cluster IDs you care about
    all_cluster_ids = range(0, n_clusters)  # or max cluster ID + 1
    palette = sns.color_palette("husl", n_clusters)  # or try "hls"
    cluster_color_dict = {cid: palette[i] for i, cid in enumerate(all_cluster_ids)}

    # Use filtered clusters in your current sample
    colors_for_clusters = [cluster_color_dict[int(cid)] for cid in clusters]

    # Sort clusters by count and keep top 20
    top_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    top_cluster_ids = [clust for clust, _ in top_clusters]
    mask_top = np.isin(clusters, top_cluster_ids)

    # Now only plot those
    x = adata.obsm['spatial'][mask][mask_top, 0]
    y = adata.obsm['spatial'][mask][mask_top, 1]
    filtered_clusters = clusters[mask_top]

    sc = ax.scatter(
        coords_filt[:, 0], coords_filt[:, 1],
        c=colors_for_clusters, s=5, alpha=0.8
    )
    ax.set_aspect('equal')
    for idx, clust in enumerate(top_cluster_ids):
        color = cluster_color_dict[clust]
        handles.append(Line2D([0], [0],
                              marker='o',
                              color='w',
                              markerfacecolor=color,
                              markersize=5,
                              linestyle=''))
        genes = marker_genes.get(str(int(clust)))
        count = cluster_counts.get(int(clust))
        labels.append(f"{str(int(clust))}: {{{genes}}}, {count}")
    ax.legend(handles, labels,
              title='Cluster',
              ncol=1,
              bbox_to_anchor=(0.5, 0.08),
              loc='upper center',
              fontsize='x-small',
              frameon=True)
    ax.axis('off')
    #plt.show()
    return fig

def plot_sample_core(adata, marker_genes, key, n_clusters, method, state): 

    fig = plot_mtap_neg(adata, marker_genes, key, n_clusters, method, state)
    return fig

    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    
    cmap = plt.cm.tab20c
    handles = []
    labels = []
    
    if adata.obsm.get('spatial') is None:
        xenium_cells = '/project/banerjee/MDACC_CholangioData/output-XETG00074__0024840__TMA_1__20240607__194742/cells.csv'
        cells = pd.read_csv(xenium_cells)
        coords = cells.set_index('cell_id')[['x_centroid', 'y_centroid']]
        common_barcodes = coords.index.intersection(adata.obs.index)
        if len(common_barcodes) != adata.n_obs:
            print(f"Warning: Only {len(common_barcodes)} of {adata.n_obs} barcodes match.")
        coords = coords.loc[common_barcodes, ['x_centroid', 'y_centroid']].reindex(adata.obs.index).values
        adata.obsm['spatial'] = coords
        print(f"Loaded Xenium coordinates for {len(coords)} cells.")
        
    coords = adata.obsm['spatial']  # shape (n_cells, 2)
    x, y = coords[:, 0], coords[:, 1]
    mask = (x >= 6100) & (x <= 7300) & (y >= 4100) & (y <= 5300) # Sample 34
    coords_filt = coords[mask]

    # Get cluster labels for filtered cells
    clusters = adata.obs[key].values.astype(int)[mask]
    unique_clusters = np.unique(clusters)
    cluster_counts = dict(Counter(clusters))

    # -------- NEW: dump full cluster summary, sorted by size --------
    summary_rows = [
        {
            'cluster_id'  : cid,
            'marker_genes': marker_genes.get(str(cid), ''),  # empty if not in dict
            'n_cells'     : cnt
        }
        for cid, cnt in cluster_counts.items()
    ]

    df_summary = (pd.DataFrame(summary_rows)
                    .sort_values('n_cells', ascending=False)
                    .reset_index(drop=True))

    out_path = f'{method}_{state}_cluster_summary_sample34.csv'       # customise as needed
    df_summary.to_csv(out_path, index=False)
    print(f"Saved full cluster summary → {out_path}")
    # ---------------------------------------------------------------

    
    # Fixed range of cluster IDs you care about
    all_cluster_ids = range(0, n_clusters)  # or max cluster ID + 1
    palette = sns.color_palette("husl", n_clusters)  # or try "hls"
    cluster_color_dict = {cid: palette[i] for i, cid in enumerate(all_cluster_ids)}

    # Use filtered clusters in your current sample
    colors_for_clusters = [cluster_color_dict[int(cid)] for cid in clusters]

    # Sort clusters by count and keep top 20
    top_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    top_cluster_ids = [clust for clust, _ in top_clusters]
    mask_top = np.isin(clusters, top_cluster_ids)

    # Now only plot those
    x = adata.obsm['spatial'][mask][mask_top, 0]
    y = adata.obsm['spatial'][mask][mask_top, 1]
    filtered_clusters = clusters[mask_top]

    sc = ax.scatter(
        coords_filt[:, 0], coords_filt[:, 1],
        c=colors_for_clusters, s=5, alpha=0.8
    )
    ax.set_aspect('equal')
    for idx, clust in enumerate(top_cluster_ids):
        color = cluster_color_dict[clust]
        handles.append(Line2D([0], [0],
                              marker='o',
                              color='w',
                              markerfacecolor=color,
                              markersize=5,
                              linestyle=''))
        genes = marker_genes.get(str(int(clust)))
        count = cluster_counts.get(int(clust))
        labels.append(f"{str(int(clust))}: {{{genes}}}, {count}")
    ax.legend(handles, labels,
              title='Cluster',
              ncol=1,
              bbox_to_anchor=(0.5, 0.08),
              loc='upper center',
              fontsize='x-small',
              frameon=True)
    ax.axis('off')
    #plt.show()
    return fig

