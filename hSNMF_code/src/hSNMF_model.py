"""nmf_sweep.py – Rank × Resolution sweep for Non-negative Matrix Factorisation

This script evaluates NMF embeddings at several ranks (component counts) and
Leiden resolution values. It includes spatial graph metrics (Adjacency Score)
and performs spatial smoothing of NMF embeddings using a hybrid spatial graph.

Run:
    python hSNMF_model.py /path/to/my_data.h5ad

Dependencies: scanpy, numpy, pandas, scikit-learn, networkx, scipy
"""

from __future__ import annotations

import argparse
import pathlib
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.decomposition import NMF
import networkx as nx
from scipy import sparse

from utils import _subsample
from utils import compute_silhouette_score
from utils import compute_db_index
from utils import compute_marker_genes
from utils import compute_enrichment
from utils import compute_reconstruction_metrics
from utils import get_marker_genes_dict
from utils import cluster_marker_fraction
from utils import weighted_marker_fraction
from utils import compute_specificity
from utils import compute_marker_exclusion_rate

from utils_ishtyaq import compute_morans_i
from utils_ishtyaq import CHAOS

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale

# -----------------------------------------------------------------------------
# CONFIG – edit to taste
# -----------------------------------------------------------------------------
RANK_GRID: List[int] = [10]
RES_GRID: List[float] = [0.4]
N_NEIGHBORS: int = 15
RANDOM_STATE: int = 0
SAMPLE_SIZE: int = 10_000
UMAP_SEEDS: int = 0
MIN_DIST: float = 0.4
MAX_ITER: int = 400

# Spatial graph params
CONTACT_RADIUS: float = 20.0
FALLBACK_RADIUS: float = 80.0

# Advisor-requested behavior
MIN_CLUSTER_SIZE: int = 100       # drop clusters with fewer than this many cells for DE
USE_SPATIAL_FOR_LEIDEN: bool = True          # existing
USE_HYBRID_FOR_LEIDEN: bool = True           # NEW: if True, use α·A_spatial + (1-α)·A_feature
HYBRID_ALPHA: float = 0.5                   # NEW: α in [0,1] (spatial weight)
ROW_STOCH_NORMALIZE: bool = True              # NEW: use row-stochastic normalization for adjacencies
RENORMALIZE_MIX: bool = True  # add to CONFIG


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def smooth_W_with_graph(W, spatial_graph: nx.Graph, beta: float = 0.6, steps: int = 1, weight_attr: str = "weight"):
    n = W.shape[0]
    A = nx.to_scipy_sparse_array(
        spatial_graph,
        nodelist=range(n),
        weight=weight_attr,
        dtype=np.float64,
        format="csr",
    )
    A = A + sparse.eye(n, format="csr", dtype=np.float64)
    row_sums = np.asarray(A.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    D_inv = sparse.diags(1.0 / row_sums)
    P = D_inv @ A
    W_s = W.copy()
    for _ in range(steps):
        W_s = (1 - beta) * W_s + beta * (P @ W_s)
    np.maximum(W_s, 0, out=W_s)
    return W_s


def get_marker_genes_safe(adata_layer: AnnData, cluster_key: str, min_cluster_size: int = 100, n_genes: int = 50):
    """
    Wrapper that:
      - runs get_marker_genes_dict on clusters >= min_cluster_size,
      - suppresses scanpy errors,
      - returns a dict with an entry for every cluster label (fallback top-expression genes for small/missing clusters).
    Keys in the returned dict are strings.
    """
    counts = adata_layer.obs[cluster_key].value_counts()
    all_clusters = list(counts.index)
    small_clusters = counts[counts < min_cluster_size].index.tolist()

    # Subset excluding small clusters for statistical DE testing
    if small_clusters:
        adata_for_markers = adata_layer[~adata_layer.obs[cluster_key].isin(small_clusters)].copy()
    else:
        adata_for_markers = adata_layer

    # Try to compute statistical markers
    try:
        marker_genes = get_marker_genes_dict(adata_for_markers, groupby=cluster_key)
    except ValueError as e:
        print(f"Warning: get_marker_genes_dict ValueError: {e} — continuing with empty marker dict.")
        marker_genes = {}
    except Exception as e:
        print(f"Warning: get_marker_genes_dict exception: {e} — continuing with empty marker dict.")
        marker_genes = {}

    # Normalize keys to strings
    marker_genes = {str(k): v for k, v in marker_genes.items()}

    # Provide non-statistical fallback for any cluster missing markers
    for cl in all_clusters:
        k = str(cl)
        if k in marker_genes and marker_genes[k]:
            continue
        mask = adata_layer.obs[cluster_key] == cl
        if mask.sum() == 0:
            marker_genes[k] = []
            continue
        sub = adata_layer[mask]
        if sparse.issparse(sub.X):
            mean_expr = np.asarray(sub.X.mean(axis=0)).ravel()
        else:
            mean_expr = np.asarray(sub.X).mean(axis=0).ravel()
        top_idx = np.argsort(mean_expr)[-n_genes:][::-1]
        marker_genes[k] = adata_layer.var_names[top_idx].tolist()

    return marker_genes


# Graph builders (kept as in your script)
def build_spatial_graph(
    adata: AnnData,
    radius: float,
    weight_value: float,
    store_dist: bool = True,
    self_loop_eps: float = 1e-6,
) -> nx.Graph:
    coords = adata.obsm["spatial"]
    nbrs = NearestNeighbors(radius=radius)
    nbrs.fit(coords)
    distances, indices = nbrs.radius_neighbors(coords, return_distance=True)

    G = nx.Graph()
    G.add_nodes_from(range(adata.n_obs))

    for i in range(adata.n_obs):
        neigh_idx = indices[i]
        neigh_dist = distances[i]
        for j, d in zip(neigh_idx, neigh_dist):
            if i == j:
                continue
            if d <= self_loop_eps:
                continue
            if G.has_edge(i, j):
                continue
            if store_dist:
                G.add_edge(i, j, weight=float(weight_value), dist=float(d))
            else:
                G.add_edge(i, j, weight=float(weight_value))
    return G


def build_contact_graph(adata: AnnData, contact_radius: float, contact_scale: float = 2.5, **kwargs) -> nx.Graph:
    print(f"  ➜ Building contact graph (radius={contact_radius}, weight={contact_scale})...")
    G = build_spatial_graph(adata, radius=contact_radius, weight_value=contact_scale, **kwargs)
    print(f"    - Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} contact edges.")
    return G


def build_radius_graph(adata: AnnData, fallback_radius: float, radius_scale: float = 1.0, **kwargs) -> nx.Graph:
    print(f"  ➜ Building radius graph (radius={fallback_radius}, weight={radius_scale})...")
    G = build_spatial_graph(adata, radius=fallback_radius, weight_value=radius_scale, **kwargs)
    print(f"    - Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} radius edges.")
    return G


def build_hybrid_graph(contact_graph: nx.Graph, radius_graph: nx.Graph, combine: str = "max", weight_attr: str = "weight", cap: float | None = None) -> nx.Graph:
    def _combine(wc, wr):
        if wc is None: return wr
        if wr is None: return wc
        if combine == "max":   w = max(wc, wr)
        elif combine == "sum": w = wc + wr
        elif combine == "mean":w = 0.5 * (wc + wr)
        elif combine == "contact": w = wc
        elif combine == "radius":  w = wr
        else: raise ValueError(f"Unknown combine='{combine}'")
        return min(w, cap) if cap is not None else w

    H = nx.Graph()
    H.add_nodes_from(set(contact_graph.nodes) | set(radius_graph.nodes))

    for u, v, d in radius_graph.edges(data=True):
        w = d.get(weight_attr, 1.0)
        if cap is not None: w = min(w, cap)
        H.add_edge(u, v, **{weight_attr: float(w)}, in_contact=False, in_radius=True, type="radius")

    for u, v, d in contact_graph.edges(data=True):
        wc = d.get(weight_attr, 1.0)
        if cap is not None: wc = min(wc, cap)
        if H.has_edge(u, v):
            wr = H[u][v][weight_attr]
            w   = _combine(wc, wr)
            H[u][v][weight_attr] = float(w)
            H[u][v]["in_contact"] = True
            H[u][v]["in_radius"]  = True
            H[u][v]["type"]       = "both"
        else:
            H.add_edge(u, v, **{weight_attr: float(wc)}, in_contact=True, in_radius=False, type="contact")
    return H


def calculate_adjacency_score(graph: nx.Graph, clusters: pd.Series) -> float:
    if graph.number_of_edges() == 0:
        return 0.0
    same_cluster_edges = 0
    for u, v in graph.edges():
        if clusters.iloc[u] == clusters.iloc[v]:
            same_cluster_edges += 1
    return same_cluster_edges / graph.number_of_edges()

def get_connectivities(adata, neighbors_key: str | None = None):
    """
    Return the CSR connectivities matrix for a given neighbors_key.
    Falls back to the default 'connectivities' if key_added wasn't used.
    """
    if neighbors_key:
        k = f"{neighbors_key}_connectivities"
        if k in adata.obsp:
            return adata.obsp[k].tocsr()
    # fallback to the default slot
    if "connectivities" in adata.obsp:
        return adata.obsp["connectivities"].tocsr()
    raise KeyError("No connectivities matrix found in .obsp")

def _row_stochastic(A: sparse.spmatrix) -> sparse.spmatrix:
    """Return row-stochastic version of a CSR matrix."""
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    row_sums = np.asarray(A.sum(axis=1)).ravel()
    row_sums[row_sums == 0.0] = 1.0
    D_inv = sparse.diags(1.0 / row_sums)
    return D_inv @ A

# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------
def ensure_nmf_rep(adata: sc.AnnData, rank: int, spatial_graph: nx.Graph) -> Tuple[str, str, str]:
    rep_key = f"X_nmf_{rank}"
    neigh_key = f"neighbors_nmf{rank}"
    recon_key = f"X_recon_{rank}"

    if rep_key not in adata.obsm:
        print(f"  ➜ Running spatial NMF rank={rank} …")
        model = NMF(n_components=rank, init="nndsvda", max_iter=500, random_state=RANDOM_STATE)
        W = model.fit_transform(adata.X)
        H = model.components_
        print("    - Smoothing NMF embedding using the provided spatial graph...")
        smoothed_W = smooth_W_with_graph(W, spatial_graph=spatial_graph, beta=0.6, steps=2, weight_attr="weight")
        adata.obsm[rep_key] = smoothed_W.astype(np.float32).copy()
        adata.uns[f'H_X_nmf_{rank}'] = H
        X_reconstructed = np.dot(smoothed_W, H)
        adata.obsm[recon_key] = X_reconstructed.copy()

    if neigh_key not in adata.uns:
        sc.pp.neighbors(
            adata,
            use_rep=rep_key,
            n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
            key_added=neigh_key,
        )
    return rep_key, neigh_key, recon_key


def run_sweep(adata: sc.AnnData) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    obsm_dict = {}
    if 'spatial' in adata.obsm:
        obsm_dict['spatial'] = adata.obsm['spatial'].copy()
    adata_layer = AnnData(
        X=adata.layers['lognorm'].copy(),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        obsm=obsm_dict
    )

    # Ensure spatial coords exist
    if adata_layer.obsm.get('spatial') is None:
        print("Loading spatial data before sweep...")
        xenium_cells = '/project/banerjee/MDACC_CholangioData/output-XETG00074__0024840__TMA_1__20240607__194742/cells.csv'
        cells = pd.read_csv(xenium_cells)
        coords = cells.set_index('cell_id')[['x_centroid', 'y_centroid']]
        common_barcodes = coords.index.intersection(adata_layer.obs.index)
        if len(common_barcodes) != adata_layer.n_obs:
            print(f"Warning: Only {len(common_barcodes)} of {adata_layer.n_obs} barcodes match.")
        coords = coords.loc[common_barcodes, ['x_centroid', 'y_centroid']].reindex(adata_layer.obs.index).values
        adata_layer.obsm['spatial'] = coords
        print(f"Loaded Xenium coordinates for {len(coords)} cells.")

    print("\n--- Pre-calculating Spatial Graphs ---")
    contact_graph = build_contact_graph(adata_layer, CONTACT_RADIUS, contact_scale=2.5)
    radius_graph = build_radius_graph(adata_layer, FALLBACK_RADIUS, radius_scale=1.0)
    hybrid_graph = build_hybrid_graph(contact_graph, radius_graph)
    print("------------------------------------\n")

    num_total_nodes = contact_graph.number_of_nodes()
    num_isolated_nodes = nx.number_of_isolates(contact_graph)
    num_contact_cells = num_total_nodes - num_isolated_nodes
    print(f"    - Of these, {num_contact_cells} cells have at least one contact.")

    A_spatial = nx.to_scipy_sparse_array(
        hybrid_graph,
        nodelist=range(adata_layer.n_obs),
        weight="weight",
        dtype=np.float64,
        format="csr",
    )
    # Normalize spatial adjacency
    if ROW_STOCH_NORMALIZE:
        A_spatial = _row_stochastic(A_spatial)
    else:
        if A_spatial.max() > 0:
            A_spatial = A_spatial / A_spatial.max()  # your original scale normalization

    for k in RANK_GRID:
        rep_key, neigh_key, recon_key = ensure_nmf_rep(adata_layer, k, spatial_graph=hybrid_graph)

        # --- NEW: feature adjacency from NMF neighbors (Scanpy connectivities) ---
        A_feature = get_connectivities(adata_layer, neighbors_key=neigh_key)
        if ROW_STOCH_NORMALIZE:
            A_feature = _row_stochastic(A_feature)
        else:
            maxv = A_feature.max()
            if maxv > 0:
                A_feature = A_feature / maxv
        # ------------------------------------------------------------------------

        for res in RES_GRID:
            cluster_key = f"leiden_nmf{k}_r{res}"

            if USE_HYBRID_FOR_LEIDEN:
                # --- NEW: mix the two graphs (Leiden will use both spaces) ---
                alpha = float(HYBRID_ALPHA)
                # Safety: coerce to CSR before arithmetic
                if not sparse.isspmatrix_csr(A_spatial):  A_s = A_spatial.tocsr()
                else:                                     A_s = A_spatial
                if not sparse.isspmatrix_csr(A_feature):  A_f = A_feature.tocsr()
                else:                                     A_f = A_feature

                A_mix = alpha * A_s + (1.0 - alpha) * A_f
                # Optional: renormalize (keeps weights in a comparable range)
                if ROW_STOCH_NORMALIZE and RENORMALIZE_MIX:
                    A_mix = _row_stochastic(A_mix)

                sc.tl.leiden(
                    adata_layer,
                    resolution=res,
                    adjacency=A_mix,
                    random_state=RANDOM_STATE,
                    key_added=cluster_key,
                    n_iterations=2,
                    directed=False,
                )
            elif USE_SPATIAL_FOR_LEIDEN:
                # Your existing spatial-only path (kept intact)
                sc.tl.leiden(
                    adata_layer,
                    resolution=res * 0.8,  # your original scaling
                    adjacency=A_spatial,
                    random_state=RANDOM_STATE,
                    key_added=cluster_key,
                    n_iterations=2,
                    directed=False,
                )
            else:
                # Your existing feature-only path using neighbors_key (unchanged)
                sc.tl.leiden(
                    adata_layer,
                    resolution=res,
                    neighbors_key=neigh_key,
                    random_state=RANDOM_STATE,
                    key_added=cluster_key,
                    n_iterations=2,
                    directed=False,
                )
            # Calculate spatial metrics
            clusters = adata_layer.obs[cluster_key]
            cluster_sizes = clusters.value_counts().sort_values()
            print(f"Cluster size summary (smallest 10):\n{cluster_sizes.head(10)}")

            adj_score_contact = calculate_adjacency_score(contact_graph, clusters)
            adj_score_radius = calculate_adjacency_score(radius_graph, clusters)
            adj_score_hybrid = calculate_adjacency_score(hybrid_graph, clusters)
            chaos_score = CHAOS(clusters, adata_layer.obsm['spatial'])
            y = adata_layer.obsm[rep_key][:, 0]
            moran_I = compute_morans_i(y, hybrid_graph)

            # Existing metrics (unchanged)
            sil = compute_silhouette_score(adata_layer, cluster_key, rep_key)
            # IMPORTANT: do not compute UMAP in Python; pass umap_key_base=None
            db_rep, db_umap = compute_db_index(adata_layer, cluster_key, rep_key, umap_key_base=None, neighbors_key=neigh_key)
            n_clust = adata_layer.obs[cluster_key].nunique()

            # ---- marker detection: safe wrapper ensures marker_genes exists for all clusters ----
            marker_genes = get_marker_genes_safe(adata_layer, cluster_key, min_cluster_size=MIN_CLUSTER_SIZE, n_genes=50)

            # The rest of your metric computations can remain unchanged
            overall_frac, _  = cluster_marker_fraction(adata_layer, cluster_key, marker_genes)
            weighted_frac = weighted_marker_fraction(adata_layer, cluster_key, marker_genes)
            mer, zero, _ = compute_marker_exclusion_rate(adata_layer, cluster_key, marker_genes)
            avg_spec, _ = compute_specificity(adata_layer, cluster_key, marker_genes)
            recon_metrics = compute_reconstruction_metrics(adata_layer, recon_key)

            all_genes = adata_layer.var_names.tolist()
            cluster_ids = sorted(marker_genes.keys(), key=int)
            top_genes = [marker_genes[c] for c in cluster_ids]
            try:
                enrich_score = compute_enrichment(adata_layer, cluster_key, all_genes, top_genes, gene_set_library='KEGG_2021_Human')
            except Exception as e:
                print(f"Warning: compute_enrichment failed: {e}. Setting Enrich=NaN.")
                enrich_score = np.nan

            rows.append(
                dict(
                    Rank=k,
                    Resolution=res,
                    n_clusters=n_clust,
                    Silhouette=sil,
                    DBI_NMF=db_rep,
                    Adj_Score_Contact=adj_score_contact,
                    Adj_Score_Radius=adj_score_radius,
                    Adj_Score_Hybrid=adj_score_hybrid,
                    Moran_I_Comp1=moran_I,
                    CHAOS=chaos_score,
                    MarkerFraction=overall_frac,
                    WeightedMF=weighted_frac,
                    MarkerExclusionRate=mer,
                    MarkerZeroRate=zero,
                    Specificity=avg_spec,
                    Reconstruction=(recon_metrics.get('reconstruction_error') if isinstance(recon_metrics, dict) else getattr(recon_metrics, 'reconstruction_error', np.nan)),
                    Expl_Var=(recon_metrics.get('explained_variance') if isinstance(recon_metrics, dict) else getattr(recon_metrics, 'explained_variance', np.nan)),
                    Enrich=enrich_score,
                    DBI_UMAP=db_umap,
                )
            )

            # Print a concise summary (handle possible NaNs)
            def _fmt(x):
                return f"{x:.3f}" if isinstance(x, (float, np.floating)) and not np.isnan(x) else "nan"
            print(
                f"NMF{k:>2} · res={res:<3} → k={n_clust:<3}  "
                f"Sil={_fmt(sil)} DB(NMF)={_fmt(db_rep)} | "
                f"Adj(Hyb)={_fmt(adj_score_hybrid)} Adj(Cont)={_fmt(adj_score_contact)} CHAOS={_fmt(chaos_score)} MoranI={_fmt(moran_I)}| "
                f"Frac={_fmt(overall_frac)} WtFrac={_fmt(weighted_frac)}"
            )

    # -------------------------
    # Save enriched AnnData for R (embeddings, neighbors, clusters, spatial)
    # -------------------------
    try:
        out_dir = pathlib.Path(__file__).parent / "SNMF_v4_hybrid_3_NEW_for_R"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_h5ad = out_dir / "snmf_rank_resolution_sweep_with_spatial_v4_hybrid_v2_mincl100_0.5_0.6_ALL_10_0.4for_R.h5ad"

        # Defensive: remove any UMAP keys that might have been created by helpers
        for key in list(adata_layer.obsm.keys()):
            if 'umap' in key.lower() or key.startswith('X_umap'):
                del adata_layer.obsm[key]

        # Add provenance and params
        adata_layer.uns['sweep_params'] = dict(
            rank_grid=RANK_GRID,
            res_grid=RES_GRID,
            n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
            hybrid_alpha=HYBRID_ALPHA,
            min_cluster_size=MIN_CLUSTER_SIZE,
        )

        print(f"Saving enriched AnnData (no UMAP) to: {out_h5ad}")
        adata_layer.write_h5ad(out_h5ad)
        print("Saved .h5ad successfully.")
    except Exception as e:
        print(f"Warning: failed to save h5ad: {e}")

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="NMF rank × resolution sweep")
    ap.add_argument("adata_path", type=pathlib.Path, help="Input .h5ad file")
    args = ap.parse_args()

    print("Loading AnnData …")
    adata = sc.read_h5ad(args.adata_path)

    print("Running NMF sweep …")
    df = run_sweep(adata)

    out_dir = pathlib.Path(__file__).parent / "SNMF_v4_hybrid_3_NEW_for_R"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "snmf_rank_resolution_sweep_with_spatial_v4_hybrid_v2_mincl100_0.5_0.6_ALL_for_R.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSweep complete → {out_csv}")


if __name__ == "__main__":
    main()
