import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors
# --- NEW IMPORTS FOR MORAN'S I ---
import networkx as nx
from esda.moran import Moran
from pysal.lib import weights


def CHAOS(clusterlabel: pd.Series, location: np.ndarray) -> float:
    """
    Calculates a CHAOS score, quantifying spatial cluster compactness.
    A lower score indicates more compact, less "chaotic" clusters.
    """
    # Prepare data for processing
    matched_location = np.array(location)
    clusterlabel_arr = np.array(clusterlabel)

    # Remove any cells that don't have a cluster label
    is_na = pd.isna(clusterlabel)
    if np.any(is_na):
        clusterlabel_arr = clusterlabel_arr[~is_na]
        matched_location = matched_location[~is_na]

    # Standardize coordinates to prevent bias from scale
    matched_location = scale(matched_location)

    unique_labels = np.unique(clusterlabel_arr)
    total_dist = 0.0

    # Process one cluster at a time
    for k in unique_labels:
        # Isolate coordinates for the current cluster
        location_cluster = matched_location[clusterlabel_arr == k]

        # A cluster must have at least 2 points to measure distance
        if location_cluster.shape[0] <= 1:
            continue

        # Find the single nearest neighbor for all points in the cluster
        # n_neighbors=2 finds each point itself and its closest neighbor
        nn_model = NearestNeighbors(n_neighbors=2).fit(location_cluster)
        distances, _ = nn_model.kneighbors(location_cluster)

        # The first column of distances is always 0 (distance to self),
        # so we sum the second column (distance to the nearest neighbor).
        total_dist += np.sum(distances[:, 1])

    # Return the final score, normalized by the number of cells
    return total_dist / len(clusterlabel_arr)


def compute_morans_i(feature_vector: np.ndarray, graph: nx.Graph) -> float:
    """
    Calculates Moran's I to measure spatial autocorrelation of a feature.
    """
    if graph.number_of_edges() == 0:
        return 0.0

    # --- MODIFIED SECTION: Manually create the spatial weights object ---
    # This is a more fundamental method that works with older pysal versions.
    
    neighbors = {i: list(graph.neighbors(i)) for i in graph.nodes()}
    edge_weights = {i: [graph[i][j]['weight'] for j in neighbors[i]] for i in graph.nodes()}

    w = weights.W(neighbors, weights=edge_weights)
    # --- END OF MODIFICATION ---
    
    # Row-standardize the weights matrix, which is a standard practice
    w.transform = 'r'

    # Calculate and return the Moran's I statistic
    moran = Moran(feature_vector, w)
    return moran.I