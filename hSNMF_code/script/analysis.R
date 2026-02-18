
#!/usr/bin/env Rscript

# ------------------------------
# UMAP PLOTTING WITH CLUSTER COLORS
# ------------------------------

# Tell reticulate which Python to use
Sys.setenv(RETICULATE_PYTHON = "/project/banerjee/ishtyaq/genomic_ishtyaq_R_env/bin/python")

library(reticulate)
use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)

library(uwot)
library(ggplot2)

# ------------------------------
# 1. Load your h5ad data
# ------------------------------
h5ad_path <- "**/snmf_rank_resolution_sweep_with_spatial_v4_hybrid_v2_mincl100_0.5_0.6_ALL_10_0.4for_R.h5ad"

anndata <- import("anndata")
adata <- anndata$read_h5ad(h5ad_path)
# --- ADD THIS LINE TO FIND YOUR KEYS ---
cat("Available keys in adata.obs:", paste(adata$obs_keys(), collapse=", "), "\n")
# ----------------------------------------

# ------------------------------
# 2. Choose embedding and clusters
# ------------------------------
# Example: use NMF rank = 15
nmf_key <- "X_nmf_10"   # <---- CHANGE THIS to your desired NMF rank


# Specify the key for your cluster labels from adata.obs
cluster_key <- "leiden_nmf10_r0.4"  

# Extract NMF matrix
nmf_mat <- py_to_r(adata$obsm[[nmf_key]])
nmf_mat <- as.matrix(nmf_mat)

# Extract cluster labels
clusters <- py_to_r(adata$obs[[cluster_key]])

# ------------------------------
# 3. Compute UMAP
# ------------------------------
set.seed(0)
umap_coords <- uwot::umap(
  nmf_mat,
  n_neighbors = 15,
  min_dist = 0.4,
  metric = "euclidean"
)

# Create data frame with UMAP coordinates AND clusters
df <- data.frame(
  UMAP1 = umap_coords[,1],
  UMAP2 = umap_coords[,2],
  cluster_label = clusters # Add clusters as a new column
)

# ------------------------------
# 4. Plot UMAP (with colors)
# ------------------------------
p <- ggplot(df, aes(UMAP1, UMAP2, color = as.factor(cluster_label))) +
  geom_point(size = 0.4, alpha = 0.7) +
  theme_minimal(base_size = 14) +
  coord_equal() +
  ggtitle(paste("UMAP from", nmf_key)) +
  labs(color = "Cluster") + # Adds a legend title
  guides(color = guide_legend(override.aes = list(size = 3))) # Makes legend dots bigger

# --- MODIFIED SAVE PATH ---
# 1. Get the directory path from the h5ad file
output_dir <- dirname(h5ad_path)

# 2. Create the desired filename
png_filename <- paste0("umap_color_", nmf_key, "_", cluster_key, ".png")

# 3. Combine them into the full save path
png_path <- file.path(output_dir, png_filename)
# --- END MODIFICATION ---

ggsave(png_path, p, width = 7.5, height = 6, dpi = 300)

cat("Saved UMAP →", png_path, "\n")