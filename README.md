# hSNMF: Hybrid Spatially Regularized NMF for Spatial Transcriptomics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![ISBI 2026](https://img.shields.io/badge/Conference-ISBI%202026-d50000)](https://biomedicalimaging.org/2026/)

> **Official implementation of the paper accepted at IEEE ISBI 2026.**

---

## 🧬 Overview

**hSNMF** (Hybrid Spatially Regularized Non-negative Matrix Factorization) is a novel dimensionality reduction framework designed for high-resolution spatial transcriptomics data, such as **10x Xenium** and **Nanostring CosMx**.

Standard methods either ignore spatial context (like standard NMF) or over-smooth local details. hSNMF solves this by introducing a **hybrid graph regularization** term that mathematically bridges the gap between:
1.  **Transcriptomic Identity:** What genes a cell expresses.
2.  **Spatial Proximity:** Where the cell is located in the tissue.

This results in biologically interpretable spatial domains with sharper boundaries and higher coherence than existing state-of-the-art methods [cite paper: 36-38].

## ⚡ Key Innovations

Why use hSNMF over other tools?

* **🎯 Hybrid Spatial Awareness:** Unlike methods using a single fixed radius, hSNMF integrates short-range "contact" (<20μm) and long-range "context" (<80μm) graphs to capture multi-scale tissue architecture.
* **🔍 Interpretable by Design:** Built on NMF, providing non-negative latent factors that directly map to distinct biological programs or cell types (unlike PCA/UMAP).
* **🚀 Scalable to Millions:** Optimized sparse matrix operations allow it to handle massive datasets (e.g., 100k+ cells from Xenium runs) efficiently on standard hardware.
* **📈 Superior Metrics:** Demonstrates quantitatively better cluster compactness (CHAOS score) and spatial autocorrelation (Moran's I) compared to standard NMF and RASP.

---

## 🛠️ Methodology Snapshot

The core mechanism of hSNMF involves an iterative diffusion process guided by a hybrid spatial adjacency matrix.

<img width="2816" height="1536" alt="overview_hSNMF" src="https://github.com/user-attachments/assets/09e61fc6-955e-41a8-97c9-76fc7ebb0f7d" />

**Figure 1: Schematic overview of the hSNMF framework.** The methodology seamlessly integrates transcriptomic and spatial modalities across four key stages: (1) **Base Decomposition** of gene expression into latent factors via standard NMF; (2) **Hybrid Graph Construction** capturing multi-scale spatial architecture; (3) **Spatial Regularization** via iterative diffusion to smooth latent factors based on tissue proximity; and (4) **Dual-Graph Clustering** that combines smoothed transcriptomic features with spatial connectivity to identify spatially coherent biological domains.


**The four-step process (as seen above):**

1.  **Base Decomposition:** Decompose the gene expression matrix ($X$) using standard NMF into latent factors ($W$) and gene programs ($H$).
2.  **Hybrid Graph Construction:** Build two spatial graphs based on physical coordinates—a dense short-range "contact" graph and a sparse long-range "context" graph—merging them into a hybrid adjacency matrix ($A_s$).
3.  **Iterative Diffusion (Regularization):** Smooth the latent factors ($W$) using a diffusion operator derived from $A_s$. This encourages neighboring cells to share similar factor weights (Eq. 4 & 5 in paper).
4.  **Dual-Graph Clustering:** Combine the smoothed spatial factors with transcriptomic features to generate final, spatially coherent clusters using Leiden algortihm.

---
## 📦 Installation

### **Prerequisites**
* Python >= 3.10
* RAM: ~16GB (for typical Xenium datasets)

### **1. Clone the Repository**
```bash
git clone [https://github.com/ishtyaqmahmud/hSNMF.git](https://github.com/ishtyaqmahmud/hSNMF.git)
cd hSNMF
```
### 2. Create environment
We recommend using a virtual environment (Conda or venv).

```bash
conda create -n hsnmf_env python=3.10
conda activate hsnmf_env
```
### 3. Install Dependencies
All required high-level Python dependencies are listed in the requirements.txt file located at the repository root. Install them via pip:
```bash
# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```



## 📜 Citation
If you find this code or methodology useful in your research, please cite our ISBI 2026 paper:

Mahmud, M. I., Kochat, V., Satpati, S., Dwarampudi, J. M. R., Anzum, H., Rai, K., & Banerjee, T. (2026). hSNMF: Hybrid spatially regularized NMF for image-derived spatial transcriptomics. In Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI '26). IEEE.

```bibtex
@inproceedings{Mahmud_ISBI_2026_hSNMF,
  title={{hSNMF}: Hybrid Spatially Regularized {NMF} for Image-Derived Spatial Transcriptomics},
  author={Mahmud, Md. Ishtyaq and Kochat, Veena and Satpati, Suresh and Dwarampudi, Jagan Mohan Reddy and Anzum, Humaira and Rai, Kunal and Banerjee, Tania},
  booktitle={Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI '26)},
  year={2026},
  publisher={IEEE}
}
``` 
