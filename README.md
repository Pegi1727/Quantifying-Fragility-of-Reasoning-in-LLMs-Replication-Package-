# Quantifying the Fragility of Reasoning in Successive Generations of Large Language Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23777125.svg)](https://doi.org/10.5281/zenodo.23777125)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete replication package, empirical datasets, analysis scripts, and high-resolution figures for the research study **"Quantifying the Fragility of Reasoning in Successive Generations of Large Language Models"**.

---

## 📌 Quick Access & Links
- **Zenodo DOI (Data & Code Archive):** [10.5281/zenodo.23777125](https://doi.org/10.5281/zenodo.23777125)
- **High-Resolution Figures:** [View Figures Directory](./figures/)
- **Analysis Scripts & Notebooks:** [View Scripts Directory](./scripts/)
- **Empirical Datasets:** [View Data Directory](./data/)

---

## 📊 Core Framework & Key Results

The study establishes a mathematical framework to measure reasoning degradation in successive text generations using three core components:
1. **Reasoning Predictive Equity (RPE):** Tracks semantic fidelity via SBERT (`all-mpnet-base-v2`, cosine similarity scaled to $[0,1]$) against reference trajectories.
2. **Fragility Coefficient ($\lambda$):** Parameter tracking exponential decay rate: 
   $$\text{RPE}(g) = \text{RPE}_0 \cdot e^{-\lambda g}$$
3. **Critical Stability Horizon ($g^*$):** Threshold defined as 50% loss of semantic alignment:
   $$g^* = \frac{\ln(2)}{\lambda}$$

### Summary of Empirical Decay Parameters & Stability Horizons

| Model Scale | Fragility Coefficient ($\lambda$) | Coefficient of Determination ($R^2$) | Critical Stability Horizon ($g^*$) |
| :--- | :---: | :---: | :---: |
| **Llama-2-7B** | $0.22$ | $0.97$ | $3.1$ generations |
| **Llama-2-13B** | $0.16$ | $0.98$ | $4.3$ generations |
| **Llama-2-70B** | $0.10$ | $0.99$ | $6.9$ generations |

---

## 📈 Key Figures Overview

All high-resolution (300 DPI) publication-ready figures are located in the [`figures/`](./figures/) directory:
* [`graphical_abstract.png`](./figures/graphical_abstract.png) – High-impact visual summary of the methodology and core findings.
* [`rpe_trajectories_reported_vs_fit.png`](./figures/rpe_trajectories_reported_vs_fit.png) – Comparison of empirical RPE trajectories versus exponential fits.
* [`stability_horizon_bar.png`](./figures/stability_horizon_bar.png) – Comparative bar charts of the critical stability horizon ($g^*$) across model scales.
* [`reconstructed_params_comparison.png`](./figures/reconstructed_params_comparison.png) – Parameter comparison across successive model generations.
* [`raw_like_distribution_overview.png`](./figures/raw_like_distribution_overview.png) – Statistical distribution overview of raw-like records.

---

## 🔬 Conclusions

1. **Exponential Reasoning Degradation:** Reasoning capability in large language models decays exponentially over successive self-generated rollouts rather than remaining stable.
2. **Model Scale Mitigates but Does Not Halt Fragility:** While larger parameter scales (e.g., Llama-2-70B with $g^* = 6.9$) substantially delay semantic drift compared to smaller counterparts (e.g., Llama-2-7B with $g^* = 3.1$), scaling alone cannot entirely prevent eventual reasoning collapse.
3. **Predictive Equity as a Diagnostic Tool:** RPE and the critical stability horizon provide robust quantitative metrics for auditing multi-step reasoning reliability in autoregressive architectures.

---

## 📂 Repository Structure
```text
├── data/                      # Empirical CSV datasets, Bayesian HDI records, and reconstructed tables
├── figures/                   # High-resolution (300 DPI) publication figures and graphical abstract
├── scripts/                   # Jupyter notebooks and Python scripts for replication
└── paper/                     # Manuscript documentation and compressed drafts
---------------------
bibtex
@software{merrikhi_2026_fragility,
  author       = {Merrikhi, Pegah},
  title        = {Replication Package and Empirical Datasets for: Quantifying the Fragility of Reasoning in Successive Generations of Large Language Models},
  month        = {October},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1.00},
  doi          = {10.5281/zenodo.23777125},
  url          = {https://doi.org/10.5281/zenodo.23777125}
}
