# ⚗️ PolyVAE: Generative Deep Learning for Conductive Polymer Design

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Method](https://img.shields.io/badge/Method-Variational_Autoencoder-purple)
![Domain](https://img.shields.io/badge/Domain-Materials_Informatics-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b)

> A Variational Autoencoder (VAE) trained on 500 conductive polymers
> that generates novel candidates predicted to exceed known conductivity
> benchmarks — with Jacobian XAI to identify key molecular drivers.

---

## Key Results

| Model | R² | Notes |
|---|---|---|
| Ridge (raw descriptors) | ~0.0 | Linear baseline |
| GBM (raw descriptors) | ~0.0 | Non-linear but no compression |
| **GBM (VAE latent z)** | **0.83** | VAE enables downstream prediction |

**Generated:** 200 novel polymers · **Valid:** 196 (98%) · **Novel (>0.3):** 196

---

## Dashboard Tabs

| Tab | What It Shows |
|---|---|
| 📊 EDA | Conductivity by family, π-conjugation scatter, feature correlations |
| 🧠 Latent Space | VAE training curves (ELBO, recon, KL) + PCA latent space |
| 🧪 Generation | Training vs generated conductivity + novelty scatter |
| 📈 XAI | Permutation importance + model comparison |
| ⚡ Predictor | 9 sliders → instant conductivity prediction + gauge |

---

## 6 Polymer Families

| Family | Base log(σ) | π-Conjugation |
|---|---|---|
| PEDOT | 5.1 | 0.98 |
| Polythiophene | 4.2 | 0.95 |
| Polypyrrole | 3.8 | 0.88 |
| Polyaniline | 3.5 | 0.82 |
| Polyfluorene | 1.8 | 0.78 |
| Polycarbazole | 1.2 | 0.72 |

---

## 13 Molecular Descriptors

| Descriptor | Physical Meaning |
|---|---|
| π-Conjugation | Extended backbone conjugation (strongest driver) |
| Backbone Rigidity | Chain stiffness → crystallinity |
| Doping Affinity | Ease of charge carrier addition |
| HOMO-LUMO Gap | Electronic activation barrier |
| Ionization Potential | Energy to remove electron |
| Electron Affinity | Energy gained by adding electron |
| Charge Mobility | Speed of charge carrier hopping |
| Disorder Parameter | Structural defects that trap charges |
| Crystallinity | Ordered microstructure fraction |
| Dopant Concentration | Molar fraction of dopant added |
| Chain Length | Number of repeat units |
| Molecular Weight | Total chain mass |
| Side-Chain Polarity | Solvent interaction of pendant groups |

---

## Project Structure
```
polyvae/
├── app/
│   └── streamlit_app.py      ← Interactive dashboard
├── src/
│   └── polyvae_pipeline.py   ← Full pipeline (Stages 1–9)
├── data/
│   └── polymer_dataset.csv
├── results/
│   └── figures/
│       ├── fig1_eda.png
│       ├── fig2_training.png
│       ├── fig3_latent.png
│       ├── fig4_generation.png
│       ├── fig5_interpolation.png
│       ├── fig6_xai.png
│       ├── fig7_model_comparison.png
│       ├── fig8_ablation.png
│       └── fig9_landscape.png
├── README.md
├── requirements.txt
└── runtime.txt
```

---

## Quick Start
```bash
git clone https://github.com/roydipta-ux/polyvae
cd polyvae
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Physics Foundation

- **Marcus transport**: log(σ) ∝ π-conjugation + doping − disorder
- **Morgan fingerprints**: 64-bit circular structural fragments
- **VAE latent space**: L=8 dimensions = chemical design knobs
- **Reparameterisation trick**: z = μ + ε·σ, ε ~ N(0,I) — Kingma & Welling 2014
- **Dataset reference**: Dryad doi:10.5061/dryad.5ht3n + NeurIPS Open Polymer 2025

---

## The 9 Publication Figures

| Figure | Key Finding |
|---|---|
| fig1_eda | PEDOT highest σ, Polycarbazole lowest — matches literature |
| fig2_training | ELBO converges, KL healthy ~0.5 — no posterior collapse |
| fig3_latent | PCA shows family clusters — VAE learned chemical structure |
| fig4_generation | Generated distribution shifts right — surpasses training set |
| fig5_interpolation | Smooth PEDOT→Polycarbazole walk — valid latent geometry |
| fig6_xai | π-conjugation top driver — consistent with Marcus theory |
| fig7_model_comparison | GBM-latent R²=0.83 vs GBM-phys R²≈0 — VAE adds value |
| fig8_ablation | L=8 optimal — posterior collapse at β=4 detected |
| fig9_landscape | Top-5 generated polymers cluster in high-σ region |

---
