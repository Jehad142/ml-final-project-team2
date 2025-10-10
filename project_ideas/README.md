# ML Project Ideas — Materials ML Pipeline

This document outlines ten machine learning project ideas for Chem 277B that leverage our interdisciplinary team’s strengths in chemistry, software engineering, systems analysis, and data science. Each idea is structured around a reproducible ML pipeline: EDA, Featurization, Modeling, Interpretation, and Reporting.

---

## 1. Electronic Classification from Crystal Structure

**Goal**: Predict whether a material is a conductor, semiconductor, or insulator using DFT-derived features.

**Pipeline Roles**:
- **EDA**: Analyze band gap distributions, class balance, and feature correlations.
- **Featurization**: Use Matminer and Pymatgen to extract composition and structure features.
- **Modeling**: Build a custom GNN using atomic graphs and symmetry-aware descriptors.
- **Interpretation**: Use SHAP and attention maps to explain predictions.
- **Reporting**: Validate predictions against known materials and literature.

---

## 2. Piezoelectric and Photoelectric Screening

**Goal**: Classify materials with piezoelectric or photoelectric potential using symmetry and electronic features.

**Pipeline Roles**:
- **EDA**: Explore dielectric constants, band gaps, and space groups.
- **Featurization**: Extract tensor features and symmetry indicators from JARVIS-DFT.
- **Modeling**: Train a multitask GNN or ensemble classifier.
- **Interpretation**: Visualize feature importance and candidate rankings.
- **Reporting**: Highlight promising materials for optoelectronic applications.

---

## 3. Crystal System Classification from CIF Files

**Goal**: Predict crystal system (e.g., cubic, hexagonal) from atomic coordinates.

**Pipeline Roles**:
- **EDA**: Analyze class distribution and lattice parameter ranges.
- **Featurization**: Convert CIFs to graphs; extract symmetry and bonding features.
- **Modeling**: Train a GNN or symmetry-aware classifier.
- **Interpretation**: Visualize decision boundaries and confusion matrices.
- **Reporting**: Compare predictions to known crystallographic assignments.

---

## 4. Spectral Fingerprinting with Vision Models

**Goal**: Classify IR, UV/Vis, or Raman spectra using CNNs or transformers.

**Pipeline Roles**:
- **EDA**: Normalize spectra, detect peaks, and visualize class clusters.
- **Featurization**: Convert spectra to image-like formats or embeddings.
- **Modeling**: Train a CNN or ViT on spectral fingerprints.
- **Interpretation**: Use Grad-CAM to highlight informative spectral regions.
- **Reporting**: Link spectral features to chemical functionality.

---

## 5. Predicting Toxicity from Composition and Structure

**Goal**: Classify materials by toxicity risk using elemental and structural features.

**Pipeline Roles**:
- **EDA**: Explore toxicity labels, elemental distributions, and feature correlations.
- **Featurization**: Use Matminer to extract electronegativity, oxidation states, and coordination.
- **Modeling**: Train a classifier (e.g., XGBoost or GNN).
- **Interpretation**: Use SHAP to identify toxicological drivers.
- **Reporting**: Validate predictions against known toxic compounds.

---

## 6. Multimodal ML for Material Property Prediction

**Goal**: Predict multiple properties (e.g., band gap, stability, toxicity) using structure, composition, and spectra.

**Pipeline Roles**:
- **EDA**: Analyze cross-modal correlations and missing data patterns.
- **Featurization**: Integrate tabular, graph, and spectral features.
- **Modeling**: Build a multitask GNN or transformer with modality-specific encoders.
- **Interpretation**: Use attention weights to assess modality contributions.
- **Reporting**: Highlight tradeoffs and cross-property insights.

---

## 7. ML-Guided Synthesis Planning

**Goal**: Predict synthesis conditions (temperature, pressure, precursors) from target material properties.

**Pipeline Roles**:
- **EDA**: Analyze synthesis metadata and property distributions.
- **Featurization**: Encode precursor text and structural features.
- **Modeling**: Train a sequence-to-property model or synthesis recommender.
- **Interpretation**: Visualize predicted conditions and compare to literature.
- **Reporting**: Propose synthesis routes for novel candidates.

---

## 8. Solubility and Reactivity Prediction

**Goal**: Predict aqueous solubility and acid/base reactivity from composition.

**Pipeline Roles**:
- **EDA**: Explore solubility ranges and reactivity class balance.
- **Featurization**: Use composition-based descriptors (e.g., ionic character, valence).
- **Modeling**: Train regression and classification models.
- **Interpretation**: Map predictions to chemical space.
- **Reporting**: Validate against known solubility and reactivity data.

---

## 9. Diagnostic Marker Prediction from Spectral Data

**Goal**: Predict diagnostic markers (e.g., metal toxicity) from lab spectra or compound features.

**Pipeline Roles**:
- **EDA**: Preprocess spectra and analyze marker distributions.
- **Featurization**: Combine spectral and structural features.
- **Modeling**: Train a CNN or hybrid model.
- **Interpretation**: Use saliency maps and SHAP for clinical relevance.
- **Reporting**: Compare predictions to diagnostic thresholds.

---

## 10. Material Stability Under Environmental Stress

**Goal**: Classify materials as stable/unstable under heat, moisture, or radiation.

**Pipeline Roles**:
- **EDA**: Explore stress annotations and thermodynamic features.
- **Featurization**: Extract formation energy, packing factor, and symmetry.
- **Modeling**: Train a classifier with stress-aware features.
- **Interpretation**: Identify structural traits linked to resilience.
- **Reporting**: Recommend candidates for harsh environments.

---

## Next Steps

- Select 2–3 candidate projects for proposal development  
- Assign milestone leads and GitHub issues per pipeline role  
- Scaffold starter notebooks and featurization scripts  
- Begin EDA and dataset validation


