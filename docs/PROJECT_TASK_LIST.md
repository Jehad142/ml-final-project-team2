# Project Task List

Tasks Staged by ML pipeline.

---

## 1. Data Acquisition & Filtering
- [X] Download datasets from JARVIS-DFT (3D data set) and NOMAD2018 (Filtered data set - discuss for presentation if time allows)
- [X] Apply filters:
  - [X] Band gap > 3.0 eV (transparency)
  - [X] Formation energy < 0.5 eV/atom (stability)
  - [X] Exclude toxic or non-biocompatible elements
- [X] Document filtering logic in a reproducible notebook - internal testing required
- [ ] Save and version filtered datasets (`filtered_materials.csv`) with metadata *manipulation of features (conv. numeric or categorical) & tensors (flatened)

---

## 2. Exploratory Data Analysis (EDA)
- [X] Visualize distributions of key properties (band gap, dielectric constant, symmetry class) 
- [X] Identify correlations between features and targets *use baseline
- [ ] Generate summary statistics and feature importance plots *thumbnails in doc with appendix (overleaf) [
- [ ] Share annotated notebooks (`eda_report.ipynb`) for team-wide reuse

---

## 3. Feature Engineering & Representation
- [X] Extract structural features (lattice parameters, symmetry, coordination)
- [X] Extract electronic features (band gap, DOS, dielectric constant)
- [X] Extract compositional features (atomic number, electronegativity, oxidation states)
- [X] Construct graph representations (nodes: atomic descriptors, edges: bond distances)
- [ ] Save feature matrices (`features.pkl`) and graph objects (`graphs.pt`) with schema documentation

---

## 4. Baseline Modeling 
- [X] Train Random Forest model
- [X] Train XGBoost model
- [X] Predict band gap and dielectric constant using structured features
- [ ] Benchmark performance (MAE, RMSE, R², F1-score)
- [ ] Save trained models and inference scripts ** See cleaned version as example / internal reproduction and testing

---

## 5. Advanced Modeling
- [ ] Implement and train GNNs (CGCNN, SchNet, MatGL)
- [ ] Implement and train CNNs (spectral data)
- [ ] Implement and train Transformers (e.g., MatFormer for composition)
- [ ] Apply multitask learning (band gap, stability, dielectric constant, toxicity)
- [ ] Save adapter weights and model checkpoints **REPLACE with pruning for fine tuning

---

## 6. Interpretation & Scoring
- [ ] Apply SHAP values for feature importance
- [ ] Generate attention maps for Transformers
- [ ] Perform node/edge saliency analysis for GNNs
- [ ] Cluster materials into families
- [X] Develop composite scoring rubric (transparency + conductivity + biocompatibility)
- [ ] Rank candidate materials and document rationale

---

## 7. DFT Validation
- [ ] Select top-ranked candidates for DFT calculations
- [ ] Run DFT simulations and collect results
- [ ] Compare predicted vs. DFT-derived properties
- [ ] Benchmark against literature
- [ ] Update candidate list and feed results back into the pipeline

---

## 8. Final Reporting & Reproducibility
- [ ] Maintain GitHub repository with versioned code, data, and models
- [ ] Add environment files (`requirements.txt`, `environment.yml`)
- [ ] Write modular README files for each stage
- [ ] Assemble final report and presentation slides *
- [ ] Ensure reproducibility from raw data to results
- [ ] Conduct peer reviews and internal validation

