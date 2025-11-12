# Project Task List

Tasks Staged by ML pipeline.

---

## 1. Data Acquisition & Filtering
- [ ] Download datasets from JARVIS-DFT, Materials Project, and OQMD
- [ ] Apply filters:
  - [ ] Band gap > 3.0 eV (transparency)
  - [ ] Formation energy < 0.5 eV/atom (stability)
  - [ ] Exclude toxic or non-biocompatible elements
- [ ] Document filtering logic in a reproducible notebook
- [ ] Save and version filtered datasets (`filtered_materials.csv`) with metadata

---

## 2. Exploratory Data Analysis (EDA)
- [ ] Visualize distributions of key properties (band gap, dielectric constant, symmetry class)
- [ ] Identify correlations between features and targets
- [ ] Generate summary statistics and feature importance plots
- [ ] Share annotated notebooks (`eda_report.ipynb`) for team-wide reuse

---

## 3. Feature Engineering & Representation
- [ ] Extract structural features (lattice parameters, symmetry, coordination)
- [ ] Extract electronic features (band gap, DOS, dielectric constant)
- [ ] Extract compositional features (atomic number, electronegativity, oxidation states)
- [ ] Construct graph representations (nodes: atomic descriptors, edges: bond distances)
- [ ] Save feature matrices (`features.pkl`) and graph objects (`graphs.pt`) with schema documentation

---

## 4. Baseline Modeling
- [ ] Train Random Forest model
- [ ] Train XGBoost model
- [ ] Predict band gap and dielectric constant using structured features
- [ ] Benchmark performance (MAE, RMSE, R², F1-score)
- [ ] Save trained models and inference scripts

---

## 5. Advanced Modeling
- [ ] Implement and train GNNs (CGCNN, SchNet, MatGL)
- [ ] Implement and train CNNs (spectral data)
- [ ] Implement and train Transformers (e.g., MatFormer for composition)
- [ ] Apply multitask learning (band gap, stability, dielectric constant, toxicity)
- [ ] Integrate LoRA-style adapters for efficient fine-tuning
- [ ] Save adapter weights and model checkpoints

---

## 6. Interpretation & Scoring
- [ ] Apply SHAP values for feature importance
- [ ] Generate attention maps for Transformers
- [ ] Perform node/edge saliency analysis for GNNs
- [ ] Cluster materials into families
- [ ] Develop composite scoring rubric (transparency + conductivity + biocompatibility)
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
- [ ] Assemble final report and presentation slides
- [ ] Ensure reproducibility from raw data to results
- [ ] Conduct peer reviews and internal validation

