# Kaggle — Enveda CASMI26 Molecule Identification from Mass Spectra

Starter & competitive retrieval-hybrid workspace for the **[Enveda CASMI26 — Molecule Identification from MS/MS Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)** competition on Kaggle.

> **Action Needed Before Pushing / Downloading Data:** Competition rules must be manually accepted on the Kaggle Web UI ([`kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra`](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)) before attaching competition data or submitting via CLI (`403 Forbidden` until joined).

---

## Competition Contract & Submission Format

- **Data Files:**
  - `train.parquet`: Reference & training tandem mass spectra (`ingest_lib`, `inchikey14`, `normalized_smiles`, `precursor_mz`, `ms2_mzs`, `ms2_normalized_intensities`).
  - `test.parquet`: Unknown spectra grouped by `molecule_id` (`precursor_mz`, `ms2_mzs`, `ms2_normalized_intensities`).
  - `sample_submission.csv`: Target order & row schema (`molecule_id`, `smiles`).
- **Submission Format:** `submission.csv` with columns `['molecule_id', 'smiles']`, containing up to **25 semicolon `;`-separated canonical SMILES** per molecule, **deduplicated strictly on the 14-character InChIKey block (`inchikey14`)**, with `'CCO'` fallback if empty.

---

## Retrieval + De Novo Hybrid Baseline (`baseline_casmi26_retrieval_hybrid.ipynb`)

Rather than spending 2+ hours training a from-scratch autoregressive Transformer for a single epoch before obtaining a valid baseline, this workspace pairs **ultra-fast SciPy sparse mass-window MS/MS spectral retrieval** with an **ensemble blend hook for De Novo Transformer candidates**:

1. **Stein-Scott Fragment + Neutral-Loss Sparse L2 Representation (`0.05 Da` bins):**
   - Encodes up to `128` strongest MS/MS peaks into L2-normalized CSR matrices for both fragment `m/z` ($w = I^{0.5} \cdot m^{1.25}$) and neutral losses $\Delta m = \text{precursor\_mz} - m/z$.
2. **Precursor-Sorted Multi-Tier Window Lookup (`±0.015 Da -> ±0.05 Da -> ±0.25 Da -> ±2.0 Da`):**
   - Uses `np.searchsorted` $O(\log N)$ precursor mass slicing followed by vectorized CSR cosine similarity across candidate spectra.
3. **Multi-Spectrum Replicate Voting & `InChIKey14` Deduplication:**
   - Pools similarity scores across all replicate spectra sharing the same `molecule_id`, ranks unique `inchikey14` scaffolds descending, and formats `;`-separated canonical SMILES (`<= 25` guesses).
4. **Dual Validation & De Novo Blend Hook:**
   - Evaluates both **replicate-spectrum exact-match recall** (when structure is in library) and **structure-disjoint scaffold analog `Tanimoto@1 / Max@25`** via `RDKit`, plus merges optional `denovo_predictions.csv` via reciprocal rank fusion (RRF).

---

## Local Verification & Build

```powershell
& 'G:\ReplitProjects\enterprise-banking-ds-platform\.venv\Scripts\python.exe' build_baseline_nb.py
```

Generates `kaggle_notebook/baseline_casmi26_retrieval_hybrid.ipynb`, runs AST compilation checks across all cells, and executes a local synthetic MS/MS retrieval + multi-spectrum voting + InChIKey14 deduplication unit test.
