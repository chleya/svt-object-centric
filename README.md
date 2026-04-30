# SVT-v2: Against Spurious Structure

This project implements the Structure Validity Tests v2 (SVT-v2) framework to audit whether agents genuinely learn transferable structure, or whether apparent structure can be explained by retrieval, interpolation, data augmentation, leakage, underfitting, hand-crafted rules, or private communication protocols.

## Core Principle

> High performance is not evidence of structure unless strong non-structural explanations fail.

## Current Stage

The current stage focuses on k-NN Retrieval Attack:

1. First-pass absolute-output k-NN (v1)
2. Strong delta-output k-NN (v2)
3. Re-anchored normalized k-NN
4. Velocity-based k-NN
5. Permutation-consistent k-NN
6. Last-velocity heuristic baseline
7. Scale sweep over training set size

## Project Structure

```
svt_agents/
  configs/
    smoke.yaml          # Quick smoke test config
    svt_v2.yaml         # Full experiment config
  envs/
    motion_world.py     # 2D motion world simulation
    interventions.py    # Counterfactual & compositional interventions
    physics_oracle.py   # Oracle upper bound (perfect physics)
  data/
    generate_2d_motion.py  # Dataset generation script
  baselines/
    knn_retriever.py    # k-NN v1 (absolute output)
    knn_retriever_v2.py # k-NN v2 (delta output + re-anchoring)
  metrics/
    prediction_metrics.py   # MSE, skill score, normalized MSE
    identity_metrics.py     # Identity accuracy, skill over random
    gated_svt_score.py      # Gated SVTScore + old SMSS
  scripts/
    run_oracle_upper_bound.py   # Verify task solvability
    run_knn_attack.py           # Run k-NN v1 attack
    run_all_smoke.py            # One-click smoke test
  analysis/
    make_failure_fingerprint_map.py  # Generate fingerprint heatmap
  results/
    oracle/             # Oracle results
    knn_attack/         # k-NN v1 results
    knn_attack_v2/      # k-NN v2 results
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Smoke Test (Fast)

```bash
python scripts/run_all_smoke.py
```

This will:
1. Generate a small dataset (1000 train + 200 test per split)
2. Run Oracle upper bound
3. Run k-NN Retrieval Attack v1
4. Generate results in `results/`

### 3. Run Individual Steps

```bash
# Generate dataset
python data/generate_2d_motion.py --config configs/smoke.yaml

# Run Oracle
python scripts/run_oracle_upper_bound.py --config configs/smoke.yaml

# Run k-NN v1
python scripts/run_knn_attack.py --config configs/smoke.yaml

# Generate fingerprint map
python analysis/make_failure_fingerprint_map.py --input results/knn_attack/summary.csv
```

## Expected Outputs

- `results/oracle/oracle_upper_bound.json` — Oracle performance per split
- `results/knn_attack/summary.csv` — k-NN results table
- `results/knn_attack/failure_fingerprint.png` — Model × Gate heatmap

## Interpretation

If k-NN performs well, SVT-v1 contains retrieval shortcuts.
If k-NN fails clean prediction but passes identity, identity tests contain trajectory-similarity shortcuts.
If old SMSS is high while prediction is poor, old SMSS is invalid and gated scoring is required.
