# Structure Validity Tests for Object-Centric Agents

> Clean feature matching is not object-file identity.

This repository implements the Structure Validity Tests (SVT) framework to audit whether object-centric agents genuinely learn transferable identity binding, or whether apparent performance can be explained by retrieval, interpolation, feature matching, or hand-crafted rules.

## Core Principle

> High performance is not evidence of structure unless strong non-structural explanations fail.

## Current Mainline: v3.6 → v4 → v4.1 → v4.2

The current diagnostic chain establishes that clean feature matching can read out identity under benign conditions, but fails to constitute an object-file mechanism:

| Stage | Key Finding | What it shows |
|-------|-------------|---------------|
| **v3.6** Temporal-aligned feature key | FeatureOnly swap-only = 1.000 | Positive control: feature matching *can* solve identity |
| **v4** Minimal ObjectFile stress test | FeatureOnly conflict = 0.000; MinimalObjectFile conflict = 0.933 | Clean matching ≠ structural adjudication |
| **v4.1** Learned trajectory state | Swap-only 0.096 → 0.558; conflict 0.933 → 0.610 | Learned state improves normal performance but degrades conflict resolution |
| **v4.2** Conflict-first gate | Conflict resolution = 0.648; confidence calibration = 0.637 | Explicit conflict detection + adjudication outperforms weighted fusion |

### Primary Metric (2 objects)

For 2-object episodes, the primary metric is **`identity_swap_only`**: accuracy on swap episodes only, which avoids no-swap bias.

### Primary Metric (N > 2 objects)

For N > 2, use **permutation-level assignment accuracy** plus a **sanity audit** (label permutation check, object order randomization, and position ablation) to rule out artifactual solutions.

## Supplementary Checks: v4.3 / v5 / v5.1

These stages probe specific boundary conditions and scaling properties:

| Stage | Focus |
|-------|-------|
| **v4.3** Trajectory robustness | Approach detection, trajectory augmentation — negative result: diminishing returns on gate heuristics |
| **v5** Scaling up | Larger datasets, more complex dynamics — preliminary scaling check (not a full verification) |
| **v5.1** Scaling sanity audit | Conflict construction audit, permutation metric audit, continuous feature oracle — hardens the metric itself |

## Legacy Experiments: v2 k-NN Attack

The v2 experiments (k-NN retrieval attack) remain available as legacy demonstrations of non-structural baseline performance. They show that apparent structure can often be explained by retrieval and interpolation.

See `results/knn_attack/` and `results/knn_attack_v2/` for historical outputs.

## Project Structure

```
svt_agents/
  models/                    # Object-centric models
    object_file_models.py    # TrajectoryOnly, ConflictFirst, MinimalObjectFile
    learned_object_file.py   # Learned hybrid models
    counterfactual_object_file.py      # Counterfactual training + SMH
    probabilistic_structure_object_file.py  # Probabilistic structure selection
    slot_attention_model.py  # Published model adapters (experimental)
    rims_model.py            # Published model adapters (experimental)
    savi_model.py            # Published model adapters (experimental)
    dinosaur_model.py        # Published model adapters (experimental)
  scripts/                   # Experiment runners
    run_current_main.py      # Current main diagnostic (v4.2 + v5.1)
    run_svt_v4_2_conflict_first_object_file.py
    run_svt_v5_1_scaling_sanity_audit.py
    run_all_smoke.py         # Legacy v2 smoke test
  diagnostics/               # Diagnostic tools
    subspace_intervention.py # Neural probe + subspace removal
  data/                      # Dataset generation
    generate_nonlinear_feature_ood.py
  metrics/                   # Evaluation metrics
    identity_breakdown.py    # identity_swap_only, conflict resolution
    gated_svt_score.py       # Gated SVT score
  envs/                      # Simulation environments
    nonlinear_force_world.py
    interventions.py
  reports/                   # Technical reports
    SVT_v3_6_to_v4_2_Technical_Report.md
    Paper_Outline.md
    Reviewer_Hardening.md
  results/                   # Experiment outputs (not committed)
```

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Current Main Diagnostic

Run the current mainline experiments (v4.2 conflict-first + v5.1 scaling sanity):

```bash
python scripts/run_current_main.py
```

This will:
1. Run ConflictFirstObjectFile stress tests
2. Run scaling sanity audits
3. Output results to `results/svt_v4_2_conflict_first_object_file/` and `results/svt_v5_1_scaling_sanity_audit/`

> Note: The current main pipeline may take significant time to complete. A lightweight `smoke-current` script is planned (optional — run individual experiments above for faster iteration).

### Individual Experiments

```bash
# v4.2: Conflict-first ObjectFile
python scripts/run_svt_v4_2_conflict_first_object_file.py

# v5.1: Scaling sanity audit
python scripts/run_svt_v5_1_scaling_sanity_audit.py
```

### Legacy v2 Smoke Test

```bash
python scripts/run_all_smoke.py
```

Runs the legacy k-NN retrieval attack and oracle upper bound for quick sanity checking.

## Key Claims (and What We Do NOT Claim)

**We claim:**
- Clean feature matching can read out identity under benign conditions, but this does not constitute an object-file mechanism.
- Weighted hybrid fusion is insufficient for conflict resolution; explicit conflict detection + adjudication is necessary.
- The remaining bottleneck in learned models is trajectory-state quality, not gate design.

**We do NOT claim:**
- Object permanence is solved.
- SVT "passes" or any model is fully validated.
- These findings automatically generalize to complex visual environments without further testing.

## Citation

If you use this framework, please cite:

```bibtex
@misc{svt_object_centric,
  title={From Feature Matching to Object Files: Stress-Testing Identity Binding in Agents},
  author={[Authors]},
  year={2026},
  howpublished={\url{https://github.com/chleya/svt-object-centric}},
  note={Work in progress}
}
```

## License

MIT
