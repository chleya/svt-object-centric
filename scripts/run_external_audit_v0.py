"""
External Audit v0: Placeholder script

This script will run SVT stress tests on one external object-centric model
(currently: Slot Attention-like baseline).

Current status: PLACEHOLDER — prints expected inputs/outputs, does not
run a full model.

Expected inputs:
  - observed_positions: [B, T_obs, N, 2]
  - observed_features:  [B, T_obs, N, feature_dim] or [B, N, feature_dim]
  - future_positions:   [B, T_pred, N, 2]
  - future_features:    [B, T_pred, N, feature_dim] or [B, N, feature_dim]
  - identity_labels:    [B, N]

Expected outputs (structural fingerprint):
  - identity_swap_only
  - feature_dependency_score
  - trajectory_dependency_score
  - conflict_resolution
  - occlusion_persistence
  - confidence_calibration
  - no_swap_bias_gap
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 70)
    print("External Audit v0 — PLACEHOLDER")
    print("=" * 70)
    print()
    print("TODO: Implement full external audit pipeline")
    print()
    print("Expected inputs:")
    print("  observed_positions: [B, T_obs, N, 2]")
    print("  observed_features:  [B, T_obs, N, feature_dim]")
    print("  future_positions:   [B, T_pred, N, 2]")
    print("  future_features:    [B, T_pred, N, feature_dim]")
    print("  identity_labels:    [B, N]")
    print()
    print("Expected outputs (structural fingerprint):")
    print("  identity_swap_only")
    print("  feature_dependency_score")
    print("  trajectory_dependency_score")
    print("  conflict_resolution")
    print("  occlusion_persistence")
    print("  confidence_calibration")
    print("  no_swap_bias_gap")
    print()
    print("Target model: Slot Attention-like baseline")
    print("Adapter: adapters/base_adapter.py → SlotAttentionAdapter")
    print()
    print("See reports/External_Audit_Roadmap.md for full plan.")


if __name__ == "__main__":
    main()
