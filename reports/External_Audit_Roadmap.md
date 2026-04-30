# External Audit Roadmap: SVT for Published Object-Centric Models

## Goal

Extend SVT from testing our own ObjectFile variants to auditing structural claims of published object-centric models.

## 1. Target Model Priority

| Priority | Model | Reason |
|----------|-------|--------|
| 1 | Slot Attention / SAVi-like | Most widely cited; slot structure is closest to object-file |
| 2 | MONet / SPACE | Attention-based segmentation; explicit object slots |
| 3 | DINOSAUR / object-centric encoder | Recent; strong reconstruction but untested identity binding |

## 2. Minimal External Audit Interface

All external models connect through a unified adapter:

```python
class ObjectCentricAdapter:
    def encode(self, observations):
        # Returns slots or object representations
        raise NotImplementedError

    def predict_identity(self, observed_positions, observed_features,
                         future_positions, future_features):
        # Returns assignment [B, N]
        raise NotImplementedError

    def predict_confidence(self, observed_positions, observed_features,
                           future_positions, future_features):
        # Optional: returns confidence [B, N]
        raise NotImplementedError
```

See `adapters/base_adapter.py` for the interface definition.

## 3. External Audit v0: One Model Only

Start with a Slot Attention-like lightweight baseline. Do not connect all published models at once.

The v0 adapter wraps the existing `models/slot_attention_model.py` and runs it through the SVT stress test pipeline.

## 4. SVT Stress Tests for External Models

| Test | What it reveals |
|------|----------------|
| Clean identity | Baseline: can the model assign identity at all? |
| Feature ablation | Does the model rely on features? |
| Occlusion without feature | Can the model persist identity without features? |
| Feature-trajectory conflict | Can the model adjudicate conflicting cues? |
| Confidence calibration | Does the model know when it doesn't know? |

## 5. Structural Fingerprint Output

Each model produces a structural fingerprint:

| Metric | What it measures |
|--------|-----------------|
| identity_swap_only | Identity accuracy on swap episodes |
| feature_dependency_score | Normal - shuffled feature accuracy |
| trajectory_dependency_score | Zero-feature accuracy |
| conflict_resolution | Accuracy under feature-trajectory conflict |
| occlusion_persistence | Accuracy under feature occlusion |
| confidence_calibration | Correct confidence - incorrect confidence |
| no_swap_bias_gap | Overall - swap-only accuracy |

## 6. Interpretation Patterns

| Pattern | Interpretation |
|---------|---------------|
| Clean high, conflict collapses | Feature-reader profile: reads features but lacks structural adjudication |
| Occlusion collapses | No object persistence: cannot maintain identity without features |
| Trajectory-only too strong | Trajectory shortcut: uses motion cues without binding |
| Conflict handled | Object-file-like structural bias |

## 7. Language Guidelines

- Do NOT say published models "lack intelligence" or "fail"
- Do NOT say "Slot Attention fails at identity binding"
- DO say: "Under SVT stress tests, the model shows a feature-reader-like profile"
- DO say: "The model demonstrates object-file-like structural bias under conflict"
- DO say: "The model exhibits a shortcut-like profile under occlusion"

## 8. What NOT to Do

- Do not download large pretrained models
- Do not train published models from scratch on new data
- Do not introduce complex visual datasets
- Do not break the current mainline (v4.2)
