# Long-Term Roadmap: Relation-Internalized ObjectFile and General SVT Benchmark

## 1. Overall Framework

| Concept | Role |
|---------|------|
| SVT | Structure claim discriminator — tests whether structural claims are genuine |
| ObjectFile | First minimal structural sample — demonstrates feature-trajectory adjudication |
| Relation Internalization | Explains how external structure becomes internally updatable, adjudicable, and transferable |

The long-term goal is to generalize SVT from testing object identity to testing any structural claim about agent cognition.

## 2. Relations Already Present in Current ObjectFile

The current ObjectFile implicitly contains several relations:

| Relation | Current implementation | Limitation |
|----------|----------------------|------------|
| Feature-identity relation | Feature matching channel | No uncertainty; binary match/no-match |
| Trajectory-continuity relation | Trajectory prediction channel | OOD generalization bottleneck |
| Occlusion-persistence relation | Zero-feature fallback | Rule-based, not learned |
| Conflict-resolution relation | Margin-gated conflict detection | Heuristic, not principled |
| Confidence relation | Margin-based confidence | Fragile; not calibrated uncertainty |

Each relation has:
- A **state** (what is the current estimate?)
- An **uncertainty** (how reliable is this estimate?)
- An **update rule** (how does new evidence change the estimate?)
- An **inspection trigger** (when should this relation be re-examined?)
- A **failure mode** (how does this relation break down?)

## 3. Next Stage: Relation-Specific ObjectFile

### 3.0 v18 Finding: Training Signal Was the Bottleneck, Not Architecture

The v18 experiment (DualPathwayObjectFile with corrected conflict training) produced a critical positive result: conditional identity binding IS achievable with the correct training signal. Key findings:

- **v16/v17 bug**: Conflict augmentation swapped features AND identity labels together, training the model to follow swapped features under conflict. This is the OPPOSITE of conditional binding.
- **v18 fix**: Keep identity labels unchanged when features are swapped. This trains the model to follow trajectory under conflict.
- **Result**: DualPath_pconf02 achieves conflict resolution = 0.879, clean accuracy = 0.879, State D (causal).
- **Architecture**: Dual independent scorers (feature + trajectory) with agreement-based switching.
- **Agreement detection**: Clean agreement 95%, conflict disagreement 99% — reliable conflict detection without explicit training.

**Implication**: The substrate ladder needs revision:
- S1 (Flat predictor): MLP binding, no conditional capacity
- S2 (Flat + edit pressure): MLP + CF training, destroys identity
- S4 (Differentiable graph): State D but no conditional binding
- **S5 (Dual pathway + corrected training)**: State D + conditional binding (0.879)

The remaining bottleneck is trajectory scorer quality (84-88%), limited by OOD trajectory prediction.

### 3.1 v14 Finding: MLP Binding Cannot Do Conditional Adjudication (Superseded by v18)

The v14 experiment (counterfactual training with clean masking) produced a critical negative result: counterfactual training completely destroys identity encoding in MLP-based binding networks. This means:

- MLP pairwise matching (current ObjectFile) corresponds to R4 substrate S1/S2
- S1/S2 cannot achieve conditional identity binding ("follow feature when it agrees, follow trajectory when it conflicts")
- The next architectural step requires S3 (relation slot) or S4 (differentiable graph)

**Note**: v18 shows that the issue was not purely architectural — the training signal was also wrong. With corrected training, even a simpler dual-pathway architecture achieves conditional binding.

**Implication**: Before implementing relation-specific inspection, we need BOTH a substrate that can represent conditional dependencies AND a training signal that correctly specifies which signal to follow under conflict.

### 3.2 Relations Already Present in Current ObjectFile

Current ObjectFile uses a single blanket confidence score. This is insufficient because:

- Feature uncertainty and trajectory uncertainty have different sources
- Occlusion uncertainty is about absence of evidence
- Conflict uncertainty is about contradictory evidence
- These require different inspection and recovery strategies

### 3.3 Inspection Design Space

| Variant | Description | Expected behavior |
|---------|-------------|-------------------|
| NoInspectionObjectFile | No inspection; always use default channel | Feature-hijacked under conflict |
| BlanketInspectionObjectFile | Single inspection score for all relations | Over-conservative; abstains too much |
| RelationSpecificInspectionObjectFile | Separate inspection per relation | Targeted; inspects only when specific relation is unreliable |
| OracleInspectionObjectFile | Oracle knows which relation is unreliable | Upper bound; shows ceiling of relation-specific inspection |

### 3.4 Key Hypothesis

Relation-specific inspection outperforms blanket inspection because:
- Different relations fail for different reasons
- Blanket inspection cannot distinguish "feature is noisy" from "feature and trajectory conflict"
- Relation-specific inspection can trigger targeted recovery (e.g., fall back to trajectory when feature is noisy, but flag conflict when both are reliable but disagree)

## 4. General SVT Benchmark

Beyond object identity, SVT can test any structural claim:

| Domain | Structural claim | SVT stress test |
|--------|-----------------|-----------------|
| Object identity | "The model maintains object identity" | Feature-trajectory conflict, occlusion |
| Relation reasoning | "The model learns relational structure" | Relation ablation, counterfactual intervention |
| Causal intervention | "The model understands causation" | Do-intervention, confound removal |
| Tool use | "The model understands tool-object relations" | Tool substitution, tool removal |
| Graph structure | "The model represents graph topology" | Edge perturbation, node removal |
| Latent reasoning | "The model reasons in latent space" | Chain length variation, shortcut injection |
| Communication | "Symbols carry meaning" | Symbol permutation, channel noise |

## 5. Core Principles (for any structural claim)

For any structural claim, SVT asks:

1. **Does it exceed strong memory baselines?** — Can the structure be explained by retrieval?
2. **Does perturbing structure content degrade performance?** — Is the structure causally used?
3. **Is it more stable than memory under capacity constraints?** — Does the structure generalize?
4. **Can wrong priors be overturned by feedback?** — Is the structure updatable?
5. **Does confidence correspond to utility?** — Is the uncertainty calibrated?
6. **Does it maintain function under OOD / conflict / intervention?** — Is the structure robust?

## 6. Connection to Relation-Internalization Program

The Relation-Internalization program (F:\relation-internalization-program) provides complementary findings:

| Relation-Internalization finding | Implication for SVT | v18 status |
|--------------------------------|---------------------|------------|
| Probe readability != causal use | Subspace intervention test (v12) directly tests this | Confirmed: State D achieved |
| Edit-pressure training is false positive | Conflict-augmented training (v10) has same issue | Confirmed: wrong training signal was the bug |
| Counterfactual training is strongest | Should be adapted for ObjectFile training (v13) | Partially: corrected conflict training works better |
| Structure states A/B/C/D | Directly applicable to identity encoding diagnosis | v18 achieves State D + conditional binding |
| S4 differentiable graph substrate | Potential architecture for Relation-Specific ObjectFile | S4 alone insufficient; S5 (dual pathway) needed |

### 6.1 Revised Substrate Ladder

| Substrate | Architecture | Conditional Binding | Conflict Resolution |
|-----------|-------------|--------------------|--------------------|
| S1 (Flat predictor) | MLP binding | No | 0.000 |
| S2 (Flat + edit pressure) | MLP + CF training | No (destroys identity) | 0.000 |
| S4 (Differentiable graph) | GraphObjectFile | No (State D only) | 0.000 |
| S4 + wrong training | GatedGraphObjectFile | No (training bug) | 0.000 |
| **S5 (Dual pathway + corrected training)** | **DualPathwayObjectFile** | **Yes** | **0.879** |

Key insight: S4 structure is necessary for State D but not sufficient for conditional binding. The training signal (which signal to follow under conflict) is equally important. S5 = S4-level structure + corrected training signal + explicit dual pathway.

## 7. What NOT to Do

- Do not implement long-term models now
- Do not add new experiments
- Only write planning and theoretical connections
- Keep current mainline (v4.2) stable
