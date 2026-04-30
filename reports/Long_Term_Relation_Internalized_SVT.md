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

Each relation should be treated as a first-class entity with its own state, uncertainty, update rule, and failure mode.

### 3.1 Relation-Specific Uncertainty

Current ObjectFile uses a single blanket confidence score. This is insufficient because:

- Feature uncertainty and trajectory uncertainty have different sources
- Occlusion uncertainty is about absence of evidence
- Conflict uncertainty is about contradictory evidence
- These require different inspection and recovery strategies

### 3.2 Inspection Design Space

| Variant | Description | Expected behavior |
|---------|-------------|-------------------|
| NoInspectionObjectFile | No inspection; always use default channel | Feature-hijacked under conflict |
| BlanketInspectionObjectFile | Single inspection score for all relations | Over-conservative; abstains too much |
| RelationSpecificInspectionObjectFile | Separate inspection per relation | Targeted; inspects only when specific relation is unreliable |
| OracleInspectionObjectFile | Oracle knows which relation is unreliable | Upper bound; shows ceiling of relation-specific inspection |

### 3.3 Key Hypothesis

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

| Relation-Internalization finding | Implication for SVT |
|--------------------------------|---------------------|
| Probe readability ≠ causal use | Subspace intervention test (v12) directly tests this |
| Edit-pressure training is false positive | Conflict-augmented training (v10) has same issue |
| Counterfactual training is strongest | Should be adapted for ObjectFile training (v13) |
| Structure states A/B/C/D | Directly applicable to identity encoding diagnosis |
| S4 differentiable graph substrate | Potential architecture for Relation-Specific ObjectFile |

## 7. What NOT to Do

- Do not implement long-term models now
- Do not add new experiments
- Only write planning and theoretical connections
- Keep current mainline (v4.2) stable
