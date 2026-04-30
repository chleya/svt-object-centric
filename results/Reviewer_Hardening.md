# Reviewer Hardening: Anticipated Objections and Responses

## Objection 1: "This is just a toy problem. 2 objects in 2D with one-hot features is trivial."

**Severity**: HIGH — This is the most likely and most damaging objection.

**Response Strategy**:

1. **Reframe the contribution**: The contribution is not solving object permanence in a complex environment. It is establishing a structural discriminative chain — a method for determining whether a model's identity binding is genuine or superficial. The toy environment is a *controlled setting* that allows clean causal attribution, not a limitation.

2. **Analogy to physics**: Galileo's inclined plane was a toy setup, but it established the law of inertia. The value of controlled toy experiments is in the clarity of the diagnostic, not the complexity of the setup.

3. **Explicit acknowledgment**: We explicitly state in Limitations that this is a 2D toy world with simplified features. We do not claim object permanence is solved.

4. **Constructive framing**: The stress tests (feature ablation, conflict, occlusion) are *generalizable* — they can be applied to any environment and any model. The specific numbers will change, but the structural diagnostic chain (clean matching ≠ object-file) is likely to hold.

**Key sentence for paper**: "We do not claim that object permanence is solved in this toy environment. We claim that the diagnostic chain — clean feature matching can read out identity but fails under conflict, while structural adjudication maintains correct bias — is a generalizable finding that should be tested in more complex settings."

---

## Objection 2: "The ObjectFile is just a rule-based system. Of course it handles conflict better — you hard-coded the conflict resolution."

**Severity**: HIGH — This challenges whether ObjectFile's conflict resolution is a genuine finding or a tautology.

**Response Strategy**:

1. **Acknowledge the rule-based nature**: Yes, ObjectFile is rule-based. That's the point. It's a *minimal structural unit* for testing whether maintaining separate feature and trajectory channels with conflict adjudication is a necessary structure.

2. **The finding is not that ObjectFile works, but that FeatureOnly/Hybrid fail**: The core finding is not "ObjectFile is great" but "FeatureOnly and Hybrid are structurally deficient." ObjectFile serves as an existence proof that a mechanism with correct structural bias can resolve conflicts, even if its absolute performance is low.

3. **The conflict resolution is not hard-coded**: ObjectFile does not have a "when conflict, choose trajectory" rule. It has separate feature matching and trajectory matching, and the conflict arises naturally when they disagree. The v4 ObjectFile uses simple nearest-neighbor matching for both channels — no hard-coded conflict resolution logic.

4. **v4.1 and v4.2 show the difficulty**: If conflict resolution were trivially hard-coded, v4.1's confidence-based weighting wouldn't have failed, and v4.2's margin-gated strategy wouldn't have been necessary. The progression from v4→v4.1→v4.2 shows that even with explicit conflict detection, correct adjudication is non-trivial.

**Key sentence**: "ObjectFile's conflict resolution is not a tautology — it is an existence proof that structural bias toward trajectory continuity can resist feature hijacking, a property that learned models (FeatureOnly, Hybrid) lack."

---

## Objection 3: "The feature-trajectory conflict test is artificial. In real scenarios, features don't get 'flipped'."

**Severity**: MEDIUM — The conflict test is a deliberate adversarial manipulation.

**Response Strategy**:

1. **Purpose of stress testing**: Stress tests are designed to be extreme. The feature-trajectory conflict is the identity-binding equivalent of an adversarial attack — it tests the *structure* of the mechanism, not its typical performance.

2. **Real-world analogs**: Feature corruption happens in practice: sensor malfunction, adversarial perturbation, domain shift, lighting changes that alter appearance features. The conflict test is a clean abstraction of "what happens when the feature signal is wrong?"

3. **The finding generalizes beyond the specific manipulation**: The key finding is not "flipped features break FeatureOnly" but "FeatureOnly has no mechanism to detect or recover from feature errors." This structural deficiency exists regardless of how the feature error arises.

4. **Occlusion test as non-adversarial complement**: The occlusion without feature test is not adversarial — it's a natural scenario. FeatureOnly also fails this test (0% at full occlusion), confirming the structural deficiency is not specific to adversarial manipulation.

**Key sentence**: "The feature-trajectory conflict is a diagnostic probe, not a realistic scenario. Its value is in revealing that FeatureOnly has no error-recovery mechanism — a structural deficiency that would manifest whenever features are unreliable, regardless of the cause."

---

## Objection 4: "The swap-only identity metric is too narrow. You should report overall accuracy, precision, recall, etc."

**Severity**: MEDIUM — Metric choice affects interpretation.

**Response Strategy**:

1. **Why swap-only**: In a dataset with 50%+ no-swap episodes, overall accuracy is inflated by the no-swap majority. A model that always predicts "no swap" would achieve >50% overall accuracy while being completely uninformative about identity binding.

2. **We do report overall accuracy**: The mechanism_comparison.csv includes identity_overall alongside identity_swap_only.

3. **Swap-only is the harder test**: Identity binding is only tested when identities actually change. No-swap episodes don't test binding — they test whether the model can correctly predict "nothing happened," which is trivially solvable.

4. **Feature dependency score as complementary metric**: We also report feature_dependency_score (normal - shuffled) and trajectory_dependency_score (zero-feature), which measure the *source* of identity information, not just accuracy.

**Key sentence**: "Swap-only identity is the appropriate metric for testing identity binding because it isolates the cases where identity actually changes — the harder and more diagnostic test."

---

## Objection 5: "ConflictFirst_margin's swap-only=0.519 is barely above chance. How is this a meaningful result?"

**Severity**: MEDIUM-HIGH — The absolute numbers are unimpressive.

**Response Strategy**:

1. **The contribution is the diagnostic chain, not the absolute numbers**: The finding is not "ConflictFirst_margin is a great model" but "the progression from FeatureOnly→Hybrid→ObjectFile→ImprovedObjectFile→ConflictFirst reveals a structural trade-off that cannot be resolved by gate heuristics alone."

2. **The numbers are diagnostic, not competitive**: This is not a benchmark paper. The numbers serve to demonstrate structural properties (feature dependency, conflict resolution, confidence calibration), not to achieve state-of-the-art performance.

3. **The trajectory predictor is the bottleneck**: TrajectoryOnly's swap-only=0.135 on OOD data means the trajectory signal itself is weak. No gate can produce good identity from a weak signal. The ObjectFile's performance ceiling is set by trajectory quality.

4. **v4.3 confirms this**: Approach detection and augmentation didn't help, showing that the bottleneck is not in the gate but in the signal quality.

**Key sentence**: "The absolute numbers are not the contribution. The contribution is the diagnostic chain showing that: (1) perfect clean performance masks structural deficiency, (2) structural bias toward trajectory continuity enables conflict resolution, and (3) the remaining bottleneck is signal quality, not gate design."

---

## Objection 6: "You're comparing learned models (FeatureOnly, Hybrid) with rule-based systems (ObjectFile). This is an unfair comparison."

**Severity**: MEDIUM — Apples-to-oranges concern.

**Response Strategy**:

1. **The comparison is intentional**: We are not comparing "which model is better." We are comparing "which structure is correct under stress." The learned models have higher clean performance but fail under conflict. The rule-based system has lower clean performance but correct structural bias. This asymmetry IS the finding.

2. **The comparison tests structure, not performance**: If we only compared learned models, we would conclude that FeatureOnly is perfect (1.000 clean accuracy). The rule-based system serves as a structural probe — it demonstrates that an alternative structure exists that resolves conflicts correctly.

3. **v4.1 and v4.2 bridge the gap**: ImprovedObjectFile and ConflictFirstObjectFile incorporate learned components (trajectory predictor) while maintaining the rule-based conflict resolution structure. They represent a middle ground.

**Key sentence**: "The comparison is not about which model is better, but about which structure is correct under stress. The rule-based ObjectFile serves as a structural probe, demonstrating that an alternative to pure feature matching can resist feature hijacking."

---

## Objection 7: "The confidence calibration 'breakthrough' in v4.2 is trivially achieved by assigning low confidence to uncertain decisions."

**Severity**: LOW-MEDIUM — This is partially valid.

**Response Strategy**:

1. **Partially valid**: Yes, the calibration improvement in v4.2 is partly because the gate assigns low confidence when it's uncertain. But this is exactly what calibration should do — the previous versions (v4, v4.1) couldn't even do this.

2. **The calibration is not trivial**: v4.1 also had a confidence mechanism but failed to calibrate (correct=incorrect=1.0). The difference is that v4.2's conflict-first structure produces meaningful confidence variation, while v4.1's confidence-based weighting produces uniform high confidence.

3. **The calibration is fragile**: We acknowledge in the report that v4.2's calibration is a byproduct of the gate's decision mode, not a principled uncertainty model. This is why we recommend "add_uncertainty_model" as a future direction.

**Key sentence**: "The calibration improvement is not trivial — v4.1 also had a confidence mechanism but failed to calibrate. The difference is structural: conflict-first gating produces meaningful confidence variation, while confidence-based weighting produces uniform high confidence."

---

## Objection 8: "This work doesn't connect to the broader object permanence literature in developmental psychology."

**Severity**: MEDIUM — Interdisciplinary gap.

**Response Strategy**:

1. **Acknowledge the gap**: We should add a more thorough connection to developmental psychology in Related Work.

2. **Key connection**: The violation-of-expectation paradigm (Baillargeon) is analogous to our stress tests — infants look longer at "impossible" events (objects disappearing, identity violations), suggesting they maintain object files. Our stress tests similarly probe whether models maintain object files under violation.

3. **Object-file theory**: Kahneman & Treisman's object-file theory is directly relevant — they proposed that object identity is maintained through a file-like structure that binds features, locations, and temporal continuity. Our ObjectFile is a computational implementation of this theory.

4. **Key difference**: Developmental research tests whether infants *have* object permanence; we test whether models *have* the structural prerequisites for object permanence. This is a weaker but more precise claim.

**Key sentence**: "Our stress tests are analogous to violation-of-expectation experiments in developmental psychology: they probe whether models maintain structural prerequisites for object permanence, not whether they possess full object permanence."

---

## Objection 9: "You only tested 2 objects with one-hot features. Does the diagnostic pattern even generalize?"

**Severity**: MEDIUM — Valid concern about scope.

**Response Strategy**:

1. **We have supplementary evidence (v5/v5.1)**: We repeated the diagnostic pattern under 3 objects + 16-dim continuous features. FeatureOnly still achieves swap-only=1.000 under clean conditions and conflict=0.000 under feature-trajectory conflict. ConflictFirstObjectFile achieves swap-only=0.650 and conflict=0.343~0.389.

2. **Sanity audit passed all checks**: v5.1 confirmed that (a) N=3 permutation metric is valid, (b) continuous feature oracle achieves 1.000 under clean conditions, (c) conflict construction induces genuine feature-trajectory disagreement (conflict_rate=0.889), and (d) FeatureOnly conflict=0 is not a metric artifact (restoring correct features recovers accuracy to 1.000).

3. **Honest framing**: We do NOT claim "SVT generalizes to complex environments." We say: "preliminary evidence suggests the diagnostic pattern is not limited to two one-hot objects, while broader scaling remains future work."

4. **The supplementary result strengthens the main claim**: The main claim is "clean feature matching ≠ object-file." If this pattern also holds under 3 objects + continuous features, it's harder to dismiss as a 2-object artifact.

**Key sentence**: "As a preliminary scaling sanity check, we repeated the diagnostic pattern under a 3-object setting with continuous 16-dimensional features. The audit confirmed that the diagnostic pattern persists: FeatureOnly achieves perfect clean accuracy but fails completely under conflict, and this failure is not a metric artifact."

---

## Objection 10: "The zero-feature oracle accuracy is 0.720 in v5.1. Doesn't this mean the metric is unreliable?"

**Severity**: LOW — This is a tie-breaking artifact, not a real issue.

**Response Strategy**:

1. **Explain the artifact**: When all features are zero, cosine similarity is undefined. The oracle falls back to tie-breaking (typically first-available assignment), which can produce above-chance accuracy for some permutation distributions.

2. **The key checks don't use zero-feature**: The main evidence uses normal, shuffled, wrong, and restored-feature conditions. Zero-feature is included for completeness but not used as primary evidence.

3. **This doesn't affect the main result**: FeatureOnly conflict=0 is confirmed by Audit 4 (restoring features recovers accuracy to 1.000), not by zero-feature analysis.

**Key sentence**: "Zero-feature results are not used as main evidence due to tie-breaking ambiguity; the key checks are normal, shuffled, wrong, and restoration conditions."

---

## Summary: Top 3 Most Dangerous Objections

1. **Toy problem** (Objection 1) — Must reframe contribution as diagnostic chain, not model performance
2. **Rule-based tautology** (Objection 2) — Must emphasize that the finding is FeatureOnly/Hybrid failure, not ObjectFile success
3. **Low absolute numbers** (Objection 5) — Must emphasize diagnostic value over competitive performance

## Defensive Writing Principles

1. Never claim "object permanence solved" or "model understands objects"
2. Always frame as "structural diagnostic chain" and "existence proof"
3. Lead with negative results (FeatureOnly/Hybrid failure) before positive results (ObjectFile conflict resolution)
4. Explicitly state limitations before conclusions
5. Use "demonstrate" not "prove"; "suggest" not "show"; "diagnostic" not "solution"
6. For v5 supplementary: use "preliminary" not "validated"; "supplementary sanity check" not "scaling experiment"
