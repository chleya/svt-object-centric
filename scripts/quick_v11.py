import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from data.generate_nonlinear_feature_ood import generate_v3_dataset, _generate_single_episode, _stack_episodes
from metrics.identity_breakdown import compute_identity_breakdown
from models.probabilistic_structure_object_file import ProbabilisticStructureObjectFile
from models.object_file_models import TrajectoryOnlyAssignment, ConflictFirstObjectFile
from utils.torch_training import train_model

def flip_ff(ff):
    f = ff.copy()
    f[:,:,0,:], f[:,:,1,:] = f[:,:,1,:].copy(), f[:,:,0,:].copy()
    return f

def gen_train(n=1000, nobj=2, fdim=2, seed=0):
    rng = np.random.RandomState(seed)
    eps = []
    for _ in range(n):
        ep = _generate_single_episode(t_obs=10, t_pred=20, num_objects=nobj, arena_size=64.0,
            feature_mode="feature_bearing", feature_dim=fdim, randomize_object_order=True,
            identity_test=True, swap_probability=0.5, force_type="attractor",
            field_strength=0.5, damping=0.95, noise_std=0.1, rng=rng)
        eps.append(ep)
    return _stack_episodes(eps, "feature_bearing")

seed = 0
eval_ds = generate_v3_dataset(n_train=1000, n_test=200, num_objects=2, feature_dim=2,
    feature_mode="feature_bearing", force_train_type="attractor", force_test_type="vortex",
    randomize_object_order=True, disjoint_init_split=True, seed=seed)
train_data = gen_train(n=1000, seed=seed)
swap_test = eval_ds["identity_test_swap_only"]
clean_test = eval_ds["clean_test_id"]

obs_pos_t = train_data["observed_positions"]
fut_pos_t = train_data["future_positions"]
obs_feat_t = train_data.get("object_features_obs")
fut_feat_t = train_data.get("object_features_fut")
ids_t = train_data["identity_labels"]

results = []

for p_conf in [0.0, 0.3, 0.5]:
    print(f"\nTraining PS-ObjectFile (p_conf={p_conf})...")
    model = ProbabilisticStructureObjectFile(
        num_objects=2, feature_dim=2, hidden_dim=128, slot_dim=64,
        identity_weight=1.0, structure_weight=0.5,
        p_conflict=p_conf, p_feature_drop=0.05)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_batches = len(obs_pos_t) // 64

    for epoch in range(30):
        indices = np.random.permutation(len(obs_pos_t))
        for bi in range(n_batches):
            idx = indices[bi*64:(bi+1)*64]
            loss, _, _, _ = model.compute_loss(
                torch.FloatTensor(obs_pos_t[idx]),
                torch.FloatTensor(fut_pos_t[idx]),
                torch.LongTensor(ids_t[idx]),
                torch.FloatTensor(obs_feat_t[idx]),
                torch.FloatTensor(fut_feat_t[idx]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for method in ["combined", "feature", "trajectory"]:
        obs_pos = swap_test["observed_positions"]
        obs_feat = swap_test.get("object_features_obs")
        fut_feat = swap_test.get("object_features_fut")
        fut_pos = swap_test["future_positions"]
        true_id = swap_test["identity_labels"]

        pred_id, pi = model.predict_identity(obs_pos, obs_feat, future_positions=fut_pos,
                                              future_features=fut_feat, method=method)
        if isinstance(pred_id, torch.Tensor):
            pred_id = pred_id.cpu().numpy()
        bd = compute_identity_breakdown(pred_id, true_id)

        is_swap_c = clean_test["is_swap"]
        no_swap_idx = np.where(~is_swap_c)[0]
        conflict_a = float("nan")
        if len(no_swap_idx) > 0:
            ns = {k: v[no_swap_idx] for k, v in clean_test.items()}
            ff = flip_ff(ns.get("object_features_fut"))
            pid_c, _ = model.predict_identity(ns["observed_positions"], ns.get("object_features_obs"),
                                              future_positions=ns["future_positions"],
                                              future_features=ff, method=method)
            if isinstance(pid_c, torch.Tensor):
                pid_c = pid_c.cpu().numpy()
            correct = (pid_c == ns["identity_labels"]).all(axis=1)
            conflict_a = float(correct.mean())

        mean_pi = float(np.mean(pi)) if pi is not None else 0
        print(f"  pconf={p_conf}/{method}: swap={bd['identity_swap_only']:.4f} conflict_a={conflict_a:.4f} mean_pi={mean_pi:.4f}")
        results.append((p_conf, method, bd["identity_swap_only"], conflict_a, mean_pi))

print("\n" + "="*70)
print("CFObjectFile baseline:")
traj = TrajectoryOnlyAssignment(num_objects=2)
train_model(traj, train_data, val_data=clean_test, epochs=30, batch_size=64, lr=1e-3,
            device="cpu", uses_features=False, uses_future_features=False, verbose=False)
cf = ConflictFirstObjectFile(traj_model=traj, strategy="margin_gated", num_objects=2, feature_dim=2)
obs_pos = swap_test["observed_positions"]
obs_feat = swap_test.get("object_features_obs")
fut_feat = swap_test.get("object_features_fut")
fut_pos = swap_test["future_positions"]
true_id = swap_test["identity_labels"]
r = cf.predict_identity(obs_pos, obs_feat, fut_pos, fut_feat, return_conflict_info=True)
cf_bd = compute_identity_breakdown(r[0], true_id)
ns = {k: v[np.where(~clean_test["is_swap"])[0]] for k, v in clean_test.items()}
ff = flip_ff(ns.get("object_features_fut"))
r2 = cf.predict_identity(ns["observed_positions"], ns.get("object_features_obs"),
                          ns["future_positions"], ff, return_conflict_info=True)
cf_conflict = float((r2[0] == ns["identity_labels"]).all(axis=1).mean())
print(f"  CF rule: swap={cf_bd['identity_swap_only']:.4f} conflict_a={cf_conflict:.4f}")

print("\n" + "="*70)
print("SUMMARY: Probabilistic Structure Selection + Delta Trajectory")
print("="*70)
print(f"{'Config':<25} {'Method':<12} {'Swap':>8} {'Conflict':>10} {'Product':>10} {'Mean_Pi':>10}")
print("-"*75)
for p_conf, method, swap, conflict, mean_pi in results:
    prod = swap * conflict if not np.isnan(conflict) else 0
    print(f"pconf={p_conf:<18} {method:<12} {swap:>8.4f} {conflict:>10.4f} {prod:>10.4f} {mean_pi:>10.4f}")
cf_prod = cf_bd['identity_swap_only'] * cf_conflict
print(f"{'CFObjectFile_rule':<25} {'rule':<12} {cf_bd['identity_swap_only']:>8.4f} {cf_conflict:>10.4f} {cf_prod:>10.4f}")
