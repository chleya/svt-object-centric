import sys
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import csv
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_predictor import MLPPredictor
from models.transformer_predictor import TransformerPredictor
from models.identity_head import DualHeadModel
from models.object_centric import ObservationConditionedIdentityModel, ObjectCentricPredictor
from models.self_supervised import VelocityContinuityModel, ContrastiveIdentityModel
from models.slot_persistence import SlotPersistenceModel
from metrics.prediction_metrics import compute_prediction_metrics
from metrics.identity_metrics import compute_identity_accuracy, velocity_continuity_identity
from metrics.gated_svt_score import compute_gated_svt_score, compute_old_smss


class MotionDataset(Dataset):
    def __init__(self, data_dict, use_velocity=False):
        self.obs_pos = torch.FloatTensor(data_dict["observed_positions"])
        self.obs_vel = torch.FloatTensor(data_dict["observed_velocities"])
        self.fut_pos = torch.FloatTensor(data_dict["future_positions"])
        self.fut_vel = torch.FloatTensor(data_dict["future_velocities"])
        self.identity = torch.LongTensor(data_dict["identity_labels"])
        self.use_velocity = use_velocity

    def __len__(self):
        return len(self.obs_pos)

    def __getitem__(self, idx):
        obs = self.obs_pos[idx]
        if self.use_velocity:
            obs = torch.cat([obs, self.obs_vel[idx]], dim=-1)
        return obs, self.fut_pos[idx], self.identity[idx]


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(data_dir, split_name):
    path = os.path.join(data_dir, f"{split_name}.npz")
    data = np.load(path)
    return {
        "observed_positions": data["observed_positions"],
        "observed_velocities": data["observed_velocities"],
        "future_positions": data["future_positions"],
        "future_velocities": data["future_velocities"],
        "identity_labels": data["identity_labels"],
    }


def build_model(model_type, cfg, world_cfg):
    t_obs = world_cfg["t_obs"]
    t_pred = world_cfg["t_pred"]
    n_objects = world_cfg["num_objects"]
    dim = 2

    if model_type == "mlp":
        model = MLPPredictor(
            t_obs=t_obs,
            t_pred=t_pred,
            n_objects=n_objects,
            dim=dim,
            hidden_dim=cfg.get("hidden_dim", 256),
            n_layers=cfg.get("n_layers", 4),
            dropout=cfg.get("dropout", 0.1),
        )
    elif model_type == "transformer":
        model = TransformerPredictor(
            t_obs=t_obs,
            t_pred=t_pred,
            n_objects=n_objects,
            dim=dim,
            d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 4),
            n_encoder_layers=cfg.get("n_encoder_layers", 3),
            n_decoder_layers=cfg.get("n_decoder_layers", 3),
            dim_feedforward=cfg.get("dim_feedforward", 512),
            dropout=cfg.get("dropout", 0.1),
        )
    elif model_type == "transformer_small":
        model = TransformerPredictor(
            t_obs=t_obs,
            t_pred=t_pred,
            n_objects=n_objects,
            dim=dim,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.1,
        )
    elif model_type == "object_centric":
        model = ObjectCentricPredictor(
            t_obs=t_obs,
            t_pred=t_pred,
            n_objects=n_objects,
            dim=dim,
            hidden_dim=cfg.get("hidden_dim", 128),
            n_layers=cfg.get("n_layers", 3),
            dropout=cfg.get("dropout", 0.1),
        )
    elif model_type == "velocity_continuity":
        model = VelocityContinuityModel(
            t_obs=t_obs,
            t_pred=t_pred,
            n_objects=n_objects,
            dim=dim,
            hidden_dim=cfg.get("hidden_dim", 256),
            n_layers=cfg.get("n_layers", 4),
            dropout=cfg.get("dropout", 0.1),
        )
    elif model_type == "contrastive":
        model = ContrastiveIdentityModel(
            t_obs=t_obs,
            t_pred=t_pred,
            n_objects=n_objects,
            dim=dim,
            hidden_dim=cfg.get("hidden_dim", 256),
            n_layers=cfg.get("n_layers", 4),
            dropout=cfg.get("dropout", 0.1),
        )
    elif model_type == "slot_persistence":
        model = SlotPersistenceModel(
            t_obs=t_obs,
            t_pred=t_pred,
            n_objects=n_objects,
            dim=dim,
            slot_dim=cfg.get("slot_dim", 32),
            hidden_dim=cfg.get("hidden_dim", 128),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def train_model(model, train_loader, val_loader, n_epochs, lr, device, model_save_path,
                identity_weight=0.0, patience=15):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for obs, fut, ids in train_loader:
            obs, fut = obs.to(device), fut.to(device)
            ids = ids.to(device)

            optimizer.zero_grad()

            if hasattr(model, "compute_loss"):
                total_loss, pred_loss, id_loss = model.compute_loss(obs, fut, ids)
            else:
                pred = model(obs)
                pred_loss = nn.functional.mse_loss(pred, fut)
                total_loss = pred_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += pred_loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for obs, fut, ids in val_loader:
                obs, fut = obs.to(device), fut.to(device)
                pred = model.predict_future(obs) if hasattr(model, "predict_future") else model(obs)
                val_loss += nn.functional.mse_loss(pred, fut).item()
                val_batches += 1

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(val_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: train_mse={avg_train:.4f}, val_mse={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    return model


def evaluate_model(model, data_dict, device, metrics_cfg):
    model.eval()
    obs = torch.FloatTensor(data_dict["observed_positions"]).to(device)
    fut = data_dict["future_positions"]
    ids = data_dict["identity_labels"]

    with torch.no_grad():
        pred = model.predict_future(obs) if hasattr(model, "predict_future") else model(obs)
        pred_np = pred.cpu().numpy()

    pred_metrics = compute_prediction_metrics(pred_np, fut)

    pred_identity_via_traj = _identity_by_trajectory_match(pred_np, fut)
    traj_id_acc = compute_identity_accuracy(pred_identity_via_traj, ids)

    vel_ids = velocity_continuity_identity(
        data_dict["observed_positions"],
        future_positions=fut,
    )
    vel_id_acc = compute_identity_accuracy(vel_ids, ids)

    if hasattr(model, "predict_identity"):
        with torch.no_grad():
            model_ids = model.predict_identity(obs).cpu().numpy()
        model_id_acc = compute_identity_accuracy(model_ids, ids)
    else:
        model_id_acc = None

    return {
        "mse": pred_metrics["mse"],
        "mean_predictor_mse": pred_metrics["mean_predictor_mse"],
        "skill_score": pred_metrics["skill_score"],
        "normalized_mse": pred_metrics["normalized_mse"],
        "traj_identity_accuracy": traj_id_acc,
        "vel_identity_accuracy": vel_id_acc,
        "model_identity_accuracy": model_id_acc,
    }


def _identity_by_trajectory_match(pred_future, test_future):
    B, T, N, D = pred_future.shape
    pred_ids = np.tile(np.arange(N), (B, 1))
    if N != 2:
        return pred_ids
    for i in range(B):
        mse_no_swap = np.mean((pred_future[i] - test_future[i]) ** 2)
        swapped_pred = pred_future[i].copy()
        swapped_pred[:, [0, 1]] = swapped_pred[:, [1, 0]]
        mse_swap = np.mean((swapped_pred - test_future[i]) ** 2)
        if mse_swap < mse_no_swap:
            pred_ids[i] = np.array([1, 0])
    return pred_ids


def run_training(cfg, model_types=None, n_epochs=80, lr=1e-3, batch_size=64,
                 use_identity_head=False, identity_weight=1.0,
                 obs_conditioned_identity=False):
    data_dir = cfg["data"]["save_dir"]
    world_cfg = cfg["world"]
    metrics_cfg = cfg["metrics"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data = load_split(data_dir, "train")
    clean_data = load_split(data_dir, "clean_test")
    cf_data = load_split(data_dir, "counterfactual_test")
    comp_data = load_split(data_dir, "compositional_test")
    identity_data = load_split(data_dir, "identity_test")

    train_dataset = MotionDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    clean_dataset = MotionDataset(clean_data)
    val_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False)

    if model_types is None:
        model_types = ["mlp", "transformer"]

    all_results = {}

    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type}")
        print(f"{'='*60}")

        model = build_model(model_type, {}, world_cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        if use_identity_head and model_type not in ("object_centric", "velocity_continuity", "contrastive", "slot_persistence"):
            if obs_conditioned_identity:
                model = ObservationConditionedIdentityModel(
                    model, t_obs=world_cfg["t_obs"],
                    n_objects=world_cfg["num_objects"], dim=2,
                    identity_weight=identity_weight,
                )
            else:
                model = DualHeadModel(model, identity_weight=identity_weight)

        model_save_path = f"checkpoints/{model_type}_{'dual' if use_identity_head else 'base'}.pt"
        model = train_model(model, train_loader, val_loader, n_epochs, lr, device, model_save_path)

        model.eval()
        model_results = {}

        for split_name, split_data in [
            ("clean", clean_data),
            ("counterfactual", cf_data),
            ("compositional", comp_data),
            ("identity", identity_data),
        ]:
            print(f"\n  Evaluating on {split_name}...")
            result = evaluate_model(model, split_data, device, metrics_cfg)
            model_results[split_name] = result
            print(f"    Skill: {result['skill_score']:.3f}, "
                  f"Traj-ID: {result['traj_identity_accuracy']:.3f}, "
                  f"Vel-ID: {result['vel_identity_accuracy']:.3f}"
                  + (f", Model-ID: {result['model_identity_accuracy']:.3f}" if result['model_identity_accuracy'] is not None else ""))

        clean = model_results["clean"]
        cf = model_results["counterfactual"]
        comp = model_results["compositional"]
        identity = model_results["identity"]

        for id_method in ["traj", "vel"]:
            id_key = f"{id_method}_identity_accuracy"
            gated = compute_gated_svt_score(
                clean["skill_score"],
                cf["skill_score"],
                comp["skill_score"],
                identity[id_key],
                clean_skill_threshold=metrics_cfg["clean_skill_threshold"],
            )
            old = compute_old_smss(
                clean["mse"], cf["mse"], comp["mse"], identity[id_key]
            )
            model_results[f"gated_{id_method}"] = gated
            model_results[f"old_smss_{id_method}"] = old
            print(f"\n  Gated SVT ({id_method}): {gated['gated_svt_score']:.4f}, "
                  f"Old SMSS: {old:.4f}, Gate passed: {gated['gate_passed']}")

        all_results[model_type] = model_results

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke_hard.yaml")
    parser.add_argument("--models", type=str, nargs="+", default=["mlp", "transformer"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--identity-head", action="store_true")
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--obs-conditioned-identity", action="store_true")
    parser.add_argument("--output", type=str, default="results/learned_models")
    args = parser.parse_args()

    cfg = load_config(args.config)

    results = run_training(
        cfg,
        model_types=args.models,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        use_identity_head=args.identity_head,
        identity_weight=args.identity_weight,
        obs_conditioned_identity=args.obs_conditioned_identity,
    )

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "learned_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")
