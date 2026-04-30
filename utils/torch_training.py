"""
SVT Training Utilities (v3.4+)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def train_model(
    model,
    train_data,
    val_data=None,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    device="cpu",
    uses_features=False,
    uses_future_features=False,
    verbose=True,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_obs = torch.FloatTensor(train_data["observed_positions"]).to(device)
    train_fut = torch.FloatTensor(train_data["future_positions"]).to(device)
    train_ids = torch.FloatTensor(train_data["identity_labels"]).to(device)

    train_feat = None
    if uses_features and "object_features_obs" in train_data:
        train_feat = torch.FloatTensor(train_data["object_features_obs"]).to(device)

    train_fut_feat = None
    if uses_future_features and "object_features_fut" in train_data:
        train_fut_feat = torch.FloatTensor(train_data["object_features_fut"]).to(device)

    tensors = [train_obs, train_fut, train_ids]
    if train_feat is not None:
        tensors.append(train_feat)
    if train_fut_feat is not None:
        tensors.append(train_fut_feat)

    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_feat_tensors = int(train_feat is not None) + int(train_fut_feat is not None)

    training_log = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_id = 0.0
        epoch_bind = 0.0
        n_batches = 0

        for batch in loader:
            obs_b = batch[0]
            fut_b = batch[1]
            ids_b = batch[2]

            feat_b = None
            fut_feat_b = None

            idx = 3
            if train_feat is not None:
                feat_b = batch[idx]
                idx += 1
            if train_fut_feat is not None:
                fut_feat_b = batch[idx]
                idx += 1

            optimizer.zero_grad()

            if hasattr(model, 'compute_loss'):
                try:
                    loss, mse_loss, id_loss, bind_loss = model.compute_loss(
                        obs_b, fut_b, ids_b, feat_b, fut_feat_b)
                    epoch_bind += bind_loss.item() if isinstance(bind_loss, torch.Tensor) else bind_loss
                except TypeError:
                    try:
                        loss, mse_loss, id_loss = model.compute_loss(obs_b, fut_b, ids_b, feat_b)
                    except TypeError:
                        loss, mse_loss, id_loss = model.compute_loss(obs_b, fut_b, ids_b)
                    bind_loss = torch.tensor(0.0)
            else:
                pred = model(obs_b)
                if isinstance(pred, tuple):
                    pred = pred[0]
                mse_loss = nn.functional.mse_loss(pred, fut_b)
                id_loss = torch.tensor(0.0, device=device)
                bind_loss = torch.tensor(0.0, device=device)
                loss = mse_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_id += id_loss.item() if isinstance(id_loss, torch.Tensor) else id_loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_mse = epoch_mse / max(n_batches, 1)
        avg_id = epoch_id / max(n_batches, 1)
        avg_bind = epoch_bind / max(n_batches, 1)

        val_skill = float("nan")
        if val_data is not None:
            val_skill = _compute_val_skill(model, val_data, device, uses_features)

        training_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "mse_loss": avg_mse,
            "identity_loss": avg_id,
            "binding_loss": avg_bind,
            "val_clean_skill": val_skill,
        })

        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} mse={avg_mse:.4f} id={avg_id:.4f} bind={avg_bind:.4f} val_skill={val_skill:.4f}")

    return training_log


def _compute_val_skill(model, val_data, device, uses_features):
    model.eval()
    with torch.no_grad():
        obs = torch.FloatTensor(val_data["observed_positions"]).to(device)
        feat = None
        if uses_features and "object_features_obs" in val_data:
            feat = torch.FloatTensor(val_data["object_features_obs"]).to(device)

        if hasattr(model, 'predict_future'):
            pred = model.predict_future(val_data["observed_positions"], feat)
        else:
            pred = model(obs)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred.cpu().numpy()

        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        target = val_data["future_positions"]
        mse = np.mean((pred - target) ** 2)
        mean_pred = target.mean(axis=(0, 1), keepdims=True)
        mean_mse = np.mean((target - mean_pred) ** 2)

        if mean_mse > 1e-10:
            skill = 1.0 - mse / mean_mse
        else:
            skill = 0.0

    return float(skill)
