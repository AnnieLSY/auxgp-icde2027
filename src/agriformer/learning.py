"""Training utilities shared by AgriFormer and the command-line runner."""

from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import AgriFormer

LOSS_WEIGHTS = {
    "high": 1.6478412980814696,
    "moderate": 0.1733336874998804,
    "low": 0.14109997472065477,
    "independent": 0.9084366353281252,
}
YIELD_FACTOR = 2.5730918734709154
LEARNING_RATE = 0.00021884237189521226


def reset_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def task_groups(targets, target_index: int, trait_names, dataset=""):
    names = [str(name).lower() for name in trait_names]
    if dataset.upper().startswith("AUS"):
        expected = {
            "high": ["yield", "harvest_index", "maturity_2", "maturity_3"],
            "moderate": ["biomass", "spike_no"],
            "low": ["plant_height", "tgw", "maturity_1"],
            "independent": [],
        }
        if set(sum(expected.values(), [])) <= set(names):
            return {group: [names.index(name) for name in values] for group, values in expected.items()}
    groups = {"high": [target_index], "moderate": [], "low": [], "independent": []}
    primary = np.asarray(targets, dtype=float)[:, target_index]
    for index in range(np.asarray(targets).shape[1]):
        if index == target_index:
            continue
        values = np.asarray(targets, dtype=float)[:, index]
        valid = np.isfinite(primary) & np.isfinite(values)
        correlation = abs(np.corrcoef(primary[valid], values[valid])[0, 1]) if valid.sum() > 2 else 0.0
        group = "high" if correlation >= 0.50 else "moderate" if correlation >= 0.25 else "low" if correlation >= 0.10 else "independent"
        groups[group].append(index)
    return groups


def fill_targets(training, evaluation):
    training = np.asarray(training, dtype=np.float32).copy()
    evaluation = np.asarray(evaluation, dtype=np.float32).copy()
    means = np.nanmean(training, axis=0).astype(np.float32)
    means[~np.isfinite(means)] = 0.0
    return np.where(np.isnan(training), means, training), np.where(np.isnan(evaluation), means, evaluation)


def fit_predict_agriformer(
    context_train,
    effect_train,
    y_train,
    context_test,
    effect_test,
    *,
    target_index,
    trait_names,
    dataset,
    seed=42,
    epochs=100,
    batch_size=8,
    device="auto",
):
    """Fit the formal architecture for a fixed, externally prepared fold."""
    reset_seed(seed)
    selected_device = choose_device(device)
    y_train, _ = fill_targets(y_train, np.zeros((len(context_test), np.asarray(y_train).shape[1])))
    groups = task_groups(y_train, target_index, trait_names, dataset)
    model = AgriFormer(
        context_dim=np.asarray(context_train).shape[1],
        effect_dim=np.asarray(effect_train).shape[1],
        output_dim=y_train.shape[1],
        task_groups=groups,
    ).to(selected_device)
    weights = torch.ones(y_train.shape[1], device=selected_device)
    for group, indices in groups.items():
        if indices:
            weights[indices] = LOSS_WEIGHTS[group]
    weights[target_index] *= YIELD_FACTOR
    tensors = TensorDataset(
        torch.as_tensor(context_train, dtype=torch.float32),
        torch.as_tensor(effect_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(tensors, batch_size=min(batch_size, len(tensors)), shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    model.train()
    for _ in range(int(epochs)):
        for context, effects, targets in loader:
            context, effects, targets = context.to(selected_device), effects.to(selected_device), targets.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(context, effects)
            loss = (((prediction - targets) ** 2) * weights).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite AgriFormer training loss")
            loss.backward()
            optimizer.step()
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(context_test), batch_size):
            context = torch.as_tensor(context_test[start:start + batch_size], dtype=torch.float32, device=selected_device)
            effects = torch.as_tensor(effect_test[start:start + batch_size], dtype=torch.float32, device=selected_device)
            predictions.append(model(context, effects).cpu().numpy())
    return np.vstack(predictions), model, {"task_groups": groups, "device": str(selected_device), "epochs": int(epochs)}
