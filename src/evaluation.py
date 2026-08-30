from typing import Mapping

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

def accuracy_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[int, int]:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs).argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
    return correct, total

def evaluate_clients(
    model: nn.Module, loaders: Mapping[int, DataLoader], device: torch.device
) -> pd.DataFrame:
    model = model.to(device)
    rows = []
    for client_id, loader in sorted(loaders.items()):
        correct, total = accuracy_on_loader(model, loader, device)
        if total == 0:
            raise ValueError(
                f"Validation loader for client {client_id} is empty."
            )
        rows.append({
            "client_id": int(client_id),
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
        })
    return pd.DataFrame(rows)

def summarize_client_evaluations(
    client_results: pd.DataFrame,
) -> dict:
    total_correct = client_results["correct"].sum()
    total_examples = client_results["total"].sum()

    if total_examples <= 0:
        raise ValueError("Cannot summarize an empty evaluation.")

    return {
        "weighted_accuracy": total_correct / total_examples,
        "mean_client_accuracy": client_results["accuracy"].mean(),
        "worst_client_accuracy": client_results["accuracy"].min(),
        "std_client_accuracy": client_results["accuracy"].std(ddof=0),
    }

def create_global_test_loader(
    test_ds,
    batch_size: int,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(test_ds) == 0:
        raise ValueError("test_ds must be nonempty")

    return DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False
    )

def evaluate_global_model(
    model: nn.Module,
    global_test_loader: DataLoader,
    device: torch.device,
) -> dict:
    correct, total = accuracy_on_loader(
        model=model,
        loader=global_test_loader,
        device=device,
    )
    if total == 0:
        raise ValueError(
            "cannot evaluate an empty global test loader"
        )

    return {
        "global_test_correct": correct,
        "global_test_total": total,
        "global_test_accuracy": correct/total,
    }