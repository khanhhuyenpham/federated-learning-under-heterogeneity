import math
from itertools import combinations

import numpy as np
import torch

from .diagnostics import (
    cosine_similarity_updates,
    model_to_vector,
    model_update_vector,
    update_magnitude,
)
from .fedprox import client_update_fedprox
from .training import aggregate, evaluate

from dataclasses import dataclass
import pandas as pd

import random

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_client_loaders(
        model,
        client_loaders,
        loss_fn,
        device,
):
    sorted_client_id = sorted(client_loaders.keys())
    client_evaluation_rows = []
    weighted_accuracy = 0
    weighted_loss = 0
    total_examples_size = 0

    for client_id in sorted_client_id:
        loss, accuracy = evaluate(model, client_loaders[client_id], loss_fn, device)
        num_examples = len(client_loaders[client_id].dataset)
        client_evaluation_rows.append({
            "client_id": client_id,
            "num_examples": num_examples,
            "loss": loss,
            "accuracy": accuracy, 
        })
        weighted_accuracy += accuracy * num_examples
        weighted_loss += loss * num_examples
        total_examples_size += num_examples

    assert total_examples_size > 0
    weighted_accuracy /= total_examples_size
    weighted_loss /= total_examples_size

    accuracy_rows = [row['accuracy'] for row in client_evaluation_rows]
    mean_client_accuracy = np.mean(accuracy_rows)
    std_client_accuracy = np.std(accuracy_rows)
    worst_client_accuracy = np.min(accuracy_rows)
    p10_client_accuracy = np.percentile(accuracy_rows, 10)

    summary = {
        "weighted_accuracy": float(weighted_accuracy),
        "weighted_loss": float(weighted_loss),
        "mean_client_accuracy": float(mean_client_accuracy),
        "std_client_accuracy": float(std_client_accuracy),
        "worst_client_accuracy": float(worst_client_accuracy),
        "p10_client_accuracy": float(p10_client_accuracy),
    }    

    return (
        client_evaluation_rows,
        summary,
    )

def run_diagnostic_round(
        global_model,
        client_loaders,
        round_idx,
        mu,
        config, 
        loss_fn,
        device,
):
    local_models_by_id = {}
    client_sizes_by_id = {}
    client_updates = {}
    num_client = len(client_loaders)
    sorted_client_id = sorted(client_loaders.keys())

    global_vector = model_to_vector(global_model).clone()
    global_model_magnitude = torch.linalg.norm(global_vector).item()
    aggregate_update = torch.zeros_like(global_vector)
    total_client_size = 0

    client_diagnostic_rows = []

    for client_id in sorted_client_id:
        client_loader = client_loaders[client_id]
        local_model = client_update_fedprox(
            global_model=global_model,
            client_loader=client_loader,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            loss_fn=loss_fn,
            device=device,
            mu=mu,
        )
        client_size=len(client_loader.dataset)
        delta = model_update_vector(local_model=local_model, global_model=global_model)
        local_models_by_id[client_id] = local_model
        client_sizes_by_id[client_id] = client_size
        total_client_size += client_size
        aggregate_update += client_size * delta
        client_updates[client_id] = delta

    assert len(local_models_by_id) == num_client
    assert len(client_sizes_by_id) == num_client
    assert len(client_updates) == num_client
    assert total_client_size == sum(len(loader.dataset) for loader in client_loaders.values())
    assert total_client_size > 0

    aggregate_update /= total_client_size
    epsilon = 1e-12

    for client_id in sorted_client_id:
        delta = client_updates[client_id]
        client_size = client_sizes_by_id[client_id]
        update_magnitude_value = update_magnitude(delta).item()
        peer_total_size = total_client_size - client_size
        assert peer_total_size > 0
        peer_only_aggregate = (aggregate_update * total_client_size - client_size * delta) / peer_total_size
        client_diagnostic_rows.append({
            "seed": config.seed,
            "alpha": config.alpha,
            "mu": mu,
            "round": round_idx,
            "local_epochs": config.local_epochs,
            "client_id": client_id,
            "client_size": client_size,
            "client_weight": client_size / total_client_size,
            "update_magnitude": update_magnitude_value,
            "relative_update_magnitude": update_magnitude_value / (epsilon + global_model_magnitude),
            "aggregate_alignment": cosine_similarity_updates(delta, aggregate_update).item(),
            "loo_alignment": cosine_similarity_updates(delta, peer_only_aggregate).item(),
        })

    assert len(client_diagnostic_rows) == num_client
    assert math.isclose(sum(rows["client_weight"] for rows in client_diagnostic_rows), 1.0, rel_tol=0.0, abs_tol=epsilon)
    assert aggregate_update.shape == global_vector.shape
    assert torch.isfinite(aggregate_update).all()

    pairwise_rows = []
    for client_i, client_j in combinations(sorted_client_id, 2):
        pairwise_rows.append({
            "seed": config.seed,
            "alpha": config.alpha,
            "mu": mu,
            "round": round_idx,
            "local_epochs": config.local_epochs,
            "client_i": client_i,
            "client_j": client_j,
            "cosine_similarity": cosine_similarity_updates(client_updates[client_i], client_updates[client_j]).item(),
        })

    assert len(pairwise_rows) == num_client * (num_client - 1) // 2

    round_rows = [{
        "seed": config.seed,
        "alpha": config.alpha,
        "mu": mu,
        "round": round_idx,
        "local_epochs": config.local_epochs,
        "global_model_magnitude": global_model_magnitude,
        "aggregate_update_magnitude": update_magnitude(aggregate_update).item(),
        "total_client_size": total_client_size,
        "num_participating_clients": num_client,
    }]

    ordered_local_models = [local_models_by_id[client_id] for client_id in sorted_client_id]
    ordered_client_size = [client_sizes_by_id[client_id] for client_id in sorted_client_id]

    new_global_model = aggregate(global_model=global_model, local_models=ordered_local_models, client_sizes=ordered_client_size)

    server_update = model_to_vector(new_global_model) - global_vector
    assert torch.allclose(
            server_update, 
            aggregate_update,
            rtol=1e-5,
            atol=1e-6,
    )

    return (
        new_global_model,
        client_diagnostic_rows,
        pairwise_rows,
        round_rows,
    )

@dataclass
class RunResult:
    performance_history: pd.DataFrame
    client_performance_history: pd.DataFrame
    client_update_diagnostics: pd.DataFrame
    pairwise_update_diagnostics: pd.DataFrame
    round_diagnostics: pd.DataFrame


import copy
from .data import create_client_loaders
def run_experiment(
        config,
        mu_values,
        initial_model,
        train_ds,
        client_train_indices,
        validation_loaders,
        loss_fn,
        device,
        verbose: bool = False,
):
    performance_rows = []
    client_performance_rows = []
    client_update_rows = []
    pairwise_update_rows = []
    all_round_rows = []

    for mu_value in mu_values:
        set_seed(config.seed)
        global_model = copy.deepcopy(initial_model).to(device)
        client_loaders = create_client_loaders(train_ds, client_train_indices, config.batch_size, config.seed, True)
        client_rows, summary = evaluate_client_loaders(global_model, validation_loaders, loss_fn, device)
        if verbose:
            print(
                f"mu={mu_value:g} | "
                f"round=00/{config.num_rounds:02d} | "
                f"validation_loss={summary['weighted_loss']:.4f} | "
                f"validation_accuracy={summary['weighted_accuracy']:.2%} | "
                f"worst_client_accuracy={summary['worst_client_accuracy']:.2%}"
            )
        extended_summary = {
            "seed": config.seed,
            "alpha": config.alpha,
            "mu": mu_value,
            "round": 0,
            "local_epochs": config.local_epochs,
            "split": "validation",
            **summary,
        }
        extended_client_rows = [
            {
                "seed": config.seed,
                "alpha": config.alpha,
                "mu": mu_value,
                "round": 0,
                "local_epochs": config.local_epochs,
                "split": "validation",
                **client_row,
            }
            for client_row in client_rows
        ]
        performance_rows.append(extended_summary)
        client_performance_rows.extend(extended_client_rows)

        for round_idx in range(1, config.num_rounds + 1):
            global_model, client_diagnostic_rows, pairwise_rows, current_round_rows = run_diagnostic_round(
                global_model=global_model, client_loaders=client_loaders, round_idx=round_idx, mu=mu_value, config=config, loss_fn=loss_fn, device=device
            )
            client_rows, summary = evaluate_client_loaders(global_model, validation_loaders, loss_fn, device)
            extended_summary = {
                "seed": config.seed,
                "alpha": config.alpha,
                "mu": mu_value,
                "round": round_idx,
                "local_epochs": config.local_epochs,
                "split": "validation",
                **summary,
            }
            extended_client_rows = [
                {
                    "seed": config.seed,
                    "alpha": config.alpha,
                    "mu": mu_value,
                    "round": round_idx,
                    "local_epochs": config.local_epochs,
                    "split": "validation",
                    **client_row,
                }
                for client_row in client_rows
            ]

            if verbose:
                print(
                    f"mu={mu_value:g} | "
                    f"round={round_idx:02d}/{config.num_rounds:02d} | "
                    f"validation_loss={summary['weighted_loss']:.4f} | "
                    f"validation_accuracy={summary['weighted_accuracy']:.2%} | "
                    f"worst_client_accuracy={summary['worst_client_accuracy']:.2%}"
                )

            performance_rows.append(extended_summary)
            client_performance_rows.extend(extended_client_rows)
            client_update_rows.extend(client_diagnostic_rows)
            pairwise_update_rows.extend(pairwise_rows)
            all_round_rows.extend(current_round_rows)

    result = RunResult(
        performance_history=pd.DataFrame(performance_rows),
        client_performance_history=pd.DataFrame(client_performance_rows),
        client_update_diagnostics=pd.DataFrame(client_update_rows),
        pairwise_update_diagnostics=pd.DataFrame(pairwise_update_rows),
        round_diagnostics=pd.DataFrame(all_round_rows),
    )

    M = len(mu_values)
    R = config.num_rounds
    K = len(validation_loaders)
    P = K * (K - 1) // 2

    assert len(result.performance_history) == M * (R + 1)
    assert len(result.client_performance_history) == M * (R + 1) * K
    assert len(result.client_update_diagnostics) == M * R * K
    assert len(result.pairwise_update_diagnostics) == M * R * P
    assert len(result.round_diagnostics) == M * R

    return result
