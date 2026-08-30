import time

import torch
from torch import nn
import math

from .benchmark_results import (
    AlgorithmName,
    BenchmarkRunResult,
    adapt_fed_run_result,
    adapt_scaffold_run_result,
)
from .condition_metadata import ConditionMetadata
from .data import create_client_loaders
from .experiment import run_experiment, set_seed
from .scaffold import run_scaffold_experiment

def _synchronize_device(
    device: torch.device,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark_condition(
    *,
    algorithm: AlgorithmName,
    mu: float | None,
    config,
    condition_metadata: ConditionMetadata,
    initial_model: nn.Module,
    train_ds,
    client_train_indices,
    validation_loaders,
    local_test_loaders,
    global_test_loader,
    loss_fn: nn.Module,
    device: torch.device,
    client_feature_transforms=None,
    verbose: bool = False,
) -> BenchmarkRunResult:
    if algorithm == "fedavg":
        if mu != 0.0:
            raise ValueError(
                "FedAvg requires mu=0.0"
            )

    elif algorithm == "fedprox":
        if (
            mu is None
            or not math.isfinite(mu)
            or mu <= 0.0
        ):
            raise ValueError(
                "FedProx requires a finite mu > 0"
            )

    elif algorithm == "scaffold":
        if mu is not None:
            raise ValueError(
                "SCAFFOLD requires mu=None"
            )

    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}"
        )

    if condition_metadata.seed != config.seed:
        raise ValueError(
            "Condition metadata seed must match "
            "the experiment config seed"
        )

    if not client_train_indices:
        raise ValueError(
                "client partition cannot be empty"
            )

    if condition_metadata.num_clients != len(client_train_indices):
        raise ValueError(
            "num_clients in condition_metadata must match with the size of client_train_indices"
        )

    if client_feature_transforms is not None:
        if (
            set(client_feature_transforms)
            != set(client_train_indices)
        ):
            raise ValueError(
                "Client IDs in client_feature_transforms "
                "must match the partition client IDs"
            )

    train_client_ids = set(client_train_indices)
    validation_client_ids = set(validation_loaders)
    local_test_client_ids = set(local_test_loaders)

    if (
        train_client_ids != validation_client_ids
        or train_client_ids != local_test_client_ids
    ):
        raise ValueError(
            "Training, validation, and local-test "
            "client IDs must match"
        )
    
    set_seed(config.seed)
    
    if algorithm in {"fedavg", "fedprox"}:  
        _synchronize_device(device)
        start_time = time.perf_counter()

        raw_result = run_experiment(
            config=config,
            mu_values=(float(mu),),
            initial_model=initial_model,
            train_ds=train_ds,
            client_train_indices=client_train_indices,
            validation_loaders=validation_loaders,
            loss_fn=loss_fn,
            device=device,
            client_feature_transforms=client_feature_transforms,
            verbose=verbose,
        )

        _synchronize_device(device)

        runtime_seconds = time.perf_counter() - start_time

        return adapt_fed_run_result(
            run_result=raw_result,
            algorithm=algorithm,
            mu=mu,
            seed=config.seed,
            initial_model=initial_model,
            condition_metadata=condition_metadata,
            local_test_loaders=local_test_loaders,
            global_test_loader=global_test_loader,
            device=device,
            runtime_seconds=runtime_seconds,
        )

    elif algorithm == "scaffold":
        _synchronize_device(device=device)
        start_time = time.perf_counter()

        train_loaders = create_client_loaders(
            train_ds=train_ds,
            client_indices=client_train_indices,
            batch_size=config.batch_size,
            seed=config.seed,
            shuffle=True,
            client_feature_transforms=client_feature_transforms,
        )

        raw_result = run_scaffold_experiment(
            seed=config.seed,
            initial_model=initial_model,
            train_loaders=train_loaders,
            validation_loaders=validation_loaders,
            config=config,
            loss_fn=loss_fn,
            device=device,
        )

        _synchronize_device(device=device)
        runtime_seconds= time.perf_counter() - start_time

        return adapt_scaffold_run_result(
            scaffold_result=raw_result,
            seed=config.seed,
            condition_metadata=condition_metadata,
            local_test_loaders=local_test_loaders,
            global_test_loader=global_test_loader,
            runtime_seconds=runtime_seconds,
            device=device,
        )