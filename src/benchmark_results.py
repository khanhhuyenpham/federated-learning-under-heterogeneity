from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
import copy
from torch.utils.data import DataLoader
from .condition_metadata import ConditionMetadata

from .evaluation import (
    evaluate_clients,
    evaluate_global_model,
    summarize_client_evaluations,
)

from .experiment import RunResult
from .scaffold import ScaffoldExperimentResult

AlgorithmName = Literal[
    "fedavg",
    "fedprox",
    "scaffold",
]

def normalize_client_validation_history(
    history: pd.DataFrame,
    algorithm: AlgorithmName,
    seed: int
) -> pd.DataFrame:
    if history.empty:
        raise ValueError("client history cannot be empty")
    requires_columns = {
        "round",
        "client_id",
        "accuracy",
    }
    missing_columns = requires_columns - set(history.columns)
    if missing_columns:
        raise ValueError(f"client history is missing columns: {sorted(missing_columns)}")

    if "num_examples" in history.columns:
        num_examples = history["num_examples"]
    elif "total" in history.columns:
        num_examples = history['total']
    else:
        raise ValueError("num_examples is missing")

    if "seed" in history.columns:
        observed_seeds = set(
            history["seed"].astype(int).tolist()
        )

        if observed_seeds != {int(seed)}:
            raise ValueError(
                f"Expected seed {seed}, but history "
                f"contains seeds {observed_seeds}"
            )

    normalized = pd.DataFrame({
        "algorithm": algorithm,
        "seed": int(seed),
        "round": history["round"].astype(int),
        "client_id": history["client_id"].astype(int),
        "num_examples": num_examples.astype(int),
        "accuracy": history["accuracy"].astype(float),
    })

    if not normalized["accuracy"].between(0, 1).all():
        raise ValueError(
            "accuracy must be within the range [0, 1]"
        )
    if (normalized["num_examples"] <= 0).any():
        raise ValueError(
            "num_examples must be positive"
        )

    return normalized

def summarize_validation_history(
    client_history: pd.DataFrame,
) -> pd.DataFrame:
    if client_history.empty:
        raise ValueError(
            "Client history cannot be empty"
        )
    if client_history["algorithm"].nunique() != 1:
        raise ValueError("Client history must contain one algorithm")

    if client_history["seed"].nunique() != 1:
        raise ValueError("Client history must contain one seed")

    rows = []
    for round_idx, round_frame in client_history.groupby(
        "round",
        sort=True,
    ):
        accuracies = round_frame["accuracy"].to_numpy(dtype=float)
        weights = round_frame["num_examples"].to_numpy(dtype=float)
        weighted_accuracy = np.average(
            accuracies,
            weights=weights,
        )
        mean_client_accuracy = np.mean(accuracies)
        worst_client_accuracy = np.min(accuracies)
        std_client_accuracy = np.std(accuracies)
        p10_client_accuracy = np.percentile(
            accuracies,
            10,
        )
        rows.append({
            "algorithm": round_frame["algorithm"].iloc[0],
            "seed": int(round_frame["seed"].iloc[0]),
            "round": int(round_idx),
            "weighted_accuracy": float(weighted_accuracy),
            "mean_client_accuracy": float(
                mean_client_accuracy
            ),
            "worst_client_accuracy": float(
                worst_client_accuracy
            ),
            "std_client_accuracy": float(
                std_client_accuracy
            ),
            "p10_client_accuracy": float(
                p10_client_accuracy
            ),
        })
    return pd.DataFrame(rows)

@dataclass
class BenchmarkRunResult:
    algorithm: AlgorithmName
    seed: int
    mu: float | None

    final_model: nn.Module
    condition_metadata: ConditionMetadata

    validation_by_round: pd.DataFrame
    validation_by_client: pd.DataFrame

    client_update_diagnostics: pd.DataFrame
    pairwise_update_diagnostics: pd.DataFrame
    round_diagnostics: pd.DataFrame

    local_test_by_client: pd.DataFrame
    local_test_summary: dict
    global_test_summary: dict

    runtime_seconds: float
    extra_artifacts: dict = field(
        default_factory=dict
    )

    def __post_init__(self):
        if self.seed != self.condition_metadata.seed:
            raise ValueError(
                "Run seed must match condition metadata seed"
            )

        if self.runtime_seconds < 0:
            raise ValueError(
                "runtime_seconds must be nonnegative"
            )

        if self.algorithm == "fedavg":
            if self.mu != 0.0:
                raise ValueError(
                    "FedAvg must use mu=0.0"
                )

        elif self.algorithm == "fedprox":
            if self.mu is None or self.mu <= 0:
                raise ValueError(
                    "FedProx must use a positive mu"
                )

        elif self.algorithm == "scaffold":
            if self.mu is not None:
                raise ValueError(
                    "SCAFFOLD does not use mu"
                )

        else:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm}"
            )

    def to_summary_record(self) -> dict:
        record = self.condition_metadata.to_record()

        record.update({
            "algorithm": self.algorithm,
            "seed": self.seed,
            "mu": self.mu,
            "runtime_seconds": self.runtime_seconds,

            "local_test_weighted_accuracy":
                self.local_test_summary[
                    "weighted_accuracy"
                ],

            "local_test_mean_client_accuracy":
                self.local_test_summary[
                    "mean_client_accuracy"
                ],

            "local_test_worst_client_accuracy":
                self.local_test_summary[
                    "worst_client_accuracy"
                ],

            "local_test_std_client_accuracy":
                self.local_test_summary[
                    "std_client_accuracy"
                ],

            **self.global_test_summary,
        })

        return record

def adapt_fed_run_result(
    *,
    run_result: RunResult,
    algorithm: Literal["fedavg", "fedprox"],
    mu: float,
    seed: int,
    initial_model: nn.Module,
    condition_metadata: ConditionMetadata,
    local_test_loaders: dict[int, DataLoader],
    global_test_loader: DataLoader,
    device: torch.device,
    runtime_seconds: float,
) -> BenchmarkRunResult:
    actual_model_keys = set(run_result.final_model_states.keys())
    if len(run_result.final_model_states) > 1:
        raise ValueError(
            "Fed benchmark adaptation requires exactly "
            f"one mu={mu}. Received model keys: "
            f"{actual_model_keys}"
            )

    raw_client_history = run_result.client_performance_history
    if "mu" in raw_client_history.columns:
        expected_mu_values = {float(mu)}

        observed_mu_values = set(
            raw_client_history["mu"].astype(float).unique().tolist()
        )

        if observed_mu_values != expected_mu_values:
            raise ValueError(
                "Client history contains unexpected "
                f"mu values: {observed_mu_values}. "
                f"Expected: {expected_mu_values}"
            )
        
    final_model = copy.deepcopy(initial_model)
    final_model.load_state_dict(run_result.final_model_states[float(mu)])
    final_model = final_model.to(device)
    local_test_by_client=evaluate_clients(
        model=final_model, 
        loaders=local_test_loaders,
        device=device,
    )
    validation_by_client = normalize_client_validation_history(
        history=run_result.client_performance_history,
        algorithm=algorithm,
        seed=seed,
    )
    validation_by_round = summarize_validation_history(
        validation_by_client
    )
    return BenchmarkRunResult(
        algorithm=algorithm,
        seed=seed,
        mu=mu,
        final_model=final_model,
        condition_metadata=condition_metadata,
        validation_by_client=validation_by_client,
        validation_by_round=validation_by_round,
        client_update_diagnostics=run_result.client_update_diagnostics.copy(),
        pairwise_update_diagnostics=run_result.pairwise_update_diagnostics.copy(),
        round_diagnostics=run_result.round_diagnostics.copy(),
        local_test_by_client=local_test_by_client,
        local_test_summary=summarize_client_evaluations(local_test_by_client),
        global_test_summary=evaluate_global_model(
            model=final_model,
            global_test_loader=global_test_loader,
            device=device,
        ),
        runtime_seconds=runtime_seconds,
        extra_artifacts={
            "raw_validation_by_round":
                run_result.performance_history.copy(),

            "raw_validation_by_client":
                run_result
                .client_performance_history
                .copy(),
        },
    )

def adapt_scaffold_run_result(
    *,
    scaffold_result: ScaffoldExperimentResult,
    seed: int,
    condition_metadata: ConditionMetadata,
    local_test_loaders: dict[int, DataLoader],
    global_test_loader: DataLoader,
    runtime_seconds: float,
    device: torch.device,
) -> BenchmarkRunResult:
    final_model = copy.deepcopy(scaffold_result.final_model).to(device)

    validation_by_client = normalize_client_validation_history(
        history=scaffold_result.validation_history,
        algorithm="scaffold",
        seed=seed,
    )

    validation_by_round = summarize_validation_history(
        validation_by_client
    )

    local_test_by_client = evaluate_clients(
        model=final_model,
        loaders=local_test_loaders,
        device=device,
    )

    local_test_summary = summarize_client_evaluations(
        local_test_by_client,
    )

    global_test_summary = evaluate_global_model(
        model=final_model,
        global_test_loader=global_test_loader,
        device=device,
    )

    final_server_control = {
        name: tensor.detach().cpu().clone()
        for name, tensor
        in scaffold_result.final_server_control.items()
    }

    final_client_controls = {
        client_id: {
            name: tensor.detach().cpu().clone()
            for name, tensor in control.items()
        }
        for client_id, control
        in scaffold_result.final_client_controls.items()
    }

    return BenchmarkRunResult(
        algorithm="scaffold",
        seed=seed,
        mu=None,
        final_model=final_model,
        condition_metadata=condition_metadata,
        validation_by_client=validation_by_client,
        validation_by_round=validation_by_round,
        client_update_diagnostics=scaffold_result.client_diagnostics,
        pairwise_update_diagnostics=scaffold_result.pairwise_diagnostics,
        round_diagnostics=scaffold_result.round_diagnostics,
        local_test_by_client=local_test_by_client,
        local_test_summary=local_test_summary,
        global_test_summary=global_test_summary,
        runtime_seconds=runtime_seconds,
        extra_artifacts={
            "raw_validation_by_round": scaffold_result.performance_history.copy(),
            "raw_validation_by_client": scaffold_result.validation_history.copy(),
            "scaffold_cost_summary": scaffold_result.cost_summary.copy(),
            "final_server_control": final_server_control,
            "final_client_controls": final_client_controls,
            "training_runtime_seconds": scaffold_result.training_runtime_seconds,
        }
    )