from dataclasses import dataclass

import pandas as pd
import math
from .benchmark_results import (
    AlgorithmName,
    BenchmarkRunResult,
)

from collections.abc import Callable, Sequence

import torch 
from torch import nn
from .condition_artifacts import (
    build_condition_artifacts,
)
from .experiment import set_seed
from .heterogeneity import HeterogeneityConfig
from .benchmark_runner import (
    run_benchmark_condition,
)

@dataclass(frozen=True)
class AlgorithmSpec:
    algorithm: AlgorithmName
    mu: float

@dataclass
class BenchmarkSuiteResult:
    runs: list[BenchmarkRunResult]
    summary: pd.DataFrame

    def __post_init__(self) -> None:
        if len(self.summary) != len(self.runs):
            raise ValueError(
                "Benchmark suite summary must contain "
                "exactly one row per run"
            )

def summarize_benchmark_runs(
    runs: list[BenchmarkRunResult],
) -> pd.DataFrame:
    records = [
        run.to_summary_record()
        for run in runs
    ]

    return pd.DataFrame(records)

def _validate_suite_inputs(
    condition_configs: Sequence[
        HeterogeneityConfig
    ],
    algorithm_specs: Sequence[AlgorithmSpec],
) -> None:
    if not condition_configs:
        raise ValueError(
            "condition_configs cannot be empty"
        )

    if not algorithm_specs:
        raise ValueError(
            "algorithm_specs cannot be empty"
        )
    
def _validate_matching_alpha(
    *,
    training_alpha: float | None,
    condition_alpha: float | None,
) -> None:
    if (
        training_alpha is None
        and condition_alpha is None
    ):
        return

    if (
        training_alpha is None
        or condition_alpha is None
    ):
        raise ValueError(
            "Training config alpha must match "
            "the heterogeneity condition alpha"
        )

    if not math.isclose(
        float(training_alpha),
        float(condition_alpha),
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError(
            "Training config alpha must match "
            "the heterogeneity condition alpha"
        )

def _validate_training_config(
    *,
    training_config,
    condition_config: HeterogeneityConfig,
) -> None:
    required_attributes = (
        "seed",
        "alpha",
        "batch_size",
        "num_rounds",
        "local_epochs",
        "learning_rate",
    )

    missing_attributes = [
        attribute
        for attribute in required_attributes
        if not hasattr(
            training_config,
            attribute,
        )
    ]

    if missing_attributes:
        raise TypeError(
            "Training config is missing required "
            "attributes: "
            f"{missing_attributes}"
        )

    if training_config.seed != condition_config.seed:
        raise ValueError(
            "Training config seed must match "
            "the heterogeneity condition seed"
        )

    _validate_matching_alpha(
        training_alpha=training_config.alpha,
        condition_alpha=condition_config.alpha,
    )

    if training_config.batch_size <= 0:
        raise ValueError(
            "Training config batch_size must "
            "be positive"
        )

    if training_config.num_rounds <= 0:
        raise ValueError(
            "Training config num_rounds must "
            "be positive"
        )

    if training_config.local_epochs <= 0:
        raise ValueError(
            "Training config local_epochs must "
            "be positive"
        )

    if (
        not math.isfinite(
            float(training_config.learning_rate)
        )
        or training_config.learning_rate <= 0.0
    ):
        raise ValueError(
            "Training config learning_rate "
            "must be finite and positive"
        )

def run_benchmark_suite(
    *,
    train_ds,
    global_test_loader,
    condition_configs: Sequence[
        HeterogeneityConfig
    ],
    algorithm_specs: Sequence[AlgorithmSpec],
    training_config_factory: Callable,
    model_factory: Callable[[], nn.Module],
    loss_fn: nn.Module,
    device: torch.device,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    verbose: bool = False,
    is_run_complete: Callable[
        [HeterogeneityConfig, AlgorithmSpec],
        bool,
    ] | None = None,
    on_run_complete: Callable[
        [
            HeterogeneityConfig,
            AlgorithmSpec,
            BenchmarkRunResult,
        ],
        None,
    ] | None = None,
) -> BenchmarkSuiteResult:
    runs = []

    _validate_suite_inputs(
        condition_configs=condition_configs,
        algorithm_specs=algorithm_specs,
    )

    for condition_config in condition_configs:
        pending_algorithm_specs = [
            algorithm_spec
            for algorithm_spec in algorithm_specs
            if (
                is_run_complete is None
                or not is_run_complete(
                    condition_config,
                    algorithm_spec,
                )
            )
        ]

        if not pending_algorithm_specs:
            continue

        training_config = training_config_factory(
            condition_config
        )

        _validate_training_config(
            training_config=training_config,
            condition_config=condition_config,
        )

        artifacts = build_condition_artifacts(
            train_ds=train_ds,
            config=condition_config,
            batch_size=training_config.batch_size,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )

        set_seed(condition_config.seed)
        initial_model = model_factory()

        for algorithm_spec in pending_algorithm_specs:
            run = run_benchmark_condition(
                algorithm=algorithm_spec.algorithm,
                mu=algorithm_spec.mu,
                config=training_config,
                condition_metadata=artifacts.metadata,
                initial_model=initial_model,
                train_ds=train_ds,
                client_train_indices=(
                    artifacts.client_train_indices
                ),
                validation_loaders=(
                    artifacts.loaders.validation
                ),
                local_test_loaders=(
                    artifacts.loaders.test
                ),
                global_test_loader=(
                    global_test_loader
                ),
                loss_fn=loss_fn,
                device=device,
                client_feature_transforms=(
                    artifacts
                    .federated_data
                    .client_feature_transforms
                ),
                verbose=verbose,
            )

            if on_run_complete is not None:
                on_run_complete(
                    condition_config,
                    algorithm_spec,
                    run,
                )

            runs.append(run)

    summary = summarize_benchmark_runs(runs)

    return BenchmarkSuiteResult(
        runs=runs,
        summary=summary,
    )

