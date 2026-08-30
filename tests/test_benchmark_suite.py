import dataclasses

import pandas as pd
import pytest
from torch import nn

from src.benchmark_results import BenchmarkRunResult
from src.benchmark_suite import (
    AlgorithmSpec,
    BenchmarkSuiteResult,
    summarize_benchmark_runs,
)
from src.condition_metadata import ConditionMetadata
from types import SimpleNamespace

import torch

import src.benchmark_suite as benchmark_suite
from src.heterogeneity import HeterogeneityConfig
from torch.utils.data import DataLoader, Dataset

from src.benchmark_storage import (
    BenchmarkRunStore,
    make_run_key,
)

def make_condition_metadata(
    *,
    seed: int = 42,
) -> ConditionMetadata:
    return ConditionMetadata(
        mode="iid",
        seed=seed,
        num_clients=2,
        alpha=None,
        min_samples_per_client=1,
        max_abs_angle=0.0,
        dataset_size=8,
        min_client_size=4,
        max_client_size=4,
        quantity_cv=0.0,
        mean_label_tvd=0.0,
        rotation_std_degrees=0.0,
    )

def make_label_skew_config(
    *,
    seed: int = 42,
    alpha: float = 0.1,
) -> HeterogeneityConfig:
    return HeterogeneityConfig(
        mode="label_skew",
        num_clients=5,
        seed=seed,
        alpha=alpha,
        min_samples_per_client=1,
        max_abs_angle=0.0,
    )


def make_benchmark_result(
    *,
    algorithm="fedavg",
    seed: int = 42,
    mu: float | None = 0.0,
    runtime_seconds: float = 1.5,
) -> BenchmarkRunResult:
    return BenchmarkRunResult(
        algorithm=algorithm,
        seed=seed,
        mu=mu,
        final_model=nn.Linear(2, 2),
        condition_metadata=(
            make_condition_metadata(seed=seed)
        ),
        validation_by_round=pd.DataFrame({
            "algorithm": [algorithm, algorithm],
            "seed": [seed, seed],
            "round": [0, 1],
            "weighted_accuracy": [0.5, 0.9],
            "mean_client_accuracy": [0.5, 0.9],
            "worst_client_accuracy": [0.4, 0.8],
            "std_client_accuracy": [0.1, 0.05],
            "p10_client_accuracy": [0.42, 0.82],
        }),
        validation_by_client=pd.DataFrame({
            "algorithm": [
                algorithm,
                algorithm,
                algorithm,
                algorithm,
            ],
            "seed": [seed, seed, seed, seed],
            "round": [0, 0, 1, 1],
            "client_id": [0, 1, 0, 1],
            "num_examples": [2, 2, 2, 2],
            "accuracy": [0.4, 0.6, 0.8, 1.0],
        }),
        client_update_diagnostics=pd.DataFrame({
            "round": [1, 1],
            "client_id": [0, 1],
            "update_norm": [0.4, 0.6],
        }),
        pairwise_update_diagnostics=pd.DataFrame({
            "round": [1],
            "client_a": [0],
            "client_b": [1],
            "cosine_similarity": [0.8],
        }),
        round_diagnostics=pd.DataFrame({
            "round": [1],
            "mean_update_norm": [0.5],
        }),
        local_test_by_client=pd.DataFrame({
            "client_id": [0, 1],
            "correct": [8, 9],
            "total": [10, 10],
            "accuracy": [0.8, 0.9],
        }),
        local_test_summary={
            "weighted_accuracy": 0.85,
            "mean_client_accuracy": 0.85,
            "worst_client_accuracy": 0.8,
            "std_client_accuracy": 0.05,
        },
        global_test_summary={
            "global_test_correct": 18,
            "global_test_total": 20,
            "global_test_accuracy": 0.9,
        },
        runtime_seconds=runtime_seconds,
    )


def test_algorithm_spec_stores_condition():
    spec = AlgorithmSpec(
        algorithm="fedprox",
        mu=0.1,
    )

    assert spec.algorithm == "fedprox"
    assert spec.mu == pytest.approx(0.1)


def test_algorithm_spec_is_frozen():
    spec = AlgorithmSpec(
        algorithm="fedavg",
        mu=0.0,
    )

    with pytest.raises(
        dataclasses.FrozenInstanceError
    ):
        spec.mu = 0.1


def test_summarize_one_benchmark_run():
    run = make_benchmark_result()

    summary = summarize_benchmark_runs([run])

    assert len(summary) == 1
    assert summary.loc[0, "algorithm"] == "fedavg"
    assert summary.loc[0, "seed"] == 42
    assert summary.loc[0, "mu"] == pytest.approx(0.0)
    assert summary.loc[
        0,
        "runtime_seconds",
    ] == pytest.approx(1.5)
    assert summary.loc[
        0,
        "global_test_accuracy",
    ] == pytest.approx(0.9)


def test_summarize_multiple_benchmark_runs():
    fedavg_result = make_benchmark_result(
        algorithm="fedavg",
        mu=0.0,
        runtime_seconds=1.5,
    )

    fedprox_result = make_benchmark_result(
        algorithm="fedprox",
        mu=0.1,
        runtime_seconds=2.0,
    )

    summary = summarize_benchmark_runs([
        fedavg_result,
        fedprox_result,
    ])

    assert len(summary) == 2

    assert summary["algorithm"].tolist() == [
        "fedavg",
        "fedprox",
    ]

    assert summary["mu"].tolist() == pytest.approx([
        0.0,
        0.1,
    ])

    assert summary[
        "runtime_seconds"
    ].tolist() == pytest.approx([
        1.5,
        2.0,
    ])


def test_summarize_empty_run_list():
    summary = summarize_benchmark_runs([])

    assert isinstance(summary, pd.DataFrame)
    assert summary.empty


def test_suite_result_accepts_matching_summary():
    runs = [
        make_benchmark_result(
            algorithm="fedavg",
            mu=0.0,
        ),
        make_benchmark_result(
            algorithm="fedprox",
            mu=0.1,
        ),
    ]

    summary = summarize_benchmark_runs(runs)

    result = BenchmarkSuiteResult(
        runs=runs,
        summary=summary,
    )

    assert result.runs == runs
    assert len(result.summary) == 2


def test_suite_result_requires_one_row_per_run():
    runs = [make_benchmark_result()]

    with pytest.raises(
        ValueError,
        match="one row per run",
    ):
        BenchmarkSuiteResult(
            runs=runs,
            summary=pd.DataFrame(),
        )

def make_heterogeneity_config():
    return HeterogeneityConfig(
        mode="iid",
        num_clients=2,
        seed=42,
        alpha=None,
        min_samples_per_client=1,
        max_abs_angle=0.0,
    )


def make_training_config(
    condition_config,
):
    return SimpleNamespace(
        seed=condition_config.seed,
        alpha=condition_config.alpha,
        batch_size=2,
        num_rounds=1,
        local_epochs=1,
        learning_rate=0.01,
    )


def test_suite_prepares_condition_once_and_runs_all_algorithms(
    monkeypatch,
):
    condition_config = (
        make_heterogeneity_config()
    )

    algorithm_specs = [
        AlgorithmSpec(
            algorithm="fedavg",
            mu=0.0,
        ),
        AlgorithmSpec(
            algorithm="fedprox",
            mu=0.1,
        ),
        AlgorithmSpec(
            algorithm="scaffold",
            mu=None,
        ),
    ]

    train_indices = {
        0: [0, 1],
        1: [2, 3],
    }

    validation_loaders = {
        0: object(),
        1: object(),
    }

    local_test_loaders = {
        0: object(),
        1: object(),
    }

    metadata = make_condition_metadata(
        seed=42
    )

    fake_artifacts = SimpleNamespace(
        client_train_indices=train_indices,
        metadata=metadata,
        loaders=SimpleNamespace(
            validation=validation_loaders,
            test=local_test_loaders,
        ),
        federated_data=SimpleNamespace(
            client_feature_transforms=None,
        ),
    )

    artifact_calls = []
    model_calls = []
    run_calls = []

    def fake_build_condition_artifacts(
        **kwargs,
    ):
        artifact_calls.append(kwargs)
        return fake_artifacts

    def fake_model_factory():
        model = nn.Linear(2, 2)
        model_calls.append(model)
        return model

    def fake_run_benchmark_condition(
        **kwargs,
    ):
        run_calls.append(kwargs)

        return make_benchmark_result(
            algorithm=kwargs["algorithm"],
            seed=kwargs["config"].seed,
            mu=kwargs["mu"],
        )

    monkeypatch.setattr(
        benchmark_suite,
        "build_condition_artifacts",
        fake_build_condition_artifacts,
    )

    monkeypatch.setattr(
        benchmark_suite,
        "run_benchmark_condition",
        fake_run_benchmark_condition,
    )

    train_ds = object()
    global_test_loader = object()
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    result = benchmark_suite.run_benchmark_suite(
        train_ds=train_ds,
        global_test_loader=global_test_loader,
        condition_configs=[condition_config],
        algorithm_specs=algorithm_specs,
        training_config_factory=(
            make_training_config
        ),
        model_factory=fake_model_factory,
        loss_fn=loss_fn,
        device=device,
        verbose=False,
    )

    assert len(artifact_calls) == 1
    assert len(model_calls) == 1
    assert len(run_calls) == 3

    assert [
        (
            call["algorithm"],
            call["mu"],
        )
        for call in run_calls
    ] == [
        ("fedavg", 0.0),
        ("fedprox", 0.1),
        ("scaffold", None),
    ]

    shared_initial_model = model_calls[0]

    assert all(
        call["initial_model"]
        is shared_initial_model
        for call in run_calls
    )

    assert all(
        call["client_train_indices"]
        is train_indices
        for call in run_calls
    )

    assert all(
        call["validation_loaders"]
        is validation_loaders
        for call in run_calls
    )

    assert all(
        call["local_test_loaders"]
        is local_test_loaders
        for call in run_calls
    )

    assert all(
        call["condition_metadata"]
        is metadata
        for call in run_calls
    )

    assert all(
        call["train_ds"] is train_ds
        for call in run_calls
    )

    assert all(
        call["global_test_loader"]
        is global_test_loader
        for call in run_calls
    )

    assert len(result.runs) == 3
    assert len(result.summary) == 3

    assert result.summary[
        "algorithm"
    ].tolist() == [
        "fedavg",
        "fedprox",
        "scaffold",
    ]

def test_suite_runs_every_algorithm_for_every_condition(
    monkeypatch,
):
    condition_configs = [
        HeterogeneityConfig(
            mode="iid",
            num_clients=2,
            seed=42,
            alpha=None,
        ),
        HeterogeneityConfig(
            mode="iid",
            num_clients=2,
            seed=43,
            alpha=None,
        ),
    ]

    algorithm_specs = [
        AlgorithmSpec("fedavg", 0.0),
        AlgorithmSpec("fedprox", 0.1),
        AlgorithmSpec("scaffold", None),
    ]

    artifact_calls = []
    model_calls = []
    run_calls = []

    def fake_build_condition_artifacts(
        *,
        train_ds,
        config,
        batch_size,
        train_fraction,
        validation_fraction,
    ):
        artifacts = SimpleNamespace(
            client_train_indices={
                0: [0, 1],
                1: [2, 3],
            },
            metadata=make_condition_metadata(
                seed=config.seed
            ),
            loaders=SimpleNamespace(
                validation={
                    0: object(),
                    1: object(),
                },
                test={
                    0: object(),
                    1: object(),
                },
            ),
            federated_data=SimpleNamespace(
                client_feature_transforms=None,
            ),
        )

        artifact_calls.append({
            "config": config,
            "artifacts": artifacts,
        })

        return artifacts

    def fake_model_factory():
        model = nn.Linear(2, 2)
        model_calls.append(model)
        return model

    def fake_run_benchmark_condition(
        **kwargs,
    ):
        run_calls.append(kwargs)

        return make_benchmark_result(
            algorithm=kwargs["algorithm"],
            seed=kwargs["config"].seed,
            mu=kwargs["mu"],
        )

    monkeypatch.setattr(
        benchmark_suite,
        "build_condition_artifacts",
        fake_build_condition_artifacts,
    )

    monkeypatch.setattr(
        benchmark_suite,
        "run_benchmark_condition",
        fake_run_benchmark_condition,
    )

    result = benchmark_suite.run_benchmark_suite(
        train_ds=object(),
        global_test_loader=object(),
        condition_configs=condition_configs,
        algorithm_specs=algorithm_specs,
        training_config_factory=(
            make_training_config
        ),
        model_factory=fake_model_factory,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    assert len(artifact_calls) == 2
    assert len(model_calls) == 2
    assert len(run_calls) == 6
    assert len(result.runs) == 6
    assert len(result.summary) == 6

    assert [
        (
            call["config"].seed,
            call["algorithm"],
            call["mu"],
        )
        for call in run_calls
    ] == [
        (42, "fedavg", 0.0),
        (42, "fedprox", 0.1),
        (42, "scaffold", None),
        (43, "fedavg", 0.0),
        (43, "fedprox", 0.1),
        (43, "scaffold", None),
    ]

    first_condition_model = (
        run_calls[0]["initial_model"]
    )

    second_condition_model = (
        run_calls[3]["initial_model"]
    )

    assert all(
        call["initial_model"]
        is first_condition_model
        for call in run_calls[:3]
    )

    assert all(
        call["initial_model"]
        is second_condition_model
        for call in run_calls[3:]
    )

    assert (
        first_condition_model
        is not second_condition_model
    )

def test_suite_rejects_empty_conditions():
    with pytest.raises(
        ValueError,
        match="condition_configs",
    ):
        benchmark_suite.run_benchmark_suite(
            train_ds=object(),
            global_test_loader=object(),
            condition_configs=[],
            algorithm_specs=[
                AlgorithmSpec("fedavg", 0.0)
            ],
            training_config_factory=(
                make_training_config
            ),
            model_factory=lambda: nn.Linear(2, 2),
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )


def test_suite_rejects_empty_algorithm_specs():
    with pytest.raises(
        ValueError,
        match="algorithm_specs",
    ):
        benchmark_suite.run_benchmark_suite(
            train_ds=object(),
            global_test_loader=object(),
            condition_configs=[
                make_heterogeneity_config()
            ],
            algorithm_specs=[],
            training_config_factory=(
                make_training_config
            ),
            model_factory=lambda: nn.Linear(2, 2),
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )

def test_training_config_rejects_missing_attributes():
    training_config = SimpleNamespace(
        seed=42,
        alpha=None,
    )

    with pytest.raises(
        TypeError,
        match="missing required attributes",
    ):
        benchmark_suite._validate_training_config(
            training_config=training_config,
            condition_config=(
                make_heterogeneity_config()
            ),
        )


def test_training_config_rejects_seed_mismatch():
    training_config = make_training_config(
        make_heterogeneity_config()
    )
    training_config.seed = 43

    with pytest.raises(
        ValueError,
        match="seed",
    ):
        benchmark_suite._validate_training_config(
            training_config=training_config,
            condition_config=(
                make_heterogeneity_config()
            ),
        )


def test_training_config_rejects_alpha_mismatch():
    condition_config = HeterogeneityConfig(
        mode="label_skew",
        num_clients=2,
        seed=42,
        alpha=0.1,
    )

    training_config = make_training_config(
        condition_config
    )
    training_config.alpha = 0.5

    with pytest.raises(
        ValueError,
        match="alpha",
    ):
        benchmark_suite._validate_training_config(
            training_config=training_config,
            condition_config=condition_config,
        )

class TinyBinaryDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 40,
    ):
        self.targets = torch.tensor(
            [
                index % 2
                for index in range(num_samples)
            ],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        label = self.targets[index]

        if label.item() == 0:
            features = torch.tensor(
                [-1.0],
                dtype=torch.float32,
            )
        else:
            features = torch.tensor(
                [1.0],
                dtype=torch.float32,
            )

        return features, label

def test_benchmark_suite_end_to_end_smoke():
    dataset = TinyBinaryDataset(
        num_samples=40
    )

    global_test_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
    )

    condition_config = HeterogeneityConfig(
        mode="iid",
        num_clients=2,
        seed=42,
        alpha=None,
        min_samples_per_client=1,
        max_abs_angle=0.0,
    )

    algorithm_specs = [
        AlgorithmSpec(
            algorithm="fedavg",
            mu=0.0,
        ),
        AlgorithmSpec(
            algorithm="fedprox",
            mu=0.1,
        ),
        AlgorithmSpec(
            algorithm="scaffold",
            mu=None,
        ),
    ]

    def training_config_factory(
        condition,
    ):
        return SimpleNamespace(
            seed=condition.seed,
            alpha=condition.alpha,
            batch_size=4,
            num_rounds=1,
            local_epochs=1,
            learning_rate=0.1,
        )

    def model_factory():
        return nn.Linear(
            in_features=1,
            out_features=2,
        )

    result = benchmark_suite.run_benchmark_suite(
        train_ds=dataset,
        global_test_loader=global_test_loader,
        condition_configs=[condition_config],
        algorithm_specs=algorithm_specs,
        training_config_factory=(
            training_config_factory
        ),
        model_factory=model_factory,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        train_fraction=0.6,
        validation_fraction=0.2,
        verbose=False,
    )

    assert len(result.runs) == 3
    assert len(result.summary) == 3

    assert [
        run.algorithm
        for run in result.runs
    ] == [
        "fedavg",
        "fedprox",
        "scaffold",
    ]

    for run in result.runs:
        assert run.seed == 42

        assert run.validation_by_round[
            "round"
        ].tolist() == [0, 1]

        assert set(
            run.local_test_by_client[
                "client_id"
            ]
        ) == {0, 1}

        assert (
            0.0
            <= run.local_test_summary[
                "weighted_accuracy"
            ]
            <= 1.0
        )

        assert (
            0.0
            <= run.global_test_summary[
                "global_test_accuracy"
            ]
            <= 1.0
        )

        assert run.runtime_seconds >= 0.0

def make_fake_suite_artifacts(
    *,
    seed: int = 42,
):
    return SimpleNamespace(
        client_train_indices={
            0: [0, 1],
            1: [2, 3],
        },
        metadata=make_condition_metadata(
            seed=seed
        ),
        loaders=SimpleNamespace(
            validation={
                0: object(),
                1: object(),
            },
            test={
                0: object(),
                1: object(),
            },
        ),
        federated_data=SimpleNamespace(
            client_feature_transforms=None,
        ),
    )

def test_suite_skips_completed_run_and_saves_new_runs(
    monkeypatch,
):
    condition_config = (
        make_heterogeneity_config()
    )

    algorithm_specs = [
        AlgorithmSpec("fedavg", 0.0),
        AlgorithmSpec("fedprox", 0.1),
        AlgorithmSpec("scaffold", None),
    ]

    checker_calls = []
    build_calls = []
    model_calls = []
    run_calls = []
    save_calls = []

    def fake_is_run_complete(
        condition,
        algorithm_spec,
    ):
        checker_calls.append(
            (condition, algorithm_spec)
        )

        return (
            algorithm_spec.algorithm
            == "fedavg"
        )

    def fake_build_condition_artifacts(
        **kwargs,
    ):
        build_calls.append(kwargs)

        return make_fake_suite_artifacts(
            seed=kwargs["config"].seed
        )

    def fake_model_factory():
        model = nn.Linear(2, 2)
        model_calls.append(model)
        return model

    def fake_run_benchmark_condition(
        **kwargs,
    ):
        run_calls.append(kwargs)

        return make_benchmark_result(
            algorithm=kwargs["algorithm"],
            seed=kwargs["config"].seed,
            mu=kwargs["mu"],
        )

    def fake_on_run_complete(
        condition,
        algorithm_spec,
        result,
    ):
        save_calls.append((
            condition,
            algorithm_spec,
            result,
        ))

    monkeypatch.setattr(
        benchmark_suite,
        "build_condition_artifacts",
        fake_build_condition_artifacts,
    )

    monkeypatch.setattr(
        benchmark_suite,
        "run_benchmark_condition",
        fake_run_benchmark_condition,
    )

    suite_result = (
        benchmark_suite.run_benchmark_suite(
            train_ds=object(),
            global_test_loader=object(),
            condition_configs=[
                condition_config
            ],
            algorithm_specs=algorithm_specs,
            training_config_factory=(
                make_training_config
            ),
            model_factory=fake_model_factory,
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            is_run_complete=(
                fake_is_run_complete
            ),
            on_run_complete=(
                fake_on_run_complete
            ),
        )
    )

    assert len(checker_calls) == 3
    assert len(build_calls) == 1
    assert len(model_calls) == 1

    assert [
        (
            call["algorithm"],
            call["mu"],
        )
        for call in run_calls
    ] == [
        ("fedprox", 0.1),
        ("scaffold", None),
    ]

    assert len(save_calls) == 2

    assert [
        (
            algorithm_spec.algorithm,
            algorithm_spec.mu,
        )
        for (
            _,
            algorithm_spec,
            _,
        ) in save_calls
    ] == [
        ("fedprox", 0.1),
        ("scaffold", None),
    ]

    assert len(suite_result.runs) == 2
    assert len(suite_result.summary) == 2

def test_suite_does_not_prepare_fully_completed_condition(
    monkeypatch,
):
    condition_config = (
        make_heterogeneity_config()
    )

    algorithm_specs = [
        AlgorithmSpec("fedavg", 0.0),
        AlgorithmSpec("fedprox", 0.1),
        AlgorithmSpec("scaffold", None),
    ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Completed condition should not "
            "be prepared or executed"
        )

    monkeypatch.setattr(
        benchmark_suite,
        "build_condition_artifacts",
        fail_if_called,
    )

    monkeypatch.setattr(
        benchmark_suite,
        "run_benchmark_condition",
        fail_if_called,
    )

    result = benchmark_suite.run_benchmark_suite(
        train_ds=object(),
        global_test_loader=object(),
        condition_configs=[condition_config],
        algorithm_specs=algorithm_specs,
        training_config_factory=fail_if_called,
        model_factory=fail_if_called,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        is_run_complete=(
            lambda condition, spec: True
        ),
    )

    assert result.runs == []
    assert result.summary.empty

def test_suite_stops_when_checkpoint_save_fails(
    monkeypatch,
):
    condition_config = (
        make_heterogeneity_config()
    )

    algorithm_specs = [
        AlgorithmSpec("fedavg", 0.0),
        AlgorithmSpec("fedprox", 0.1),
        AlgorithmSpec("scaffold", None),
    ]

    run_calls = []
    save_calls = []

    monkeypatch.setattr(
        benchmark_suite,
        "build_condition_artifacts",
        lambda **kwargs: (
            make_fake_suite_artifacts(
                seed=kwargs["config"].seed
            )
        ),
    )

    def fake_run_benchmark_condition(
        **kwargs,
    ):
        run_calls.append(kwargs)

        return make_benchmark_result(
            algorithm=kwargs["algorithm"],
            seed=kwargs["config"].seed,
            mu=kwargs["mu"],
        )

    def failing_save_callback(
        condition,
        algorithm_spec,
        result,
    ):
        save_calls.append((
            condition,
            algorithm_spec,
            result,
        ))

        raise RuntimeError(
            "simulated checkpoint failure"
        )

    monkeypatch.setattr(
        benchmark_suite,
        "run_benchmark_condition",
        fake_run_benchmark_condition,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated checkpoint failure",
    ):
        benchmark_suite.run_benchmark_suite(
            train_ds=object(),
            global_test_loader=object(),
            condition_configs=[
                condition_config
            ],
            algorithm_specs=algorithm_specs,
            training_config_factory=(
                make_training_config
            ),
            model_factory=lambda: (
                nn.Linear(2, 2)
            ),
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            on_run_complete=(
                failing_save_callback
            ),
        )

    assert len(run_calls) == 1
    assert len(save_calls) == 1
    assert (
        run_calls[0]["algorithm"]
        == "fedavg"
    )

def test_suite_resumes_after_interrupted_execution(
    tmp_path,
    monkeypatch,
):
    condition_config = (
        make_heterogeneity_config()
    )

    algorithm_specs = [
        AlgorithmSpec("fedavg", 0.0),
        AlgorithmSpec("fedprox", 0.1),
        AlgorithmSpec("scaffold", None),
    ]

    store = BenchmarkRunStore(
        tmp_path / "benchmark_v1"
    )

    monkeypatch.setattr(
        benchmark_suite,
        "build_condition_artifacts",
        lambda **kwargs: (
            make_fake_suite_artifacts(
                seed=kwargs["config"].seed
            )
        ),
    )

    first_phase_calls = []

    def interrupted_runner(**kwargs):
        first_phase_calls.append((
            kwargs["algorithm"],
            kwargs["mu"],
        ))

        if kwargs["algorithm"] == "fedprox":
            raise RuntimeError(
                "simulated training interruption"
            )

        return make_benchmark_result(
            algorithm=kwargs["algorithm"],
            seed=kwargs["config"].seed,
            mu=kwargs["mu"],
        )

    monkeypatch.setattr(
        benchmark_suite,
        "run_benchmark_condition",
        interrupted_runner,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated training interruption",
    ):
        benchmark_suite.run_benchmark_suite(
            train_ds=object(),
            global_test_loader=object(),
            condition_configs=[
                condition_config
            ],
            algorithm_specs=algorithm_specs,
            training_config_factory=(
                make_training_config
            ),
            model_factory=lambda: (
                nn.Linear(2, 2)
            ),
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            is_run_complete=(
                store.is_spec_complete
            ),
            on_run_complete=(
                store.save_spec_result
            ),
        )

    assert first_phase_calls == [
        ("fedavg", 0.0),
        ("fedprox", 0.1),
    ]

    fedavg_key = make_run_key(
        condition_config=condition_config,
        algorithm_spec=algorithm_specs[0],
    )

    fedprox_key = make_run_key(
        condition_config=condition_config,
        algorithm_spec=algorithm_specs[1],
    )

    scaffold_key = make_run_key(
        condition_config=condition_config,
        algorithm_spec=algorithm_specs[2],
    )

    assert store.is_complete(fedavg_key)
    assert not store.is_complete(fedprox_key)
    assert not store.is_complete(scaffold_key)

    second_phase_calls = []

    def successful_runner(**kwargs):
        second_phase_calls.append((
            kwargs["algorithm"],
            kwargs["mu"],
        ))

        return make_benchmark_result(
            algorithm=kwargs["algorithm"],
            seed=kwargs["config"].seed,
            mu=kwargs["mu"],
        )

    monkeypatch.setattr(
        benchmark_suite,
        "run_benchmark_condition",
        successful_runner,
    )

    resumed_result = (
        benchmark_suite.run_benchmark_suite(
            train_ds=object(),
            global_test_loader=object(),
            condition_configs=[
                condition_config
            ],
            algorithm_specs=algorithm_specs,
            training_config_factory=(
                make_training_config
            ),
            model_factory=lambda: (
                nn.Linear(2, 2)
            ),
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            is_run_complete=(
                store.is_spec_complete
            ),
            on_run_complete=(
                store.save_spec_result
            ),
        )
    )

    assert second_phase_calls == [
        ("fedprox", 0.1),
        ("scaffold", None),
    ]

    assert len(resumed_result.runs) == 2
    assert len(resumed_result.summary) == 2

    assert resumed_result.summary[
        "algorithm"
    ].tolist() == [
        "fedprox",
        "scaffold",
    ]

    assert store.is_complete(fedavg_key)
    assert store.is_complete(fedprox_key)
    assert store.is_complete(scaffold_key)

    completed_directories = [
        path
        for path in (
            store.root_directory / "runs"
        ).iterdir()
        if (
            path.is_dir()
            and (
                path / "COMPLETED.json"
            ).is_file()
        )
    ]

    assert len(completed_directories) == 3