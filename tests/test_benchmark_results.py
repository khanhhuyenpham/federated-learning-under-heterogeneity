import pandas as pd
import pytest
from torch import nn

from src.benchmark_results import (
    BenchmarkRunResult,
    normalize_client_validation_history,
    summarize_validation_history,
)
from src.condition_metadata import ConditionMetadata

def test_normalize_fedavg_client_history():
    history = pd.DataFrame({
        "seed": [42, 42],
        "round": [0, 0],
        "client_id": [0, 1],
        "num_examples": [10, 20],
        "loss": [0.8, 0.6],
        "accuracy": [0.5, 0.75],
    })

    normalized = normalize_client_validation_history(
        history=history,
        algorithm="fedavg",
        seed=42,
    )

    assert list(normalized.columns) == [
        "algorithm",
        "seed",
        "round",
        "client_id",
        "num_examples",
        "accuracy",
    ]

    assert normalized["algorithm"].tolist() == [
        "fedavg",
        "fedavg",
    ]
    assert normalized["seed"].tolist() == [42, 42]
    assert normalized["num_examples"].tolist() == [
        10,
        20,
    ]
    assert normalized["accuracy"].tolist() == [
        0.5,
        0.75,
    ]


def test_normalize_scaffold_renames_total():
    history = pd.DataFrame({
        "seed": [42, 42],
        "round": [0, 0],
        "client_id": [0, 1],
        "correct": [8, 15],
        "total": [10, 20],
        "accuracy": [0.8, 0.75],
    })

    normalized = normalize_client_validation_history(
        history=history,
        algorithm="scaffold",
        seed=42,
    )

    assert normalized["num_examples"].tolist() == [
        10,
        20,
    ]
    assert normalized["algorithm"].tolist() == [
        "scaffold",
        "scaffold",
    ]


def test_normalization_does_not_modify_input():
    history = pd.DataFrame({
        "round": [0],
        "client_id": [0],
        "num_examples": [10],
        "accuracy": [0.5],
    })

    original = history.copy(deep=True)

    normalize_client_validation_history(
        history=history,
        algorithm="fedavg",
        seed=42,
    )

    pd.testing.assert_frame_equal(
        history,
        original,
    )


@pytest.mark.parametrize(
    "missing_column",
    [
        "round",
        "client_id",
        "accuracy",
    ],
)
def test_normalization_rejects_missing_required_column(
    missing_column,
):
    history = pd.DataFrame({
        "round": [0],
        "client_id": [0],
        "num_examples": [10],
        "accuracy": [0.5],
    }).drop(columns=missing_column)

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        normalize_client_validation_history(
            history=history,
            algorithm="fedavg",
            seed=42,
        )


def test_normalization_rejects_missing_sample_count():
    history = pd.DataFrame({
        "round": [0],
        "client_id": [0],
        "accuracy": [0.5],
    })

    with pytest.raises(
        ValueError,
        match="num_examples|total|sample",
    ):
        normalize_client_validation_history(
            history=history,
            algorithm="fedavg",
            seed=42,
        )


def test_normalization_rejects_inconsistent_seed():
    history = pd.DataFrame({
        "seed": [43],
        "round": [0],
        "client_id": [0],
        "num_examples": [10],
        "accuracy": [0.5],
    })

    with pytest.raises(ValueError, match="seed"):
        normalize_client_validation_history(
            history=history,
            algorithm="fedavg",
            seed=42,
        )


@pytest.mark.parametrize("invalid_count", [0, -1])
def test_normalization_rejects_nonpositive_sample_count(
    invalid_count,
):
    history = pd.DataFrame({
        "round": [0],
        "client_id": [0],
        "num_examples": [invalid_count],
        "accuracy": [0.5],
    })

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        normalize_client_validation_history(
            history=history,
            algorithm="fedavg",
            seed=42,
        )


@pytest.mark.parametrize(
    "invalid_accuracy",
    [-0.1, 1.1],
)
def test_normalization_rejects_invalid_accuracy(
    invalid_accuracy,
):
    history = pd.DataFrame({
        "round": [0],
        "client_id": [0],
        "num_examples": [10],
        "accuracy": [invalid_accuracy],
    })

    with pytest.raises(
        ValueError,
        match="accuracy",
    ):
        normalize_client_validation_history(
            history=history,
            algorithm="fedavg",
            seed=42,
        )


def test_normalization_rejects_empty_history():
    history = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        normalize_client_validation_history(
            history=history,
            algorithm="fedavg",
            seed=42,
        )


def test_summarize_validation_history_calculates_metrics():
    client_history = pd.DataFrame({
        "algorithm": [
            "fedavg",
            "fedavg",
            "fedavg",
            "fedavg",
        ],
        "seed": [42, 42, 42, 42],
        "round": [0, 0, 1, 1],
        "client_id": [0, 1, 0, 1],
        "num_examples": [1, 3, 1, 3],
        "accuracy": [0.0, 1.0, 0.5, 0.5],
    })

    summary = summarize_validation_history(
        client_history
    )

    assert summary["round"].tolist() == [0, 1]

    round_zero = summary.loc[
        summary["round"] == 0
    ].iloc[0]

    assert round_zero["weighted_accuracy"] == (
        pytest.approx(0.75)
    )
    assert round_zero["mean_client_accuracy"] == (
        pytest.approx(0.5)
    )
    assert round_zero["worst_client_accuracy"] == (
        pytest.approx(0.0)
    )
    assert round_zero["std_client_accuracy"] == (
        pytest.approx(0.5)
    )
    assert round_zero["p10_client_accuracy"] == (
        pytest.approx(0.1)
    )

    round_one = summary.loc[
        summary["round"] == 1
    ].iloc[0]

    assert round_one["weighted_accuracy"] == (
        pytest.approx(0.5)
    )
    assert round_one["mean_client_accuracy"] == (
        pytest.approx(0.5)
    )
    assert round_one["worst_client_accuracy"] == (
        pytest.approx(0.5)
    )
    assert round_one["std_client_accuracy"] == (
        pytest.approx(0.0)
    )
    assert round_one["p10_client_accuracy"] == (
        pytest.approx(0.5)
    )


def test_summary_rejects_multiple_algorithms():
    client_history = pd.DataFrame({
        "algorithm": ["fedavg", "scaffold"],
        "seed": [42, 42],
        "round": [0, 0],
        "client_id": [0, 1],
        "num_examples": [10, 10],
        "accuracy": [0.5, 0.5],
    })

    with pytest.raises(
        ValueError,
        match="one algorithm",
    ):
        summarize_validation_history(
            client_history
        )


def test_summary_rejects_multiple_seeds():
    client_history = pd.DataFrame({
        "algorithm": ["fedavg", "fedavg"],
        "seed": [42, 43],
        "round": [0, 0],
        "client_id": [0, 1],
        "num_examples": [10, 10],
        "accuracy": [0.5, 0.5],
    })

    with pytest.raises(
        ValueError,
        match="one seed",
    ):
        summarize_validation_history(
            client_history
        )

def make_condition_metadata(
    seed: int = 42,
) -> ConditionMetadata:
    return ConditionMetadata(
        mode="iid",
        seed=seed,
        num_clients=2,
        alpha=None,
        min_samples_per_client=1,
        max_abs_angle=0.0,
        dataset_size=20,
        min_client_size=10,
        max_client_size=10,
        quantity_cv=0.0,
        mean_label_tvd=0.0,
        rotation_std_degrees=0.0,
    )


def make_benchmark_result(
    *,
    algorithm="fedavg",
    seed=42,
    mu=0.0,
    metadata_seed=42,
    runtime_seconds=1.5,
):
    return BenchmarkRunResult(
        algorithm=algorithm,
        seed=seed,
        mu=mu,
        final_model=nn.Linear(2, 2),
        condition_metadata=make_condition_metadata(
            seed=metadata_seed
        ),
        validation_by_round=pd.DataFrame(),
        validation_by_client=pd.DataFrame(),
        client_update_diagnostics=pd.DataFrame(),
        pairwise_update_diagnostics=pd.DataFrame(),
        round_diagnostics=pd.DataFrame(),
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


@pytest.mark.parametrize(
    ("algorithm", "mu"),
    [
        ("fedavg", 0.0),
        ("fedprox", 0.1),
        ("scaffold", None),
    ],
)
def test_benchmark_result_accepts_valid_algorithm_contract(
    algorithm,
    mu,
):
    result = make_benchmark_result(
        algorithm=algorithm,
        mu=mu,
    )

    assert result.algorithm == algorithm
    assert result.mu == mu


@pytest.mark.parametrize(
    ("algorithm", "mu", "message"),
    [
        ("fedavg", 0.1, "FedAvg"),
        ("fedavg", None, "FedAvg"),
        ("fedprox", 0.0, "FedProx"),
        ("fedprox", -0.1, "FedProx"),
        ("fedprox", None, "FedProx"),
        ("scaffold", 0.0, "SCAFFOLD"),
    ],
)
def test_benchmark_result_rejects_invalid_mu_contract(
    algorithm,
    mu,
    message,
):
    with pytest.raises(ValueError, match=message):
        make_benchmark_result(
            algorithm=algorithm,
            mu=mu,
        )


def test_benchmark_result_rejects_unknown_algorithm():
    with pytest.raises(
        ValueError,
        match="Unknown algorithm",
    ):
        make_benchmark_result(
            algorithm="unknown",
            mu=None,
        )


def test_benchmark_result_rejects_metadata_seed_mismatch():
    with pytest.raises(
        ValueError,
        match="seed",
    ):
        make_benchmark_result(
            seed=42,
            metadata_seed=43,
        )


def test_benchmark_result_rejects_negative_runtime():
    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        make_benchmark_result(
            runtime_seconds=-1.0,
        )


def test_benchmark_result_creates_summary_record():
    result = make_benchmark_result(
        algorithm="fedprox",
        mu=0.1,
    )

    record = result.to_summary_record()

    assert isinstance(record, dict)

    # Condition metadata
    assert record["mode"] == "iid"
    assert record["num_clients"] == 2
    assert record["dataset_size"] == 20

    # Run metadata
    assert record["algorithm"] == "fedprox"
    assert record["seed"] == 42
    assert record["mu"] == pytest.approx(0.1)
    assert record["runtime_seconds"] == pytest.approx(
        1.5
    )

    # Local held-out evaluation
    assert record[
        "local_test_weighted_accuracy"
    ] == pytest.approx(0.85)

    assert record[
        "local_test_mean_client_accuracy"
    ] == pytest.approx(0.85)

    assert record[
        "local_test_worst_client_accuracy"
    ] == pytest.approx(0.8)

    assert record[
        "local_test_std_client_accuracy"
    ] == pytest.approx(0.05)

    # Clean global evaluation
    assert record["global_test_correct"] == 18
    assert record["global_test_total"] == 20
    assert record[
        "global_test_accuracy"
    ] == pytest.approx(0.9)


def test_extra_artifacts_uses_independent_dictionary():
    first = make_benchmark_result()
    second = make_benchmark_result()

    first.extra_artifacts["example"] = "value"

    assert "example" not in second.extra_artifacts
