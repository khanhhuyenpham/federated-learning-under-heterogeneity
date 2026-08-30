from types import SimpleNamespace

import pytest
import torch
from torch import nn

import src.benchmark_runner as benchmark_runner
from src.condition_metadata import ConditionMetadata

def make_config():
    return SimpleNamespace(
        seed=42,
        batch_size=2,
    )

def make_metadata(
    *,
    seed=42,
    num_clients=2,
):
    return ConditionMetadata(
        mode="iid",
        seed=seed,
        num_clients=num_clients,
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

def make_runner_arguments():
    return {
        "config": make_config(),
        "condition_metadata": make_metadata(),
        "initial_model": nn.Linear(2, 2),
        "train_ds": object(),
        "client_train_indices": {
            0: [0, 1],
            1: [2, 3],
        },
        "validation_loaders": {
            0: object(),
            1: object(),
        },
        "local_test_loaders": {
            0: object(),
            1: object(),
        },
        "global_test_loader": object(),
        "loss_fn": nn.CrossEntropyLoss(),
        "device": torch.device("cpu"),
        "client_feature_transforms": None,
        "verbose": False,
    }

def test_fedavg_dispatches_one_mu(
    monkeypatch,
):
    raw_result = object()
    expected_result = object()
    calls = {}

    def fake_run_experiment(**kwargs):
        calls["runner"] = kwargs
        return raw_result

    def fake_adapter(**kwargs):
        calls["adapter"] = kwargs
        return expected_result

    monkeypatch.setattr(
        benchmark_runner,
        "run_experiment",
        fake_run_experiment,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "adapt_fed_run_result",
        fake_adapter,
    )

    result = (
        benchmark_runner.run_benchmark_condition(
            algorithm="fedavg",
            mu=0.0,
            **make_runner_arguments(),
        )
    )

    assert result is expected_result

    assert calls["runner"]["mu_values"] == (
        0.0,
    )

    assert calls["adapter"]["run_result"] is (
        raw_result
    )
    assert calls["adapter"]["algorithm"] == (
        "fedavg"
    )
    assert calls["adapter"]["mu"] == 0.0
    assert (
        calls["adapter"]["runtime_seconds"]
        >= 0.0
    )

@pytest.mark.parametrize(
    ("algorithm", "mu", "message"),
    [
        ("fedavg", 0.1, "FedAvg"),
        ("fedavg", None, "FedAvg"),
        ("fedprox", 0.0, "FedProx"),
        ("fedprox", -0.1, "FedProx"),
        ("fedprox", None, "FedProx"),
        ("fedprox", float("nan"), "FedProx"),
        ("fedprox", float("inf"), "FedProx"),
        ("scaffold", 0.0, "SCAFFOLD"),
    ],
)
def test_rejects_invalid_mu_contract(
    algorithm,
    mu,
    message,
):
    with pytest.raises(
        ValueError,
        match=message,
    ):
        benchmark_runner.run_benchmark_condition(
            algorithm=algorithm,
            mu=mu,
            **make_runner_arguments(),
        )