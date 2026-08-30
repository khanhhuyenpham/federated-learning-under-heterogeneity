import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.benchmark_results import (
    BenchmarkRunResult,
    adapt_fed_run_result,
    adapt_scaffold_run_result,
)
from src.condition_metadata import ConditionMetadata
from src.experiment import RunResult
from src.scaffold import ScaffoldExperimentResult

class BinaryDataset(Dataset):
    def __init__(self, labels):
        self.targets = torch.tensor(
            labels,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        label = self.targets[index]

        features = torch.tensor(
            [float(label)],
            dtype=torch.float32,
        )

        return features, label


def make_perfect_binary_model():
    model = nn.Linear(
        in_features=1,
        out_features=2,
        bias=False,
    )

    with torch.no_grad():
        model.weight.copy_(
            torch.tensor([
                [-1.0],
                [1.0],
            ])
        )

    return model


def copy_model_state(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def make_metadata():
    return ConditionMetadata(
        mode="iid",
        seed=42,
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


def make_raw_client_history(mu=0.0):
    return pd.DataFrame({
        "seed": [42, 42, 42, 42],
        "mu": [mu, mu, mu, mu],
        "round": [0, 0, 1, 1],
        "client_id": [0, 1, 0, 1],
        "num_examples": [2, 2, 2, 2],
        "loss": [0.8, 0.7, 0.4, 0.3],
        "accuracy": [0.5, 0.5, 1.0, 1.0],
    })


def make_run_result(mu=0.0):
    trained_model = make_perfect_binary_model()

    return RunResult(
        performance_history=pd.DataFrame({
            "seed": [42, 42],
            "mu": [mu, mu],
            "round": [0, 1],
            "weighted_accuracy": [0.5, 1.0],
        }),
        client_performance_history=(
            make_raw_client_history(mu)
        ),
        client_update_diagnostics=pd.DataFrame({
            "round": [1],
            "client_id": [0],
            "update_norm": [0.5],
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
        final_model_states={
            float(mu): copy_model_state(trained_model)
        },
    )


def make_local_test_loaders():
    return {
        0: DataLoader(
            BinaryDataset([0, 1]),
            batch_size=2,
            shuffle=False,
        ),
        1: DataLoader(
            BinaryDataset([0, 1]),
            batch_size=2,
            shuffle=False,
        ),
    }


def make_global_test_loader():
    return DataLoader(
        BinaryDataset([0, 1, 0, 1]),
        batch_size=2,
        shuffle=False,
    )


def test_adapt_fedavg_run_result():
    initial_model = nn.Linear(
        1,
        2,
        bias=False,
    )

    result = adapt_fed_run_result(
        run_result=make_run_result(mu=0.0),
        algorithm="fedavg",
        mu=0.0,
        seed=42,
        initial_model=initial_model,
        condition_metadata=make_metadata(),
        local_test_loaders=(
            make_local_test_loaders()
        ),
        global_test_loader=(
            make_global_test_loader()
        ),
        device=torch.device("cpu"),
        runtime_seconds=1.25,
    )

    assert isinstance(result, BenchmarkRunResult)
    assert result.algorithm == "fedavg"
    assert result.mu == pytest.approx(0.0)
    assert result.runtime_seconds == pytest.approx(
        1.25
    )

    assert list(result.validation_by_client.columns) == [
        "algorithm",
        "seed",
        "round",
        "client_id",
        "num_examples",
        "accuracy",
    ]

    assert result.validation_by_round[
        "round"
    ].tolist() == [0, 1]

    assert result.local_test_by_client[
        "accuracy"
    ].tolist() == pytest.approx([1.0, 1.0])

    assert result.local_test_summary[
        "weighted_accuracy"
    ] == pytest.approx(1.0)

    assert result.global_test_summary[
        "global_test_accuracy"
    ] == pytest.approx(1.0)

    assert (
        "raw_validation_by_round"
        in result.extra_artifacts
    )
    assert (
        "raw_validation_by_client"
        in result.extra_artifacts
    )


def test_adapter_reconstructs_final_model_state():
    result = adapt_fed_run_result(
        run_result=make_run_result(mu=0.0),
        algorithm="fedavg",
        mu=0.0,
        seed=42,
        initial_model=nn.Linear(
            1,
            2,
            bias=False,
        ),
        condition_metadata=make_metadata(),
        local_test_loaders=(
            make_local_test_loaders()
        ),
        global_test_loader=(
            make_global_test_loader()
        ),
        device=torch.device("cpu"),
        runtime_seconds=1.0,
    )

    expected_model = make_perfect_binary_model()

    for parameter, expected_parameter in zip(
        result.final_model.parameters(),
        expected_model.parameters(),
    ):
        assert torch.equal(
            parameter.detach().cpu(),
            expected_parameter.detach().cpu(),
        )


def test_adapt_fedprox_run_result():
    result = adapt_fed_run_result(
        run_result=make_run_result(mu=0.1),
        algorithm="fedprox",
        mu=0.1,
        seed=42,
        initial_model=nn.Linear(
            1,
            2,
            bias=False,
        ),
        condition_metadata=make_metadata(),
        local_test_loaders=(
            make_local_test_loaders()
        ),
        global_test_loader=(
            make_global_test_loader()
        ),
        device=torch.device("cpu"),
        runtime_seconds=1.0,
    )

    assert result.algorithm == "fedprox"
    assert result.mu == pytest.approx(0.1)


def test_fed_adapter_rejects_multiple_final_models():
    run_result = make_run_result(mu=0.0)

    run_result.final_model_states[0.1] = (
        copy_model_state(
            make_perfect_binary_model()
        )
    )

    with pytest.raises(
        ValueError,
        match="exactly one",
    ):
        adapt_fed_run_result(
            run_result=run_result,
            algorithm="fedavg",
            mu=0.0,
            seed=42,
            initial_model=nn.Linear(
                1,
                2,
                bias=False,
            ),
            condition_metadata=make_metadata(),
            local_test_loaders=(
                make_local_test_loaders()
            ),
            global_test_loader=(
                make_global_test_loader()
            ),
            device=torch.device("cpu"),
            runtime_seconds=1.0,
        )


def test_fed_adapter_rejects_unexpected_history_mu():
    run_result = make_run_result(mu=0.0)

    run_result.client_performance_history["mu"] = 0.1

    with pytest.raises(
        ValueError,
        match="unexpected.*mu",
    ):
        adapt_fed_run_result(
            run_result=run_result,
            algorithm="fedavg",
            mu=0.0,
            seed=42,
            initial_model=nn.Linear(
                1,
                2,
                bias=False,
            ),
            condition_metadata=make_metadata(),
            local_test_loaders=(
                make_local_test_loaders()
            ),
            global_test_loader=(
                make_global_test_loader()
            ),
            device=torch.device("cpu"),
            runtime_seconds=1.0,
        )

def make_scaffold_result(
    seed: int = 42,
) -> ScaffoldExperimentResult:
    final_model = make_perfect_binary_model()

    server_control = {
        "weight": torch.zeros((2, 1)),
    }

    client_controls = {
        0: {
            "weight": torch.ones((2, 1)),
        },
        1: {
            "weight": torch.full(
                (2, 1),
                2.0,
            ),
        },
    }

    validation_history = pd.DataFrame({
        "method": [
            "scaffold",
            "scaffold",
            "scaffold",
            "scaffold",
        ],
        "seed": [seed, seed, seed, seed],
        "round": [0, 0, 1, 1],
        "client_id": [0, 1, 0, 1],
        "correct": [1, 1, 2, 2],
        "total": [2, 2, 2, 2],
        "accuracy": [0.5, 0.5, 1.0, 1.0],
    })

    return ScaffoldExperimentResult(
        final_model=final_model,
        final_server_control=server_control,
        final_client_controls=client_controls,
        validation_history=validation_history,
        performance_history=pd.DataFrame({
            "method": ["scaffold", "scaffold"],
            "seed": [seed, seed],
            "round": [0, 1],
            "weighted_accuracy": [0.5, 1.0],
        }),
        client_diagnostics=pd.DataFrame({
            "round": [1],
            "client_id": [0],
            "update_norm": [0.5],
        }),
        pairwise_diagnostics=pd.DataFrame({
            "round": [1],
            "client_a": [0],
            "client_b": [1],
            "cosine_similarity": [0.8],
        }),
        round_diagnostics=pd.DataFrame({
            "round": [1],
            "mean_update_norm": [0.5],
        }),
        cost_summary=pd.DataFrame({
            "metric": ["communication_bytes"],
            "value": [1000],
        }),
        training_runtime_seconds=2.5,
    )


def test_adapt_scaffold_run_result():
    result = adapt_scaffold_run_result(
        scaffold_result=make_scaffold_result(),
        seed=42,
        condition_metadata=make_metadata(),
        local_test_loaders=(
            make_local_test_loaders()
        ),
        global_test_loader=(
            make_global_test_loader()
        ),
        runtime_seconds=3.0,
        device=torch.device("cpu"),
    )

    assert isinstance(result, BenchmarkRunResult)
    assert result.algorithm == "scaffold"
    assert result.mu is None
    assert result.runtime_seconds == pytest.approx(
        3.0
    )
    assert result.extra_artifacts[
        "training_runtime_seconds"
    ] == pytest.approx(2.5)

    assert list(result.validation_by_client.columns) == [
        "algorithm",
        "seed",
        "round",
        "client_id",
        "num_examples",
        "accuracy",
    ]

    assert result.validation_by_client[
        "num_examples"
    ].tolist() == [2, 2, 2, 2]

    assert result.validation_by_round[
        "round"
    ].tolist() == [0, 1]

    assert result.validation_by_round[
        "weighted_accuracy"
    ].tolist() == pytest.approx([0.5, 1.0])

    assert result.local_test_by_client[
        "accuracy"
    ].tolist() == pytest.approx([1.0, 1.0])

    assert result.local_test_summary[
        "weighted_accuracy"
    ] == pytest.approx(1.0)

    assert result.global_test_summary[
        "global_test_accuracy"
    ] == pytest.approx(1.0)


def test_scaffold_adapter_preserves_extra_artifacts():
    result = adapt_scaffold_run_result(
        scaffold_result=make_scaffold_result(),
        seed=42,
        condition_metadata=make_metadata(),
        local_test_loaders=(
            make_local_test_loaders()
        ),
        global_test_loader=(
            make_global_test_loader()
        ),
        runtime_seconds=3.0,
        device=torch.device("cpu"),
    )

    assert (
        "raw_validation_by_round"
        in result.extra_artifacts
    )
    assert (
        "raw_validation_by_client"
        in result.extra_artifacts
    )
    assert (
        "scaffold_cost_summary"
        in result.extra_artifacts
    )
    assert (
        "final_server_control"
        in result.extra_artifacts
    )
    assert (
        "final_client_controls"
        in result.extra_artifacts
    )


def test_scaffold_control_snapshots_are_independent():
    scaffold_result = make_scaffold_result()

    result = adapt_scaffold_run_result(
        scaffold_result=scaffold_result,
        seed=42,
        condition_metadata=make_metadata(),
        local_test_loaders=(
            make_local_test_loaders()
        ),
        global_test_loader=(
            make_global_test_loader()
        ),
        runtime_seconds=3.0,
        device=torch.device("cpu"),
    )

    saved_server_control = (
        result.extra_artifacts[
            "final_server_control"
        ]["weight"]
    )

    saved_client_control = (
        result.extra_artifacts[
            "final_client_controls"
        ][0]["weight"]
    )

    expected_server_control = (
        saved_server_control.clone()
    )
    expected_client_control = (
        saved_client_control.clone()
    )

    # Mutate the original result after adaptation.
    scaffold_result.final_server_control[
        "weight"
    ].add_(10.0)

    scaffold_result.final_client_controls[
        0
    ]["weight"].add_(10.0)

    assert torch.equal(
        saved_server_control,
        expected_server_control,
    )
    assert torch.equal(
        saved_client_control,
        expected_client_control,
    )


def test_scaffold_adapter_copies_final_model():
    scaffold_result = make_scaffold_result()

    result = adapt_scaffold_run_result(
        scaffold_result=scaffold_result,
        seed=42,
        condition_metadata=make_metadata(),
        local_test_loaders=(
            make_local_test_loaders()
        ),
        global_test_loader=(
            make_global_test_loader()
        ),
        runtime_seconds=3.0,
        device=torch.device("cpu"),
    )

    saved_parameters = [
        parameter.detach().clone()
        for parameter in result.final_model.parameters()
    ]

    with torch.no_grad():
        for parameter in (
            scaffold_result.final_model.parameters()
        ):
            parameter.add_(10.0)

    for saved, current in zip(
        saved_parameters,
        result.final_model.parameters(),
    ):
        assert torch.equal(
            saved,
            current.detach(),
        )


def test_scaffold_adapter_rejects_history_seed_mismatch():
    scaffold_result = make_scaffold_result(
        seed=43
    )

    with pytest.raises(ValueError, match="seed"):
        adapt_scaffold_run_result(
            scaffold_result=scaffold_result,
            seed=42,
            condition_metadata=make_metadata(),
            local_test_loaders=(
                make_local_test_loaders()
            ),
            global_test_loader=(
                make_global_test_loader()
            ),
            runtime_seconds=3.0,
            device=torch.device("cpu"),
        )
