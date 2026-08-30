import re
import json
from dataclasses import asdict
from pathlib import Path
import pytest

from src.benchmark_storage import (
    BenchmarkRunKey,
    BenchmarkRunStore,
    _format_optional_float,
    _make_run_key_from_result,
    _run_key_mismatches,
    make_run_key,
)
from src.benchmark_suite import AlgorithmSpec
from src.heterogeneity import HeterogeneityConfig
from src.benchmark_protocol import (
    build_fedprox_pilot_algorithms,
    build_fedprox_pilot_conditions,
)

import pandas as pd
import torch
from torch import nn

from src.benchmark_results import BenchmarkRunResult
from src.condition_metadata import build_condition_metadata
from src.condition_metadata import ConditionMetadata
from src.heterogeneity import build_federated_data


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


def make_rotation_config(
    *,
    seed: int = 42,
    max_abs_angle: float = 20.0,
) -> HeterogeneityConfig:
    return HeterogeneityConfig(
        mode="rotation_shift",
        num_clients=5,
        seed=seed,
        alpha=None,
        min_samples_per_client=1,
        max_abs_angle=max_abs_angle,
    )


def test_format_optional_float_handles_none():
    assert _format_optional_float(None) == "NA"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0"),
        (0.1, "0p1"),
        (1.0, "1"),
        (20.0, "20"),
        (-0.25, "m0p25"),
    ],
)
def test_format_optional_float_produces_safe_text(
    value,
    expected,
):
    assert (
        _format_optional_float(value)
        == expected
    )


def test_make_run_key_copies_condition_and_algorithm_fields():
    condition_config = make_label_skew_config(
        seed=42,
        alpha=0.1,
    )

    num_clients = 5

    algorithm_spec = AlgorithmSpec(
        algorithm="fedprox",
        mu=0.1,
    )

    key = make_run_key(
        condition_config=condition_config,
        algorithm_spec=algorithm_spec,
    )

    assert key == BenchmarkRunKey(
        mode="label_skew",
        seed=42,
        num_clients=5,
        alpha=0.1,
        min_samples_per_client=1,
        max_abs_angle=0.0,
        algorithm="fedprox",
        mu=0.1,
    )


def test_identical_run_settings_produce_identical_keys():
    first_key = make_run_key(
        condition_config=(
            make_label_skew_config()
        ),
        algorithm_spec=AlgorithmSpec(
            algorithm="fedprox",
            mu=0.1,
        ),
    )

    second_key = make_run_key(
        condition_config=(
            make_label_skew_config()
        ),
        algorithm_spec=AlgorithmSpec(
            algorithm="fedprox",
            mu=0.1,
        ),
    )

    assert first_key == second_key
    assert hash(first_key) == hash(second_key)
    assert (
        first_key.to_slug()
        == second_key.to_slug()
    )


def test_different_seeds_produce_different_keys():
    first_key = make_run_key(
        condition_config=(
            make_label_skew_config(seed=42)
        ),
        algorithm_spec=AlgorithmSpec(
            algorithm="fedavg",
            mu=0.0,
        ),
    )

    second_key = make_run_key(
        condition_config=(
            make_label_skew_config(seed=43)
        ),
        algorithm_spec=AlgorithmSpec(
            algorithm="fedavg",
            mu=0.0,
        ),
    )

    assert first_key != second_key
    assert (
        first_key.to_slug()
        != second_key.to_slug()
    )


def test_different_alpha_values_produce_different_keys():
    first_key = make_run_key(
        condition_config=(
            make_label_skew_config(alpha=0.1)
        ),
        algorithm_spec=AlgorithmSpec(
            algorithm="fedavg",
            mu=0.0,
        ),
    )

    second_key = make_run_key(
        condition_config=(
            make_label_skew_config(alpha=0.5)
        ),
        algorithm_spec=AlgorithmSpec(
            algorithm="fedavg",
            mu=0.0,
        ),
    )

    assert first_key != second_key
    assert (
        first_key.to_slug()
        != second_key.to_slug()
    )


def test_different_rotation_angles_produce_different_keys():
    first_key = make_run_key(
        condition_config=(
            make_rotation_config(
                max_abs_angle=10.0
            )
        ),
        algorithm_spec=AlgorithmSpec(
            algorithm="scaffold",
            mu=None,
        ),
    )

    second_key = make_run_key(
        condition_config=(
            make_rotation_config(
                max_abs_angle=20.0
            )
        ),
        algorithm_spec=AlgorithmSpec(
            algorithm="scaffold",
            mu=None,
        ),
    )

    assert first_key != second_key
    assert (
        first_key.to_slug()
        != second_key.to_slug()
    )


def test_different_algorithms_produce_different_keys():
    condition_config = make_label_skew_config()

    fedavg_key = make_run_key(
        condition_config=condition_config,
        algorithm_spec=AlgorithmSpec(
            algorithm="fedavg",
            mu=0.0,
        ),
    )

    scaffold_key = make_run_key(
        condition_config=condition_config,
        algorithm_spec=AlgorithmSpec(
            algorithm="scaffold",
            mu=None,
        ),
    )

    assert fedavg_key != scaffold_key
    assert (
        fedavg_key.to_slug()
        != scaffold_key.to_slug()
    )


def test_different_fedprox_mu_values_produce_different_keys():
    condition_config = make_label_skew_config()

    first_key = make_run_key(
        condition_config=condition_config,
        algorithm_spec=AlgorithmSpec(
            algorithm="fedprox",
            mu=0.01,
        ),
    )

    second_key = make_run_key(
        condition_config=condition_config,
        algorithm_spec=AlgorithmSpec(
            algorithm="fedprox",
            mu=0.1,
        ),
    )

    assert first_key != second_key
    assert (
        first_key.to_slug()
        != second_key.to_slug()
    )


def test_slug_has_expected_human_readable_form():
    key = make_run_key(
        condition_config=(
            make_label_skew_config(
                seed=42,
                alpha=0.1,
            )
        ),
        algorithm_spec=AlgorithmSpec(
            algorithm="fedprox",
            mu=0.1,
        ),
    )

    assert key.to_slug() == (
        "label_skew"
        "_seed42"
        "_clients5"
        "_alpha0p1"
        "_min1"
        "_angle0"
        "_fedprox"
        "_mu0p1"
    )


def test_slug_represents_none_as_na():
    key = make_run_key(
        condition_config=make_rotation_config(
            seed=43,
            max_abs_angle=20.0,
        ),
        algorithm_spec=AlgorithmSpec(
            algorithm="scaffold",
            mu=None,
        ),
    )

    assert key.to_slug() == (
        "rotation_shift"
        "_seed43"
        "_clients5"
        "_alphaNA"
        "_min1"
        "_angle20"
        "_scaffold"
        "_muNA"
    )


def test_slug_contains_only_path_safe_characters():
    key = make_run_key(
        condition_config=(
            make_label_skew_config(
                alpha=0.1,
            )
        ),
        algorithm_spec=AlgorithmSpec(
            algorithm="fedprox",
            mu=0.01,
        ),
    )

    slug = key.to_slug()

    assert re.fullmatch(
        r"[A-Za-z0-9_]+",
        slug,
    )

    assert " " not in slug
    assert "." not in slug
    assert "/" not in slug
    assert "\\" not in slug

def make_fedprox_key():
    return make_run_key(
        condition_config=(
            make_label_skew_config(
                seed=42,
                alpha=0.1,
            )
        ),
        algorithm_spec=AlgorithmSpec(
            algorithm="fedprox",
            mu=0.1,
        ),
    )


def test_store_normalizes_root_directory(
    tmp_path,
):
    store = BenchmarkRunStore(
        root_directory=tmp_path
    )

    assert isinstance(
        store.root_directory,
        Path,
    )
    assert store.root_directory == tmp_path


def test_store_returns_expected_run_directory(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)
    key = make_fedprox_key()

    expected = (
        tmp_path
        / "runs"
        / key.to_slug()
    )

    assert store.run_directory(key) == expected


def test_new_run_is_not_complete(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)
    key = make_fedprox_key()

    assert not store.is_complete(key)


def test_directory_without_marker_is_incomplete(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)
    key = make_fedprox_key()

    run_directory = store.run_directory(key)
    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    partial_file = (
        run_directory
        / "summary.json"
    )

    partial_file.write_text(
        "{}",
        encoding="utf-8",
    )

    assert not store.is_complete(key)


def test_completion_marker_marks_run_complete(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)
    key = make_fedprox_key()

    run_directory = store.run_directory(key)
    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    completion_file = (
        run_directory
        / "COMPLETED.json"
    )

    completion_file.write_text(
        json.dumps({
            "status": "completed",
            "run_key": key.to_slug(),
        }),
        encoding="utf-8",
    )

    assert store.is_complete(key)

def test_different_client_counts_produce_different_keys():
    first_config = HeterogeneityConfig(
        mode="label_skew",
        num_clients=5,
        seed=42,
        alpha=0.1,
    )

    second_config = HeterogeneityConfig(
        mode="label_skew",
        num_clients=10,
        seed=42,
        alpha=0.1,
    )

    algorithm_spec = AlgorithmSpec(
        algorithm="fedprox",
        mu=0.1,
    )

    first_key = make_run_key(
        condition_config=first_config,
        algorithm_spec=algorithm_spec,
    )

    second_key = make_run_key(
        condition_config=second_config,
        algorithm_spec=algorithm_spec,
    )

    assert first_key != second_key
    assert (
        first_key.to_slug()
        != second_key.to_slug()
    )

def make_storage_result(
    *,
    algorithm="fedprox",
    mu=0.1,
    seed=42,
) -> BenchmarkRunResult:
    metadata = ConditionMetadata(
        mode="label_skew",
        seed=seed,
        num_clients=5,
        alpha=0.1,
        min_samples_per_client=1,
        max_abs_angle=0.0,
        dataset_size=50,
        min_client_size=10,
        max_client_size=10,
        quantity_cv=0.0,
        mean_label_tvd=0.5,
        rotation_std_degrees=0.0,
    )

    return BenchmarkRunResult(
        algorithm=algorithm,
        seed=seed,
        mu=mu,
        final_model=nn.Linear(2, 2),
        condition_metadata=metadata,
        validation_by_round=pd.DataFrame({
            "algorithm": [
                algorithm,
                algorithm,
            ],
            "seed": [seed, seed],
            "round": [0, 1],
            "weighted_accuracy": [0.5, 0.8],
        }),
        validation_by_client=pd.DataFrame({
            "algorithm": [
                algorithm,
                algorithm,
            ],
            "seed": [seed, seed],
            "round": [0, 1],
            "client_id": [0, 0],
            "num_examples": [10, 10],
            "accuracy": [0.5, 0.8],
        }),
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
        local_test_by_client=pd.DataFrame({
            "client_id": [0, 1, 2, 3, 4],
            "correct": [8, 9, 7, 8, 8],
            "total": [10, 10, 10, 10, 10],
            "accuracy": [
                0.8,
                0.9,
                0.7,
                0.8,
                0.8,
            ],
        }),
        local_test_summary={
            "weighted_accuracy": 0.8,
            "mean_client_accuracy": 0.8,
            "worst_client_accuracy": 0.7,
            "std_client_accuracy": 0.063,
        },
        global_test_summary={
            "global_test_correct": 45,
            "global_test_total": 50,
            "global_test_accuracy": 0.9,
        },
        runtime_seconds=1.5,
        extra_artifacts={
            "training_runtime_seconds": 1.2,
            "example_tensor": torch.tensor(
                [1.0, 2.0]
            ),
        },
    )

class TinyPilotDataset:
    def __init__(self):
        self.targets = torch.tensor(
            [label for label in range(5)] * 10,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (
            torch.tensor(
                [float(index)],
                dtype=torch.float32,
            ),
            self.targets[index],
        )

def test_real_pilot_condition_key_matches_result_metadata_key():
    condition_config = (
        build_fedprox_pilot_conditions()[0]
    )
    algorithm_spec = (
        build_fedprox_pilot_algorithms()[1]
    )
    dataset = TinyPilotDataset()
    federated_data = build_federated_data(
        train_ds=dataset,
        config=condition_config,
    )
    metadata = build_condition_metadata(
        train_ds=dataset,
        config=condition_config,
        federated_data=federated_data,
    )

    requested_key = make_run_key(
        condition_config=condition_config,
        algorithm_spec=algorithm_spec,
    )

    result = make_storage_result(
        algorithm=algorithm_spec.algorithm,
        mu=algorithm_spec.mu,
        seed=condition_config.seed,
    )
    result.condition_metadata = metadata

    result_key = _make_run_key_from_result(result)
    mismatches = _run_key_mismatches(
        requested_key=requested_key,
        result_key=result_key,
    )

    assert asdict(requested_key) == {
        "mode": "label_skew",
        "seed": 42,
        "num_clients": 5,
        "alpha": 0.1,
        "min_samples_per_client": 1,
        "max_abs_angle": 0.0,
        "algorithm": "fedprox",
        "mu": 0.1,
    }
    assert asdict(result_key) == asdict(
        requested_key
    )
    assert mismatches == {}

def test_store_saves_complete_run(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)
    key = make_fedprox_key()
    result = make_storage_result()

    saved_directory = store.save(
        key=key,
        result=result,
    )

    assert saved_directory == (
        store.run_directory(key)
    )
    assert store.is_complete(key)

    expected_files = {
        "summary.json",
        "condition_metadata.json",
        "local_test_summary.json",
        "global_test_summary.json",
        "validation_by_round.csv",
        "validation_by_client.csv",
        "client_update_diagnostics.csv",
        "pairwise_update_diagnostics.csv",
        "round_diagnostics.csv",
        "local_test_by_client.csv",
        "model_state.pt",
        "extra_artifacts.pt",
        "COMPLETED.json",
    }

    actual_files = {
        path.name
        for path in saved_directory.iterdir()
        if path.is_file()
    }

    assert actual_files == expected_files

def test_saved_json_contains_expected_values(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)
    key = make_fedprox_key()

    directory = store.save(
        key=key,
        result=make_storage_result(),
    )

    summary = json.loads(
        (
            directory / "summary.json"
        ).read_text(encoding="utf-8")
    )

    metadata = json.loads(
        (
            directory
            / "condition_metadata.json"
        ).read_text(encoding="utf-8")
    )

    completion = json.loads(
        (
            directory / "COMPLETED.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["algorithm"] == "fedprox"
    assert summary["seed"] == 42
    assert summary["mu"] == pytest.approx(0.1)
    assert summary[
        "global_test_accuracy"
    ] == pytest.approx(0.9)

    assert metadata["mode"] == "label_skew"
    assert metadata["num_clients"] == 5
    assert metadata["alpha"] == pytest.approx(
        0.1
    )

    assert completion["status"] == "completed"
    assert completion["schema_version"] == 1
    assert completion[
        "run_key"
    ] == key.to_slug()

def test_saved_dataframe_matches_result(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)
    key = make_fedprox_key()
    result = make_storage_result()

    directory = store.save(
        key=key,
        result=result,
    )

    saved_frame = pd.read_csv(
        directory / "validation_by_round.csv"
    )

    pd.testing.assert_frame_equal(
        saved_frame,
        result.validation_by_round,
        check_dtype=False,
    )

def test_saved_model_state_is_independent(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)
    key = make_fedprox_key()
    result = make_storage_result()

    directory = store.save(
        key=key,
        result=result,
    )

    saved_before_mutation = torch.load(
        directory / "model_state.pt",
        map_location="cpu",
        weights_only=True,
    )

    with torch.no_grad():
        for parameter in (
            result.final_model.parameters()
        ):
            parameter.add_(10.0)

    saved_after_mutation = torch.load(
        directory / "model_state.pt",
        map_location="cpu",
        weights_only=True,
    )

    for name in saved_before_mutation:
        assert torch.equal(
            saved_before_mutation[name],
            saved_after_mutation[name],
        )

def test_store_rejects_completed_run_overwrite(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)
    key = make_fedprox_key()
    result = make_storage_result()

    store.save(
        key=key,
        result=result,
    )

    with pytest.raises(
        FileExistsError,
        match="already complete",
    ):
        store.save(
            key=key,
            result=result,
        )

def test_store_rejects_key_result_mismatch(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)
    key = make_fedprox_key()

    mismatched_result = make_storage_result(
        algorithm="fedavg",
        mu=0.0,
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ) as error:
        store.save(
            key=key,
            result=mismatched_result,
        )

    assert "Mismatches" in str(error.value)
    assert "algorithm" in str(error.value)
    assert "mu" in str(error.value)
    assert not store.run_directory(key).exists()

def test_failed_save_removes_temporary_directory(
    tmp_path,
    monkeypatch,
):
    store = BenchmarkRunStore(tmp_path)
    key = make_fedprox_key()
    result = make_storage_result()

    def fail_during_write(**kwargs):
        raise RuntimeError(
            "simulated write failure"
        )

    monkeypatch.setattr(
        store,
        "_write_result_files",
        fail_during_write,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated write failure",
    ):
        store.save(
            key=key,
            result=result,
        )

    assert not store.run_directory(key).exists()
    assert not store.is_complete(key)

    runs_directory = tmp_path / "runs"

    temporary_directories = list(
        runs_directory.glob(
            f".{key.to_slug()}.tmp-*"
        )
    )

    assert temporary_directories == []

def test_is_spec_complete_builds_correct_key(
    tmp_path,
    monkeypatch,
):
    store = BenchmarkRunStore(tmp_path)
    condition_config = (
        make_label_skew_config()
    )
    algorithm_spec = AlgorithmSpec(
        algorithm="fedprox",
        mu=0.1,
    )

    expected_key = make_run_key(
        condition_config=condition_config,
        algorithm_spec=algorithm_spec,
    )

    observed_keys = []

    def fake_is_complete(key):
        observed_keys.append(key)
        return True

    monkeypatch.setattr(
        store,
        "is_complete",
        fake_is_complete,
    )

    assert store.is_spec_complete(
        condition_config,
        algorithm_spec,
    )

    assert observed_keys == [expected_key]

def test_save_spec_result_builds_correct_key(
    tmp_path,
    monkeypatch,
):
    store = BenchmarkRunStore(tmp_path)
    condition_config = (
        make_label_skew_config()
    )
    algorithm_spec = AlgorithmSpec(
        algorithm="fedprox",
        mu=0.1,
    )
    result = make_storage_result()

    expected_key = make_run_key(
        condition_config=condition_config,
        algorithm_spec=algorithm_spec,
    )

    save_calls = []

    def fake_save(
        *,
        key,
        result,
    ):
        save_calls.append((key, result))

    monkeypatch.setattr(
        store,
        "save",
        fake_save,
    )

    store.save_spec_result(
        condition_config,
        algorithm_spec,
        result,
    )

    assert save_calls == [
        (expected_key, result)
    ]

def test_load_summary_returns_saved_record(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)
    key = make_fedprox_key()
    result = make_storage_result()

    store.save(
        key=key,
        result=result,
    )

    summary = store.load_summary(key)

    assert summary["algorithm"] == "fedprox"
    assert summary["seed"] == 42
    assert summary["mu"] == pytest.approx(0.1)
    assert summary[
        "global_test_accuracy"
    ] == pytest.approx(0.9)

def test_load_summary_rejects_incomplete_run(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)
    key = make_fedprox_key()

    with pytest.raises(
        FileNotFoundError,
        match="not complete",
    ):
        store.load_summary(key)

def test_load_completed_summary_combines_all_runs(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)

    condition_config = (
        make_label_skew_config()
    )

    specifications = [
        AlgorithmSpec("fedavg", 0.0),
        AlgorithmSpec("fedprox", 0.1),
        AlgorithmSpec("scaffold", None),
    ]

    for specification in specifications:
        result = make_storage_result(
            algorithm=specification.algorithm,
            mu=specification.mu,
        )

        store.save_spec_result(
            condition_config,
            specification,
            result,
        )

    summary = store.load_completed_summary()

    assert len(summary) == 3

    assert set(summary["algorithm"]) == {
        "fedavg",
        "fedprox",
        "scaffold",
    }

    assert set(summary["seed"]) == {42}

def test_load_completed_summary_ignores_incomplete_directories(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)

    incomplete_directory = (
        store.root_directory
        / "runs"
        / "incomplete_run"
    )

    incomplete_directory.mkdir(
        parents=True,
    )

    (
        incomplete_directory
        / "summary.json"
    ).write_text(
        json.dumps({
            "algorithm": "fedavg",
        }),
        encoding="utf-8",
    )

    summary = store.load_completed_summary()

    assert summary.empty

def test_completed_run_missing_summary_is_reported(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)

    corrupted_directory = (
        store.root_directory
        / "runs"
        / "corrupted_run"
    )

    corrupted_directory.mkdir(
        parents=True,
    )

    (
        corrupted_directory
        / "COMPLETED.json"
    ).write_text(
        json.dumps({
            "status": "completed",
        }),
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
        match="missing summary.json",
    ):
        store.load_completed_summary()

def test_load_completed_validation_histories(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)
    condition = make_label_skew_config()

    specifications = [
        AlgorithmSpec("fedprox", 0.01),
        AlgorithmSpec("fedprox", 0.1),
    ]

    for specification in specifications:
        store.save_spec_result(
            condition,
            specification,
            make_storage_result(
                algorithm="fedprox",
                mu=specification.mu,
            ),
        )

    histories = store.load_completed_table(
        "validation_by_round"
    )

    assert len(histories) == 4

    assert set(histories["mu"]) == {
        0.01,
        0.1,
    }

    assert set(histories["algorithm"]) == {
        "fedprox"
    }

    assert set(histories["seed"]) == {42}
    assert set(histories["mode"]) == {
        "label_skew"
    }
    assert set(histories["alpha"]) == {
        0.1
    }

    assert histories["run_key"].nunique() == 2

def test_load_completed_table_rejects_unknown_name(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)

    with pytest.raises(
        ValueError,
        match="Unknown saved benchmark table",
    ):
        store.load_completed_table(
            "unknown_table"
        )

def test_load_completed_table_ignores_incomplete_run(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)

    incomplete_directory = (
        tmp_path / "runs" / "incomplete"
    )

    incomplete_directory.mkdir(
        parents=True
    )

    pd.DataFrame({
        "round": [0],
    }).to_csv(
        incomplete_directory
        / "validation_by_round.csv",
        index=False,
    )

    result = store.load_completed_table(
        "validation_by_round"
    )

    assert result.empty

def test_load_completed_table_detects_identity_mismatch(
    tmp_path,
):
    store = BenchmarkRunStore(tmp_path)
    key = make_fedprox_key()

    directory = store.save(
        key=key,
        result=make_storage_result(),
    )

    corrupted_history = pd.read_csv(
        directory / "validation_by_round.csv"
    )

    corrupted_history["algorithm"] = "fedavg"

    corrupted_history.to_csv(
        directory / "validation_by_round.csv",
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="identity does not match",
    ):
        store.load_completed_table(
            "validation_by_round"
        )
