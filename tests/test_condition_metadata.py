import pytest
import torch

from src.condition_metadata import (
    ConditionMetadata,
    build_condition_metadata,
)
from src.heterogeneity import (
    FederatedData,
    HeterogeneityConfig,
    build_federated_data,
)


class ToyDataset:
    def __init__(self, targets):
        self.targets = torch.tensor(
            targets,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        features = torch.tensor(
            [float(index)],
            dtype=torch.float32,
        )
        return features, self.targets[index]


def test_iid_condition_metadata_records_basic_information():
    dataset = ToyDataset([0, 1] * 6)

    config = HeterogeneityConfig(
        mode="iid",
        num_clients=3,
        seed=42,
    )

    federated_data = build_federated_data(
        train_ds=dataset,
        config=config,
    )

    metadata = build_condition_metadata(
        train_ds=dataset,
        config=config,
        federated_data=federated_data,
    )

    assert isinstance(metadata, ConditionMetadata)
    assert metadata.mode == "iid"
    assert metadata.seed == 42
    assert metadata.num_clients == 3
    assert metadata.dataset_size == 12
    assert metadata.min_client_size == 4
    assert metadata.max_client_size == 4
    assert metadata.quantity_cv == pytest.approx(0.0)
    assert metadata.rotation_std_degrees == pytest.approx(0.0)


def test_label_skew_metadata_preserves_alpha():
    dataset = ToyDataset(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    )

    config = HeterogeneityConfig(
        mode="label_skew",
        num_clients=3,
        seed=42,
        alpha=0.5,
    )

    federated_data = build_federated_data(dataset, config)

    metadata = build_condition_metadata(
        dataset,
        config,
        federated_data,
    )

    assert metadata.mode == "label_skew"
    assert metadata.alpha == pytest.approx(0.5)
    assert metadata.min_client_size == 4
    assert metadata.max_client_size == 4


def test_quantity_skew_metadata_preserves_configuration():
    dataset = ToyDataset([0, 1] * 50)

    config = HeterogeneityConfig(
        mode="quantity_skew",
        num_clients=5,
        seed=42,
        alpha=1.0,
        min_samples_per_client=4,
    )

    federated_data = build_federated_data(dataset, config)

    metadata = build_condition_metadata(
        dataset,
        config,
        federated_data,
    )

    assert metadata.mode == "quantity_skew"
    assert metadata.alpha == pytest.approx(1.0)
    assert metadata.min_samples_per_client == 4
    assert metadata.min_client_size >= 4
    assert metadata.max_client_size >= metadata.min_client_size


def test_rotation_metadata_records_rotation_variation():
    dataset = ToyDataset([0, 1] * 6)

    config = HeterogeneityConfig(
        mode="rotation_shift",
        num_clients=3,
        seed=42,
        max_abs_angle=20.0,
    )

    federated_data = build_federated_data(dataset, config)

    metadata = build_condition_metadata(
        dataset,
        config,
        federated_data,
    )

    assert metadata.max_abs_angle == pytest.approx(20.0)
    assert metadata.rotation_std_degrees > 0
    assert metadata.min_client_size == 4
    assert metadata.max_client_size == 4


def test_condition_metadata_converts_to_dictionary():
    dataset = ToyDataset([0, 1] * 4)

    config = HeterogeneityConfig(
        mode="iid",
        num_clients=2,
        seed=42,
    )

    federated_data = build_federated_data(dataset, config)

    metadata = build_condition_metadata(
        dataset,
        config,
        federated_data,
    )

    record = metadata.to_record()

    assert isinstance(record, dict)
    assert record["mode"] == "iid"
    assert record["seed"] == 42
    assert record["dataset_size"] == 8


def test_condition_metadata_rejects_empty_partition():
    dataset = ToyDataset([0, 1])

    config = HeterogeneityConfig(
        mode="iid",
        num_clients=2,
        seed=42,
    )

    empty_federated_data = FederatedData(
        client_indices={},
        client_feature_transforms=None,
    )

    with pytest.raises(ValueError, match="empty"):
        build_condition_metadata(
            train_ds=dataset,
            config=config,
            federated_data=empty_federated_data,
        )