import pytest
import torch

from src.data import validate_partition
from src.heterogeneity import (
    FederatedData,
    HeterogeneityConfig,
    build_federated_data,
)


class ToyDataset:
    def __init__(self, targets):
        self.targets = torch.tensor(targets)

    def __len__(self):
        return len(self.targets)

def test_build_federated_data_iid():
    dataset = ToyDataset([0, 1, 2] * 20)

    config = HeterogeneityConfig(
        mode="iid",
        num_clients=3,
        seed=42,
    )

    result = build_federated_data(dataset, config)

    assert isinstance(result, FederatedData)
    assert result.client_feature_transforms is None

    validate_partition(
        result.client_indices,
        dataset_size=len(dataset),
        expected_num_clients=3,
    )

    client_sizes = [
        len(indices)
        for indices in result.client_indices.values()
    ]

    assert max(client_sizes) - min(client_sizes) <= 1

def test_build_federated_data_label_skew():
    dataset = ToyDataset([0, 1, 2] * 20)

    config = HeterogeneityConfig(
        mode="label_skew",
        num_clients=3,
        seed=42,
        alpha=0.1,
    )

    result = build_federated_data(dataset, config)

    assert result.client_feature_transforms is None

    validate_partition(
        result.client_indices,
        dataset_size=len(dataset),
        expected_num_clients=3,
    )

    client_sizes = [
        len(indices)
        for indices in result.client_indices.values()
    ]

    assert max(client_sizes) - min(client_sizes) <= 1

def test_build_federated_data_quantity_skew():
    dataset = ToyDataset([0, 1, 2] * 100)

    config = HeterogeneityConfig(
        mode="quantity_skew",
        num_clients=5,
        seed=42,
        alpha=0.1,
        min_samples_per_client=2,
    )

    result = build_federated_data(dataset, config)

    assert result.client_feature_transforms is None

    validate_partition(
        result.client_indices,
        dataset_size=len(dataset),
        expected_num_clients=5,
    )

    client_sizes = [
        len(indices)
        for indices in result.client_indices.values()
    ]

    assert min(client_sizes) >= 2

def test_build_federated_data_rotation_shift():
    dataset = ToyDataset([0, 1, 2] * 20)

    config = HeterogeneityConfig(
        mode="rotation_shift",
        num_clients=5,
        seed=42,
        max_abs_angle=20,
    )

    result = build_federated_data(dataset, config)

    validate_partition(
        result.client_indices,
        dataset_size=len(dataset),
        expected_num_clients=5,
    )

    assert result.client_feature_transforms is not None
    assert set(result.client_feature_transforms) == {
        0, 1, 2, 3, 4
    }

    angles = [
        result.client_feature_transforms[client_id].angle
        for client_id in range(5)
    ]

    assert angles == pytest.approx(
        [-20, -10, 0, 10, 20]
    )

@pytest.mark.parametrize(
    "mode",
    ["label_skew", "quantity_skew"],
)
def test_build_federated_data_requires_alpha(mode):
    dataset = ToyDataset([0, 1] * 20)

    config = HeterogeneityConfig(
        mode=mode,
        num_clients=2,
        seed=42,
        alpha=None,
    )

    with pytest.raises(ValueError, match="alpha.*required"):
        build_federated_data(dataset, config)

def test_build_federated_data_rejects_unknown_mode():
    dataset = ToyDataset([0, 1] * 20)

    config = HeterogeneityConfig(
        mode="unknown",
        num_clients=2,
        seed=42,
        alpha=0.5,
    )

    with pytest.raises(ValueError, match="mode"):
        build_federated_data(dataset, config)