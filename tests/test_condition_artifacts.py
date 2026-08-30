import pytest
import torch
from torch.utils.data import Dataset

from src.condition_artifacts import (
    ConditionArtifacts,
    build_condition_artifacts,
)
from src.data import (
    ClientSplitLoaders,
    validate_partition,
)
from src.heterogeneity import HeterogeneityConfig


class ToyDataset(Dataset):
    def __init__(self, targets):
        self.targets = torch.tensor(
            targets,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # Image-shaped features make this dataset compatible
        # with torchvision rotation transforms.
        features = torch.full(
            (1, 4, 4),
            fill_value=float(index),
            dtype=torch.float32,
        )

        return features, self.targets[index]


def test_build_condition_artifacts_creates_complete_result():
    dataset = ToyDataset([0, 1, 2] * 10)

    config = HeterogeneityConfig(
        mode="iid",
        num_clients=3,
        seed=42,
    )

    artifacts = build_condition_artifacts(
        train_ds=dataset,
        config=config,
        batch_size=4,
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    assert isinstance(artifacts, ConditionArtifacts)
    assert isinstance(artifacts.loaders, ClientSplitLoaders)

    assert artifacts.config == config

    assert set(artifacts.federated_data.client_indices) == {
        0,
        1,
        2,
    }
    assert set(artifacts.client_train_indices) == {0, 1, 2}
    assert set(artifacts.client_validation_indices) == {
        0,
        1,
        2,
    }
    assert set(artifacts.client_test_indices) == {0, 1, 2}

    assert set(artifacts.loaders.train) == {0, 1, 2}
    assert set(artifacts.loaders.validation) == {0, 1, 2}
    assert set(artifacts.loaders.test) == {0, 1, 2}


def test_condition_artifacts_full_partition_is_valid():
    dataset = ToyDataset([0, 1, 2] * 10)

    config = HeterogeneityConfig(
        mode="label_skew",
        num_clients=3,
        seed=42,
        alpha=0.5,
    )

    artifacts = build_condition_artifacts(
        train_ds=dataset,
        config=config,
        batch_size=4,
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    validate_partition(
        client_indices=(
            artifacts.federated_data.client_indices
        ),
        dataset_size=len(dataset),
        expected_num_clients=3,
    )


def test_local_splits_are_disjoint_and_reconstruct_partition():
    dataset = ToyDataset([0, 1, 2] * 10)

    config = HeterogeneityConfig(
        mode="iid",
        num_clients=3,
        seed=42,
    )

    artifacts = build_condition_artifacts(
        train_ds=dataset,
        config=config,
        batch_size=4,
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    for client_id in range(config.num_clients):
        full_indices = set(
            artifacts.federated_data.client_indices[
                client_id
            ]
        )

        train_indices = set(
            artifacts.client_train_indices[client_id]
        )
        validation_indices = set(
            artifacts.client_validation_indices[client_id]
        )
        test_indices = set(
            artifacts.client_test_indices[client_id]
        )

        assert train_indices.isdisjoint(validation_indices)
        assert train_indices.isdisjoint(test_indices)
        assert validation_indices.isdisjoint(test_indices)

        assert (
            train_indices
            | validation_indices
            | test_indices
        ) == full_indices

        # Each client has 10 samples:
        # train = int(10 * 0.6) = 6
        # validation = int(10 * 0.2) = 2
        # test = 2
        assert len(train_indices) == 6
        assert len(validation_indices) == 2
        assert len(test_indices) == 2


def test_condition_artifacts_are_reproducible():
    dataset = ToyDataset([0, 1, 2] * 10)

    config = HeterogeneityConfig(
        mode="label_skew",
        num_clients=3,
        seed=42,
        alpha=0.5,
    )

    first = build_condition_artifacts(
        train_ds=dataset,
        config=config,
        batch_size=4,
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    second = build_condition_artifacts(
        train_ds=dataset,
        config=config,
        batch_size=4,
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    assert (
        first.federated_data.client_indices
        == second.federated_data.client_indices
    )
    assert (
        first.client_train_indices
        == second.client_train_indices
    )
    assert (
        first.client_validation_indices
        == second.client_validation_indices
    )
    assert (
        first.client_test_indices
        == second.client_test_indices
    )


def test_rotation_transforms_are_propagated_to_all_local_splits():
    dataset = ToyDataset([0, 1, 2] * 10)

    config = HeterogeneityConfig(
        mode="rotation_shift",
        num_clients=3,
        seed=42,
        max_abs_angle=20.0,
    )

    artifacts = build_condition_artifacts(
        train_ds=dataset,
        config=config,
        batch_size=4,
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    transforms = (
        artifacts.federated_data.client_feature_transforms
    )

    assert transforms is not None

    for client_id in range(config.num_clients):
        expected_transform = transforms[client_id]

        train_subset = (
            artifacts.loaders.train[client_id].dataset
        )
        validation_subset = (
            artifacts.loaders.validation[client_id].dataset
        )
        test_subset = (
            artifacts.loaders.test[client_id].dataset
        )

        assert (
            train_subset.feature_transform
            is expected_transform
        )
        assert (
            validation_subset.feature_transform
            is expected_transform
        )
        assert (
            test_subset.feature_transform
            is expected_transform
        )


def test_condition_artifacts_metadata_matches_condition():
    dataset = ToyDataset([0, 1, 2] * 10)

    config = HeterogeneityConfig(
        mode="label_skew",
        num_clients=3,
        seed=42,
        alpha=0.5,
    )

    artifacts = build_condition_artifacts(
        train_ds=dataset,
        config=config,
        batch_size=4,
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    metadata = artifacts.metadata

    assert metadata.mode == "label_skew"
    assert metadata.seed == 42
    assert metadata.num_clients == 3
    assert metadata.alpha == pytest.approx(0.5)
    assert metadata.dataset_size == 30
    assert metadata.min_client_size == 10
    assert metadata.max_client_size == 10


def test_condition_artifacts_rejects_empty_local_split():
    # Three IID clients receive two samples each.
    # int(2 * 0.1) gives zero validation samples.
    dataset = ToyDataset([0, 1] * 3)

    config = HeterogeneityConfig(
        mode="iid",
        num_clients=3,
        seed=42,
    )

    with pytest.raises(
        ValueError,
        match="validation.*empty|empty.*validation",
    ):
        build_condition_artifacts(
            train_ds=dataset,
            config=config,
            batch_size=2,
            train_fraction=0.8,
            validation_fraction=0.1,
        )