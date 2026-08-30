import pytest
import torch
from torch.utils.data import (
    Dataset,
    RandomSampler,
    SequentialSampler,
)

from src.data import (
    ClientSplitLoaders,
    create_client_split_loaders,
)


class ToyDataset(Dataset):
    def __init__(self, size: int):
        self.features = torch.arange(
            size,
            dtype=torch.float32,
        ).unsqueeze(1)

        self.targets = torch.tensor(
            [index % 2 for index in range(size)],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


class AddConstant:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, features):
        return features + self.value


@pytest.fixture
def dataset():
    return ToyDataset(size=12)


@pytest.fixture
def client_splits():
    return {
        "train": {
            0: [0, 1],
            1: [6, 7],
        },
        "validation": {
            0: [2, 3],
            1: [8, 9],
        },
        "test": {
            0: [4, 5],
            1: [10, 11],
        },
    }


def collect_features(loader):
    collected = []

    for features, _ in loader:
        collected.extend(
            features.flatten().tolist()
        )

    return collected


def test_create_client_split_loaders_returns_all_splits(
    dataset,
    client_splits,
):
    result = create_client_split_loaders(
        train_ds=dataset,
        client_train_indices=client_splits["train"],
        client_validation_indices=client_splits["validation"],
        client_test_indices=client_splits["test"],
        batch_size=2,
        seed=42,
    )

    assert isinstance(result, ClientSplitLoaders)

    assert set(result.train) == {0, 1}
    assert set(result.validation) == {0, 1}
    assert set(result.test) == {0, 1}

    assert len(result.train[0].dataset) == 2
    assert len(result.validation[0].dataset) == 2
    assert len(result.test[0].dataset) == 2


def test_create_client_split_loaders_uses_correct_shuffling(
    dataset,
    client_splits,
):
    result = create_client_split_loaders(
        train_ds=dataset,
        client_train_indices=client_splits["train"],
        client_validation_indices=client_splits["validation"],
        client_test_indices=client_splits["test"],
        batch_size=2,
        seed=42,
    )

    assert isinstance(
        result.train[0].sampler,
        RandomSampler,
    )

    assert isinstance(
        result.validation[0].sampler,
        SequentialSampler,
    )

    assert isinstance(
        result.test[0].sampler,
        SequentialSampler,
    )


def test_create_client_split_loaders_applies_transforms_to_every_split(
    dataset,
    client_splits,
):
    transforms = {
        0: AddConstant(100.0),
        1: AddConstant(200.0),
    }

    result = create_client_split_loaders(
        train_ds=dataset,
        client_train_indices=client_splits["train"],
        client_validation_indices=client_splits["validation"],
        client_test_indices=client_splits["test"],
        batch_size=2,
        seed=42,
        client_feature_transforms=transforms,
    )

    assert sorted(collect_features(result.train[0])) == [
        100.0,
        101.0,
    ]
    assert sorted(collect_features(result.validation[0])) == [
        102.0,
        103.0,
    ]
    assert sorted(collect_features(result.test[0])) == [
        104.0,
        105.0,
    ]

    assert sorted(collect_features(result.train[1])) == [
        206.0,
        207.0,
    ]
    assert sorted(collect_features(result.validation[1])) == [
        208.0,
        209.0,
    ]
    assert sorted(collect_features(result.test[1])) == [
        210.0,
        211.0,
    ]


def test_create_client_split_loaders_preserves_features_without_transforms(
    dataset,
    client_splits,
):
    result = create_client_split_loaders(
        train_ds=dataset,
        client_train_indices=client_splits["train"],
        client_validation_indices=client_splits["validation"],
        client_test_indices=client_splits["test"],
        batch_size=2,
        seed=42,
        client_feature_transforms=None,
    )

    assert sorted(collect_features(result.train[0])) == [
        0.0,
        1.0,
    ]
    assert sorted(collect_features(result.validation[0])) == [
        2.0,
        3.0,
    ]
    assert sorted(collect_features(result.test[0])) == [
        4.0,
        5.0,
    ]


def test_create_client_split_loaders_rejects_mismatched_client_ids(
    dataset,
    client_splits,
):
    mismatched_validation_indices = {
        0: [2, 3],
    }

    with pytest.raises(
        ValueError,
        match="identical client IDs",
    ):
        create_client_split_loaders(
            train_ds=dataset,
            client_train_indices=client_splits["train"],
            client_validation_indices=mismatched_validation_indices,
            client_test_indices=client_splits["test"],
            batch_size=2,
            seed=42,
        )