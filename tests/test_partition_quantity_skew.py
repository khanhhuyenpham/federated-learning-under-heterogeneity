import pandas as pd
import pytest
import torch
import numpy as np
from src.data import validate_partition, partition_quantity_skew


class ToyDataset:
    def __init__(self, targets):
        self.targets = torch.tensor(targets)

    def __len__(self):
        return len(self.targets)

def test_quantity_skew_produces_valid_full_partition():
    dataset = ToyDataset([0, 1] * 50)

    partition = partition_quantity_skew(
        train_ds=dataset,
        num_clients=5,
        alpha=0.5,
        seed=42,
        min_samples_per_client=2,
    )

    validate_partition(
        client_indices=partition,
        dataset_size=len(dataset),
        expected_num_clients=5,
    )

def test_quantity_skew_respects_minimum_client_size():
    dataset = ToyDataset([0, 1] * 50)

    partition = partition_quantity_skew(
        train_ds=dataset,
        num_clients=5,
        alpha=0.1,
        seed=42,
        min_samples_per_client=4,
    )

    client_sizes = [
        len(indices)
        for indices in partition.values()
    ]

    assert min(client_sizes) >= 4
    assert sum(client_sizes) == len(dataset)

def test_quantity_skew_handles_no_remaining_samples():
    dataset = ToyDataset([0, 1] * 6)

    partition = partition_quantity_skew(
        train_ds=dataset,
        num_clients=3,
        alpha=0.5,
        seed=42,
        min_samples_per_client=4,
    )

    client_sizes = [
        len(indices)
        for indices in partition.values()
    ]

    assert client_sizes == [4, 4, 4]

def test_quantity_skew_is_reproducible():
    dataset = ToyDataset([0, 1, 2] * 100)

    first = partition_quantity_skew(
        dataset,
        num_clients=5,
        alpha=0.5,
        seed=42,
        min_samples_per_client=2,
    )

    second = partition_quantity_skew(
        dataset,
        num_clients=5,
        alpha=0.5,
        seed=42,
        min_samples_per_client=2,
    )

    assert first == second

def test_quantity_skew_changes_with_seed():
    dataset = ToyDataset([0, 1, 2] * 100)

    first = partition_quantity_skew(
        dataset,
        num_clients=5,
        alpha=0.5,
        seed=42,
        min_samples_per_client=2,
    )

    second = partition_quantity_skew(
        dataset,
        num_clients=5,
        alpha=0.5,
        seed=43,
        min_samples_per_client=2,
    )

    assert first != second

@pytest.mark.parametrize("num_clients", [0, -1])
def test_quantity_skew_rejects_nonpositive_num_clients(num_clients):
    dataset = ToyDataset([0, 0, 1, 1])

    with pytest.raises(ValueError, match="num_clients.*positive"):
        partition_quantity_skew(
            dataset,
            num_clients=num_clients,
            alpha=0.5,
            seed=42,
        )

@pytest.mark.parametrize("alpha", [0, -0.1])
def test_quantity_skew_rejects_nonpositive_alpha(alpha):
    dataset = ToyDataset([0, 0, 1, 1])

    with pytest.raises(ValueError, match="alpha.*positive"):
        partition_quantity_skew(
            dataset,
            num_clients=2,
            alpha=alpha,
            seed=42,
        )

@pytest.mark.parametrize("minimum", [0, -1])
def test_quantity_skew_rejects_nonpositive_minimum(minimum):
    dataset = ToyDataset([0, 0, 1, 1])

    with pytest.raises(
        ValueError,
        match="min_samples_per_client.*positive",
    ):
        partition_quantity_skew(
            dataset,
            num_clients=2,
            alpha=0.5,
            seed=42,
            min_samples_per_client=minimum,
        )

def test_quantity_skew_rejects_impossible_minimum_allocation():
    dataset = ToyDataset([0] * 10)

    with pytest.raises(ValueError, match="minimum|cannot"):
        partition_quantity_skew(
            dataset,
            num_clients=4,
            alpha=0.5,
            seed=42,
            min_samples_per_client=3,
        )

def test_smaller_alpha_produces_stronger_quantity_skew():
    dataset = ToyDataset([0, 1, 2] * 2000)

    strongly_skewed = partition_quantity_skew(
        dataset,
        num_clients=6,
        alpha=0.05,
        seed=42,
        min_samples_per_client=10,
    )

    nearly_balanced = partition_quantity_skew(
        dataset,
        num_clients=6,
        alpha=100.0,
        seed=42,
        min_samples_per_client=10,
    )

    skewed_sizes = np.array(
        [len(indices) for indices in strongly_skewed.values()]
    )

    balanced_sizes = np.array(
        [len(indices) for indices in nearly_balanced.values()]
    )

    skewed_coefficient_of_variation = (
        skewed_sizes.std() / skewed_sizes.mean()
    )

    balanced_coefficient_of_variation = (
        balanced_sizes.std() / balanced_sizes.mean()
    )

    assert (
        skewed_coefficient_of_variation
        > balanced_coefficient_of_variation
    )

class LengthOnlyDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def test_quantity_skew_does_not_require_labels():
    dataset = LengthOnlyDataset(size=100)

    partition = partition_quantity_skew(
        train_ds=dataset,
        num_clients=5,
        alpha=0.5,
        seed=42,
        min_samples_per_client=2,
    )

    validate_partition(
        client_indices=partition,
        dataset_size=len(dataset),
        expected_num_clients=5,
    )