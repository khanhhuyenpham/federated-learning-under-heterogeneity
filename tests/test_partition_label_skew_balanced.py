import pandas as pd
import pytest
import torch
import numpy as np
from src.data import summarize_partition, validate_partition, partition_label_skew_balanced


class ToyDataset:
    def __init__(self, targets):
        self.targets = torch.tensor(targets)

    def __len__(self):
        return len(self.targets)

def test_balanced_label_skew_has_full_valid_partition():
    dataset = ToyDataset([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    partition = partition_label_skew_balanced(
        train_ds=dataset,
        num_clients=3,
        alpha=0.5,
        seed=42,
    )

    validate_partition(
        client_indices=partition,
        dataset_size=12,
        expected_num_clients=3,
    )

def test_balanced_label_skew_balances_client_sizes():
    dataset = ToyDataset([0] * 5 + [1] * 5 + [2] * 5)

    partition = partition_label_skew_balanced(
        train_ds=dataset,
        num_clients=4,
        alpha=0.5,
        seed=42,
    )

    client_sizes = [len(indices) for indices in partition.values()]

    assert max(client_sizes) - min(client_sizes) <= 1

def test_balanced_label_skew_is_reproducible():
    dataset = ToyDataset([0] * 10 + [1] * 10 + [2] * 10)

    first = partition_label_skew_balanced(
        dataset,
        num_clients=3,
        alpha=0.5,
        seed=42,
    )

    second = partition_label_skew_balanced(
        dataset,
        num_clients=3,
        alpha=0.5,
        seed=42,
    )

    assert first == second

def test_balanced_label_skew_changes_with_seed():
    dataset = ToyDataset([0] * 10 + [1] * 10 + [2] * 10)

    first = partition_label_skew_balanced(
        dataset,
        num_clients=3,
        alpha=0.5,
        seed=42,
    )

    second = partition_label_skew_balanced(
        dataset,
        num_clients=3,
        alpha=0.5,
        seed=43,
    )

    assert first != second

@pytest.mark.parametrize("alpha", [0, -0.1])
def test_balanced_label_skew_rejects_nonpositive_alpha(alpha):
    dataset = ToyDataset([0, 0, 1, 1])

    with pytest.raises(ValueError, match="alpha.*positive"):
        partition_label_skew_balanced(
            dataset,
            num_clients=2,
            alpha=alpha,
            seed=42,
        )

def test_balanced_label_skew_supports_noncontiguous_class_labels():
    dataset = ToyDataset(
        [2, 2, 2, 2, 5, 5, 5, 5, 9, 9, 9, 9]
    )

    partition = partition_label_skew_balanced(
        train_ds=dataset,
        num_clients=3,
        alpha=0.5,
        seed=42,
    )

    validate_partition(
        client_indices=partition,
        dataset_size=len(dataset),
        expected_num_clients=3,
    )

    assigned_labels = {
        int(dataset.targets[sample_idx])
        for indices in partition.values()
        for sample_idx in indices
    }

    assert assigned_labels == {2, 5, 9}

def test_smaller_alpha_produces_stronger_label_skew():
    dataset = ToyDataset(
        [0] * 1000
        + [1] * 1000
        + [2] * 1000
    )

    strongly_skewed = partition_label_skew_balanced(
        train_ds=dataset,
        num_clients=6,
        alpha=0.05,
        seed=42,
    )

    nearly_uniform = partition_label_skew_balanced(
        train_ds=dataset,
        num_clients=6,
        alpha=100.0,
        seed=42,
    )

    skewed_summary = summarize_partition(dataset, strongly_skewed)
    uniform_summary = summarize_partition(dataset, nearly_uniform)

    skewed_dominance = (
        skewed_summary
        .groupby("client_id")["proportion"]
        .max()
        .mean()
    )

    uniform_dominance = (
        uniform_summary
        .groupby("client_id")["proportion"]
        .max()
        .mean()
    )

    assert skewed_dominance > uniform_dominance

