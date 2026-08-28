import pandas as pd
import pytest
import torch

from src.data import summarize_partition

class ToyDataset:
    def __init__(self, targets):
        self.targets = torch.tensor(targets)

    def __len__(self):
        return len(self.targets)

def test_summarize_partition_returns_correct_counts():
    dataset = ToyDataset([0, 0, 1, 1, 2, 2])

    partition = {
        0: [0 ,2, 4],
        1: [1, 3, 5],
    }

    summary = summarize_partition(dataset, partition)

    expected_counts = {
        (0, 0): 1,
        (0, 1): 1,
        (0, 2): 1, 
        (1, 0): 1,
        (1, 1): 1,
        (1, 2): 1,
    }

    actual_counts = {
        (row.client_id, row.class_id): row.count
        for row in summary.itertuples()
    }

    assert actual_counts == expected_counts

def test_summarize_partition_includes_zero_counts():
    dataset = ToyDataset([0, 0, 1, 1])

    partition = {
        0: [0, 1],
        1: [2, 3],
    }

    summary = summarize_partition(dataset, partition)

    client_zero_class_one = summary[
        (summary["client_id"] == 0)
        & (summary["class_id"] == 1)
    ]

    assert client_zero_class_one.iloc[0]["count"] == 0
    assert client_zero_class_one.iloc[0]["proportion"] == pytest.approx(0.0)

def test_summarize_partition_proportions_sum_to_one():
    dataset = ToyDataset([0, 0, 1, 1, 2, 2])

    partition = {
        0: [0, 2, 4],
        1: [1, 3, 5],
    }

    summary = summarize_partition(dataset, partition)

    proportion_sums = summary.groupby("client_id")['proportion'].sum()

    assert proportion_sums.loc[0] == pytest.approx(1.0)
    assert proportion_sums.loc[1] == pytest.approx(1.0)

def test_summarize_partition_rejects_empty_client():
    dataset = ToyDataset([0, 0, 1, 1])

    partition = {
        0: [0, 1, 2, 3],
        1: [],
    }

    with pytest.raises(ValueError, match="client 1 has no assigned samples"):
        summarize_partition(dataset, partition)