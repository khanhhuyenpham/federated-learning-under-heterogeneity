import pytest
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
)

from src.evaluation import (
    create_global_test_loader,
    evaluate_global_model,
)


class ToyDataset(Dataset):
    def __init__(self, labels):
        self.targets = torch.tensor(
            labels,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        label = self.targets[index]

        # The feature directly represents the binary label.
        features = torch.tensor(
            [float(label)],
            dtype=torch.float32,
        )

        return features, label


class PerfectBinaryModel(nn.Module):
    def forward(self, inputs):
        values = inputs.reshape(
            inputs.shape[0],
            -1,
        )[:, 0]

        # Input 0 produces class-0 logits [1, 0].
        # Input 1 produces class-1 logits [0, 1].
        return torch.stack(
            [
                1.0 - values,
                values,
            ],
            dim=1,
        )


def test_global_test_loader_uses_original_dataset():
    dataset = ToyDataset([0, 1, 0, 1])

    loader = create_global_test_loader(
        test_ds=dataset,
        batch_size=2,
    )

    # The global dataset must not be wrapped in
    # TransformedSubset or receive a client transform.
    assert loader.dataset is dataset


def test_global_test_loader_does_not_shuffle():
    dataset = ToyDataset([0, 1, 0, 1])

    loader = create_global_test_loader(
        test_ds=dataset,
        batch_size=2,
    )

    assert isinstance(
        loader.sampler,
        SequentialSampler,
    )


def test_global_evaluation_counts_every_sample():
    dataset = ToyDataset([0, 1, 0, 1, 1])

    loader = create_global_test_loader(
        test_ds=dataset,
        batch_size=2,
    )

    result = evaluate_global_model(
        model=PerfectBinaryModel(),
        global_test_loader=loader,
        device=torch.device("cpu"),
    )

    assert result["global_test_correct"] == 5
    assert result["global_test_total"] == 5
    assert result["global_test_accuracy"] == pytest.approx(
        1.0
    )


@pytest.mark.parametrize("invalid_batch_size", [0, -1])
def test_global_test_loader_rejects_invalid_batch_size(
    invalid_batch_size,
):
    dataset = ToyDataset([0, 1])

    with pytest.raises(ValueError, match="positive"):
        create_global_test_loader(
            test_ds=dataset,
            batch_size=invalid_batch_size,
        )


def test_global_test_loader_rejects_empty_dataset():
    empty_dataset = ToyDataset([])

    with pytest.raises(ValueError, match="empty"):
        create_global_test_loader(
            test_ds=empty_dataset,
            batch_size=2,
        )


def test_global_evaluation_rejects_empty_loader():
    # Construct the loader directly because the official
    # loader builder correctly rejects an empty dataset.
    empty_loader = DataLoader(
        ToyDataset([]),
        batch_size=2,
        shuffle=False,
    )

    with pytest.raises(ValueError, match="empty"):
        evaluate_global_model(
            model=PerfectBinaryModel(),
            global_test_loader=empty_loader,
            device=torch.device("cpu"),
        )