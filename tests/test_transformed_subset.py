import pandas as pd
import pytest
import torch
import numpy as np
from src.data import TransformedSubset, create_client_loaders

class ToyFeatureDataset:
    def __init__(self):
        self.features = [
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([3.0]),
        ]
        self.targets = torch.tensor([0, 1, 2])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], int(self.targets[index])

def test_transformed_subset_maps_local_to_original_indices():
    dataset = ToyFeatureDataset()

    subset = TransformedSubset(
        base_dataset=dataset,
        indices=[2, 0],
    )

    feature, label = subset[0]

    assert torch.equal(feature, torch.tensor([3.0]))
    assert label == 2
    assert len(subset) == 2

def test_transformed_subset_applies_feature_transform():
    dataset = ToyFeatureDataset()

    subset = TransformedSubset(
        base_dataset=dataset,
        indices=[0, 1],
        feature_transform=lambda feature: feature + 10,
    )

    feature, label = subset[0]

    assert torch.equal(feature, torch.tensor([11.0]))
    assert label == 0

def test_transformed_subset_preserves_labels():
    dataset = ToyFeatureDataset()

    subset = TransformedSubset(
        base_dataset=dataset,
        indices=[2],
        feature_transform=lambda feature: feature * -1,
    )

    feature, label = subset[0]

    assert torch.equal(feature, torch.tensor([-3.0]))
    assert label == 2

def test_transformed_subset_copies_indices():
    dataset = ToyFeatureDataset()
    indices = [0, 1]

    subset = TransformedSubset(
        base_dataset=dataset,
        indices=indices,
    )

    indices.append(2)

    assert len(subset) == 2

def test_create_client_loaders_applies_client_specific_transform():
    dataset = ToyFeatureDataset()

    client_indices = {
        0: [0],
        1: [1],
    }

    transforms = {
        1: lambda feature: feature + 10,
    }

    loaders = create_client_loaders(
        train_ds=dataset,
        client_indices=client_indices,
        batch_size=1,
        shuffle=False,
        client_feature_transforms=transforms,
    )

    client_0_features, client_0_labels = next(iter(loaders[0]))
    client_1_features, client_1_labels = next(iter(loaders[1]))

    assert torch.equal(
        client_0_features,
        torch.tensor([[1.0]]),
    )
    assert torch.equal(
        client_1_features,
        torch.tensor([[12.0]]),
    )

    assert client_0_labels.item() == 0
    assert client_1_labels.item() == 1

def test_create_client_loaders_works_without_feature_transforms():
    dataset = ToyFeatureDataset()

    loaders = create_client_loaders(
        train_ds=dataset,
        client_indices={0: [2, 0]},
        batch_size=2,
        shuffle=False,
    )

    features, labels = next(iter(loaders[0]))

    assert torch.equal(
        features,
        torch.tensor([[3.0], [1.0]]),
    )
    assert torch.equal(
        labels,
        torch.tensor([2, 0]),
    )