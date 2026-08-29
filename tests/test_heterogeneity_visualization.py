import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
import numpy as np
import torch

from src.heterogeneity_visualization import (
    plot_client_label_distribution,
    plot_client_rotation_examples,
    plot_client_sizes,
    FixedRotation
)

class ToyDataset:
    def __init__(self, targets):
        self.targets = torch.tensor(targets)

    def __len__(self):
        return len(self.targets)

class ToyImageDataset:
    def __init__(self):
        image = torch.zeros((1, 5, 5))
        image[0, 0, 2] = 1.0

        self.features = [image]
        self.targets = torch.tensor([7])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], int(self.targets[index])

from src.heterogeneity_visualization import (
    plot_client_sizes,
    plot_client_label_distribution,
)

def test_plot_client_sizes_returns_figure():
    partition = {
        0: [0, 1],
        1: [2, 3, 4],
    }

    figure = plot_client_sizes(partition)

    assert isinstance(figure, Figure)

    plt.close(figure)

def test_plot_client_sizes_creates_one_bar_per_client():
    partition = {
        2: [0, 1, 2],
        0: [3],
        1: [4, 5],
    }

    figure = plot_client_sizes(partition)
    axis = figure.axes[0]

    bar_heights = [
        patch.get_height()
        for patch in axis.patches
    ]

    assert bar_heights == [1, 2, 3]

    plt.close(figure)

def test_plot_client_sizes_labels_axes():
    partition = {
        0: [0, 1],
        1: [2, 3],
    }

    figure = plot_client_sizes(partition)
    axis = figure.axes[0]

    assert axis.get_xlabel() == "Client"
    assert axis.get_ylabel() == "Number of samples"
    assert axis.get_title() == "Client dataset sizes"

    plt.close(figure)

def test_plot_client_sizes_rejects_empty_partition():
    with pytest.raises(ValueError, match="partition.*empty"):
        plot_client_sizes({})

def test_plot_client_sizes_rejects_empty_client():
    partition = {
        0: [0, 1],
        1: [],
    }

    with pytest.raises(ValueError, match="Client 1.*empty"):
        plot_client_sizes(partition)

def test_plot_client_label_distribution_creates_correct_matrix():
    dataset = ToyDataset([0, 0, 1, 1])

    partition = {
        0: [0, 1],
        1: [2, 3],
    }

    figure = plot_client_label_distribution(
        dataset,
        partition,
        annotate=False,
    )

    axis = figure.axes[0]
    heatmap_values = np.asarray(
        axis.images[0].get_array()
    )

    expected = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    np.testing.assert_allclose(
        heatmap_values,
        expected,
    )

    plt.close(figure)

def test_plot_client_label_distribution_labels_axes():
    dataset = ToyDataset([0, 1, 0, 1])

    partition = {
        0: [0, 1],
        1: [2, 3],
    }

    figure = plot_client_label_distribution(
        dataset,
        partition,
    )

    axis = figure.axes[0]

    assert axis.get_xlabel() == "Class"
    assert axis.get_ylabel() == "Client"
    assert axis.get_title() == "Client label distributions"

    plt.close(figure)

def test_plot_client_rotation_examples_creates_client_panels():
    dataset = ToyImageDataset()

    transforms = {
        0: FixedRotation(-20),
        1: FixedRotation(0),
        2: FixedRotation(20),
    }

    figure = plot_client_rotation_examples(
        train_ds=dataset,
        client_feature_transforms=transforms,
    )

    titles = [
        axis.get_title()
        for axis in figure.axes
    ]

    assert titles == [
        "Client 0\n-20.0°",
        "Client 1\n+0.0°",
        "Client 2\n+20.0°",
    ]

    plt.close(figure)

def test_plot_client_rotation_examples_changes_image():
    dataset = ToyImageDataset()

    transforms = {
        0: FixedRotation(0),
        1: FixedRotation(90),
    }

    figure = plot_client_rotation_examples(
        train_ds=dataset,
        client_feature_transforms=transforms,
    )

    original_view = np.asarray(
        figure.axes[0].images[0].get_array()
    )
    rotated_view = np.asarray(
        figure.axes[1].images[0].get_array()
    )

    assert not np.array_equal(
        original_view,
        rotated_view,
    )

    plt.close(figure)

def test_plot_client_rotation_examples_rejects_invalid_index():
    dataset = ToyImageDataset()

    with pytest.raises(ValueError, match="sample_index.*range"):
        plot_client_rotation_examples(
            train_ds=dataset,
            client_feature_transforms={
                0: FixedRotation(0),
            },
            sample_index=1,
        )