import pytest
import torch

from src.feature_shifts import FixedRotation
from src.heterogeneity_metrics import (
    quantity_coefficient_of_variation,
    mean_client_label_total_variation,
    rotation_angle_standard_deviation,
    HeterogeneityReport,
    compute_heterogeneity_report,
)

class ToyDataset:
    def __init__(self, targets):
        self.targets = torch.tensor(targets)

    def __len__(self):
        return len(self.targets)

def test_quantity_cv_is_zero_for_equal_client_sizes():
    partition = {
        0: [0, 1],
        1: [2, 3],
        2: [4, 5],
    }

    result = quantity_coefficient_of_variation(partition)

    assert result == pytest.approx(0.0)

def test_quantity_cv_detects_unequal_client_sizes():
    partition = {
        0: [0, 1],
        1: [2, 3, 4, 5],
        2: [6, 7, 8, 9, 10, 11],
    }

    result = quantity_coefficient_of_variation(partition)

    assert result == pytest.approx(0.4082482905)

def test_quantity_cv_is_scale_invariant():
    small = {
        0: [0],
        1: [1, 2],
        2: [3, 4, 5],
    }

    large = {
        0: list(range(10)),
        1: list(range(10, 30)),
        2: list(range(30, 60)),
    }

    assert quantity_coefficient_of_variation(
        small
    ) == pytest.approx(
        quantity_coefficient_of_variation(large)
    )

def test_quantity_cv_rejects_empty_partition():
    with pytest.raises(ValueError, match="partition.*empty"):
        quantity_coefficient_of_variation({})

def test_quantity_cv_rejects_empty_client():
    partition = {
        0: [0, 1],
        1: [],
    }

    with pytest.raises(ValueError, match="client 1.*empty"):
        quantity_coefficient_of_variation(partition)

def test_label_tvd_is_zero_for_identical_distributions():
    dataset = ToyDataset([0, 1, 0, 1])

    partition = {
        0: [0, 1],
        1: [2, 3],
    }

    result = mean_client_label_total_variation(
        dataset,
        partition,
    )

    assert result == pytest.approx(0.0)

def test_label_tvd_detects_extreme_label_skew():
    dataset = ToyDataset([0, 0, 1, 1])

    partition = {
        0: [0, 1],
        1: [2, 3],
    }

    result = mean_client_label_total_variation(
        dataset,
        partition,
    )

    assert result == pytest.approx(0.5)

def test_label_tvd_supports_noncontiguous_labels():
    dataset = ToyDataset([2, 2, 5, 5])

    partition = {
        0: [0, 2],
        1: [1, 3],
    }

    result = mean_client_label_total_variation(
        dataset,
        partition,
    )

    assert result == pytest.approx(0.0)

def test_label_tvd_rejects_empty_partition():
    dataset = ToyDataset([0, 1])

    with pytest.raises(ValueError, match="partition.*empty"):
        mean_client_label_total_variation(
            dataset,
            {},
        )

def test_label_tvd_rejects_empty_client():
    dataset = ToyDataset([0, 1])

    partition = {
        0: [0, 1],
        1: [],
    }

    with pytest.raises(ValueError, match="client 1.*empty"):
        mean_client_label_total_variation(
            dataset,
            partition,
        )

def test_rotation_std_is_zero_without_transforms():
    result = rotation_angle_standard_deviation(None)

    assert result == pytest.approx(0.0)

def test_rotation_std_detects_between_client_variation():
    transforms = {
        0: FixedRotation(-20),
        1: FixedRotation(-10),
        2: FixedRotation(0),
        3: FixedRotation(10),
        4: FixedRotation(20),
    }

    result = rotation_angle_standard_deviation(transforms)

    assert result == pytest.approx(14.1421356237)

def test_rotation_std_is_zero_for_identical_angles():
    transforms = {
        0: FixedRotation(15),
        1: FixedRotation(15),
        2: FixedRotation(15),
    }

    result = rotation_angle_standard_deviation(transforms)

    assert result == pytest.approx(0.0)

def test_rotation_std_rejects_empty_transform_mapping():
    with pytest.raises(ValueError, match="transform.*empty"):
        rotation_angle_standard_deviation({})

def test_rotation_std_rejects_transform_without_angle():
    transforms = {
        0: lambda features: features,
    }

    with pytest.raises(ValueError, match="angle"):
        rotation_angle_standard_deviation(transforms)

def test_heterogeneity_report_is_zero_for_controlled_iid_partition():
    dataset = ToyDataset([0, 1, 0, 1])

    partition = {
        0: [0, 1],
        1: [2, 3],
    }

    report = compute_heterogeneity_report(
        train_ds=dataset,
        client_indices=partition,
    )

    assert report.quantity_cv == pytest.approx(0.0)
    assert report.mean_label_tvd == pytest.approx(0.0)
    assert report.rotation_angle_std_degrees == pytest.approx(0.0)

def test_heterogeneity_report_detects_label_skew():
    dataset = ToyDataset([0, 0, 1, 1])

    partition = {
        0: [0, 1],
        1: [2, 3],
    }

    report = compute_heterogeneity_report(
        train_ds=dataset,
        client_indices=partition,
    )

    assert report.quantity_cv == pytest.approx(0.0)
    assert report.mean_label_tvd == pytest.approx(0.5)
    assert report.rotation_angle_std_degrees == pytest.approx(0.0)

def test_heterogeneity_report_detects_quantity_skew():
    dataset = ToyDataset(
        [0, 1]
        + [0, 1, 0, 1]
        + [0, 1, 0, 1, 0, 1]
    )

    partition = {
        0: [0, 1],
        1: [2, 3, 4, 5],
        2: [6, 7, 8, 9, 10, 11],
    }

    report = compute_heterogeneity_report(
        train_ds=dataset,
        client_indices=partition,
    )

    assert report.quantity_cv == pytest.approx(0.4082482905)
    assert report.mean_label_tvd == pytest.approx(0.0)
    assert report.rotation_angle_std_degrees == pytest.approx(0.0)

def test_heterogeneity_report_detects_quantity_skew():
    dataset = ToyDataset(
        [0, 1]
        + [0, 1, 0, 1]
        + [0, 1, 0, 1, 0, 1]
    )

    partition = {
        0: [0, 1],
        1: [2, 3, 4, 5],
        2: [6, 7, 8, 9, 10, 11],
    }

    report = compute_heterogeneity_report(
        train_ds=dataset,
        client_indices=partition,
    )

    assert report.quantity_cv == pytest.approx(0.4082482905)
    assert report.mean_label_tvd == pytest.approx(0.0)
    assert report.rotation_angle_std_degrees == pytest.approx(0.0)
