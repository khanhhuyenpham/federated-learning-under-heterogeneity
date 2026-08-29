import numpy as np
from .data import summarize_partition
from collections import Counter
from .feature_shifts import FixedRotation
from dataclasses import dataclass

def quantity_coefficient_of_variation(
    client_indices: dict[int, list[int]],
) -> float:
    if len(client_indices) == 0:
        raise ValueError("partition is empty")
    for client_id, indices in client_indices.items():
        if not indices:
            raise ValueError(f"client {client_id} is empty")

    client_sizes = np.asarray(
        [len(indices) for indices in client_indices.values()],
        dtype=float,
    )
    for client_id in range(len(client_indices)):
        if client_sizes[client_id] == 0:
            raise ValueError(f"client {client_id} is empty")
    return float(client_sizes.std() / client_sizes.mean())

def mean_client_label_total_variation(
    train_ds,
    client_indices: dict[int, list[int]],
) -> float:
    if not client_indices:
        raise ValueError("partition cannot be empty")

    for client_id, indices in client_indices.items():
        if not indices:
            raise ValueError(f"client {client_id} is empty")

    targets = train_ds.targets.tolist()
    classes = sorted(set(targets))
    class_counts = Counter(targets)
    dataset_size = len(train_ds)

    global_proportions = {
        class_id : class_counts[class_id] / dataset_size
        for class_id in classes
    }

    summary = summarize_partition(train_ds=train_ds, client_indices=client_indices)
    client_tvds = []
    for client_id in client_indices:
        distance_sum = 0
        for class_id in classes:
            mask = (
                (summary["client_id"] == client_id)
                & (summary["class_id"] == class_id)
            )
            client_proportion = summary.loc[
                mask,
                "proportion",
            ].iloc[0]
            distance_sum += abs(client_proportion - global_proportions[class_id])
        client_tvd = 0.5 * distance_sum
        client_tvds.append(client_tvd)
    return float(np.mean(client_tvds))

def rotation_angle_standard_deviation(
        client_feature_transforms: dict[int, FixedRotation] | None,
) -> float:
    if client_feature_transforms is None:
            return 0.0
    for client_id, transform in client_feature_transforms.items():
        if not hasattr(transform, "angle"):
            raise ValueError(
                f"transform for client {client_id} has no angle"
            )
    if client_feature_transforms is not None and len(client_feature_transforms) == 0:
        raise ValueError(
            "transform can be empty"
        )
    client_angles = np.asarray([transform.angle for transform in client_feature_transforms.values()], dtype=float)
    if any(angle is None for angle in client_angles):
        raise ValueError("angle is required")
    return float(client_angles.std())

@dataclass(frozen=True)
class HeterogeneityReport:
    quantity_cv: float
    mean_label_tvd: float
    rotation_angle_std_degrees: float

def compute_heterogeneity_report(
    train_ds,
    client_indices: dict[int, list[int]],
    client_feature_transforms=None,
) -> HeterogeneityReport:
    """
    Compute quantity, label, and rotation heterogeneity measurements.
    """
    quantity_cv = quantity_coefficient_of_variation(
            client_indices=client_indices
        )
    mean_label_tvd = mean_client_label_total_variation(
        train_ds=train_ds,
        client_indices=client_indices
    )
    rotation_angle_std_degrees = rotation_angle_standard_deviation(
        client_feature_transforms=client_feature_transforms,
    )
    return HeterogeneityReport(
        quantity_cv=quantity_cv,
        mean_label_tvd=mean_label_tvd,
        rotation_angle_std_degrees=rotation_angle_std_degrees,
    )