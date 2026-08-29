import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
from .data import summarize_partition

import math
import torch
from .feature_shifts import FixedRotation

def plot_client_sizes(
    client_indices: dict[int, list[int]],
) -> Figure:
    """
    Plot the number of assigned samples for every client.

    Returns:
        A Matplotlib Figure owned by the caller.
    """
    if not client_indices:
        raise ValueError("partition cannot be empty")

    client_ids = sorted(client_indices)
    client_sizes = []

    for client_id in client_ids:
        indices = client_indices[client_id]

        if not indices:
            raise ValueError(f"Client {client_id} is empty")

        client_sizes.append(len(indices))

    figure_width = max(
        6.0,
        0.9 * len(client_ids),
    )

    figure, axis = plt.subplots(
        figsize=(figure_width, 4.5),
    )

    bars = axis.bar(
        [str(client_id) for client_id in client_ids],
        client_sizes,
    )

    axis.bar_label(
        bars,
        labels=[
            f"{client_size:,}"
            for client_size in client_sizes
        ],
        padding=3,
    )

    axis.set_title("Client dataset sizes")
    axis.set_xlabel("Client")
    axis.set_ylabel("Number of samples")

    axis.set_ylim(bottom=0)
    axis.grid(
        axis="y",
        alpha=0.25,
    )
    axis.set_axisbelow(True)

    figure.tight_layout()

    return figure

def plot_client_label_distribution(
    train_ds,
    client_indices: dict[int, list[int]],
    annotate: bool = True,
) -> Figure:
    """
    Plot client label proportions as a heatmap.

    Rows represent clients, columns represent classes, and colors represent
    the proportion of each client's data belonging to each class.
    """
    if not client_indices:
        raise ValueError("partition cannot be empty")

    for client_id, indices in client_indices.items():
        if not indices:
            raise ValueError(f"Client {client_id} is empty")

    summary = summarize_partition(
        train_ds=train_ds,
        client_indices=client_indices,
    )

    proportion_matrix = (
        summary
        .pivot(
            index="client_id",
            columns="class_id",
            values="proportion",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    client_ids = proportion_matrix.index.tolist()
    class_ids = proportion_matrix.columns.tolist()
    values = proportion_matrix.to_numpy(dtype=float)

    figure_width = max(
        7.0,
        0.7 * len(class_ids) + 2.0,
    )
    figure_height = max(
        4.0,
        0.6 * len(client_ids) + 1.5,
    )

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
    )

    heatmap = axis.imshow(
        values,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )

    axis.set_xticks(
        np.arange(len(class_ids))
    )
    axis.set_xticklabels(
        [str(class_id) for class_id in class_ids]
    )

    axis.set_yticks(
        np.arange(len(client_ids))
    )
    axis.set_yticklabels(
        [str(client_id) for client_id in client_ids]
    )

    axis.set_title("Client label distributions")
    axis.set_xlabel("Class")
    axis.set_ylabel("Client")

    colorbar = figure.colorbar(
        heatmap,
        ax=axis,
    )
    colorbar.set_label("Class proportion")

    if annotate:
        for row_position in range(len(client_ids)):
            for column_position in range(len(class_ids)):
                proportion = values[
                    row_position,
                    column_position,
                ]

                text_color = (
                    "white"
                    if proportion >= 0.5
                    else "black"
                )

                axis.text(
                    column_position,
                    row_position,
                    f"{proportion:.2f}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=text_color,
                    fontsize=8,
                )

    figure.tight_layout()

    return figure

def _prepare_image_for_display(features):
    """Convert a PyTorch image tensor into Matplotlib's image layout."""
    if not torch.is_tensor(features):
        return features

    image = features.detach().cpu()

    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)

    elif image.ndim == 3 and image.shape[0] in {3, 4}:
        image = image.permute(1, 2, 0)

    return image.numpy()

def plot_client_rotation_examples(
    train_ds,
    client_feature_transforms: dict[int, FixedRotation],
    sample_index: int = 0,
) -> Figure:
    """
    Show one dataset image under every client's fixed rotation.
    """
    if not client_feature_transforms:
        raise ValueError("transform mapping cannot be empty")

    if sample_index < 0 or sample_index >= len(train_ds):
        raise ValueError(
            f"sample_index {sample_index} is outside the valid range "
            f"[0, {len(train_ds)})"
        )

    for client_id, transform in client_feature_transforms.items():
        if not hasattr(transform, "angle"):
            raise ValueError(
                f"Transform for client {client_id} has no angle"
            )

    features, label = train_ds[sample_index]

    client_ids = sorted(client_feature_transforms)
    num_clients = len(client_ids)

    num_columns = min(5, num_clients)
    num_rows = math.ceil(
        num_clients / num_columns
    )

    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(
            2.5 * num_columns,
            2.7 * num_rows,
        ),
        squeeze=False,
    )

    flattened_axes = axes.ravel()

    for position, client_id in enumerate(client_ids):
        axis = flattened_axes[position]
        transform = client_feature_transforms[client_id]

        if torch.is_tensor(features):
            transform_input = features.clone()
        elif hasattr(features, "copy"):
            transform_input = features.copy()
        else:
            transform_input = features

        transformed_features = transform(
            transform_input
        )

        display_image = _prepare_image_for_display(
            transformed_features
        )

        axis.imshow(
            display_image,
            cmap="gray",
        )

        axis.set_title(
            f"Client {client_id}\n"
            f"{transform.angle:+.1f}°"
        )
        axis.axis("off")

    for unused_position in range(
        num_clients,
        len(flattened_axes),
    ):
        flattened_axes[unused_position].axis("off")

    figure.suptitle(
        f"Fixed client rotations — label {label}"
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.92)
    )

    return figure