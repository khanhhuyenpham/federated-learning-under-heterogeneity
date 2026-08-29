from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from src.heterogeneity import (
    HeterogeneityConfig,
    build_federated_data,
)
from src.heterogeneity_visualization import (
    plot_client_label_distribution,
    plot_client_rotation_examples,
    plot_client_sizes,
)


OUTPUT_DIR = Path(
    "results/heterogeneity_visualizations"
)


def save_figure(figure, filename: str) -> None:
    output_path = OUTPUT_DIR / filename

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved {output_path}")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    cases = [
        (
            "iid",
            HeterogeneityConfig(
                mode="iid",
                num_clients=5,
                seed=42,
            ),
        ),
        (
            "label_skew_alpha_0_1",
            HeterogeneityConfig(
                mode="label_skew",
                num_clients=5,
                seed=42,
                alpha=0.1,
            ),
        ),
        (
            "quantity_skew_alpha_1_0",
            HeterogeneityConfig(
                mode="quantity_skew",
                num_clients=5,
                seed=42,
                alpha=1.0,
                min_samples_per_client=500,
            ),
        ),
        (
            "rotation_shift_20_degrees",
            HeterogeneityConfig(
                mode="rotation_shift",
                num_clients=5,
                seed=42,
                max_abs_angle=20,
            ),
        ),
    ]

    for case_name, config in cases:
        print(f"Visualizing {case_name}")

        federated_data = build_federated_data(
            train_ds=train_ds,
            config=config,
        )

        size_figure = plot_client_sizes(
            federated_data.client_indices
        )

        save_figure(
            size_figure,
            f"{case_name}_client_sizes.png",
        )

        label_figure = (
            plot_client_label_distribution(
                train_ds=train_ds,
                client_indices=(
                    federated_data.client_indices
                ),
                annotate=True,
            )
        )

        save_figure(
            label_figure,
            f"{case_name}_label_heatmap.png",
        )

        if (
            federated_data.client_feature_transforms
            is not None
        ):
            rotation_figure = (
                plot_client_rotation_examples(
                    train_ds=train_ds,
                    client_feature_transforms=(
                        federated_data
                        .client_feature_transforms
                    ),
                    sample_index=0,
                )
            )

            save_figure(
                rotation_figure,
                f"{case_name}_examples.png",
            )


if __name__ == "__main__":
    main()