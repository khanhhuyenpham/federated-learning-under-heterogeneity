from pathlib import Path

import pandas as pd
from torchvision import datasets, transforms
from dataclasses import asdict

from src.heterogeneity import (
    HeterogeneityConfig,
    build_federated_data,
)

from src.heterogeneity_metrics import (
    compute_heterogeneity_report,
)

SEEDS = (42, 43, 44)

LABEL_ALPHAS = (10.0, 1.0, 0.5, 0.1)
QUANTITY_ALPHAS = (10.0, 1.0, 0.3, 0.1)
ROTATION_ANGLES = (0.0, 10.0, 20.0, 30.0)

NUM_CLIENTS = 5
MIN_SAMPLES_PER_CLIENT = 500

OUTPUT_DIR = Path("results/heterogeneity_calibration")

def main():
    train_ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    rows = []

    calibration_cases = []

    for seed in SEEDS:
        # one iid case
        calibration_cases.append({
            "parameter_name": "none",
            "parameter_value": 0.0,
            "config": HeterogeneityConfig(
                mode="iid",
                num_clients=NUM_CLIENTS,
                seed=seed,
            )
        })
        # every label alpha * seed
        for label_alpha in LABEL_ALPHAS:
            calibration_cases.append({
                "parameter_name": "alpha",
                "parameter_value": label_alpha,
                "config": HeterogeneityConfig(
                    mode="label_skew",
                    num_clients=NUM_CLIENTS,
                    seed=seed,
                    alpha=label_alpha,
                )
            })

        #every quantity alpha * seed
        for quantity_alpha in QUANTITY_ALPHAS:
            calibration_cases.append({
                "parameter_name": "alpha",
                "parameter_value": quantity_alpha,
                "config": HeterogeneityConfig(
                    mode="quantity_skew",
                    num_clients=NUM_CLIENTS,
                    seed=seed,
                    alpha=quantity_alpha,
                    min_samples_per_client=MIN_SAMPLES_PER_CLIENT
                )
            })

        #every rotation angle * seed
        for rotation_angle in ROTATION_ANGLES:
            calibration_cases.append({
                "parameter_name": "max_abs_angle",
                "parameter_value": rotation_angle,
                "config": HeterogeneityConfig(
                    mode="rotation_shift",
                    num_clients=NUM_CLIENTS,
                    seed=seed,
                    max_abs_angle=rotation_angle
                )
            })

    assert len(calibration_cases) == 39

    for case_number, case in enumerate(
        calibration_cases,
        start=1,
    ):
        config = case["config"]
        print(
            f"[{case_number}/{len(calibration_cases)}] "
            f"mode={config.mode}, "
            f"{case['parameter_name']}=",
            f"{case['parameter_value']}, "
            f"seed={config.seed}"
        )

        federated_data = build_federated_data(
            train_ds=train_ds,
            config=config,
        )

        report = compute_heterogeneity_report(
            train_ds=train_ds,
            client_indices=federated_data.client_indices,
            client_feature_transforms=federated_data.client_feature_transforms,
        )

        client_sizes = [
            len(indices)
            for indices in federated_data.client_indices.values()
        ]

        row = {
            "mode": config.mode,
            "parameter_name": case["parameter_name"],
            "parameter_value": case["parameter_value"],
            "seed": config.seed,
            "num_clients": config.num_clients,
            "min_client_size": min(client_sizes),
            "max_client_size": max(client_sizes),
            **asdict(report),
        }

        for client_id, client_size in enumerate(client_sizes):
            row[f"client_{client_id}_size"] = client_sizes

        rows.append(row)

    results = pd.DataFrame(rows)

    assert len(results) == 39
    assert results["seed"].nunique() == 3

    print()
    print(results.to_string(index=False))

    summary = (
        results
        .groupby(
            [
                "mode",
                "parameter_name",
                "parameter_value",
            ],
            as_index=False,
        )
        .agg(
            num_seeds=("seed", "nunique"),
            quantity_cv_mean=("quantity_cv", "mean"),
            quantity_cv_std=("quantity_cv", "std"),
            label_tvd_mean=("mean_label_tvd", "mean"),
            label_tvd_std=("mean_label_tvd", "std"),
            rotation_std_degrees_mean=(
                "rotation_angle_std_degrees",
                "mean",
            ),
            rotation_std_degrees_std=(
                "rotation_angle_std_degrees",
                "std",
            ),
            smallest_client=(
                "min_client_size",
                "min",
            ),
            largest_client=(
                "max_client_size",
                "max",
            ),
        )
    )
        
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
        
    results.to_csv(
        OUTPUT_DIR / "raw_metrics.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False,
    )

    print()
    print("Calibration summary")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()