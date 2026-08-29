from dataclasses import asdict

import pandas as pd

from src.heterogeneity import (
    HeterogeneityConfig,
    build_federated_data,
)
from src.heterogeneity_metrics import (
    compute_heterogeneity_report,
)

from torchvision import datasets, transforms


train_ds = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

configs = [
    HeterogeneityConfig(
        mode="iid",
        num_clients=5,
        seed=42,
    ),
    HeterogeneityConfig(
        mode="label_skew",
        num_clients=5,
        seed=42,
        alpha=0.1,
    ),
    HeterogeneityConfig(
        mode="quantity_skew",
        num_clients=5,
        seed=42,
        alpha=0.1,
        min_samples_per_client=500,
    ),
    HeterogeneityConfig(
        mode="rotation_shift",
        num_clients=5,
        seed=42,
        max_abs_angle=20,
    ),
]

rows = []

for config in configs:
    federated_data = build_federated_data(
        train_ds=train_ds,
        config=config,
    )

    report = compute_heterogeneity_report(
        train_ds=train_ds,
        client_indices=federated_data.client_indices,
        client_feature_transforms=(
            federated_data.client_feature_transforms
        ),
    )

    client_sizes = [
        len(indices)
        for indices in federated_data.client_indices.values()
    ]
    rows.append({
        "mode": config.mode,
        "min_client_size": min(client_sizes),
        "max_client_size": max(client_sizes),
        **asdict(report),
    })

diagnostic_table = pd.DataFrame(rows)

print(
    diagnostic_table.to_string(index=False)
)