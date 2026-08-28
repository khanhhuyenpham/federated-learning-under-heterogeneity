from dataclasses import dataclass
from typing import Callable, Literal

from .data import (
    partition_iid,
    partition_quantity_skew,
    partition_label_skew_balanced,
)
from .feature_shifts import build_client_rotation_transforms

HeterogeneityMode = Literal[
    "iid",
    "label_skew",
    "quantity_skew",
    "rotation_shift",
]

@dataclass(frozen=True)
class HeterogeneityConfig:
    mode: HeterogeneityMode
    num_clients: int
    seed: int
    alpha: float | None = None
    min_samples_per_client: int = 1
    max_abs_angle: float = 0.0

@dataclass
class FederatedData:
    client_indices: dict[int, list[int]]
    client_feature_transforms: dict[int, Callable] | None = None

def build_federated_data(
    train_ds,
    config: HeterogeneityConfig,
) -> FederatedData:
    """
    Build client index assignments and optional feature transformations
    from one heterogeneity configuration.
    """
    client_feature_transforms = None
    if config.mode == "iid":
        client_indices = partition_iid(train_ds=train_ds, num_clients=config.num_clients, seed=config.seed)
    elif config.mode == "label_skew":
        if config.alpha is None:
            raise ValueError(
                f"alpha is required for mode {config.mode!r}"
            )
        client_indices = partition_label_skew_balanced(train_ds=train_ds, num_clients=config.num_clients, alpha=config.alpha, seed=config.seed)
    elif config.mode == "quantity_skew":
        if config.alpha is None:
            raise ValueError(
                f"alpha is required for mode {config.mode!r}"
            )
        client_indices = partition_quantity_skew(train_ds=train_ds, num_clients=config.num_clients, alpha=config.alpha, seed=config.seed, min_samples_per_client=config.min_samples_per_client)
    elif config.mode == "rotation_shift":
        client_indices = partition_iid(train_ds=train_ds, num_clients=config.num_clients, seed=config.seed)
        client_feature_transforms = build_client_rotation_transforms(num_clients=config.num_clients, max_abs_angle=config.max_abs_angle)
    else:
        raise ValueError(
            f"Unsupported heterogeneity mode: {config.mode!r}"
        )
    return FederatedData(client_indices=client_indices, client_feature_transforms=client_feature_transforms)
