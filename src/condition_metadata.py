from dataclasses import asdict, dataclass

from .heterogeneity import (
    FederatedData,
    HeterogeneityConfig,
)
from .heterogeneity_metrics import (
    compute_heterogeneity_report,
)

@dataclass(frozen=True)
class ConditionMetadata:
    mode: str
    seed: int
    num_clients: int

    alpha: float | None
    min_samples_per_client: int
    max_abs_angle: float

    dataset_size: int
    min_client_size: int
    max_client_size: int

    quantity_cv: float
    mean_label_tvd: float
    rotation_std_degrees: float

    def to_record(self) -> dict:
        return asdict(self)

def build_condition_metadata(
    train_ds,
    config: HeterogeneityConfig,
    federated_data: FederatedData,
) -> ConditionMetadata:
    mode = config.mode
    seed = config.seed
    num_clients = config.num_clients

    alpha = None
    min_samples_per_client = config.min_samples_per_client
    max_abs_angle = config.max_abs_angle
    if mode == "label_skew" or mode == "quantity_skew":
        alpha = config.alpha
    
    dataset_size = len(train_ds)
    client_sizes = [len(indices) for indices in federated_data.client_indices.values()]
    min_client_size = min(client_sizes)
    max_client_size = max(client_sizes)

    report = compute_heterogeneity_report(
        train_ds=train_ds,
        client_indices=federated_data.client_indices,
        client_feature_transforms=federated_data.client_feature_transforms,
    )
    quantity_cv = report.quantity_cv
    mean_label_tvd = report.mean_label_tvd
    rotation_std_degrees = report.rotation_angle_std_degrees

    return ConditionMetadata(
        mode=mode,
        seed=seed,
        num_clients=num_clients,
        alpha=alpha,
        min_samples_per_client=min_samples_per_client,
        max_abs_angle=max_abs_angle,
        dataset_size=dataset_size,
        min_client_size=min_client_size,
        max_client_size=max_client_size,
        quantity_cv=quantity_cv,
        mean_label_tvd=mean_label_tvd,
        rotation_std_degrees=rotation_std_degrees,
    )
