from dataclasses import dataclass

from .condition_metadata import (
    ConditionMetadata,
    build_condition_metadata,
)

from .data import (
    ClientSplitLoaders,
    create_client_split_loaders,
    split_client_indices,
    validate_partition,
)

from .heterogeneity import (
    FederatedData,
    HeterogeneityConfig,
    build_federated_data,
)

@dataclass
class ConditionArtifacts:
    config: HeterogeneityConfig
    federated_data: FederatedData

    client_train_indices: dict[int, list[int]]
    client_validation_indices: dict[int, list[int]]
    client_test_indices: dict[int, list[int]]

    loaders: ClientSplitLoaders
    metadata: ConditionMetadata

def build_condition_artifacts(
    train_ds,
    config: HeterogeneityConfig,
    batch_size: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> ConditionArtifacts:
    federated_data = build_federated_data(
        train_ds=train_ds,
        config=config,
    )

    validate_partition(
        client_indices=federated_data.client_indices,
        dataset_size=len(train_ds),
        expected_num_clients=config.num_clients,
    )

    (
        client_train_indices,
        client_validation_indices,
        client_test_indices
    ) = split_client_indices(
        client_indices=federated_data.client_indices,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=config.seed,
        )

    split_mappings = {
        "train": client_train_indices,
        "validation": client_validation_indices,
        "test": client_test_indices,
    }

    for split_name, mapping in split_mappings.items():
        empty_clients = [
            client_id
            for client_id, indices in mapping.items()
            if not indices
        ]

        if empty_clients:
            raise ValueError(
                f"{split_name} split is empty for clients "
                f"{empty_clients}"
            )

    loaders = create_client_split_loaders(
        train_ds=train_ds,
        client_train_indices=client_train_indices,
        client_validation_indices=client_validation_indices,
        client_test_indices=client_test_indices,
        batch_size=batch_size,
        seed=config.seed,
        client_feature_transforms=federated_data.client_feature_transforms,
    )

    metadata = build_condition_metadata(
        train_ds=train_ds,
        config=config,
        federated_data=federated_data,
    )

    return ConditionArtifacts(
        config=config,
        federated_data=federated_data,
        client_train_indices=client_train_indices,
        client_validation_indices=client_validation_indices,
        client_test_indices=client_test_indices,
        loaders=loaders,
        metadata=metadata,
    )