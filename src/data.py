
import torch
import pandas as pd
import numpy as np
from collections import Counter

def partition_shard(train_ds, num_clients : int, seed : int, shard_per_client : int):
    generator = torch.Generator().manual_seed(seed)
    labels = [train_ds[i][1] for i in range(len(train_ds))]

    sorted_indices = sorted(range(len(train_ds)), key=lambda i: labels[i])
    shard_size = len(train_ds) // (num_clients * shard_per_client)
    shards = [sorted_indices[i:i+shard_size] for i in range(0, len(sorted_indices), shard_size)]

    shuffled_shard_ids = torch.randperm(len(shards), generator=generator)
    client_indices = {}

    for pos, shard_id in enumerate(shuffled_shard_ids):
      client_id = pos // shard_per_client
      if client_id not in client_indices:
        client_indices[client_id] = []
      client_indices[client_id].extend(shards[int(shard_id)])
    
    return client_indices

def partition_dirichlet(
    train_ds,
    num_clients: int,
    alpha: float,
    seed: int,
):
    assert num_clients > 0
    assert alpha > 0

    rng = np.random.default_rng(seed)

    targets = train_ds.targets.cpu().numpy()
    
    client_indices = {
        client_id: []
        for client_id in range(num_clients)
    }

    classes = np.unique(targets)

    for label in classes:
        class_indices = np.flatnonzero(targets == label)

        rng.shuffle(class_indices)

        proportions = rng.dirichlet(
            np.full(num_clients, alpha)
        )

        split_points = (
            np.cumsum(proportions)[:-1]
            * len(class_indices)
        ).astype(int)

        chunks = np.split(
            class_indices,
            split_points
        )

        for client_id, chunk in enumerate(chunks):
            client_indices[client_id].extend(
                chunk.tolist()
            )

    for client_id in client_indices:
        indices = np.array(
            client_indices[client_id],
            dtype=np.int64
        )
        rng.shuffle(indices)
        client_indices[client_id] = indices.tolist()

    return client_indices
   
def partition_iid(train_ds, num_clients: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(len(train_ds), generator=generator)
    splits = torch.tensor_split(shuffled, num_clients)
    return {client_id: split.tolist() for client_id, split in enumerate(splits)}

def validate_partition(
    client_indices,
    dataset_size: int,
    expected_num_clients: int | None = None,
    require_full_coverage: bool = True,
    require_non_empty_clients: bool = True,
) -> None:
    if dataset_size <= 0:
        raise ValueError("dataset_size needs to be positive")

    actual_client_ids = set(client_indices.keys())

    if expected_num_clients is not None:
        if expected_num_clients <= 0:
            raise ValueError("expected_num_clients needs to be positive")
        expected_client_ids = set(range(expected_num_clients))
        if actual_client_ids != expected_client_ids:
            raise ValueError(f"Expected client IDs {expected_client_ids} but received {actual_client_ids}")

    all_sample_indices = []
    for client_id in actual_client_ids:
        if require_non_empty_clients and not client_indices[client_id]:
            raise ValueError("empty client detected")
        all_sample_indices.extend(client_indices[client_id])

    if len(set(all_sample_indices)) != len(all_sample_indices):
        raise ValueError("duplicate sample indices detected")
    
    for sample_idx in all_sample_indices:
        if sample_idx < 0 or sample_idx >= dataset_size: 
            raise ValueError("index out of range")
        
    if require_full_coverage:
        if set(all_sample_indices) != set(range(dataset_size)):
            raise ValueError("full_coverage not met")

def summarize_partition(
    train_ds, 
    client_indices,
) -> pd.DataFrame:
    client_rows = []
    class_ids = sorted(set(train_ds.targets.tolist()))
    for client_id, indices in client_indices.items():
        client_targets = train_ds.targets[indices].tolist()
        class_counts = Counter(client_targets)
        client_size = len(indices)
        if client_size == 0:
            raise ValueError(f"client {client_id} has no assigned samples")
        
        for class_id in class_ids:
            cnt = class_counts.get(class_id, 0)
            client_rows.append({
                "client_id": client_id,
                "class_id": class_id,
                "count": cnt,
                "proportion": cnt / client_size,
                "client_size": client_size,
            })
    return pd.DataFrame(client_rows)

def create_client_loaders(train_ds, client_indices, batch_size: int, seed=None, shuffle=True, client_feature_transforms=None):
    loaders = {}
    for client_id in sorted(client_indices):
        feature_transform = None
        if client_feature_transforms is not None:
            feature_transform = client_feature_transforms.get(client_id)

        indices = client_indices[client_id]

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed + int(client_id))

        loaders[client_id] = torch.utils.data.DataLoader(
            TransformedSubset(train_ds, indices, feature_transform),
            batch_size = batch_size,
            shuffle=shuffle,
            generator=generator,
        )
    return loaders

def client_label_counts(train_ds, client_indices):
    rows = []
    targets = train_ds.targets

    for client_id, indices in client_indices.items():
        counts = torch.bincount(targets[indices], minlength=10)
        row = {"client": client_id, "samples": len(indices)}
        row.update({f"digit_{d}": int(counts[d]) for d in range(10)})
        rows.append(row)

    return pd.DataFrame(rows)

def split_client_indices(client_indices, train_fraction=0.8, validation_fraction=0.1, seed=42):
    if not (0 < train_fraction and train_fraction < 1 and 0 <= validation_fraction and validation_fraction < 1 and train_fraction + validation_fraction < 1):
        raise ValueError("invalid fraction")
    client_train_indices = {}
    client_validation_indices = {}
    client_test_indices = {}

    for client_id, indices in client_indices.items():
        normalized_indices = (
            torch.as_tensor(indices, dtype=torch.long)
            .cpu()
            .tolist()
        )
        rng = np.random.default_rng(seed + int(client_id))
        shuffled_indices = rng.permutation(
            np.asarray(normalized_indices)
        ).tolist()
        num_examples = len(shuffled_indices)
        num_train = int(num_examples * train_fraction)
        num_validation = int(num_examples * validation_fraction)
        train_indices = shuffled_indices[:num_train]
        validation_indices = shuffled_indices[num_train:num_validation + num_train]
        test_indices = shuffled_indices[num_train + num_validation:]

        train_set = set(train_indices)
        validation_set = set(validation_indices)
        test_set = set(test_indices)
        original_set = set(normalized_indices)

        assert train_set.isdisjoint(validation_set)
        assert train_set.isdisjoint(test_set)
        assert validation_set.isdisjoint(test_set)
        assert train_set | validation_set | test_set == original_set
        assert len(train_indices) + len(validation_indices) + len(test_indices) == len(normalized_indices)

        client_train_indices[client_id] = train_indices
        client_validation_indices[client_id] = validation_indices
        client_test_indices[client_id] = test_indices

    return (
        client_train_indices,
        client_validation_indices,
        client_test_indices,
    )
        
def partition_label_skew_balanced(
    train_ds,
    num_clients: int,
    alpha: float,
    seed: int,
):
    """Partition data with Dirichlet label skew and balanced client sizes."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    dataset_size = len(train_ds)
    if num_clients > dataset_size:
        raise ValueError("num_clients cannot exceed dataset size when clients must be non-empty")

    rng = np.random.default_rng(seed=seed)
    targets = train_ds.targets.cpu().numpy()

    classes = np.unique(targets)

    remainder = dataset_size % num_clients
    capacities = [dataset_size // num_clients] * num_clients

    client_indices = {
        client_id : []
        for client_id in range(num_clients)
    }

    class_pools = {
        class_id : []
        for class_id in classes
    }

    for client_id in range(remainder):
        capacities[client_id] += 1

    for class_id in classes:
        class_indices = np.flatnonzero(targets == class_id)
        rng.shuffle(class_indices)
        class_pools[class_id] = class_indices.tolist()


    client_vectors = {
        client_id : rng.dirichlet(np.full(len(classes), alpha)) 
        for client_id in range(num_clients)
    }

    while any(capacity > 0 for capacity in capacities):
        for client_id in range(num_clients):
            if capacities[client_id] == 0: continue

            available_mask = np.array(
                [bool(class_pools[class_id]) for class_id in classes],
                dtype=bool,
            )

            if not available_mask.any():
                raise RuntimeError(
                    "No samples remain, but at least one client still has capacity"
                )

            client_vector = client_vectors[client_id].copy()
            client_vector[~available_mask] = 0.0

            probability_sum = client_vector.sum()

            if not np.isfinite(probability_sum) or probability_sum <= 0:
                client_vector = available_mask.astype(float)
                probability_sum = client_vector.sum()

            normalized_client_vector = client_vector / np.sum(client_vector)
            class_chosen = rng.choice(classes, p=normalized_client_vector)
            sample_idx = class_pools[class_chosen].pop()
            client_indices[client_id].append(sample_idx)
            capacities[client_id] -= 1

    return client_indices

def partition_quantity_skew(
    train_ds,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 1,
) -> dict[int, list[int]]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    if min_samples_per_client <= 0:
        raise ValueError("min_samples_per_client must be positive")

    dataset_size = len(train_ds)
    if num_clients > dataset_size:
        raise ValueError("num_clients cannot exceed dataset size when clients must be non-empty")

    if num_clients * min_samples_per_client > dataset_size:
        raise ValueError("minimum allocation cannot exceed dataset size")
    
    rng = np.random.default_rng(seed=seed)

    capacities = np.full(
        num_clients,
        min_samples_per_client,
        dtype=int,
    )

    reserved_samples = num_clients * min_samples_per_client
    remaining_samples = dataset_size - reserved_samples

    client_probabilities = rng.dirichlet(np.full(num_clients, alpha))
    additional_counts = rng.multinomial(
        remaining_samples,
        client_probabilities,
    )
    capacities += additional_counts

    shuffled_indices = rng.permutation(dataset_size)
    split_points = np.cumsum(capacities)[:-1]

    client_splits = np.split(
        shuffled_indices,
        split_points,
    )

    assert len(capacities) == num_clients
    assert int(capacities.sum()) == dataset_size
    assert np.all(capacities >= min_samples_per_client)
    return {client_id : split.tolist() for client_id, split in enumerate(client_splits)}

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(
        self, 
        base_dataset,
        indices,
        feature_transform=None,
    ):
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices]
        self.feature_transform = feature_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        features, label = self.base_dataset[self.indices[position]]
        if self.feature_transform is not None:
            features = self.feature_transform(features)
        return features, label