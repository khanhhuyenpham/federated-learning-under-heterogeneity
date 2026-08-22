
import torch
import pandas as pd
import numpy as np

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

def create_client_loaders(train_ds, client_indices, batch_size: int, seed=None, shuffle=True,):
    loaders = {}
    for client_id in sorted(client_indices):
        indices = client_indices[client_id]

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed + int(client_id))

        loaders[client_id] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_ds, indices),
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
        
