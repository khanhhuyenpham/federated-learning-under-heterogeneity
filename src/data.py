
import torch
import pandas as pd

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

def partition_iid(train_ds, num_clients: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(len(train_ds), generator=generator)
    splits = torch.tensor_split(shuffled, num_clients)
    return {client_id: split.clone() for client_id, split in enumerate(splits)}

def create_client_loaders(train_ds, client_indices, batch_size: int):
    return {
        client_id: torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_ds, indices),
            batch_size=batch_size,
            shuffle=True,
        )
        for client_id, indices in client_indices.items()
    }

def client_label_counts(train_ds, client_indices):
    rows = []
    targets = train_ds.targets

    for client_id, indices in client_indices.items():
        counts = torch.bincount(targets[indices], minlength=10)
        row = {"client": client_id, "samples": len(indices)}
        row.update({f"digit_{d}": int(counts[d]) for d in range(10)})
        rows.append(row)

    return pd.DataFrame(rows)
