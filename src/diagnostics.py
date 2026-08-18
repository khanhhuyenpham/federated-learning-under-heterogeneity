import torch
import torch.nn as nn

def model_to_vector(model : nn.Module):
    """
    Convert all trainable parameters of a model
    into one detached 1-D tensor.
    """
    model_params = [
        p.detach().flatten() for p in model.parameters()
    ]
    v = torch.cat(model_params, dim=0)
    return v

def model_update_vector(local_model, global_model):
    global_vector = model_to_vector(global_model).clone()
    delta = model_to_vector(local_model) - global_vector
    return delta

def cosine_similarity_updates(delta_u, delta_j):
    norm_u = torch.linalg.norm(delta_u)
    norm_j = torch.linalg.norm(delta_j)

    if norm_u == 0 or norm_j == 0:
        return torch.tensor(
            float("nan"),
            device = delta_u.device,
            dtype = delta_u.dtype
        )
    return torch.dot(delta_u, delta_j) / (norm_u * norm_j)

def update_magnitude(delta):
    return torch.linalg.vector_norm(delta)

