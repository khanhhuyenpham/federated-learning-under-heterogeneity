
import torch

def model_distance(model_a, model_b) -> float:
    vec_a = torch.nn.utils.parameters_to_vector(model_a.parameters())
    vec_b = torch.nn.utils.parameters_to_vector(model_b.parameters())
    return torch.dist(vec_a, vec_b, p=2).item()
