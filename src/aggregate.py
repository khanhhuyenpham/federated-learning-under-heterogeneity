
import torch
import torch.nn as nn

def aggregate(
    global_model: nn.Module,
    local_models: list[nn.Module],
    client_sizes: list[int],
) -> nn.Module:
    """Weighted FedAvg aggregation for models whose trainable state is in parameters.

    SimpleMLP has no BatchNorm-style buffers, so parameter-wise in-place
    accumulation is sufficient here.
    """
    assert len(local_models) == len(client_sizes) > 0

    total_size = sum(client_sizes)
    global_params = list(global_model.parameters())

    with torch.no_grad():
        for param in global_params:
            param.zero_()

        for local_model, size in zip(local_models, client_sizes):
            weight = size / total_size
            for global_param, local_param in zip(global_params, local_model.parameters()):
                global_param.add_(local_param, alpha=weight)

    return global_model
