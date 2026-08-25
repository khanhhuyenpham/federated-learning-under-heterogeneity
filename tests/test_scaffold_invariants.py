import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.aggregate import aggregate
from src.scaffold import (
    client_update_scaffold,
    initialize_control_state,
    run_scaffold_round,
    update_client_control,
)
from src.training import client_update


def _tiny_loader(num_examples: int = 4) -> DataLoader:
    features = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, -0.5],
        ],
        dtype=torch.float32,
    )[:num_examples]
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)[:num_examples]
    return DataLoader(TensorDataset(features, targets), batch_size=2, shuffle=False)


def _tiny_model() -> nn.Module:
    model = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.20, -0.10], [0.05, 0.30]]))
        model.bias.copy_(torch.tensor([0.01, -0.02]))
    return model


def _assert_models_close(left: nn.Module, right: nn.Module) -> None:
    for left_param, right_param in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(left_param, right_param, rtol=0, atol=1e-7)


def test_first_scaffold_local_update_matches_fedavg_with_zero_controls() -> None:
    device = torch.device("cpu")
    global_model = _tiny_model().to(device)
    loader = _tiny_loader()
    loss_fn = nn.CrossEntropyLoss()

    fedavg_model = client_update(
        global_model=copy.deepcopy(global_model),
        client_loader=loader,
        local_epochs=2,
        learning_rate=0.05,
        loss_fn=loss_fn,
        device=device,
    )

    zero_server_control = initialize_control_state(global_model)
    zero_client_control = initialize_control_state(global_model)
    scaffold_model, _, _, num_steps = client_update_scaffold(
        global_model=copy.deepcopy(global_model),
        client_loader=loader,
        server_control=zero_server_control,
        client_control=zero_client_control,
        local_epochs=2,
        learning_rate=0.05,
        loss_fn=loss_fn,
        device=device,
    )

    assert num_steps == 4
    _assert_models_close(scaffold_model, fedavg_model)


def test_option_ii_client_control_matches_closed_form() -> None:
    global_model = _tiny_model()
    local_model = _tiny_model()
    with torch.no_grad():
        local_model.weight.add_(torch.tensor([[0.10, -0.20], [0.05, 0.15]]))
        local_model.bias.add_(torch.tensor([0.03, -0.04]))

    server_control = initialize_control_state(global_model)
    client_control = initialize_control_state(global_model)
    with torch.no_grad():
        for name in server_control:
            server_control[name].fill_(0.25)
            client_control[name].fill_(-0.10)

    num_local_steps = 3
    learning_rate = 0.2
    new_client_control, control_delta = update_client_control(
        global_model=global_model,
        local_model=local_model,
        server_control=server_control,
        client_control=client_control,
        num_local_steps=num_local_steps,
        learning_rate=learning_rate,
    )

    for name, global_parameter in global_model.named_parameters():
        local_parameter = dict(local_model.named_parameters())[name]
        expected = (
            client_control[name]
            - server_control[name]
            + (global_parameter - local_parameter) / (num_local_steps * learning_rate)
        )
        torch.testing.assert_close(new_client_control[name], expected)
        torch.testing.assert_close(
            control_delta[name],
            new_client_control[name] - client_control[name],
        )


def test_full_participation_server_control_is_weighted_client_control_average() -> None:
    device = torch.device("cpu")
    global_model = _tiny_model().to(device)
    server_control = initialize_control_state(global_model)
    client_controls = {
        0: initialize_control_state(global_model),
        1: initialize_control_state(global_model),
    }
    client_loaders = {
        0: _tiny_loader(num_examples=2),
        1: _tiny_loader(num_examples=4),
    }

    result = run_scaffold_round(
        global_model=global_model,
        client_loaders=client_loaders,
        server_control=server_control,
        client_controls=client_controls,
        local_epochs=1,
        learning_rate=0.05,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
    )

    assert set(result.client_weights_by_id) == {0, 1}
    assert abs(sum(result.client_weights_by_id.values()) - 1.0) < 1e-12
    for name in result.server_control:
        weighted_average = torch.zeros_like(result.server_control[name])
        for client_id, control in result.client_controls.items():
            weighted_average.add_(
                control[name],
                alpha=result.client_weights_by_id[client_id],
            )
        torch.testing.assert_close(result.server_control[name], weighted_average)


def test_parameterwise_weighted_aggregation_matches_vectorized_average() -> None:
    global_model = _tiny_model()
    local_a = _tiny_model()
    local_b = _tiny_model()
    with torch.no_grad():
        for parameter in local_a.parameters():
            parameter.add_(1.0)
        for parameter in local_b.parameters():
            parameter.sub_(2.0)

    sizes = [1, 3]
    aggregated = aggregate(
        global_model=global_model,
        local_models=[local_a, local_b],
        client_sizes=sizes,
    )

    expected_parts = []
    for param_a, param_b in zip(local_a.parameters(), local_b.parameters()):
        expected_param = (sizes[0] * param_a + sizes[1] * param_b) / sum(sizes)
        expected_parts.append(expected_param.detach().reshape(-1))
    expected_vector = torch.cat(expected_parts)
    actual_vector = torch.cat(
        [parameter.detach().reshape(-1) for parameter in aggregated.parameters()]
    )

    torch.testing.assert_close(actual_vector, expected_vector)
