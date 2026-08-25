from typing import Dict

import torch
from torch import nn
from .evaluation import evaluate_clients, summarize_client_evaluations
import pandas as pd
from dataclasses import dataclass
from .training import aggregate
from itertools import combinations
import time

ControlState = Dict[str, torch.Tensor]

def initialize_control_state(
    model: nn.Module,
) -> ControlState:
    control_state: ControlState = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        control_state[name] = torch.zeros_like(parameter)

    return control_state

def assert_control_state_matches_model(
    model: nn.Module,
    control_state: ControlState,
) -> None:
    trainable_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    expected_names = set(trainable_parameters)
    actual_names = set(control_state)

    if actual_names != expected_names:
        missing = expected_names - actual_names
        unexpected = actual_names - expected_names

        raise ValueError(
            "Control-state parameter names do not match model. "
            f"Missing: {sorted(missing)}. "
            f"Unexpected: {sorted(unexpected)}."
        )

    for name, parameter in trainable_parameters.items():
        control_tensor = control_state[name]

        if control_tensor.shape != parameter.shape:
            raise ValueError(
                f"Shape mismatch for {name!r}: "
                f"expected {parameter.shape}, "
                f"received {control_tensor.shape}."
            )

        if control_tensor.dtype != parameter.dtype:
            raise ValueError(
                f"Dtype mismatch for {name!r}: "
                f"expected {parameter.dtype}, "
                f"received {control_tensor.dtype}."
            )

        if control_tensor.device != parameter.device:
            raise ValueError(
                f"Device mismatch for {name!r}: "
                f"expected {parameter.device}, "
                f"received {control_tensor.device}."
            )

        if control_tensor.requires_grad:
            raise ValueError(
                f"Control tensor {name!r} must not require gradients."
            )

        if not torch.isfinite(control_tensor).all():
            raise ValueError(
                f"Control tensor {name!r} contains nonfinite values."
            )

def clone_control_state(
    control_state: ControlState,
) -> ControlState:
    cloned_state: ControlState = {}

    for name, tensor in control_state.items():
        cloned_state[name] = control_state[name].detach().clone()

    return cloned_state

def apply_scaffold_gradient_correction(
    model: nn.Module,
    server_control: ControlState,
    client_control: ControlState,
) -> None:
    assert_control_state_matches_model(
        model=model,
        control_state=server_control,
    )

    assert_control_state_matches_model(
        model=model,
        control_state=client_control,
    )

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            if parameter.grad is None:
                raise RuntimeError(
                    f"Parameter {name!r} has no gradient. "
                    "Call loss.backward() before applying "
                    "the SCAFFOLD correction."
                )

            correction = server_control[name] - client_control[name]
            parameter.grad.add_(correction)

def update_client_control(
    global_model: nn.Module,
    local_model: nn.Module,
    server_control: ControlState,
    client_control: ControlState,
    num_local_steps: int,
    learning_rate: float,
) -> tuple[ControlState, ControlState]:
    if num_local_steps <= 0:
        raise ValueError("num_local_steps must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    assert_control_state_matches_model(
        model=global_model,
        control_state=server_control
    )

    assert_control_state_matches_model(
        model=global_model,
        control_state=client_control,
    )

    global_parameters = dict(global_model.named_parameters())
    local_parameters = dict(local_model.named_parameters())

    assert set(global_parameters) == set(local_parameters)

    old_client_control = clone_control_state(client_control)

    new_client_control: ControlState = {}
    control_delta: ControlState = {}

    with torch.no_grad():
        for name, global_parameter in global_parameters.items():
            if not global_parameter.requires_grad:
                continue

            local_parameter = local_parameters[name]
            global_parameter = global_parameters[name]

            trajectory_term = (global_parameter - local_parameter) / (num_local_steps * learning_rate)

            new_value = (old_client_control[name] - server_control[name] + trajectory_term)

            new_client_control[name] = new_value.detach().clone()
            control_delta[name] = (new_client_control[name] - old_client_control[name]).detach().clone()

    return new_client_control, control_delta


import copy
from torch.optim import SGD
from torch.utils.data import DataLoader

def client_update_scaffold(
    global_model: nn.Module,
    client_loader: DataLoader,
    server_control: ControlState,
    client_control: ControlState,
    local_epochs: int,
    learning_rate: float,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[nn.Module, ControlState, ControlState, int,]:

    expected_device = torch.empty(
        0,
        device=device,
    ).device

    model_devices = {
        parameter.device
        for parameter in global_model.parameters()
    }

    if model_devices != {expected_device}:
        raise ValueError(
            "global_model must already be on requested "
            f"device {expected_device}. but found {model_devices}"
        )

    if local_epochs <= 0:
        raise ValueError(
            "local_epochs must be positive."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive."
        )

    assert_control_state_matches_model(
        model=global_model,
        control_state=server_control,
    )

    assert_control_state_matches_model(
        model=global_model,
        control_state=client_control,
    )

    server_control_before = clone_control_state(server_control)
    client_control_before = clone_control_state(client_control)

    local_model = copy.deepcopy(global_model).to(device)

    local_model.train()

    optimizer = SGD(local_model.parameters(), lr = learning_rate)

    num_local_steps = 0
    for _ in range(local_epochs):
        for inputs, targets in client_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            predictions = local_model(inputs)

            loss = loss_fn(predictions, targets)

            loss.backward()

            apply_scaffold_gradient_correction(model=local_model, server_control=server_control, client_control=client_control)

            optimizer.step()

            num_local_steps += 1

    if num_local_steps <= 0:
        raise RuntimeError("The client performed no optimizer steps.")

    new_client_control, control_delta = (
        update_client_control(
            global_model=global_model,
            local_model=local_model,
            server_control=server_control,
            client_control=client_control,
            num_local_steps=num_local_steps,
            learning_rate=learning_rate,
        )
    )

    for name in server_control:
        assert torch.equal(
            server_control[name],
            server_control_before[name],
        )

    for name in client_control:
        assert torch.equal(
            client_control[name],
            client_control_before[name],
        )

    assert_control_state_matches_model(
        model=local_model,
        control_state=new_client_control,
    )

    assert_control_state_matches_model(
        model=local_model,
        control_state=control_delta,
    )

    return local_model, new_client_control, control_delta, num_local_steps

import math
from typing import Mapping

def update_server_control(
    model: nn.Module,
    server_control: ControlState,
    control_deltas_by_id: Mapping[int, ControlState],
    client_weights_by_id: Mapping[int, float],
) -> ControlState:
    if not control_deltas_by_id:
        raise ValueError(
            "At least one client control delta is required."
        )

    if set(control_deltas_by_id) != set(client_weights_by_id):
        raise ValueError(
            "Control-delta clients and weight clients "
            "must match."
        )

    assert_control_state_matches_model(
        model=model,
        control_state=server_control,
    )

    for client_id, control_delta in (
        control_deltas_by_id.items()
    ):
        assert_control_state_matches_model(
            model=model,
            control_state=control_delta,
        )

    for client_id, weight in (
        client_weights_by_id.items()
    ):
        if not math.isfinite(weight):
            raise ValueError(
                f"Client {client_id} has a nonfinite weight."
            )

        if weight < 0:
            raise ValueError(
                f"Client {client_id} has a negative weight."
            )

    total_weight = sum(
        client_weights_by_id.values()
    )

    if not math.isclose(
        total_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Client weights must sum to one; "
            f"received {total_weight}."
        )

    new_server_control = clone_control_state(server_control)

    with torch.no_grad():
        for name in new_server_control:
            for client_id, control_delta in control_deltas_by_id.items():
                weight = client_weights_by_id[client_id]
                new_server_control[name].add_(control_delta[name], alpha=weight)

    return new_server_control

@dataclass
class ScaffoldRoundResult:
    global_model: nn.Module
    server_control: ControlState
    client_controls: Dict[int, ControlState]

    local_models_by_id: Dict[int, nn.Module]
    control_deltas_by_id: Dict[int, ControlState]
    client_sizes_by_id: Dict[int, int]
    client_weights_by_id: Dict[int, float]
    local_steps_by_id: Dict[int, int]

def run_scaffold_round(
    global_model: nn.Module,
    client_loaders: Mapping[int, DataLoader],
    server_control: ControlState,
    client_controls: Mapping[int, ControlState],
    local_epochs: int,
    learning_rate: float,
    loss_fn: nn.Module,
    device: torch.device,
) -> ScaffoldRoundResult:
    if not client_loaders:
        raise ValueError(
            "At least one client is required."
        )

    if set(client_loaders) != set(client_controls):
        raise ValueError(
            "Every client loader must have exactly one "
            "persistent client control."
        )

    sorted_client_ids = sorted(client_loaders)

    round_server_control = clone_control_state(server_control)

    local_models_by_id: Dict[int, nn.Module] = {}

    new_client_controls: Dict[int, ControlState] = {}

    control_deltas_by_id: Dict[int, ControlState] = {}

    client_sizes_by_id: Dict[int, int] = {}

    local_steps_by_id: Dict[int, int] = {}

    for client_id in sorted_client_ids:
        client_loader = client_loaders[client_id]

        (
            local_model,
            new_client_control,
            control_delta,
            num_local_steps,
        ) = client_update_scaffold(
            global_model=global_model,
            client_loader=client_loader,
            server_control=round_server_control,
            client_control=client_controls[client_id],
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            device=device,
        )

        local_models_by_id[client_id] = local_model
        new_client_controls[client_id] = clone_control_state(new_client_control)
        control_deltas_by_id[client_id] = control_delta
        client_sizes_by_id[client_id] = len(client_loader.dataset)
        local_steps_by_id[client_id] = num_local_steps

    total_client_size = sum(client_sizes_by_id.values())

    if total_client_size <= 0:
        raise RuntimeError("The participating clients contain no examples.")

    client_weights_by_id = {
        client_id: (
            client_sizes_by_id[client_id] / total_client_size
        )
        for client_id in sorted_client_ids
    }

    assert math.isclose(
        sum(client_weights_by_id.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    ordered_local_models = [
        local_models_by_id[client_id] for client_id in sorted_client_ids
    ]

    ordered_client_sizes = [
        client_sizes_by_id[client_id] for client_id in sorted_client_ids
    ]

    new_global_model = aggregate(
        global_model=global_model,
        local_models=ordered_local_models,
        client_sizes=ordered_client_sizes,
    )

    new_server_control = update_server_control(
        model=global_model,
        server_control=round_server_control,
        control_deltas_by_id=control_deltas_by_id,
        client_weights_by_id=client_weights_by_id,
    )

    return ScaffoldRoundResult(
        global_model=new_global_model,
        server_control=new_server_control,
        client_controls=new_client_controls,
        local_models_by_id=local_models_by_id,
        control_deltas_by_id=control_deltas_by_id,
        client_sizes_by_id=client_sizes_by_id,
        client_weights_by_id=client_weights_by_id,
        local_steps_by_id=local_steps_by_id,
    )

def control_state_to_vector(
    model: nn.Module,
    control_state: ControlState,
) -> torch.Tensor:
    assert_control_state_matches_model(
        model=model,
        control_state=control_state,
    )
    control_parts = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        control_part = control_state[name].detach().reshape(-1)
        control_parts.append(control_part)

    if not control_parts:
        raise ValueError(
            "The model has no trainable parameters."
        )

    return torch.cat(control_parts)


from .diagnostics import (
    cosine_similarity_updates,
    model_to_vector,
    model_update_vector,
    update_magnitude,
)


def build_scaffold_cost_summary(
    *,
    model: nn.Module,
    server_control: ControlState,
    num_clients: int,
    seed: int,
) -> pd.DataFrame:
    """Build the static storage and per-round communication cost summary."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")

    assert_control_state_matches_model(
        model=model,
        control_state=server_control,
    )

    model_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    )
    control_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in server_control.values()
    )
    control_values = sum(
        tensor.numel()
        for tensor in server_control.values()
    )

    # Per client and per round:
    # server -> client: model + server control
    # client -> server: model/update + client-control delta
    baseline_communication_bytes = 2 * model_bytes * num_clients
    additional_control_bytes = 2 * control_bytes * num_clients

    return pd.DataFrame([{
        "method": "scaffold",
        "seed": seed,
        "num_clients": num_clients,
        "model_bytes": int(model_bytes),
        "control_bytes": int(control_bytes),
        "extra_server_state_bytes": int(control_bytes),
        "extra_client_state_bytes_per_client": int(control_bytes),
        "extra_client_state_bytes_total": int(
            num_clients * control_bytes
        ),
        "baseline_communication_bytes_per_round": int(
            baseline_communication_bytes
        ),
        "additional_control_bytes_per_round": int(
            additional_control_bytes
        ),
        "total_communication_bytes_per_round": int(
            baseline_communication_bytes
            + additional_control_bytes
        ),
        "additional_transmitted_values_per_round": int(
            2 * control_values * num_clients
        ),
    }])


def build_scaffold_round_diagnostics(
    global_model_before: nn.Module,
    server_control_before: ControlState,
    client_controls_before: Mapping[int, ControlState],
    round_result: ScaffoldRoundResult,
    round_idx: int,
    seed: int,
    alpha: float,
    local_epochs: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build client-, client-pair-, and round-level diagnostic rows."""
    sorted_client_ids = sorted(round_result.local_models_by_id)

    if set(sorted_client_ids) != set(client_controls_before):
        raise ValueError(
            "Previous client controls must match the participating clients."
        )

    global_vector = model_to_vector(global_model_before).detach()

    client_updates = {
        client_id: model_update_vector(
            local_model=round_result.local_models_by_id[client_id],
            global_model=global_model_before,
        )
        for client_id in sorted_client_ids
    }

    aggregate_update = torch.zeros_like(global_vector)
    for client_id in sorted_client_ids:
        aggregate_update += (
            round_result.client_weights_by_id[client_id]
            * client_updates[client_id]
        )

    actual_server_update = (
        model_to_vector(round_result.global_model).detach()
        - global_vector
    )
    assert torch.allclose(
        actual_server_update,
        aggregate_update,
        rtol=1e-5,
        atol=1e-6,
    )

    global_model_magnitude = torch.linalg.norm(global_vector).item()
    epsilon = 1e-12
    server_control_vector = control_state_to_vector(
        model=global_model_before,
        control_state=server_control_before,
    )

    client_rows: list[dict] = []
    for client_id in sorted_client_ids:
        update = client_updates[client_id]
        client_weight = round_result.client_weights_by_id[client_id]
        peer_weight = 1.0 - client_weight

        if peer_weight <= 0:
            raise ValueError(
                "LOO alignment requires at least two positive-weight clients."
            )

        peer_update = (
            aggregate_update - client_weight * update
        ) / peer_weight

        old_client_control_vector = control_state_to_vector(
            model=global_model_before,
            control_state=client_controls_before[client_id],
        )
        new_client_control_vector = control_state_to_vector(
            model=global_model_before,
            control_state=round_result.client_controls[client_id],
        )
        control_delta_vector = control_state_to_vector(
            model=global_model_before,
            control_state=round_result.control_deltas_by_id[client_id],
        )
        correction_vector = (
            server_control_vector - old_client_control_vector
        )
        update_magnitude_value = update_magnitude(update).item()

        client_rows.append({
            "method": "scaffold",
            "seed": seed,
            "alpha": alpha,
            "round": round_idx,
            "local_epochs": local_epochs,
            "client_id": client_id,
            "client_size": round_result.client_sizes_by_id[client_id],
            "client_weight": client_weight,
            "num_local_steps": round_result.local_steps_by_id[client_id],
            "update_magnitude": update_magnitude_value,
            "relative_update_magnitude": (
                update_magnitude_value
                / (global_model_magnitude + epsilon)
            ),
            "aggregate_alignment": cosine_similarity_updates(
                update,
                aggregate_update,
            ).item(),
            "loo_alignment": cosine_similarity_updates(
                update,
                peer_update,
            ).item(),
            "correction_magnitude": update_magnitude(
                correction_vector
            ).item(),
            "client_control_magnitude": update_magnitude(
                new_client_control_vector
            ).item(),
            "control_delta_magnitude": update_magnitude(
                control_delta_vector
            ).item(),
        })

    pairwise_rows: list[dict] = []
    for client_i, client_j in combinations(sorted_client_ids, 2):
        pairwise_rows.append({
            "method": "scaffold",
            "seed": seed,
            "alpha": alpha,
            "round": round_idx,
            "local_epochs": local_epochs,
            "client_i": client_i,
            "client_j": client_j,
            "cosine_similarity": cosine_similarity_updates(
                client_updates[client_i],
                client_updates[client_j],
            ).item(),
        })

    new_server_control_vector = control_state_to_vector(
        model=global_model_before,
        control_state=round_result.server_control,
    )
    model_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in global_model_before.state_dict().values()
    )
    control_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in server_control_before.values()
    )
    num_clients = len(sorted_client_ids)

    round_rows = [{
        "method": "scaffold",
        "seed": seed,
        "alpha": alpha,
        "round": round_idx,
        "local_epochs": local_epochs,
        "num_participating_clients": num_clients,
        "total_client_size": sum(
            round_result.client_sizes_by_id.values()
        ),
        "total_local_steps": sum(
            round_result.local_steps_by_id.values()
        ),
        "aggregate_update_magnitude": update_magnitude(
            aggregate_update
        ).item(),
        "server_control_magnitude": update_magnitude(
            new_server_control_vector
        ).item(),
        "total_communication_bytes": int(
            2 * (model_bytes + control_bytes) * num_clients
        ),
    }]

    return client_rows, pairwise_rows, round_rows


@dataclass
class ScaffoldExperimentResult:
    final_model: nn.Module
    final_server_control: ControlState
    final_client_controls: dict[int, ControlState]
    validation_history: pd.DataFrame
    performance_history: pd.DataFrame
    client_diagnostics: pd.DataFrame
    pairwise_diagnostics: pd.DataFrame
    round_diagnostics: pd.DataFrame
    cost_summary: pd.DataFrame
    training_runtime_seconds: float


def run_scaffold_experiment(
    *,
    seed: int,
    initial_model: nn.Module,
    train_loaders: Mapping[int, DataLoader],
    validation_loaders: Mapping[int, DataLoader],
    config,
    loss_fn: nn.Module,
    device: torch.device,
) -> ScaffoldExperimentResult:
    """Run a full-participation SCAFFOLD experiment for one seed."""
    if set(train_loaders) != set(validation_loaders):
        raise ValueError(
            "Training and validation loaders must contain the same client IDs."
        )

    global_model = copy.deepcopy(initial_model).to(device)
    server_control = initialize_control_state(global_model)
    client_controls = {
        client_id: initialize_control_state(global_model)
        for client_id in sorted(train_loaders)
    }

    assert set(client_controls) == set(train_loaders)
    assert all(
        torch.count_nonzero(tensor).item() == 0
        for tensor in server_control.values()
    )
    assert all(
        torch.count_nonzero(tensor).item() == 0
        for control in client_controls.values()
        for tensor in control.values()
    )

    cost_summary = build_scaffold_cost_summary(
        model=global_model,
        server_control=server_control,
        num_clients=len(train_loaders),
        seed=seed,
    )

    validation_frames: list[pd.DataFrame] = []
    round_zero_validation = evaluate_clients(
        model=global_model,
        loaders=validation_loaders,
        device=device,
    )
    round_zero_validation.insert(0, "round", 0)
    round_zero_validation.insert(0, "alpha", config.alpha)
    round_zero_validation.insert(0, "seed", seed)
    round_zero_validation.insert(0, "method", "scaffold")
    validation_frames.append(round_zero_validation)

    client_diagnostic_rows: list[dict] = []
    pairwise_diagnostic_rows: list[dict] = []
    round_diagnostic_rows: list[dict] = []
    training_runtime_seconds = 0.0

    for round_idx in range(1, config.num_rounds + 1):
        global_model_before = copy.deepcopy(global_model)
        server_control_before = clone_control_state(server_control)
        client_controls_before = {
            client_id: clone_control_state(control)
            for client_id, control in client_controls.items()
        }

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        round_start = time.perf_counter()

        round_result = run_scaffold_round(
            global_model=global_model,
            client_loaders=train_loaders,
            server_control=server_control,
            client_controls=client_controls,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            loss_fn=loss_fn,
            device=device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        round_runtime_seconds = time.perf_counter() - round_start
        training_runtime_seconds += round_runtime_seconds

        new_client_rows, new_pairwise_rows, new_round_rows = (
            build_scaffold_round_diagnostics(
                global_model_before=global_model_before,
                server_control_before=server_control_before,
                client_controls_before=client_controls_before,
                round_result=round_result,
                round_idx=round_idx,
                seed=seed,
                alpha=config.alpha,
                local_epochs=config.local_epochs,
            )
        )

        if len(new_round_rows) != 1:
            raise RuntimeError(
                "Expected exactly one round diagnostic row."
            )
        new_round_rows[0]["runtime_seconds"] = round_runtime_seconds

        client_diagnostic_rows.extend(new_client_rows)
        pairwise_diagnostic_rows.extend(new_pairwise_rows)
        round_diagnostic_rows.extend(new_round_rows)

        global_model = round_result.global_model
        server_control = round_result.server_control
        client_controls = round_result.client_controls

        round_validation = evaluate_clients(
            model=global_model,
            loaders=validation_loaders,
            device=device,
        )
        round_validation.insert(0, "round", round_idx)
        round_validation.insert(0, "alpha", config.alpha)
        round_validation.insert(0, "seed", seed)
        round_validation.insert(0, "method", "scaffold")
        validation_frames.append(round_validation)

    client_diagnostics = pd.DataFrame(client_diagnostic_rows)
    pairwise_diagnostics = pd.DataFrame(pairwise_diagnostic_rows)
    round_diagnostics = pd.DataFrame(round_diagnostic_rows)
    validation_history = pd.concat(
        validation_frames,
        ignore_index=True,
    )

    performance_rows: list[dict] = []
    for round_idx, round_df in validation_history.groupby(
        "round",
        sort=True,
    ):
        metrics = summarize_client_evaluations(round_df)
        performance_rows.append({
            "method": "scaffold",
            "seed": seed,
            "alpha": config.alpha,
            "round": int(round_idx),
            **metrics,
        })

    performance_history = pd.DataFrame(performance_rows)

    assert len(performance_history) == config.num_rounds + 1
    assert set(performance_history["round"]) == set(
        range(config.num_rounds + 1)
    )
    for column in (
        "weighted_accuracy",
        "mean_client_accuracy",
        "worst_client_accuracy",
        "std_client_accuracy",
    ):
        assert performance_history[column].between(0, 1).all()

    return ScaffoldExperimentResult(
        final_model=global_model,
        final_server_control=server_control,
        final_client_controls=client_controls,
        validation_history=validation_history,
        performance_history=performance_history,
        client_diagnostics=client_diagnostics,
        pairwise_diagnostics=pairwise_diagnostics,
        round_diagnostics=round_diagnostics,
        cost_summary=cost_summary,
        training_runtime_seconds=training_runtime_seconds,
    )
