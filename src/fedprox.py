import pandas as pd
import torch
import torch.nn as nn
import copy

from .aggregate import aggregate
from .training import evaluate


def train_one_epoch_fedprox(
    model: nn.Module,
    global_params,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    mu: float,
) -> float:
    """Train one local epoch with the student FedProx local objective.

    Inputs:
        model: Local model being optimized on one client.
        global_model: Fixed round-start global model reference.
        loader: Client DataLoader.
        optimizer: Optimizer for the local model.
        loss_fn: Supervised data loss, such as CrossEntropyLoss.
        device: Device used for tensors and models.
        mu: FedProx regularization strength.

    Output:
        Average scalar training objective/loss over examples in ``loader``.
    """
    model.train()
    total_loss = 0.0
    total_examples = 0
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
    
        dist = torch.tensor(0.0, device=device)
        for local_param, global_param in zip(model.parameters(), global_params):
            dist = dist + ((local_param - global_param) ** 2).sum()

        loss = loss_fn(logits, yb) + dist * mu / 2
        loss.backward()
        optimizer.step()
    
        batch_size = yb.size(0)
        total_loss = total_loss + loss.item() * batch_size
        total_examples += batch_size
    
    return total_loss / total_examples


def client_update_fedprox(
    global_model: nn.Module,
    client_loader,
    local_epochs: int,
    learning_rate: float,
    loss_fn: nn.Module,
    device: torch.device,
    mu: float,
) -> nn.Module:
    """Run one FedProx client update from a round-start global model.

    Inputs:
        global_model: Current server model at the start of the round.
        client_loader: DataLoader for one client's local dataset.
        local_epochs: Number of local epochs to run.
        learning_rate: SGD learning rate.
        loss_fn: Supervised data loss, such as CrossEntropyLoss.
        device: Device used for tensors and models.
        mu: FedProx regularization strength.

    Output:
        A local model ready for server aggregation.
    """
    local_model = copy.deepcopy(global_model).to(device)
    
    global_params = [
        p.detach().clone().to(device)
        for p in global_model.parameters()
    ]
    
    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=learning_rate
    )
    
    for _ in range(local_epochs):
        train_one_epoch_fedprox(
            local_model,
            global_params,
            client_loader,
            optimizer,
            loss_fn,
            device,
            mu,
        )
    
    return local_model


def federated_train_fedprox(
    global_model: nn.Module,
    client_loaders,
    num_rounds: int,
    local_epochs: int,
    learning_rate: float,
    loss_fn: nn.Module,
    test_loader,
    device: torch.device,
    mu: float,
    verbose: bool = False,
):
    """Run the server-side FedProx training loop.

    The server orchestration mirrors FedAvg: every round collects local client
    models, aggregates them with the existing weighted aggregate() function,
    and records test metrics. The local FedProx update is intentionally
    delegated to client_update_fedprox().

    Returns:
        (trained_global_model, history_dataframe)
    """
    history = []

    test_loss, test_acc = evaluate(global_model, test_loader, loss_fn, device)
    history.append({
        "round": 0,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "mu": mu,
    })

    for t in range(num_rounds):
        local_models = []
        client_sizes = []

        # C = 1.0: every client participates in every round.
        for client_loader in client_loaders.values():
            local_model = client_update_fedprox(
                global_model=global_model,
                client_loader=client_loader,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                loss_fn=loss_fn,
                device=device,
                mu=mu,
            )
            local_models.append(local_model)
            client_sizes.append(len(client_loader.dataset))

        global_model = aggregate(global_model, local_models, client_sizes)

        test_loss, test_acc = evaluate(global_model, test_loader, loss_fn, device)
        history.append({
            "round": t + 1,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "mu": mu,
        })

        if verbose:
            print(
                f"Round {t + 1:2d}/{num_rounds} | "
                f"mu={mu:g} | loss={test_loss:.4f} | acc={test_acc:.4f}"
            )

    return global_model, pd.DataFrame(history)
