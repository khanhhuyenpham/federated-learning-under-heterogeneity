
import copy
import torch
import torch.nn as nn
import pandas as pd

from .aggregate import aggregate

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        batch_size = yb.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            batch_size = yb.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples

def client_update(
    global_model: nn.Module,
    client_loader,
    local_epochs: int,
    learning_rate: float,
    loss_fn: nn.Module,
    device,
):
    local_model = copy.deepcopy(global_model).to(device)
    optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)

    for _ in range(local_epochs):
        train_one_epoch(local_model, client_loader, optimizer, loss_fn, device)

    return local_model

def federated_train(
    global_model,
    client_loaders,
    num_rounds,
    local_epochs,
    learning_rate,
    loss_fn,
    test_loader,
    device,
    verbose=False,
):
    history = []

    test_loss, test_acc = evaluate(global_model, test_loader, loss_fn, device)
    history.append({
        "round": 0,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    })

    for t in range(num_rounds):
        local_models = []
        client_sizes = []

        # C = 1.0: every client participates in every round.
        for client_loader in client_loaders.values():
            local_model = client_update(
                global_model,
                client_loader,
                local_epochs,
                learning_rate,
                loss_fn,
                device,
            )
            local_models.append(local_model)
            client_sizes.append(len(client_loader.dataset))

        global_model = aggregate(global_model, local_models, client_sizes)

        test_loss, test_acc = evaluate(global_model, test_loader, loss_fn, device)
        history.append({
            "round": t + 1,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        })

        if verbose:
            print(
                f"Round {t + 1:2d}/{num_rounds} | "
                f"loss={test_loss:.4f} | acc={test_acc:.4f}"
            )

    return global_model, pd.DataFrame(history)
