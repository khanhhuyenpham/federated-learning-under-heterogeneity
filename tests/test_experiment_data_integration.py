from types import SimpleNamespace

import pytest
import torch
from torch import nn

import src.experiment as experiment_module
from src.experiment import run_experiment


def test_run_experiment_rejects_mismatched_client_ids():
    config = SimpleNamespace(
        seed=42,
        batch_size=2,
    )

    with pytest.raises(
        ValueError,
        match="identical client IDs",
    ):
        run_experiment(
            config=config,
            mu_values=(0.0,),
            initial_model=nn.Linear(1, 1),
            train_ds=object(),
            client_train_indices={
                0: [0],
                1: [1],
            },
            validation_loaders={
                0: object(),
            },
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )


def test_run_experiment_forwards_client_transforms(
    monkeypatch,
):
    config = SimpleNamespace(
        seed=42,
        batch_size=2,
    )

    transforms = {
        0: lambda features: features,
    }

    class LoaderConstructionReached(Exception):
        pass

    def fake_create_client_loaders(**kwargs):
        assert kwargs["train_ds"] is train_dataset
        assert kwargs["client_indices"] == {0: [0]}
        assert kwargs["batch_size"] == 2
        assert kwargs["seed"] == 42
        assert kwargs["shuffle"] is True

        assert (
            kwargs["client_feature_transforms"]
            is transforms
        )

        # Stop after verifying the loader arguments.
        # We do not need to run federated training here.
        raise LoaderConstructionReached

    monkeypatch.setattr(
        experiment_module,
        "create_client_loaders",
        fake_create_client_loaders,
    )

    train_dataset = object()

    with pytest.raises(LoaderConstructionReached):
        run_experiment(
            config=config,
            mu_values=(0.0,),
            initial_model=nn.Linear(1, 1),
            train_ds=train_dataset,
            client_train_indices={
                0: [0],
            },
            validation_loaders={
                0: object(),
            },
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            client_feature_transforms=transforms,
        )