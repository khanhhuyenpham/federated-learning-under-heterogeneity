from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FedProxSelectionResult:
    selected_mu: float
    scores_by_mu: pd.DataFrame
    scores_by_seed: pd.DataFrame


def _validate_selection_parameters(
    *,
    expected_mu_values: Sequence[float],
    expected_seeds: Sequence[int],
    expected_num_rounds: int,
    tail_round_count: int,
    tie_tolerance: float,
) -> None:
    if not expected_mu_values:
        raise ValueError(
            "expected_mu_values cannot be empty"
        )

    if not expected_seeds:
        raise ValueError(
            "expected_seeds cannot be empty"
        )

    normalized_mu_values = [
        float(mu)
        for mu in expected_mu_values
    ]

    if len(set(normalized_mu_values)) != len(
        normalized_mu_values
    ):
        raise ValueError(
            "expected_mu_values must be unique"
        )

    if any(
        not np.isfinite(mu) or mu <= 0.0
        for mu in normalized_mu_values
    ):
        raise ValueError(
            "Expected FedProx mu values must "
            "be finite and positive"
        )

    normalized_seeds = [
        int(seed)
        for seed in expected_seeds
    ]

    if len(set(normalized_seeds)) != len(
        normalized_seeds
    ):
        raise ValueError(
            "expected_seeds must be unique"
        )

    if expected_num_rounds <= 0:
        raise ValueError(
            "expected_num_rounds must "
            "be positive"
        )

    if tail_round_count <= 0:
        raise ValueError(
            "tail_round_count must be positive"
        )

    if tail_round_count > expected_num_rounds:
        raise ValueError(
            "tail_round_count cannot exceed "
            "expected_num_rounds"
        )

    if (
        not np.isfinite(tie_tolerance)
        or tie_tolerance < 0.0
    ):
        raise ValueError(
            "tie_tolerance must be finite "
            "and nonnegative"
        )


def _prepare_fedprox_history(
    validation_by_round: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "algorithm",
        "mu",
        "seed",
        "round",
        "weighted_accuracy",
    }

    missing_columns = (
        required_columns
        - set(validation_by_round.columns)
    )

    if missing_columns:
        raise ValueError(
            "Validation history is missing "
            "required columns: "
            f"{sorted(missing_columns)}"
        )

    history = validation_by_round[
        validation_by_round["algorithm"]
        == "fedprox"
    ].copy()

    if history.empty:
        raise ValueError(
            "Validation history contains no "
            "FedProx rows"
        )

    for column in (
        "mu",
        "seed",
        "round",
        "weighted_accuracy",
    ):
        try:
            history[column] = pd.to_numeric(
                history[column],
                errors="raise",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "FedProx validation column must "
                f"be numeric: {column}"
            ) from error

    numeric_values = history[
        [
            "mu",
            "seed",
            "round",
            "weighted_accuracy",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "FedProx validation history contains "
            "non-finite numeric values"
        )

    if not np.equal(
        history["seed"],
        np.floor(history["seed"]),
    ).all():
        raise ValueError(
            "FedProx validation seeds must "
            "be integers"
        )

    if not np.equal(
        history["round"],
        np.floor(history["round"]),
    ).all():
        raise ValueError(
            "FedProx validation rounds must "
            "be integers"
        )

    history["seed"] = history[
        "seed"
    ].astype(int)

    history["round"] = history[
        "round"
    ].astype(int)

    history["mu"] = history[
        "mu"
    ].astype(float)

    history["weighted_accuracy"] = history[
        "weighted_accuracy"
    ].astype(float)

    if (
        history["mu"] <= 0.0
    ).any():
        raise ValueError(
            "FedProx mu values must be positive"
        )

    if not history[
        "weighted_accuracy"
    ].between(0.0, 1.0).all():
        raise ValueError(
            "Weighted validation accuracy must "
            "be between 0 and 1"
        )

    duplicated_rows = history.duplicated(
        subset=[
            "mu",
            "seed",
            "round",
        ],
        keep=False,
    )

    if duplicated_rows.any():
        raise ValueError(
            "FedProx validation history contains "
            "duplicate mu-seed-round rows"
        )

    return history


def _validate_expected_experiment_grid(
    *,
    history: pd.DataFrame,
    expected_mu_values: Sequence[float],
    expected_seeds: Sequence[int],
    expected_num_rounds: int,
) -> None:
    expected_mu_set = {
        float(mu)
        for mu in expected_mu_values
    }

    expected_seed_set = {
        int(seed)
        for seed in expected_seeds
    }

    observed_mu_set = set(
        history["mu"].unique()
    )

    observed_seed_set = set(
        history["seed"].unique()
    )

    if observed_mu_set != expected_mu_set:
        raise ValueError(
            "Observed FedProx mu values do not "
            "match expected_mu_values. "
            f"Observed: {observed_mu_set}; "
            f"expected: {expected_mu_set}"
        )

    if observed_seed_set != expected_seed_set:
        raise ValueError(
            "Observed FedProx seeds do not "
            "match expected_seeds. "
            f"Observed: {observed_seed_set}; "
            f"expected: {expected_seed_set}"
        )

    expected_pairs = {
        (mu, seed)
        for mu in expected_mu_set
        for seed in expected_seed_set
    }

    observed_pairs = set(
        history[
            ["mu", "seed"]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    if observed_pairs != expected_pairs:
        raise ValueError(
            "FedProx validation history does not "
            "contain every expected mu-seed pair"
        )

    expected_rounds = set(
        range(expected_num_rounds + 1)
    )

    for (
        mu,
        seed,
    ), group in history.groupby(
        ["mu", "seed"],
        sort=True,
    ):
        observed_rounds = set(group["round"])

        if observed_rounds != expected_rounds:
            missing_rounds = (
                expected_rounds
                - observed_rounds
            )

            unexpected_rounds = (
                observed_rounds
                - expected_rounds
            )

            raise ValueError(
                "FedProx validation history has "
                "an invalid round set for "
                f"mu={mu}, seed={seed}. "
                f"Missing rounds: "
                f"{sorted(missing_rounds)}; "
                f"unexpected rounds: "
                f"{sorted(unexpected_rounds)}"
            )


def select_fedprox_mu(
    validation_by_round: pd.DataFrame,
    *,
    expected_mu_values: Sequence[float],
    expected_seeds: Sequence[int],
    expected_num_rounds: int = 20,
    tail_round_count: int = 5,
    tie_tolerance: float = 0.0025,
) -> FedProxSelectionResult:
    _validate_selection_parameters(
        expected_mu_values=expected_mu_values,
        expected_seeds=expected_seeds,
        expected_num_rounds=(
            expected_num_rounds
        ),
        tail_round_count=tail_round_count,
        tie_tolerance=tie_tolerance,
    )

    history = _prepare_fedprox_history(
        validation_by_round
    )

    _validate_expected_experiment_grid(
        history=history,
        expected_mu_values=(
            expected_mu_values
        ),
        expected_seeds=expected_seeds,
        expected_num_rounds=(
            expected_num_rounds
        ),
    )

    first_tail_round = (
        expected_num_rounds
        - tail_round_count
        + 1
    )

    tail_rounds = set(
        range(
            first_tail_round,
            expected_num_rounds + 1,
        )
    )

    tail_history = history[
        history["round"].isin(tail_rounds)
    ]

    scores_by_seed = (
        tail_history
        .groupby(
            ["mu", "seed"],
            as_index=False,
        )
        .agg(
            tail_validation_accuracy=(
                "weighted_accuracy",
                "mean",
            )
        )
        .sort_values(
            ["mu", "seed"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    scores_by_mu = (
        scores_by_seed
        .groupby(
            "mu",
            as_index=False,
        )
        .agg(
            mean_validation_accuracy=(
                "tail_validation_accuracy",
                "mean",
            ),
            std_validation_accuracy=(
                "tail_validation_accuracy",
                "std",
            ),
            num_seeds=(
                "seed",
                "nunique",
            ),
        )
        .sort_values(
            "mu",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    # Pandas returns NaN for sample standard
    # deviation when only one seed is present.
    scores_by_mu[
        "std_validation_accuracy"
    ] = scores_by_mu[
        "std_validation_accuracy"
    ].fillna(0.0)

    best_mean = float(
        scores_by_mu[
            "mean_validation_accuracy"
        ].max()
    )

    mean_gap = (
        best_mean
        - scores_by_mu[
            "mean_validation_accuracy"
        ]
    )

    eligible = scores_by_mu[
        mean_gap
        <= tie_tolerance + 1e-12
    ]

    selected_row = (
        eligible
        .sort_values(
            [
                "std_validation_accuracy",
                "mu",
            ],
            ascending=[True, True],
            kind="mergesort",
        )
        .iloc[0]
    )

    selected_mu = float(
        selected_row["mu"]
    )

    scores_by_mu["selected"] = (
        scores_by_mu["mu"]
        == selected_mu
    )

    scores_by_mu = scores_by_mu[
        [
            "mu",
            "mean_validation_accuracy",
            "std_validation_accuracy",
            "num_seeds",
            "selected",
        ]
    ]

    scores_by_seed = scores_by_seed[
        [
            "mu",
            "seed",
            "tail_validation_accuracy",
        ]
    ]

    return FedProxSelectionResult(
        selected_mu=selected_mu,
        scores_by_mu=scores_by_mu,
        scores_by_seed=scores_by_seed,
    )