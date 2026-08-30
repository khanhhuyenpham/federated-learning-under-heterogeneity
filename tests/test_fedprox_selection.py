import pandas as pd
import pytest

from src.fedprox_selection import (
    FedProxSelectionResult,
    select_fedprox_mu,
)


EXPECTED_MU_VALUES = (
    0.01,
    0.1,
    1.0,
)

EXPECTED_SEEDS = (
    42,
    43,
    44,
)


def make_validation_history(
    *,
    tail_scores=None,
) -> pd.DataFrame:
    if tail_scores is None:
        tail_scores = {
            0.01: {
                42: 0.80,
                43: 0.81,
                44: 0.79,
            },
            0.1: {
                42: 0.90,
                43: 0.91,
                44: 0.89,
            },
            1.0: {
                42: 0.84,
                43: 0.85,
                44: 0.83,
            },
        }

    rows = []

    for mu in EXPECTED_MU_VALUES:
        for seed in EXPECTED_SEEDS:
            for round_idx in range(21):
                if round_idx >= 16:
                    accuracy = (
                        tail_scores[mu][seed]
                    )
                else:
                    accuracy = 0.5

                rows.append({
                    "algorithm": "fedprox",
                    "mu": mu,
                    "seed": seed,
                    "round": round_idx,
                    "weighted_accuracy": accuracy,
                })

    return pd.DataFrame(rows)


def run_selection(
    history,
    *,
    tie_tolerance=0.0025,
):
    return select_fedprox_mu(
        history,
        expected_mu_values=(
            EXPECTED_MU_VALUES
        ),
        expected_seeds=EXPECTED_SEEDS,
        expected_num_rounds=20,
        tail_round_count=5,
        tie_tolerance=tie_tolerance,
    )


def test_selects_clearly_best_mean_accuracy():
    history = make_validation_history()

    result = run_selection(history)

    assert isinstance(
        result,
        FedProxSelectionResult,
    )

    assert result.selected_mu == pytest.approx(
        0.1
    )

    assert len(result.scores_by_mu) == 3
    assert len(result.scores_by_seed) == 9

    selected_rows = result.scores_by_mu[
        result.scores_by_mu["selected"]
    ]

    assert len(selected_rows) == 1

    assert selected_rows.iloc[0][
        "mu"
    ] == pytest.approx(0.1)


def test_candidate_within_tolerance_wins_if_more_stable():
    tail_scores = {
        0.01: {
            42: 0.898,
            43: 0.898,
            44: 0.898,
        },
        0.1: {
            42: 0.88,
            43: 0.90,
            44: 0.92,
        },
        1.0: {
            42: 0.80,
            43: 0.80,
            44: 0.80,
        },
    }

    result = run_selection(
        make_validation_history(
            tail_scores=tail_scores
        )
    )

    # mu=0.01 is 0.002 below the best mean,
    # so it is eligible and wins by stability.
    assert result.selected_mu == pytest.approx(
        0.01
    )


def test_candidate_outside_tolerance_cannot_win():
    tail_scores = {
        0.01: {
            42: 0.897,
            43: 0.897,
            44: 0.897,
        },
        0.1: {
            42: 0.88,
            43: 0.90,
            44: 0.92,
        },
        1.0: {
            42: 0.80,
            43: 0.80,
            44: 0.80,
        },
    }

    result = run_selection(
        make_validation_history(
            tail_scores=tail_scores
        )
    )

    # The gap is 0.003, which is greater
    # than the 0.0025 tolerance.
    assert result.selected_mu == pytest.approx(
        0.1
    )


def test_exact_tie_selects_smaller_mu():
    tail_scores = {
        0.01: {
            42: 0.90,
            43: 0.90,
            44: 0.90,
        },
        0.1: {
            42: 0.90,
            43: 0.90,
            44: 0.90,
        },
        1.0: {
            42: 0.80,
            43: 0.80,
            44: 0.80,
        },
    }

    result = run_selection(
        make_validation_history(
            tail_scores=tail_scores
        )
    )

    assert result.selected_mu == pytest.approx(
        0.01
    )


def test_only_final_five_rounds_affect_selection():
    tail_scores = {
        0.01: {
            42: 0.80,
            43: 0.80,
            44: 0.80,
        },
        0.1: {
            42: 0.90,
            43: 0.90,
            44: 0.90,
        },
        1.0: {
            42: 0.70,
            43: 0.70,
            44: 0.70,
        },
    }

    history = make_validation_history(
        tail_scores=tail_scores
    )

    history.loc[
        (
            (history["mu"] == 0.01)
            & (history["round"] < 16)
        ),
        "weighted_accuracy",
    ] = 1.0

    history.loc[
        (
            (history["mu"] == 0.1)
            & (history["round"] < 16)
        ),
        "weighted_accuracy",
    ] = 0.0

    result = run_selection(history)

    assert result.selected_mu == pytest.approx(
        0.1
    )


def test_non_fedprox_rows_are_ignored():
    history = make_validation_history()

    extra_rows = history[
        history["mu"] == 0.01
    ].copy()

    extra_rows["algorithm"] = "fedavg"
    extra_rows["mu"] = 0.0
    extra_rows["weighted_accuracy"] = 1.0

    combined = pd.concat(
        [history, extra_rows],
        ignore_index=True,
    )

    result = run_selection(combined)

    assert result.selected_mu == pytest.approx(
        0.1
    )


def test_missing_mu_is_rejected():
    history = make_validation_history()

    history = history[
        history["mu"] != 1.0
    ]

    with pytest.raises(
        ValueError,
        match="mu values",
    ):
        run_selection(history)


def test_missing_seed_is_rejected():
    history = make_validation_history()

    history = history[
        history["seed"] != 44
    ]

    with pytest.raises(
        ValueError,
        match="seeds",
    ):
        run_selection(history)


def test_missing_round_is_rejected():
    history = make_validation_history()

    history = history[
        ~(
            (history["mu"] == 0.1)
            & (history["seed"] == 42)
            & (history["round"] == 17)
        )
    ]

    with pytest.raises(
        ValueError,
        match="invalid round set",
    ):
        run_selection(history)


def test_duplicate_mu_seed_round_is_rejected():
    history = make_validation_history()

    duplicated_row = history.iloc[
        [0]
    ].copy()

    history = pd.concat(
        [history, duplicated_row],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        run_selection(history)


@pytest.mark.parametrize(
    "invalid_accuracy",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_accuracy_is_rejected(
    invalid_accuracy,
):
    history = make_validation_history()

    history.loc[
        history.index[0],
        "weighted_accuracy",
    ] = invalid_accuracy

    with pytest.raises(ValueError):
        run_selection(history)


def test_missing_required_column_is_rejected():
    history = make_validation_history().drop(
        columns=["weighted_accuracy"]
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        run_selection(history)


def test_output_seed_scores_match_tail_means():
    history = make_validation_history()

    result = run_selection(history)

    score = result.scores_by_seed.loc[
        (
            (result.scores_by_seed["mu"] == 0.1)
            & (
                result.scores_by_seed["seed"]
                == 42
            )
        ),
        "tail_validation_accuracy",
    ].iloc[0]

    assert score == pytest.approx(0.90)


def test_output_contains_expected_seed_counts():
    result = run_selection(
        make_validation_history()
    )

    assert result.scores_by_mu[
        "num_seeds"
    ].tolist() == [3, 3, 3]


def test_rejects_invalid_tail_round_count():
    history = make_validation_history()

    with pytest.raises(
        ValueError,
        match="tail_round_count",
    ):
        select_fedprox_mu(
            history,
            expected_mu_values=(
                EXPECTED_MU_VALUES
            ),
            expected_seeds=EXPECTED_SEEDS,
            expected_num_rounds=20,
            tail_round_count=21,
        )


def test_rejects_negative_tie_tolerance():
    history = make_validation_history()

    with pytest.raises(
        ValueError,
        match="tie_tolerance",
    ):
        select_fedprox_mu(
            history,
            expected_mu_values=(
                EXPECTED_MU_VALUES
            ),
            expected_seeds=EXPECTED_SEEDS,
            expected_num_rounds=20,
            tail_round_count=5,
            tie_tolerance=-0.1,
        )