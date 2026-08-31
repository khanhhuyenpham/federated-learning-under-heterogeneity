from collections import Counter

import pytest

from src.benchmark_protocol import (
    FEDPROX_PILOT_MU_VALUES,
    FEDPROX_PILOT_SEEDS,
    MAIN_BENCHMARK_SEEDS,
    SELECTED_FEDPROX_MU,
    TrainingConfig,
    build_fedprox_pilot_algorithms,
    build_fedprox_pilot_conditions,
    build_main_benchmark_algorithms,
    build_main_benchmark_conditions,
    make_training_config,
)
from src.benchmark_storage import make_run_key


def condition_signature(condition):
    return (
        condition.mode,
        condition.alpha,
        condition.min_samples_per_client,
        condition.max_abs_angle,
    )


def algorithm_signature(specification):
    return (
        specification.algorithm,
        specification.mu,
    )


def test_pilot_uses_expected_seeds():
    assert FEDPROX_PILOT_SEEDS == (
        42,
        43,
        44,
    )


def test_pilot_uses_expected_mu_values():
    assert FEDPROX_PILOT_MU_VALUES == (
        0.01,
        0.1,
        1.0,
    )


def test_pilot_builds_three_conditions():
    conditions = (
        build_fedprox_pilot_conditions()
    )

    assert len(conditions) == 3

    assert [
        condition.seed
        for condition in conditions
    ] == [42, 43, 44]

    assert all(
        condition.mode == "label_skew"
        for condition in conditions
    )

    assert all(
        condition.num_clients == 5
        for condition in conditions
    )

    assert all(
        condition.alpha == pytest.approx(0.1)
        for condition in conditions
    )


def test_pilot_builds_three_fedprox_specs():
    specifications = (
        build_fedprox_pilot_algorithms()
    )

    assert len(specifications) == 3

    assert all(
        specification.algorithm == "fedprox"
        for specification in specifications
    )

    assert [
        specification.mu
        for specification in specifications
    ] == pytest.approx([
        0.01,
        0.1,
        1.0,
    ])


def test_pilot_produces_nine_unique_run_keys():
    conditions = (
        build_fedprox_pilot_conditions()
    )

    specifications = (
        build_fedprox_pilot_algorithms()
    )

    keys = {
        make_run_key(
            condition_config=condition,
            algorithm_spec=specification,
        )
        for condition in conditions
        for specification in specifications
    }

    assert len(keys) == 9


def test_selected_fedprox_mu_is_locked():
    assert SELECTED_FEDPROX_MU == pytest.approx(
        0.01
    )


def test_main_benchmark_uses_expected_seeds():
    assert MAIN_BENCHMARK_SEEDS == (
        42,
        43,
        44,
    )


def test_main_benchmark_builds_21_conditions():
    conditions = (
        build_main_benchmark_conditions()
    )

    assert len(conditions) == 21


def test_each_seed_has_seven_conditions():
    conditions = (
        build_main_benchmark_conditions()
    )

    counts_by_seed = Counter(
        condition.seed
        for condition in conditions
    )

    assert counts_by_seed == {
        42: 7,
        43: 7,
        44: 7,
    }


def test_each_seed_has_exact_condition_matrix():
    conditions = (
        build_main_benchmark_conditions()
    )

    expected_signatures = {
        (
            "iid",
            None,
            1,
            0.0,
        ),
        (
            "label_skew",
            0.5,
            1,
            0.0,
        ),
        (
            "label_skew",
            0.1,
            1,
            0.0,
        ),
        (
            "quantity_skew",
            0.5,
            100,
            0.0,
        ),
        (
            "quantity_skew",
            0.1,
            100,
            0.0,
        ),
        (
            "rotation_shift",
            None,
            1,
            10.0,
        ),
        (
            "rotation_shift",
            None,
            1,
            20.0,
        ),
    }

    for seed in MAIN_BENCHMARK_SEEDS:
        seed_conditions = [
            condition
            for condition in conditions
            if condition.seed == seed
        ]

        observed_signatures = {
            condition_signature(condition)
            for condition in seed_conditions
        }

        assert (
            observed_signatures
            == expected_signatures
        )


def test_all_main_conditions_use_five_clients():
    conditions = (
        build_main_benchmark_conditions()
    )

    assert all(
        condition.num_clients == 5
        for condition in conditions
    )


def test_quantity_skew_reserves_100_samples():
    conditions = (
        build_main_benchmark_conditions()
    )

    quantity_conditions = [
        condition
        for condition in conditions
        if condition.mode == "quantity_skew"
    ]

    assert len(quantity_conditions) == 6

    assert all(
        condition.min_samples_per_client
        == 100
        for condition in quantity_conditions
    )


def test_label_skew_uses_two_alpha_values():
    conditions = (
        build_main_benchmark_conditions()
    )

    observed_alpha_values = {
        condition.alpha
        for condition in conditions
        if condition.mode == "label_skew"
    }

    assert observed_alpha_values == {
        0.1,
        0.5,
    }


def test_quantity_skew_uses_two_alpha_values():
    conditions = (
        build_main_benchmark_conditions()
    )

    observed_alpha_values = {
        condition.alpha
        for condition in conditions
        if condition.mode == "quantity_skew"
    }

    assert observed_alpha_values == {
        0.1,
        0.5,
    }


def test_rotation_shift_uses_two_angle_values():
    conditions = (
        build_main_benchmark_conditions()
    )

    observed_angles = {
        condition.max_abs_angle
        for condition in conditions
        if condition.mode == "rotation_shift"
    }

    assert observed_angles == {
        10.0,
        20.0,
    }


def test_main_benchmark_builds_three_algorithms():
    specifications = (
        build_main_benchmark_algorithms()
    )

    assert [
        algorithm_signature(specification)
        for specification in specifications
    ] == [
        ("fedavg", 0.0),
        ("fedprox", 0.01),
        ("scaffold", None),
    ]


def test_main_fedprox_uses_locked_mu():
    specifications = (
        build_main_benchmark_algorithms()
    )

    fedprox_specifications = [
        specification
        for specification in specifications
        if specification.algorithm
        == "fedprox"
    ]

    assert len(fedprox_specifications) == 1

    assert fedprox_specifications[
        0
    ].mu == pytest.approx(
        SELECTED_FEDPROX_MU
    )


def test_main_benchmark_has_63_unique_run_keys():
    conditions = (
        build_main_benchmark_conditions()
    )

    specifications = (
        build_main_benchmark_algorithms()
    )

    keys = {
        make_run_key(
            condition_config=condition,
            algorithm_spec=specification,
        )
        for condition in conditions
        for specification in specifications
    }

    assert len(keys) == 63


def test_main_condition_configs_are_unique():
    conditions = (
        build_main_benchmark_conditions()
    )

    identities = {
        (
            condition.seed,
            condition_signature(condition),
        )
        for condition in conditions
    }

    assert len(identities) == 21


def test_training_config_matches_condition():
    condition = (
        build_main_benchmark_conditions()[0]
    )

    config = make_training_config(condition)

    assert isinstance(config, TrainingConfig)
    assert config.seed == condition.seed
    assert config.alpha == condition.alpha
    assert config.batch_size == 64
    assert config.num_rounds == 20
    assert config.local_epochs == 1
    assert config.learning_rate == pytest.approx(
        0.01
    )


def test_every_main_condition_produces_matching_training_config():
    conditions = (
        build_main_benchmark_conditions()
    )

    for condition in conditions:
        config = make_training_config(
            condition
        )

        assert config.seed == condition.seed
        assert config.alpha == condition.alpha
        assert config.batch_size == 64
        assert config.num_rounds == 20
        assert config.local_epochs == 1
        assert (
            config.learning_rate
            == pytest.approx(0.01)
        )