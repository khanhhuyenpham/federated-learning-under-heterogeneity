from src.benchmark_protocol import (
    FEDPROX_PILOT_MU_VALUES,
    FEDPROX_PILOT_SEEDS,
    TrainingConfig,
    build_fedprox_pilot_algorithms,
    build_fedprox_pilot_conditions,
    make_training_config,
)
from src.benchmark_storage import make_run_key


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
        condition.alpha == 0.1
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
    ] == [0.01, 0.1, 1.0]


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


def test_training_config_matches_condition():
    condition = (
        build_fedprox_pilot_conditions()[0]
    )

    config = make_training_config(condition)

    assert isinstance(config, TrainingConfig)
    assert config.seed == condition.seed
    assert config.alpha == condition.alpha
    assert config.batch_size == 64
    assert config.num_rounds == 20
    assert config.local_epochs == 1
    assert config.learning_rate == 0.01