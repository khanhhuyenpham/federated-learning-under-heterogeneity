from dataclasses import dataclass

from .benchmark_suite import AlgorithmSpec
from .heterogeneity import HeterogeneityConfig


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    alpha: float | None
    batch_size: int = 64
    num_rounds: int = 20
    local_epochs: int = 1
    learning_rate: float = 0.01

FEDPROX_PILOT_SEEDS = (
    42,
    43,
    44,
)

FEDPROX_PILOT_MU_VALUES = (
    0.01,
    0.1,
    1.0,
)

def build_fedprox_pilot_conditions(
) -> list[HeterogeneityConfig]:
    return [
        HeterogeneityConfig(
            mode="label_skew",
            num_clients=5,
            seed=seed,
            alpha=0.1,
            min_samples_per_client=1,
            max_abs_angle=0.0,
        )
        for seed in FEDPROX_PILOT_SEEDS
    ]

def build_fedprox_pilot_algorithms(
) -> list[AlgorithmSpec]:
    return [
        AlgorithmSpec(
            algorithm="fedprox",
            mu=mu,
        )
        for mu in FEDPROX_PILOT_MU_VALUES
    ]

def make_training_config(
    condition_config: HeterogeneityConfig,
) -> TrainingConfig:
    return TrainingConfig(
        seed=condition_config.seed,
        alpha=condition_config.alpha,
        batch_size=64,
        num_rounds=20,
        local_epochs=1,
        learning_rate=0.01,
    )