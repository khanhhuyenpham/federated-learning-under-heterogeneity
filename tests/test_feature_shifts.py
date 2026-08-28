import torch
import pytest

from src.feature_shifts import FixedRotation, build_client_rotation_transforms


def test_fixed_rotation_zero_degrees_preserves_features():
    features = torch.arange(
        9,
        dtype=torch.float32,
    ).reshape(1, 3, 3)

    transform = FixedRotation(angle=0)
    transformed = transform(features)

    assert torch.equal(transformed, features)

def test_fixed_rotation_preserves_shape():
    features = torch.zeros((1, 5, 5))
    features[0, 0, 2] = 1.0

    transform = FixedRotation(angle=90)
    transformed = transform(features)

    assert transformed.shape == features.shape

def test_fixed_rotation_changes_asymmetric_features():
    features = torch.zeros((1, 5, 5))
    features[0, 0, 2] = 1.0

    transform = FixedRotation(angle=90)
    transformed = transform(features)

    assert not torch.equal(transformed, features)

def test_fixed_rotation_is_deterministic():
    features = torch.zeros((1, 5, 5))
    features[0, 0, 2] = 1.0

    transform = FixedRotation(angle=30)

    first = transform(features)
    second = transform(features)

    assert torch.equal(first, second)

def test_build_client_rotation_transforms_assigns_expected_angles():
    transforms = build_client_rotation_transforms(
        num_clients=5,
        max_abs_angle=20,
    )

    assert set(transforms) == {0, 1, 2, 3, 4}

    angles = [
        transforms[client_id].angle
        for client_id in range(5)
    ]

    assert angles == pytest.approx(
        [-20, -10, 0, 10, 20]
    )

def test_build_client_rotation_transforms_returns_fixed_rotations():
    transforms = build_client_rotation_transforms(
        num_clients=3,
        max_abs_angle=15,
    )

    assert all(
        isinstance(transform, FixedRotation)
        for transform in transforms.values()
    )

def test_build_client_rotation_transforms_allows_zero_shift():
    transforms = build_client_rotation_transforms(
        num_clients=3,
        max_abs_angle=0,
    )

    assert all(
        transform.angle == 0
        for transform in transforms.values()
    )

@pytest.mark.parametrize("num_clients", [0, -1])
def test_build_client_rotation_transforms_rejects_invalid_client_count(
    num_clients,
):
    with pytest.raises(ValueError, match="num_clients must be positive"):
        build_client_rotation_transforms(
            num_clients=num_clients,
            max_abs_angle=20,
        )

def test_build_client_rotation_transforms_rejects_negative_angle():
    with pytest.raises(ValueError, match="max_abs_angle must be nonnegative"):
        build_client_rotation_transforms(
            num_clients=5,
            max_abs_angle=-20,
        )

def test_single_client_receives_zero_rotation():
    transforms = build_client_rotation_transforms(
        num_clients=1,
        max_abs_angle=20,
    )

    assert transforms[0].angle == 0.0