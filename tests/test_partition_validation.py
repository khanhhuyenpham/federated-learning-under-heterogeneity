import pytest

from src.data import validate_partition


def test_validate_partition_accepts_valid_partition():
    partition = {
        0: [0, 2],
        1: [1, 3],
    }

    result = validate_partition(
        client_indices=partition,
        dataset_size=4,
        expected_num_clients=2,
    )

    assert result is None


def test_validate_partition_rejects_incorrect_client_ids():
    partition = {
        0: [0, 2],
        2: [1, 3],
    }

    with pytest.raises(ValueError, match="Expected client IDs"):
        validate_partition(
            client_indices=partition,
            dataset_size=4,
            expected_num_clients=2,
        )


def test_validate_partition_rejects_zero_expected_clients():
    partition = {
        0: [0, 1, 2, 3],
    }

    with pytest.raises(ValueError, match="positive"):
        validate_partition(
            client_indices=partition,
            dataset_size=4,
            expected_num_clients=0,
        )

def test_validate_partition_rejects_duplicate_indices():
    partition = {
        0: [0, 1],
        1: [1, 2, 3],
    }

    with pytest.raises(ValueError, match="duplicate"):
        validate_partition(
            client_indices=partition,
            dataset_size=4,
            expected_num_clients=2,
        )


@pytest.mark.parametrize("invalid_index", [-1, 4])
def test_validate_partition_rejects_out_of_range_index(invalid_index):
    partition = {
        0: [0, 1],
        1: [2, invalid_index],
    }

    with pytest.raises(ValueError, match="out of range"):
        validate_partition(
            client_indices=partition,
            dataset_size=4,
            expected_num_clients=2,
            require_full_coverage=False,
        )

def test_validate_partition_rejects_missing_coverage():
    partition = {
        0: [0],
        1: [1, 2],
    }

    with pytest.raises(ValueError, match="full.coverage"):
        validate_partition(
            client_indices=partition,
            dataset_size=4,
            expected_num_clients=2,
            require_full_coverage=True,
        )

def test_validate_partition_allows_partial_coverage_when_disabled():
    partition = {
        0: [0],
        1: [2],
    }

    result = validate_partition(
        client_indices=partition,
        dataset_size=4,
        expected_num_clients=2,
        require_full_coverage=False,
    )

    assert result is None

def test_validate_partition_rejects_empty_client():
    partition = {
        0: [0, 1, 2, 3],
        1: [],
    }

    with pytest.raises(ValueError, match="empty client"):
        validate_partition(
            client_indices=partition,
            dataset_size=4,
            expected_num_clients=2,
            require_non_empty_clients=True,
        )
    
def test_validate_partition_allows_empty_client_when_disabled():
    partition = {
        0: [0, 1, 2, 3],
        1: [],
    }

    result = validate_partition(
        client_indices=partition,
        dataset_size=4,
        expected_num_clients=2,
        require_non_empty_clients=False,
    )

    assert result is None