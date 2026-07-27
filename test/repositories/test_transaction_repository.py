from datetime import date

import pytest

from app.data.repositories.transaction_repository import TransactionRepository

START_2024 = date(2024, 1, 1)
END_2024 = date(2024, 12, 31)


def test_find_transactions_works_without_surface_filters(duckdb_path):
    repository = TransactionRepository(duckdb_path)

    result = repository.find_transactions(
        postal_code="75015",
        property_type="Appartement",
        start_date=START_2024,
        end_date=END_2024,
    )

    # M006 is excluded because prix_m2 is NULL.
    assert result.height == 3
    assert set(result["id_mutation"].to_list()) == {"M001", "M002"}


def test_find_transactions_applies_surface_filters(duckdb_path):
    repository = TransactionRepository(duckdb_path)

    result = repository.find_transactions(
        postal_code="75015",
        property_type="Appartement",
        start_date=START_2024,
        end_date=END_2024,
        min_surface=40,
        max_surface=50,
    )

    assert result.height == 1
    assert result.row(0, named=True)["id_mutation"] == "M001"
    assert result.row(0, named=True)["surface_reelle_bati"] == pytest.approx(45.0)


def test_get_transaction_by_id_returns_all_properties(duckdb_path):
    repository = TransactionRepository(duckdb_path)

    result = repository.get_transaction_by_id("M001")

    assert result.height == 2
    assert result["id_mutation"].unique().to_list() == ["M001"]


def test_get_transaction_by_id_returns_empty_dataframe_when_missing(duckdb_path):
    repository = TransactionRepository(duckdb_path)

    result = repository.get_transaction_by_id("UNKNOWN")

    assert result.is_empty()


def test_count_transactions_counts_distinct_mutations(duckdb_path):
    repository = TransactionRepository(duckdb_path)

    result = repository.count_transactions(
        postal_code="75015",
        property_type="Appartement",
        start_date=START_2024,
        end_date=END_2024,
    )

    # M001 has two property rows, but it is one mutation.
    assert result == 3  # M001, M002 and M006


def test_count_properties_counts_rows(duckdb_path):
    repository = TransactionRepository(duckdb_path)

    result = repository.count_transactions(
        postal_code="75015",
        property_type="Appartement",
        start_date=START_2024,
        end_date=END_2024,
    )

    assert result == 3  # Two rows for M001, one for M002 and one for M006.


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [(date(2024, 12, 31), date(2024, 1, 1))],
)
def test_invalid_date_range_raises_value_error(
    duckdb_path,
    start_date,
    end_date,
):
    repository = TransactionRepository(duckdb_path)

    with pytest.raises(ValueError):
        repository.find_transactions(
            postal_code="75015",
            property_type="Appartement",
            start_date=start_date,
            end_date=end_date,
        )


@pytest.mark.parametrize(
    ("min_surface", "max_surface"),
    [(-1.0, None), (None, -1.0), (60.0, 40.0)],
)
def test_invalid_surface_range_raises_value_error(
    duckdb_path,
    min_surface,
    max_surface,
):
    repository = TransactionRepository(duckdb_path)

    with pytest.raises(ValueError):
        repository.find_transactions(
            postal_code="75015",
            property_type="Appartement",
            start_date=START_2024,
            end_date=END_2024,
            min_surface=min_surface,
            max_surface=max_surface,
        )


def test_non_positive_limit_raises_value_error(duckdb_path):
    repository = TransactionRepository(duckdb_path)

    with pytest.raises(ValueError):
        repository.find_transactions(
            postal_code="75015",
            property_type="Appartement",
            start_date=START_2024,
            end_date=END_2024,
            limit=0,
        )
