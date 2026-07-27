import pytest

from app.data.repositories.market_repository import MarketRepository


def test_get_market_metrics_returns_structured_metrics(duckdb_path):
    repository = MarketRepository(duckdb_path)

    metrics = repository.get_market_metrics(
        postal_code="75015",
        property_type="Appartement",
        year=2024,
    )

    assert metrics is not None
    assert metrics.postal_code == "75015"
    assert metrics.property_type == "Appartement"
    assert metrics.year == 2024
    assert metrics.transaction_count == 4
    assert metrics.median_price_m2 == pytest.approx(9680.0)
    assert metrics.median_vf == pytest.approx(490000.0)
    assert metrics.median_surface == pytest.approx(52.0)


def test_get_market_metrics_returns_none_when_missing(duckdb_path):
    repository = MarketRepository(duckdb_path)

    metrics = repository.get_market_metrics(
        postal_code="75020",
        property_type="Appartement",
        year=2024,
    )

    assert metrics is None


def test_get_price_history_is_ordered_by_year(duckdb_path):
    repository = MarketRepository(duckdb_path)

    history = repository.get_price_history(
        postal_code="75015",
        property_type="Appartement",
        start_year=2022,
        end_year=2024,
    )

    assert history["annee"].to_list() == [2022, 2023, 2024]
    assert history["prix_m2_median"].to_list() == [8800.0, 9200.0, 9680.0]


def test_get_price_history_rejects_invalid_period(duckdb_path):
    repository = MarketRepository(duckdb_path)

    with pytest.raises(ValueError):
        repository.get_price_history(
            postal_code="75015",
            property_type="Appartement",
            start_year=2024,
            end_year=2022,
        )


def test_get_metrics_for_areas_returns_requested_areas(duckdb_path):
    repository = MarketRepository(duckdb_path)

    result = repository.get_metrics_for_areas(
        postal_codes=["75011", "75012", "75015"],
        property_type="Appartement",
        year=2024,
    )

    assert set(result["code_postal"].to_list()) == {"75011", "75012", "75015"}
    assert result.height == 3


def test_get_metrics_for_areas_rejects_empty_sequence(duckdb_path):
    repository = MarketRepository(duckdb_path)

    with pytest.raises(ValueError):
        repository.get_metrics_for_areas(
            postal_codes=[],
            property_type="Appartement",
            year=2024,
        )
