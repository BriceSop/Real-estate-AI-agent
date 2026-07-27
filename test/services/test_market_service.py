from types import SimpleNamespace
from unittest.mock import Mock

import polars as pl
import pytest

from app.services.market_service import MarketService


def _market_metrics(**overrides):
    values = {
        "postal_code": "75015",
        "property_type": "Appartement",
        "year": 2024,
        "transaction_count": 120,
        "mean_price_m2": 9800.0,
        "median_price_m2": 9600.0,
        "q25_price_m2": 8800.0,
        "q75_price_m2": 10500.0,
        "median_vf": 480000.0,
        "median_surface": 50.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_get_market_overview_returns_structured_result():
    repository = Mock()
    repository.get_market_metrics.return_value = _market_metrics()
    service = MarketService(repository)

    result = service.get_market_overview(
        postal_code="75015",
        property_type="Appartement",
        year=2024,
    )

    assert result["status"] == "success"
    assert result["nb_transactions"] == 120
    assert result["prix_m2"]["mediane"] == 9600.0
    assert result["prix_m2"]["ecart_interquartile"] == 1700.0
    assert result["valeur_fonciere_mediane"] == 480000.0
    repository.get_market_metrics.assert_called_once_with(
        postal_code="75015",
        property_type="Appartement",
        year=2024,
    )


def test_get_market_overview_returns_not_found():
    repository = Mock()
    repository.get_market_metrics.return_value = None
    service = MarketService(repository)

    result = service.get_market_overview(
        postal_code="75020",
        property_type="Appartement",
        year=2024,
    )

    assert result["status"] == "not_found"


def test_get_price_evolution_computes_first_to_last_change():
    repository = Mock()
    repository.get_price_history.return_value = pl.DataFrame(
        {
            "annee": [2022, 2023, 2024],
            "prix_m2_mean": [9000.0, 9500.0, 9900.0],
            "prix_m2_median": [8000.0, 9000.0, 10000.0],
        }
    )
    service = MarketService(repository)

    result = service.get_price_evolution(
        postal_code="75015",
        property_type="Appartement",
        start_year=2022,
        end_year=2024,
    )

    assert result["status"] == "success"
    assert result["annee_debut_effective"] == 2022
    assert result["annee_fin_effective"] == 2024
    assert result["evolution_pourcentage"]["mediane"] == 25.0
    assert result["evolution_pourcentage"]["moyenne"] == 10.0


def test_get_price_evolution_requires_two_available_years():
    repository = Mock()
    repository.get_price_history.return_value = pl.DataFrame(
        {
            "annee": [2024],
            "prix_m2_mean": [9900.0],
            "prix_m2_median": [9680.0],
        }
    )
    service = MarketService(repository)

    result = service.get_price_evolution(
        postal_code="75015",
        property_type="Appartement",
        start_year=2022,
        end_year=2024,
    )

    assert result["status"] == "insufficient_data"


def test_get_price_evolution_rejects_invalid_requested_period():
    service = MarketService(Mock())

    with pytest.raises(ValueError):
        service.get_price_evolution(
            postal_code="75015",
            property_type="Appartement",
            start_year=2024,
            end_year=2024,
        )


def test_compare_areas_returns_ranking_and_missing_codes():
    repository = Mock()
    repository.get_metrics_for_areas.return_value = pl.DataFrame(
        {
            "code_postal": ["75011", "75012", "75015"],
            "type_bien_principal": ["Appartement"] * 3,
            "annee": [2024] * 3,
            "nb_transactions": [30, 20, 40],
            "prix_m2_mean": [10500.0, 9300.0, 9900.0],
            "prix_m2_median": [10300.0, 9100.0, 9680.0],
            "prix_m2_q25": [9300.0, 8500.0, 8800.0],
            "prix_m2_q75": [11200.0, 9900.0, 10700.0],
            "valeur_fonciere_median": [515000.0, 455000.0, 490000.0],
            "surface_totale_median": [49.0, 50.0, 52.0],
        }
    )
    service = MarketService(repository)

    result = service.compare_areas(
        postal_codes=["75011", "75012", "75015", "75020"],
        property_type="Appartement",
        year=2024,
    )

    assert result["status"] == "success"
    assert result["zone_plus_chere"]["code_postal"] == "75011"
    assert result["zone_moins_chere"]["code_postal"] == "75012"
    assert result["codes_postaux_absents"] == ["75020"]
    assert [row["code_postal"] for row in result["classement"]] == [
        "75011",
        "75015",
        "75012",
    ]


def test_compare_areas_requires_at_least_two_unique_areas():
    service = MarketService(Mock())

    with pytest.raises(ValueError):
        service.compare_areas(
            postal_codes=["75015", "75015"],
            property_type="Appartement",
            year=2024,
        )
