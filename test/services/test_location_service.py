from unittest.mock import Mock

import pytest

from app.services.location_service import LocationService


@pytest.fixture
def area_profile():
    return {
        "annee": 2024,
        "code_postal": "75015",
        "nb_sante": 120,
        "nb_enseignement": 90,
        "nb_services_publics": 35,
        "nb_transports": 80,
        "nb_commerces": 210,
        "nb_sports_loisirs": 75,
        "score_equipements": 72.5,
        "surface_km2": 8.48,
    }


def test_get_area_overview_selects_three_largest_categories(area_profile):
    repository = Mock()
    repository.get_area_profile.return_value = area_profile
    service = LocationService(repository)

    result = service.get_area_overview("75015")

    assert result["status"] == "success"
    assert result["code_postal"] == "75015"
    assert result["superficie"] == {
        "valeur_km2": 8.48,
        "categorie": "Grande",
    }
    assert result["score_equipements"] == {
        "valeur": 72.5,
        "niveau": "Élevé",
    }

    top_categories = result["principales_categories_equipements"]
    assert [item["categorie"] for item in top_categories] == [
        "nb_commerces",
        "nb_sante",
        "nb_enseignement",
    ]
    assert "surface_km2" not in result["repartition_equipements"]
    assert "score_equipements" not in result["repartition_equipements"]


def test_get_area_overview_returns_not_found():
    repository = Mock()
    repository.get_area_profile.return_value = None
    service = LocationService(repository)

    result = service.get_area_overview("75020")

    assert result["status"] == "not_found"


def test_get_area_overview_detects_missing_required_indicators(area_profile):
    repository = Mock()
    invalid_profile = area_profile.copy()
    invalid_profile.pop("score_equipements")
    repository.get_area_profile.return_value = invalid_profile
    service = LocationService(repository)

    result = service.get_area_overview("75015")

    assert result["status"] == "invalid_data"


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        (1.99, "Petite"),
        (2.0, "Moyenne"),
        (4.3, "Moyenne"),
        (4.31, "Grande"),
    ],
)
def test_evaluate_surface_boundaries(surface, expected):
    assert LocationService._evaluate_surface(surface) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (29.99, "Faible"),
        (30.0, "Moyen"),
        (59.99, "Moyen"),
        (60.0, "Élevé"),
    ],
)
def test_evaluate_equipment_score_boundaries(score, expected):
    assert LocationService._evaluate_equipment_score(score) == expected
