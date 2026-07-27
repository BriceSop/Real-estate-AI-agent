import duckdb
import pytest

from app.data.repositories.location_repository import LocationRepository


def test_get_area_profile_returns_named_dictionary(duckdb_path):
    repository = LocationRepository(duckdb_path)

    profile = repository.get_area_profile("75015")

    assert profile is not None
    assert profile["code_postal"] == "75015"
    assert profile["nb_commerces"] == 210
    assert profile["score_equipements"] == pytest.approx(72.5)
    assert profile["surface_km2"] == pytest.approx(8.48)


def test_get_area_profile_returns_none_when_missing(duckdb_path):
    repository = LocationRepository(duckdb_path)

    assert repository.get_area_profile("75020") is None


@pytest.mark.parametrize("postal_code", ["", "7501", "7501A", "  "])
def test_get_area_profile_rejects_invalid_postal_code(
    duckdb_path,
    postal_code,
):
    repository = LocationRepository(duckdb_path)

    with pytest.raises(ValueError):
        repository.get_area_profile(postal_code)


def test_get_area_profile_detects_duplicate_rows(duckdb_path):
    with duckdb.connect(str(duckdb_path)) as connection:
        connection.execute(
            """
            INSERT INTO location_features VALUES
                (2025, '75015', 125, 92, 36, 82, 215, 77, 73.0, 8.48)
            """
        )

    repository = LocationRepository(duckdb_path)

    with pytest.raises(RuntimeError):
        repository.get_area_profile("75015")
