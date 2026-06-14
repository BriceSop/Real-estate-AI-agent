from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

# Make the project root importable if tests/ is used.
CURRENT_DIR = Path(__file__).resolve().parent
POSSIBLE_ROOTS = [CURRENT_DIR, CURRENT_DIR.parent]
for root in POSSIBLE_ROOTS:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from app.core.paths import CONFIGS_DIR
from app.data.cleaning import Bpe, Dvf
from app.data.utils import load_yaml

#def _load_yaml_config() -> dict:
 #   candidates = [
 #       CURRENT_DIR / "processing.yaml",
 #       CURRENT_DIR.parent / "processing.yaml",
 #   ]
 #   for path in candidates:
 #       if path.exists():
 #           with path.open("r", encoding="utf-8") as f:
 #               return yaml.safe_load(f)
 #   raise FileNotFoundError("processing.yaml introuvable à côté du fichier de test.")


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_yaml(CONFIGS_DIR / "cleaning.yaml")


@pytest.fixture
def dvf_cfg(cfg: dict) -> dict:
    return {"DVF": cfg["DVF"]}


@pytest.fixture
def bpe_cfg(cfg: dict) -> dict:
    return {"BPE": cfg["BPE"]}


@pytest.fixture
def dvf_full_raw_df() -> pl.DataFrame:
    """Minimal raw DVF dataframe matching the YAML schema."""
    return pl.DataFrame(
        {
            "id_mutation": ["m1", "m2"],
            "date_mutation": ["2024-01-01", "2024-02-02"],
            "numero_disposition": ["1", "2"],
            "numero_volume": ["0", "1"],
            "nature_mutation": ["Vente", "Vente"],
            "valeur_fonciere": ["9500,0", "250000,5"],
            "adresse_numero": [None, "12"],
            "adresse_nom_voie": [None, "Rue de Rivoli"],
            "code_postal": [None, "75016"],
            "code_commune": ["75101", "75116"],
            "nom_commune": ["PARIS 1ER ARRONDISSEMENT", "PARIS 16E ARRONDISSEMENT"],
            "code_departement": ["75", "75"],
            "nombre_lots": ["0", "0"],
            "type_local": [None, "Appartement"],
            "surface_reelle_bati": ["4,0", "45,0"],
            "nombre_pieces_principales": [None, "2"],
            "longitude": ["2,35", "2,36"],
            "latitude": ["48,85", "48,86"],
            "lot1_surface_carrez": [None, "20,5"],
            "lot2_surface_carrez": [None, None],
            "lot3_surface_carrez": [None, None],
            "lot4_surface_carrez": [None, None],
            "lot5_surface_carrez": [None, None],
            "surface_terrain": [None, "0,0"],
        }
    )


@pytest.fixture
def bpe_raw_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "TIME_PERIOD": ["2023", "2023", "2023"],
            "OBS_VALUE": ["5", "7", "1"],
            "GEO": ["75101", "75102", "99999"],
            "FACILITY_DOM": ["D1", "D1", "D2"],
            "FACILITY_SDOM": ["SD1", "SD2", "SD3"],
            "FACILITY_TYPE": ["T1", "T2", "T3"],
        }
    )


@pytest.fixture
def codes_dom_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "code": ["D1", "D2"],
            "libelle français": ["Santé", "Sport"],
        }
    )


@pytest.fixture
def codes_sdom_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "code": ["SD1", "SD2", "SD3"],
            "libelle français": ["Médecins", "Pharmacies", "Piscines"],
        }
    )


@pytest.fixture
def codes_type_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "code": ["T1", "T2", "T3"],
            "libelle français": ["Cabinet", "Officine", "Bassin"],
        }
    )


def test_remove_header_rows_removes_duplicate_headers(dvf_cfg: dict) -> None:
    df = pl.DataFrame(
        {
            "id_mutation": ["m1", "m2"],
            "date_mutation": ["2024-01-01", "date_mutation"],
        }
    )

    cleaned = Dvf(df, dvf_cfg).remove_header_rows().df

    assert cleaned.height == 1
    assert cleaned["id_mutation"].to_list() == ["m1"]


def test_casting_columns_types_casts_expected_types(dvf_cfg: dict, dvf_full_raw_df: pl.DataFrame) -> None:
    dvf = Dvf(dvf_full_raw_df, dvf_cfg).casting_columns_types()

    assert dvf.df.schema["date_mutation"] == pl.Date
    assert dvf.df.schema["numero_disposition"] == pl.Int64
    assert dvf.df.schema["numero_volume"] == pl.Int64
    assert dvf.df.schema["valeur_fonciere"] == pl.Float64
    assert dvf.df.schema["surface_reelle_bati"] == pl.Float64
    assert dvf.df.schema["longitude"] == pl.Float64
    assert dvf.df["numero_disposition"].to_list() == [1, 2]


def test_imputing_postal_code_null_fills_only_missing_values(dvf_cfg: dict) -> None:
    df = pl.DataFrame(
        {
            "code_postal": [None, "75016"],
            "code_commune": ["75101", "75116"],
        }
    )

    dvf = Dvf(df, dvf_cfg).imputing_postal_code_null()

    assert dvf.df["code_postal"].to_list() == ["75001", "75016"]


def test_reverse_geocode_fr_returns_expected_payload(monkeypatch: pytest.MonkeyPatch, dvf_cfg: dict) -> None:
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "features": [
                    {
                        "properties": {
                            "label": "10 Rue de Rivoli 75001 Paris",
                            "street": "Rue de Rivoli",
                            "housenumber": "10",
                            "postcode": "75001",
                            "city": "Paris",
                            "citycode": "75101",
                            "score": 0.97,
                        }
                    }
                ]
            }
    
    import app.data.cleaning as cleaning

    def fake_get(url: str, params: dict, timeout: int):
        assert url == dvf_cfg["DVF"]["API_URL"]
        assert params["lon"] == 2.35
        assert params["lat"] == 48.85
        assert params["postcode"] == "75001"
        assert timeout == 10
        return DummyResponse()

    monkeypatch.setattr(cleaning.requests, "get", fake_get)

    dvf = Dvf(pl.DataFrame({"a": [1]}), dvf_cfg)
    result = dvf.reverse_geocode_fr(2.35, 48.85, postcode="75001")

    assert result == {
        "full_address_geocoded": "10 Rue de Rivoli 75001 Paris",
        "street_geocoded": "Rue de Rivoli",
        "housenumber_geocoded": "10",
        "postcode_geocoded": "75001",
        "city_geocoded": "Paris",
        "citycode_geocoded": "75101",
        "score_geocoded": 0.97,
    }


def test_reverse_geocode_fr_returns_none_when_no_feature(monkeypatch: pytest.MonkeyPatch, dvf_cfg: dict) -> None:
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"features": []}

    import app.data.cleaning as cleaning

    monkeypatch.setattr(cleaning.requests, "get", lambda *args, **kwargs: DummyResponse())

    dvf = Dvf(pl.DataFrame({"a": [1]}), dvf_cfg)
    assert dvf.reverse_geocode_fr(2.35, 48.85) is None


def test_get_address_imputation_selects_rows_with_missing_num_or_street(
    monkeypatch: pytest.MonkeyPatch, dvf_cfg: dict
) -> None:
    df = pl.DataFrame(
        {
            "id_mutation": ["m1", "m2", "m3"],
            "adresse_numero": [None, "12", "15"],
            "adresse_nom_voie": ["Rue A", None, "Rue C"],
            "longitude": [2.35, 2.36, 2.37],
            "latitude": [48.85, 48.86, 48.87],
            "code_postal": ["75001", "75002", "75003"],
            "code_commune": ["75101", "75102", "75103"],
        }
    )

    def fake_reverse(self, lon, lat, postcode=None, citycode=None):
        return {
            "full_address_geocoded": "label",
            "street_geocoded": f"street-{postcode}",
            "housenumber_geocoded": "10",
            "postcode_geocoded": postcode,
            "city_geocoded": "Paris",
            "citycode_geocoded": citycode,
            "score_geocoded": 0.95,
        }

    monkeypatch.setattr(Dvf, "reverse_geocode_fr", fake_reverse)
    import app.data.cleaning as cleaning
    monkeypatch.setattr(cleaning.time, "sleep", lambda *_: None)

    results = Dvf(df, dvf_cfg).get_address_imputation()

    assert len(results) == 2
    assert {r["id_mutation"] for r in results} == {"m1", "m2"}
    assert all(r["address_inferred_from_coords"] is True for r in results)


def test_get_address_imputation_returns_fallback_when_reverse_geocoding_fails(
    monkeypatch: pytest.MonkeyPatch, dvf_cfg: dict
) -> None:
    df = pl.DataFrame(
        {
            "id_mutation": ["m1"],
            "adresse_numero": [None],
            "adresse_nom_voie": [None],
            "longitude": [2.35],
            "latitude": [48.85],
            "code_postal": ["75001"],
            "code_commune": ["75101"],
        }
    )

    def fake_reverse(self, *args, **kwargs):
        raise RuntimeError("API down")

    monkeypatch.setattr(Dvf, "reverse_geocode_fr", fake_reverse)

    results = Dvf(df, dvf_cfg).get_address_imputation()

    assert results == [
        {
            "id_mutation": "m1",
            "street_geocoded": None,
            "postcode_geocoded": None,
            "citycode_geocoded": None,
            "score_geocoded": None,
            "address_inferred_from_coords": False,
            "lon": 2.35,
            "lat": 48.85,
        }
    ]


def test_prepare_imputation_df_cleans_housenumber_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch, dvf_cfg: dict
) -> None:
    def fake_get_address_imputation(self):
        return [
            {
                "id_mutation": "m1",
                "full_address_geocoded": "addr",
                "street_geocoded": "Rue A",
                "housenumber_geocoded": "41p1",
                "postcode_geocoded": "75001",
                "city_geocoded": "Paris",
                "citycode_geocoded": "75101",
                "score_geocoded": 0.94,
                "address_inferred_from_coords": True,
                "lon": 2.35,
                "lat": 48.85,
            },
            {
                "id_mutation": "m1",
                "full_address_geocoded": "addr",
                "street_geocoded": "Rue A",
                "housenumber_geocoded": "41p1",
                "postcode_geocoded": "75001",
                "city_geocoded": "Paris",
                "citycode_geocoded": "75101",
                "score_geocoded": 0.94,
                "address_inferred_from_coords": True,
                "lon": 2.35,
                "lat": 48.85,
            },
        ]

    monkeypatch.setattr(Dvf, "get_address_imputation", fake_get_address_imputation)

    dvf = Dvf(pl.DataFrame({"dummy": [1]}), dvf_cfg)
    out = dvf.prepare_imputation_df()

    assert out.height == 1
    assert out["adresse_numero_clean"].to_list() == ["41"]
    assert "housenumber_geocoded" not in out.columns
    assert "full_address_geocoded" not in out.columns


def test_join_impute_df_left_joins_imputation_dataframe(monkeypatch: pytest.MonkeyPatch, dvf_cfg: dict) -> None:
    source = pl.DataFrame(
        {
            "id_mutation": ["m1", "m2"],
            "longitude": [2.35, 2.36],
            "latitude": [48.85, 48.86],
        }
    )
    geo_df = pl.DataFrame(
        {
            "id_mutation": ["m1"],
            "lon": [2.35],
            "lat": [48.85],
            "street_geocoded": ["Rue A"],
        }
    )

    monkeypatch.setattr(Dvf, "prepare_imputation_df", lambda self: geo_df)

    dvf = Dvf(source, dvf_cfg).join_impute_df()

    assert dvf.df["street_geocoded"].to_list() == ["Rue A", None]


def test_impute_address_infos_updates_fields_from_high_score_and_fills_unavailable(dvf_cfg: dict) -> None:
    df = pl.DataFrame(
        {
            "address_inferred_from_coords": [True, None, False],
            "score_geocoded": [0.95, None, 0.8],
            "adresse_numero_clean": ["10", None, "99"],
            "street_geocoded": ["Rue A", None, "Rue Z"],
            "adresse_numero": [None, None, "7"],
            "adresse_nom_voie": [None, None, "Rue B"],
            "type_local": [None, "Appartement", None],
        }
    )

    dvf = Dvf(df, dvf_cfg).impute_address_infos()

    assert dvf.df["adresse_numero"].to_list() == ["10", "Indisponible", "7"]
    assert dvf.df["adresse_nom_voie"].to_list() == ["Rue A", "Indisponible", "Rue B"]
    assert dvf.df["type_local"].to_list() == ["Indisponible", "Appartement", "Indisponible"]
    assert dvf.df["address_inferred_from_coords"].to_list() == [True, False, False]


def test_normalize_string_formats_address(dvf_cfg: dict) -> None:
    dvf = Dvf(pl.DataFrame({"a": [1]}), dvf_cfg)

    value = dvf.normalize_string("Boulevard de l'Hôpital")

    assert value == "BD DE L HOPITAL"


def test_normalize_address_applies_normalization_to_column(dvf_cfg: dict) -> None:
    df = pl.DataFrame({"adresse_nom_voie": ["Passage du Caire", "Allée des Prés-Saint-Gervais"]})

    dvf = Dvf(df, dvf_cfg).normalize_address()

    assert dvf.df["adresse_nom_voie"].to_list() == [
        "PAS DU CAIRE",
        "ALL DES PRES SAINT GERVAIS",
    ]


def test_create_flag_for_null_val_creates_expected_columns(dvf_cfg: dict) -> None:
    df = pl.DataFrame(
        {
            "valeur_fonciere": [None, 100000.0],
            "surface_reelle_bati": [40.0, None],
            "nombre_pieces_principales": [None, 2],
        }
    )

    dvf = Dvf(df, dvf_cfg).create_flag_for_null_val()

    assert dvf.df["is_valeur_fonciere_missing"].to_list() == [True, False]
    assert dvf.df["is_surface_missing"].to_list() == [False, True]
    assert dvf.df["is_nombre_pieces_missing"].to_list() == [True, False]


def test_create_flag_for_atypical_val_creates_expected_columns(dvf_cfg: dict) -> None:
    df = pl.DataFrame(
        {
            "is_surface_missing": [False, True, False],
            "surface_reelle_bati": [4.0, None, 30.0],
            "is_valeur_fonciere_missing": [False, False, True],
            "valeur_fonciere": [9000.0, 15000.0, None],
            "type_local": ["Appartement", "Appartement", "Maison"],
            "nombre_lots": [0, 2, 0],
        }
    )

    dvf = Dvf(df, dvf_cfg).create_flag_for_atypical_val()

    assert dvf.df["is_surface_tres_faible"].to_list() == [True, False, False]
    assert dvf.df["is_valeur_fonciere_tres_basse"].to_list() == [True, False, False]
    assert dvf.df["is_appartement_sans_lot"].to_list() == [True, False, False]
    assert dvf.df["is_transaction_atypique_agent"].to_list() == [True, False, False]


def test_drop_duplicated_rows_keeps_unique_rows_on_selected_columns(dvf_cfg: dict) -> None:
    cols = dvf_cfg["DVF"]["COLS_TO_KEEP"]
    base_row = {col: None for col in cols}
    base_row.update(
        {
            "id_mutation": "m1",
            "date_mutation": None,
            "numero_disposition": 1,
            "nature_mutation": "Vente",
            "valeur_fonciere": 100000.0,
            "adresse_numero": "10",
            "adresse_nom_voie": "RUE A",
            "code_postal": "75001",
            "code_commune": "75101",
            "nom_commune": "PARIS 1ER ARRONDISSEMENT",
            "code_departement": "75",
            "nombre_lots": 1,
            "type_local": "Appartement",
            "surface_reelle_bati": 30.0,
            "nombre_pieces_principales": 2,
            "address_inferred_from_coords": False,
            "is_valeur_fonciere_missing": False,
            "is_surface_missing": False,
            "is_nombre_pieces_missing": False,
        }
    )
    df = pl.DataFrame([base_row, dict(base_row), {**base_row, "id_mutation": "m2"}])

    dvf = Dvf(df, dvf_cfg).drop_duplicated_rows()

    assert dvf.df.height == 2
    assert dvf.df["id_mutation"].to_list() == ["m1", "m2"]


def test_run_dvf_to_silver_runs_full_pipeline(monkeypatch: pytest.MonkeyPatch, dvf_cfg: dict, dvf_full_raw_df: pl.DataFrame) -> None:
    geo_df = pl.DataFrame(
        {
            "id_mutation": ["m1"],
            "lon": [2.35],
            "lat": [48.85],
            "street_geocoded": ["Boulevard de l'Hôpital"],
            "adresse_numero_clean": ["2"],
            "score_geocoded": [0.95],
            "address_inferred_from_coords": [True],
            "postcode_geocoded": ["75001"],
        }
    )

    monkeypatch.setattr(Dvf, "prepare_imputation_df", lambda self: geo_df)

    dvf = Dvf(dvf_full_raw_df, dvf_cfg).run_dvf_to_silver()

    assert dvf.df.schema["date_mutation"] == pl.Date
    assert dvf.df["code_postal"].to_list() == ["75001", "75016"]
    assert dvf.df["adresse_numero"].to_list()[0] == "2"
    assert dvf.df["adresse_nom_voie"].to_list()[0] == "BD DE L HOPITAL"
    assert "is_transaction_atypique_agent" not in dvf.df.columns
    assert dvf.df.height == 2


def test_bpe_casting_columns_types_casts_expected_types(bpe_cfg: dict, bpe_raw_df: pl.DataFrame) -> None:
    bpe = Bpe(bpe_raw_df, bpe_cfg).casting_columns_types()

    assert bpe.df.schema["TIME_PERIOD"] == pl.Int64
    assert bpe.df.schema["OBS_VALUE"] == pl.Int64
    assert bpe.df["TIME_PERIOD"].to_list() == [2023, 2023, 2023]


def test_slicing_by_code_arr_keeps_only_selected_arrondissements(bpe_cfg: dict, bpe_raw_df: pl.DataFrame) -> None:
    bpe = Bpe(bpe_raw_df, bpe_cfg).slicing_by_code_arr()

    assert bpe.df["GEO"].to_list() == ["75101", "75102"]


def test_join_code_nom_adds_label_column(bpe_cfg: dict, bpe_raw_df: pl.DataFrame, codes_dom_df: pl.DataFrame) -> None:
    bpe = Bpe(bpe_raw_df, bpe_cfg).join_code_nom(codes_dom_df, bpe_cfg["BPE"]["DOM_CODES"])

    assert "classification_n1" in bpe.df.columns
    assert bpe.df["classification_n1"].to_list() == ["Santé", "Santé", "Sport"]


def test_creating_postal_code_builds_paris_postcodes(bpe_cfg: dict, bpe_raw_df: pl.DataFrame) -> None:
    bpe = Bpe(bpe_raw_df, bpe_cfg).creating_postal_code()

    assert bpe.df["code_postal"].to_list() == ["75001", "75002", "75099"]


def test_renaming_columns_renames_and_filters_expected_columns(bpe_cfg: dict, bpe_raw_df: pl.DataFrame) -> None:
    df = (
        Bpe(bpe_raw_df, bpe_cfg)
        .join_code_nom(pl.DataFrame({"code": ["D1", "D2"], "libelle français": ["Santé", "Sport"]}), bpe_cfg["BPE"]["DOM_CODES"])
        .join_code_nom(pl.DataFrame({"code": ["SD1", "SD2", "SD3"], "libelle français": ["Médecins", "Pharmacies", "Piscines"]}), bpe_cfg["BPE"]["SDOM_CODES"])
        .join_code_nom(pl.DataFrame({"code": ["T1", "T2", "T3"], "libelle français": ["Cabinet", "Officine", "Bassin"]}), bpe_cfg["BPE"]["TYPES_CODES"])
        .creating_postal_code()
    )
    df.renaming_columns()

    assert df.df.columns == bpe_cfg["BPE"]["COLS_TO_KEEP"]


def test_run_bpe_to_silver_runs_full_pipeline(
    bpe_cfg: dict,
    bpe_raw_df: pl.DataFrame,
    codes_dom_df: pl.DataFrame,
    codes_sdom_df: pl.DataFrame,
    codes_type_df: pl.DataFrame,
) -> None:
    bpe = Bpe(bpe_raw_df, bpe_cfg).run_bpe_to_silver(codes_dom_df, codes_sdom_df, codes_type_df)

    assert bpe.df.columns == bpe_cfg["BPE"]["COLS_TO_KEEP"]
    assert bpe.df.height == 2
    assert bpe.df["code_postal"].to_list() == ["75001", "75002"]
    assert bpe.df["classification_n1"].to_list() == ["Santé", "Santé"]
