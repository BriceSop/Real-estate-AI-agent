from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
import pytest
import yaml
from polars.testing import assert_frame_equal

from app.data.transforming import Equipments, Transactions, Transforming


CONFIG_PATH = PROJECT_ROOT / "configs" / "transforming.yaml"


@pytest.fixture(scope="session")
def cfg() -> dict:
    """Load the real transformation configuration used by the pipeline."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _dvf_row(
    *,
    id_mutation: str,
    numero_disposition: int,
    date_mutation: str,
    valeur_fonciere: float,
    type_local: str,
    surface_reelle_bati: float,
    code_postal: str = "75001",
    is_valeur_fonciere_missing: bool = False,
    is_surface_missing: bool = False,
    is_transaction_atypique_agent: bool = False,
) -> dict:
    """Build one representative Silver DVF row with every required column."""
    return {
        "id_mutation": id_mutation,
        "date_mutation": date_mutation,
        "numero_disposition": numero_disposition,
        "nature_mutation": "Vente",
        "valeur_fonciere": valeur_fonciere,
        "adresse_numero": "10",
        "adresse_nom_voie": "RUE DE TEST",
        "code_postal": code_postal,
        "code_commune": "75101",
        "nom_commune": "PARIS 01",
        "code_departement": "75",
        "type_local": type_local,
        "surface_reelle_bati": surface_reelle_bati,
        "nombre_pieces_principales": 2,
        "address_inferred_from_coords": "10 RUE DE TEST",
        "is_valeur_fonciere_missing": is_valeur_fonciere_missing,
        "is_surface_missing": is_surface_missing,
        "is_nombre_pieces_missing": False,
        "is_surface_tres_faible": False,
        "is_valeur_fonciere_tres_basse": False,
        "is_appartement_sans_lot": False,
        "is_transaction_atypique_agent": is_transaction_atypique_agent,
    }


@pytest.fixture
def dvf_silver_df() -> pl.DataFrame:
    """Small DVF dataset covering apartment, house, mixed and complex sales."""
    rows = [
        # Appartement + dépendance: 300_000 / (40 + 10) = 6_000 €/m²
        _dvf_row(
            id_mutation="m1",
            numero_disposition=1,
            date_mutation="2024-01-15",
            valeur_fonciere=300_000.0,
            type_local="Appartement",
            surface_reelle_bati=40.0,
        ),
        _dvf_row(
            id_mutation="m1",
            numero_disposition=1,
            date_mutation="2024-01-15",
            valeur_fonciere=300_000.0,
            type_local="Dépendance",
            surface_reelle_bati=10.0,
        ),
        # Maison + dépendance: 500_000 / (80 + 20) = 5_000 €/m²
        _dvf_row(
            id_mutation="m2",
            numero_disposition=1,
            date_mutation="2024-05-10",
            valeur_fonciere=500_000.0,
            type_local="Maison",
            surface_reelle_bati=80.0,
        ),
        _dvf_row(
            id_mutation="m2",
            numero_disposition=1,
            date_mutation="2024-05-10",
            valeur_fonciere=500_000.0,
            type_local="Dépendance",
            surface_reelle_bati=20.0,
        ),
        # Appartement + maison + dépendance: 880_000 / 110 = 8_000 €/m²
        _dvf_row(
            id_mutation="m3",
            numero_disposition=1,
            date_mutation="2025-03-20",
            valeur_fonciere=880_000.0,
            type_local="Appartement",
            surface_reelle_bati=40.0,
            code_postal="75002",
        ),
        _dvf_row(
            id_mutation="m3",
            numero_disposition=1,
            date_mutation="2025-03-20",
            valeur_fonciere=880_000.0,
            type_local="Maison",
            surface_reelle_bati=60.0,
            code_postal="75002",
        ),
        _dvf_row(
            id_mutation="m3",
            numero_disposition=1,
            date_mutation="2025-03-20",
            valeur_fonciere=880_000.0,
            type_local="Dépendance",
            surface_reelle_bati=10.0,
            code_postal="75002",
        ),
        # Type explicitly considered complex by the configuration.
        _dvf_row(
            id_mutation="m4",
            numero_disposition=1,
            date_mutation="2025-06-01",
            valeur_fonciere=100_000.0,
            type_local="Indisponible",
            surface_reelle_bati=50.0,
            code_postal="75003",
        ),
        # Invalid transaction: must be removed from the final Gold dataframe.
        _dvf_row(
            id_mutation="m_bad",
            numero_disposition=1,
            date_mutation="2024-08-01",
            valeur_fonciere=200_000.0,
            type_local="Appartement",
            surface_reelle_bati=0.0,
            is_surface_missing=True,
        ),
    ]
    return pl.DataFrame(rows)


@pytest.fixture
def equipment_silver_df() -> pl.DataFrame:
    """BPE sample with two postal codes and summary rows that must be ignored."""
    return pl.DataFrame(
        {
            "annee": [2024] * 14,
            "code_postal": ["75001"] * 7 + ["75002"] * 7,
            "classification_n1": [
                "Santé et action sociale",
                "Enseignement",
                "Transports et déplacements",
                "Commerces",
                "Sports, loisirs et culture",
                "Autre",
                "Santé et action sociale",
                "Santé et action sociale",
                "Enseignement",
                "Transports et déplacements",
                "Commerces",
                "Sports, loisirs et culture",
                "Autre",
                "Santé et action sociale",
            ],
            "classification_n2": [
                "Médecins",
                "Écoles",
                "Gares",
                "Magasins",
                "Sports",
                "Services publics",
                "Total",  # Must be excluded.
                "Médecins",
                "Écoles",
                "Gares",
                "Magasins",
                "Sports",
                "Services publics",
                "Total",  # Must be excluded.
            ],
            "classification_n3": [
                "Généralistes",
                "Primaires",
                "Métro",
                "Alimentation",
                "Gymnases",
                "Mairies",
                "Détail",
                "Généralistes",
                "Primaires",
                "Métro",
                "Alimentation",
                "Gymnases",
                "Mairies",
                "Détail",
            ],
            "nombre_equipements": [10, 8, 6, 4, 2, 3, 999, 20, 16, 12, 8, 4, 6, 999],
        }
    )


@pytest.fixture
def ready_transaction_df(cfg: dict) -> pl.DataFrame:
    """Transaction-level dataframe ready for filtering and column selection."""
    cols = cfg["TRANSACTIONS"]["COLS_TO_KEEP"]

    def row(
        mutation: str,
        *,
        missing_value: bool = False,
        missing_surface: bool = False,
        atypical: bool = False,
    ) -> dict:
        values = {
            "id_mutation": mutation,
            "date_mutation": date(2024, 1, 1),
            "annee": 2024,
            "numero_disposition": 1,
            "nature_mutation": "Vente",
            "valeur_fonciere": 300_000.0,
            "prix_m2": 6_000.0,
            "adresse_numero": "10",
            "adresse_nom_voie": "RUE DE TEST",
            "code_postal": "75001",
            "type_local": "Appartement",
            "type_bien_principal": "Appartement",
            "surface_reelle_bati": 50.0,
            "surface_totale": 50.0,
            "nombre_pieces_principales": 2,
            "has_appartement": True,
            "has_maison": False,
            "has_dependance": False,
            "is_valeur_fonciere_missing": missing_value,
            "is_surface_missing": missing_surface,
            "is_nombre_pieces_missing": False,
            "is_appartement_sans_lot": False,
            "is_transaction_atypique_agent": atypical,
        }
        return {col: values[col] for col in cols}

    # m1 is duplicated to verify the final unique operation.
    return pl.DataFrame(
        [
            row("m1"),
            row("m1"),
            row("m2", missing_value=True),
            row("m3", missing_surface=True),
            row("m4", atypical=True),
        ]
    )


@pytest.fixture
def aggregated_equipment_df() -> pl.DataFrame:
    """Aggregated counts with known min/max values for score testing."""
    return pl.DataFrame(
        {
            "annee": [2024, 2024, 2025],
            "code_postal": ["75001", "75002", "75003"],
            "nb_sante": [10, 20, 5],
            "nb_enseignement": [8, 16, 5],
            "nb_services_publics": [3, 6, 5],
            "nb_transports": [6, 12, 5],
            "nb_commerces": [4, 8, 5],
            "nb_sports_loisirs": [2, 4, 5],
        }
    )


class TestTransformingBase:
    def test_constructor_clones_input_dataframe(self, cfg: dict) -> None:
        source = pl.DataFrame({"x": [1, 2]})
        transformer = Transforming(source, cfg)

        assert transformer.df is not source
        assert_frame_equal(transformer.df, source)
        assert transformer.cfg is cfg


class TestTransactions:
    def test_set_date_column_format_and_create_year(
        self, dvf_silver_df: pl.DataFrame, cfg: dict
    ) -> None:
        transformer = Transactions(dvf_silver_df, cfg)

        returned = transformer.set_date_column_format().create_year_column()

        assert returned is transformer
        assert transformer.df.schema["date_mutation"] == pl.Date
        assert transformer.df.schema["annee"] == pl.Int64
        assert transformer.df.filter(pl.col("id_mutation") == "m1")["annee"].to_list() == [2024, 2024]

    def test_prepare_prix_m2_df_aggregates_surface_once_per_transaction(
        self, dvf_silver_df: pl.DataFrame, cfg: dict
    ) -> None:
        transformer = Transactions(dvf_silver_df, cfg)

        result = transformer.prepare_prix_m2_df(transformer._cfg["PRIX_M2"])
        row = result.filter(pl.col("id_mutation") == "m1").row(0, named=True)

        assert row["valeur_fonciere"] == pytest.approx(300_000.0)
        assert row["surface_totale"] == pytest.approx(50.0)
        assert result.height == 5

    def test_build_prix_m2_column_handles_zero_surface(self, cfg: dict) -> None:
        transformer = Transactions(pl.DataFrame(), cfg)
        source = pl.DataFrame(
            {
                "id_mutation": ["m1", "m2"],
                "numero_disposition": [1, 1],
                "valeur_fonciere": [300_000.0, 100_000.0],
                "surface_totale": [50.0, 0.0],
            }
        )

        result = transformer.build_prix_m2_column(source, transformer._cfg["PRIX_M2"])

        assert result["prix_m2"].to_list() == [6_000.0, None]

    def test_prepare_type_biens_df_counts_and_deduplicates_types(
        self, dvf_silver_df: pl.DataFrame, cfg: dict
    ) -> None:
        transformer = Transactions(dvf_silver_df, cfg)

        result = transformer.prepare_type_biens_df(transformer._cfg["TYPE_BIENS"])
        row = result.filter(pl.col("id_mutation") == "m1").row(0, named=True)

        assert row["nb_biens"] == 2
        assert set(row["type_biens_unique"]) == {"Appartement", "Dépendance"}

    def test_create_flag_for_type_biens(self, cfg: dict) -> None:
        transformer = Transactions(pl.DataFrame(), cfg)
        source = pl.DataFrame(
            {
                "type_biens_unique": [
                    ["Appartement", "Dépendance"],
                    ["Maison"],
                    ["Indisponible"],
                ]
            }
        )

        result = transformer.create_flag_for_type_biens(
            source,
            transformer._cfg["TYPE_BIENS"]["FLAG_TYPE_BIENS"],
        )

        assert result.select(
            "has_appartement", "has_maison", "has_dependance"
        ).to_dict(as_series=False) == {
            "has_appartement": [True, False, False],
            "has_maison": [False, True, False],
            "has_dependance": [True, False, False],
        }

    def test_create_type_bien_princ_covers_all_business_cases(self, cfg: dict) -> None:
        transformer = Transactions(pl.DataFrame(), cfg)
        source = pl.DataFrame(
            {
                "type_biens_unique": [
                    ["Appartement"],
                    ["Appartement", "Dépendance"],
                    ["Maison", "Dépendance"],
                    ["Appartement", "Maison", "Dépendance"],
                    ["Indisponible"],
                    ["Dépendance"],
                ]
            }
        )

        result = transformer.create_type_bien_princ(
            source,
            transformer._cfg["TYPE_BIENS"],
        )

        assert result["type_bien_principal"].to_list() == [
            "Appartement",
            "Appartement",
            "Maison",
            "Mixte",
            "Complex",
            "Autre",
        ]

    def test_join_to_initial_df_preserves_left_rows(self, cfg: dict) -> None:
        transformer = Transactions(pl.DataFrame(), cfg)
        left = pl.DataFrame(
            {
                "id_mutation": ["m1", "m1", "m2"],
                "numero_disposition": [1, 1, 1],
                "value": [1, 2, 3],
            }
        )
        right = pl.DataFrame(
            {
                "id_mutation": ["m1", "m2"],
                "numero_disposition": [1, 1],
                "metric": [10.0, 20.0],
            }
        )

        result = transformer.join_to_initial_df(
            left,
            right,
            ["id_mutation", "numero_disposition"],
            ["id_mutation", "numero_disposition"],
        )

        assert result.height == left.height
        assert result["metric"].to_list() == [10.0, 10.0, 20.0]

    def test_arrange_transactions_df_filters_selects_and_deduplicates(
        self, ready_transaction_df: pl.DataFrame, cfg: dict
    ) -> None:
        transformer = Transactions(pl.DataFrame(), cfg)

        result = transformer.arrange_transactions_df(ready_transaction_df)

        assert result.columns == transformer._cfg["COLS_TO_KEEP"]
        assert result.height == 1
        assert result["id_mutation"].to_list() == ["m1"]

    def test_build_metrics_df_computes_expected_aggregations(self, cfg: dict) -> None:
        transformer = Transactions(pl.DataFrame(), cfg)
        source = pl.DataFrame(
            {
                # Four observations make q25 and q75 distinct while keeping
                # both the mean and the median equal to 6_000.
                "code_postal": ["75001", "75001", "75001", "75001", "75002"],
                "type_bien_principal": [
                    "Appartement",
                    "Appartement",
                    "Appartement",
                    "Appartement",
                    "Maison",
                ],
                "annee": [2024, 2024, 2024, 2024, 2024],
                "id_mutation": ["m1", "m2", "m3", "m4", "m5"],
                "prix_m2": [4_000.0, 5_000.0, 7_000.0, 8_000.0, 4_000.0],
                "valeur_fonciere": [
                    200_000.0,
                    250_000.0,
                    350_000.0,
                    400_000.0,
                    400_000.0,
                ],
                "surface_totale": [50.0, 50.0, 50.0, 50.0, 100.0],
            }
        )

        result = transformer.build_metrics_df(
            source,
            transformer._cfg["MARKET_METRICS"],
        )
        row = result.filter(
            (pl.col("code_postal") == "75001")
            & (pl.col("type_bien_principal") == "Appartement")
            & (pl.col("annee") == 2024)
        ).row(0, named=True)

        assert row["nb_transactions"] == 4
        assert row["prix_m2_mean"] == pytest.approx(6_000.0)
        assert row["prix_m2_median"] == pytest.approx(6_000.0)
        assert row["valeur_fonciere_mean"] == pytest.approx(300_000.0)
        assert row["surface_totale_mean"] == pytest.approx(50.0)

        # build_metrics_df uses Polars' default "nearest" interpolation.
        assert row["prix_m2_q25"] == pytest.approx(5_000.0)
        assert row["prix_m2_q75"] == pytest.approx(7_000.0)

    def test_build_gold_transaction_df_end_to_end(
        self, dvf_silver_df: pl.DataFrame, cfg: dict
    ) -> None:
        transformer = Transactions(dvf_silver_df, cfg)

        gold_transactions, market_metrics = transformer.build_gold_transaction_df()

        assert gold_transactions.height == 4  # m_bad is filtered out.
        assert gold_transactions["id_mutation"].n_unique() == 4
        assert set(gold_transactions["type_bien_principal"].to_list()) == {
            "Appartement",
            "Maison",
            "Mixte",
            "Complex",
        }
        m1 = gold_transactions.filter(pl.col("id_mutation") == "m1").row(0, named=True)
        assert m1["surface_totale"] == pytest.approx(50.0)
        assert m1["prix_m2"] == pytest.approx(6_000.0)
        assert market_metrics.height > 0


class TestEquipments:
    def test_create_equipement_count_applies_hierarchy_rules(
        self, equipment_silver_df: pl.DataFrame, cfg: dict
    ) -> None:
        transformer = Equipments(equipment_silver_df, cfg)

        result = transformer.create_equipement_count(transformer._cfg["COUNT"])

        health_detail = result.filter(
            (pl.col("code_postal") == "75001")
            & (pl.col("classification_n1") == "Santé et action sociale")
            & (pl.col("classification_n2") != "Total")
        ).row(0, named=True)
        health_total = result.filter(
            (pl.col("code_postal") == "75001")
            & (pl.col("classification_n2") == "Total")
        ).row(0, named=True)
        public_service = result.filter(
            (pl.col("code_postal") == "75001")
            & (pl.col("classification_n2") == "Services publics")
        ).row(0, named=True)

        assert health_detail["nb_sante"] == 10
        assert health_total["nb_sante"] == 0
        assert public_service["nb_services_publics"] == 3

    def test_aggregate_equipement_count_sums_by_year_and_postcode(
        self, equipment_silver_df: pl.DataFrame, cfg: dict
    ) -> None:
        transformer = Equipments(equipment_silver_df, cfg)
        counted = transformer.create_equipement_count(transformer._cfg["COUNT"])

        result = transformer.aggregate_equipement_count(counted, transformer._cfg)
        row = result.filter(pl.col("code_postal") == "75001").row(0, named=True)

        assert result.height == 2
        assert row["nb_sante"] == 10
        assert row["nb_enseignement"] == 8
        assert row["nb_services_publics"] == 3
        assert row["nb_transports"] == 6
        assert row["nb_commerces"] == 4
        assert row["nb_sports_loisirs"] == 2

    def test_create_equipement_score_normalizes_within_each_year(
        self,
        aggregated_equipment_df: pl.DataFrame,
        cfg: dict,
    ) -> None:
        transformer = Equipments(pl.DataFrame(), cfg)

        result = transformer.create_equipement_score(
            aggregated_equipment_df,
            transformer._cfg,
        ).sort(["annee", "code_postal"])

        assert result["score_equipements"].to_list() == [0.0, 100.0, 0.0]
        assert not any(column.endswith("_score") for column in result.columns)

    def test_build_gold_equipement_df_end_to_end(
        self, equipment_silver_df: pl.DataFrame, cfg: dict
    ) -> None:
        transformer = Equipments(equipment_silver_df, cfg)

        result = transformer.build_gold_equipement_df().sort("code_postal")

        assert result.height == 2
        assert result["score_equipements"].to_list() == [0.0, 100.0]
        assert result.select(transformer._cfg["EQUIP_COLS"]).null_count().sum_horizontal().item() == 0
