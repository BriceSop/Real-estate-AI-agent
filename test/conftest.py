from pathlib import Path

import duckdb
import pytest


@pytest.fixture
def duckdb_path(tmp_path: Path) -> Path:
    """Create a small isolated DuckDB database for repository tests."""
    database_path = tmp_path / "analytics_test.duckdb"

    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE market_metrics (
                code_postal VARCHAR,
                type_bien_principal VARCHAR,
                annee INTEGER,
                nb_transactions INTEGER,
                prix_m2_mean DOUBLE,
                prix_m2_median DOUBLE,
                prix_m2_q25 DOUBLE,
                prix_m2_q75 DOUBLE,
                valeur_fonciere_median DOUBLE,
                surface_totale_median DOUBLE
            )
            """
        )
        connection.execute(
            """
            INSERT INTO market_metrics VALUES
                ('75015', 'Appartement', 2022, 2, 9000, 8800, 8000, 9700, 440000, 50),
                ('75015', 'Appartement', 2023, 3, 9450, 9200, 8300, 10100, 460000, 51),
                ('75015', 'Appartement', 2024, 4, 9900, 9680, 8800, 10700, 490000, 52),
                ('75011', 'Appartement', 2024, 3, 10500, 10300, 9300, 11200, 515000, 49),
                ('75012', 'Appartement', 2024, 2, 9300, 9100, 8500, 9900, 455000, 50),
                ('75015', 'Maison', 2024, 1, 11200, 11000, 10500, 11700, 950000, 86)
            """
        )

        connection.execute(
            """
            CREATE TABLE transactions (
                id_mutation VARCHAR,
                date_mutation DATE,
                code_postal VARCHAR,
                type_bien_principal VARCHAR,
                surface_reelle_bati DOUBLE,
                nombre_pieces_principales INTEGER,
                valeur_fonciere DOUBLE,
                prix_m2 DOUBLE
            )
            """
        )
        connection.execute(
            """
            INSERT INTO transactions VALUES
                ('M001', '2024-02-01', '75015', 'Appartement', 45, 2, 500000, 10000),
                ('M001', '2024-02-01', '75015', 'Appartement', 5, NULL, 500000, 10000),
                ('M002', '2024-05-01', '75015', 'Appartement', 60, 3, 570000, 9500),
                ('M003', '2023-07-15', '75015', 'Appartement', 40, 2, 360000, 9000),
                ('M004', '2024-06-20', '75011', 'Appartement', 50, 2, 530000, 10600),
                ('M005', '2024-08-10', '75015', 'Maison', 90, 5, 990000, 11000),
                ('M006', '2024-09-10', '75015', 'Appartement', 55, 3, 520000, NULL)
            """
        )

        connection.execute(
            """
            CREATE TABLE location_features (
                annee INTEGER,
                code_postal VARCHAR,
                nb_sante INTEGER,
                nb_enseignement INTEGER,
                nb_services_publics INTEGER,
                nb_transports INTEGER,
                nb_commerces INTEGER,
                nb_sports_loisirs INTEGER,
                score_equipements DOUBLE,
                surface_km2 DOUBLE
            )
            """
        )
        connection.execute(
            """
            INSERT INTO location_features VALUES
                (2024, '75015', 120, 90, 35, 80, 210, 75, 72.5, 8.48),
                (2024, '75011', 100, 70, 28, 95, 240, 65, 76.0, 3.67)
            """
        )

    return database_path
