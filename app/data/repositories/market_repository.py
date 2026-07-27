import polars as pl
from pathlib import Path
import duckdb


from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl


@dataclass(frozen=True)
class MarketMetrics:
    postal_code: str
    property_type: str
    year: int
    transaction_count: int
    mean_price_m2: float | None
    median_price_m2: float | None
    q25_price_m2: float | None
    q75_price_m2: float | None
    median_vf: float | None
    median_surface: float | None


class MarketRepository:
    """
    Repository providing read access to aggregated real-estate market data.

    The repository executes parameterized queries against the DuckDB
    `market_metrics` view. It does not contain business rules or market
    interpretation logic.
    """

    def __init__(self, database_path: Path) -> None:
        """
        Initialize the market repository.

        Parameters
        ----------
        database_path : Path
            Path to the DuckDB database file.
        """
        self.database_path = database_path

    def get_market_metrics(
        self,
        postal_code: str,
        property_type: str,
        year: int,
    ) -> MarketMetrics | None:
        """
        Retrieve aggregated market metrics for one area and one year.

        Parameters
        ----------
        postal_code : str
            Paris postal code used to identify the area.
        property_type : str
            Principal property type, such as ``Appartement`` or ``Maison``.
        year : int
            Transaction year to retrieve.

        Returns
        -------
        MarketMetrics | None
            Aggregated metrics for the requested market segment, or ``None``
            when no matching observation exists.
        """

        query = """
            SELECT
                code_postal,
                type_bien_principal,
                annee,
                nb_transactions,
                prix_m2_mean,
                prix_m2_median,
                prix_m2_q25,
                prix_m2_q75,
                valeur_fonciere_median,
                surface_totale_median
            FROM market_metrics
            WHERE code_postal = ?
              AND type_bien_principal = ?
              AND annee = ?
        """

        parameters = [
            postal_code,
            property_type,
            year,
        ]

        with duckdb.connect(
            str(self.database_path),
            read_only=True,
        ) as connection:
            row = connection.execute(query, parameters).fetchone()

        if row is None:
            return None

        return MarketMetrics(
            postal_code=row[0],
            property_type=row[1],
            year=row[2],
            transaction_count=row[3],
            mean_price_m2=row[4],
            median_price_m2=row[5],
            q25_price_m2=row[6],
            q75_price_m2=row[7],
            median_vf=row[8],
            median_surface=row[9]
        )

    def get_price_history(
        self,
        postal_code: str,
        property_type: str,
        start_year: int,
        end_year: int,
    ) -> pl.DataFrame:
        """
        Retrieve the annual price history for a market segment.

        Parameters
        ----------
        postal_code : str
            Paris postal code used to identify the area.
        property_type : str
            Principal property type.
        start_year : int
            First year included in the result.
        end_year : int
            Last year included in the result.

        Returns
        -------
        pl.DataFrame
            Annual market metrics ordered chronologically. An empty DataFrame
            is returned when no matching data exists.

        Raises
        ------
        ValueError
            If ``start_year`` is greater than ``end_year``.
        """

        if start_year > end_year:
            raise ValueError(
                "start_year doit être inférieur ou égal à end_year."
            )

        query = """
            SELECT
                code_postal,
                type_bien_principal,
                annee,
                nb_transactions,
                prix_m2_mean,
                prix_m2_median,
                prix_m2_q25,
                prix_m2_q75
            FROM market_metrics
            WHERE code_postal = ?
              AND type_bien_principal = ?
              AND annee BETWEEN ? AND ?
            ORDER BY annee ASC
        """

        parameters = [
            postal_code,
            property_type,
            start_year,
            end_year,
        ]

        with duckdb.connect(
            str(self.database_path),
            read_only=True,
        ) as connection:
            return connection.execute(query, parameters).pl()

    def get_metrics_for_areas(
        self,
        postal_codes: Sequence[str],
        property_type: str,
        year: int,
    ) -> pl.DataFrame:
        """
        Retrieve market metrics for several areas in a given year.

        Parameters
        ----------
        postal_codes : Sequence[str]
            Postal codes of the areas to compare.
        property_type : str
            Principal property type.
        year : int
            Transaction year used for the comparison.

        Returns
        -------
        pl.DataFrame
            One row per matching area, ordered by median price per square metre.

        Raises
        ------
        ValueError
            If ``postal_codes`` is empty.
        """

        if not postal_codes:
            raise ValueError(
                "postal_codes doit contenir au moins un code postal."
            )

        placeholders = ", ".join("?" for _ in postal_codes)

        query = f"""
            SELECT
                code_postal,
                type_bien_principal,
                annee,
                nb_transactions,
                prix_m2_mean,
                prix_m2_median,
                prix_m2_q25,
                prix_m2_q75,
                valeur_fonciere_median,
                surface_totale_median
            FROM market_metrics
            WHERE code_postal IN ({placeholders})
              AND type_bien_principal = ?
              AND annee = ?
            ORDER BY prix_m2_median DESC
        """

        parameters = [
            *postal_codes,
            property_type,
            year,
        ]

        with duckdb.connect(
            str(self.database_path),
            read_only=True,
        ) as connection:
            return connection.execute(query, parameters).pl()