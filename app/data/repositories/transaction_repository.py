from datetime import date
from pathlib import Path

import duckdb
import polars as pl
from dataclasses import dataclass

@dataclass(frozen=True)
class ComparableMetrics:
    transaction_count: int
    median_price_m2: float
    q1_price_m2: float
    q3_price_m2: float

class TransactionRepository:
    """
    Repository providing read access to real-estate transactions.

    The ``transactions`` view contains one row per property belonging to a
    mutation. Consequently, several rows can share the same ``id_mutation``.

    This repository is responsible only for querying DuckDB. Business rules
    and interpretation logic must be implemented in the service layer.
    """

    def __init__(self, database_path: Path) -> None:
        """
        Initialize the transaction repository.

        Parameters
        ----------
        database_path : Path
            Path to the DuckDB database file.
        """
        self.database_path = database_path

    def find_transactions(
        self,
        postal_code: str,
        property_type: str,
        start_date: date,
        end_date: date,
        min_surface: float | None = None,
        max_surface: float | None = None,
        limit: int = 100,
    ) -> pl.DataFrame:
        """
        Retrieve property records matching the requested filters.

        Surface filters are applied only when their corresponding values
        are provided. Several rows may belong to the same mutation.

        Parameters
        ----------
        postal_code : str
            Postal code used to filter transactions.
        property_type : str
            Principal property type, such as ``Appartement`` or ``Maison``.
        start_date : date
            First transaction date included in the result.
        end_date : date
            Last transaction date included in the result.
        min_surface : float | None, default=None
            Minimum built surface in square metres.
        max_surface : float | None, default=None
            Maximum built surface in square metres.
        limit : int, default=100
            Maximum number of property records returned.

        Returns
        -------
        pl.DataFrame
            Matching property records ordered from the most recent to the
            oldest. An empty DataFrame is returned when no record matches.

        Raises
        ------
        ValueError
            If the date range, surface range or result limit is invalid.
        """
        self._validate_date_range(start_date, end_date)
        self._validate_surface_range(min_surface, max_surface)
        self._validate_limit(limit)

        conditions = [
            "code_postal = ?",
            "type_bien_principal = ?",
            "date_mutation BETWEEN ? AND ?",
            "prix_m2 IS NOT NULL",
        ]

        parameters: list[object] = [
            postal_code,
            property_type,
            start_date,
            end_date,
        ]

        if min_surface is not None:
            conditions.append("surface_reelle_bati >= ?")
            parameters.append(min_surface)

        if max_surface is not None:
            conditions.append("surface_reelle_bati <= ?")
            parameters.append(max_surface)

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                id_mutation,
                date_mutation,
                code_postal,
                type_bien_principal,
                surface_reelle_bati,
                nombre_pieces_principales,
                valeur_fonciere,
                prix_m2
            FROM transactions
            WHERE {where_clause}
            ORDER BY
                date_mutation DESC,
                id_mutation
            LIMIT ?
        """

        parameters.append(limit)

        with duckdb.connect(
            str(self.database_path),
            read_only=True,
        ) as connection:
            return connection.execute(
                query,
                parameters,
            ).pl()

    def get_comparable_metrics(
            self,
            postal_code: str,
            property_type: str,
            start_date: date,
            end_date: date,
            min_surface: float | None = None,
            max_surface: float | None = None,
            rooms: int | None = None
        ) -> ComparableMetrics:
            """
            Compute price metrics for records with similar characteristics to those entered.

            Parameters
            ----------
            postal_code : str
                Postal code used to filter transactions.
            property_type : str
                Principal property type, such as ``Appartement`` or ``Maison``.
            start_date : date
                First transaction date included in the result.
            end_date : date
                Last transaction date included in the result.
            min_surface : float | None, default=None
                Minimum built surface in square metres.
            max_surface : float | None, default=None
                Maximum built surface in square metres.
            room : int | None, default=None
                Number of rooms.

            Returns
            -------
            ComprableMetrics
                A classes containing price properties coming from comparables rows.

            Raises
            ------
            ValueError
                If the date range, surface range or result limit is invalid.
            """
            self._validate_date_range(start_date, end_date)
            self._validate_surface_range(min_surface, max_surface)

            query = """
                SELECT
                    COUNT(DISTINCT id_mutation),
                    MEDIAN(prix_m2),
                    QUANTILE_CONT(prix_m2, 0.25),
                    QUANTILE_CONT(prix_m2, 0.75)
                FROM transactions
                WHERE code_postal = ?
                  AND type_bien_principal = ?
                  AND date_mutation BETWEEN ? AND ?
                  AND surface_reelle_bati BETWEEN ? AND ?
                  AND nombre_pieces_principales = ? 
            """

            parameters = [
                postal_code,
                property_type,
                start_date,
                end_date,
                min_surface,
                max_surface,
                rooms
            ]

            with duckdb.connect(
                str(self.database_path),
                read_only=True,
            ) as connection:
                row = connection.execute(query,parameters,).fetchone()
            
            if row is None:
                return None

            return ComparableMetrics(
                transaction_count=row[0],
                median_price_m2=row[1],
                q1_price_m2=row[2],
                q3_price_m2=row[3],
            )
    
    def get_transaction_by_id(
        self,
        transaction_id: str,
    ) -> pl.DataFrame:
        """
        Retrieve all property records belonging to one mutation.

        Since one mutation can contain several properties, this method can
        return multiple rows for the same ``id_mutation``.

        Parameters
        ----------
        transaction_id : str
            Identifier of the requested mutation.

        Returns
        -------
        pl.DataFrame
            All property records associated with the mutation. An empty
            DataFrame is returned when the identifier does not exist.

        Raises
        ------
        ValueError
            If ``transaction_id`` is empty.
        """
        if not transaction_id.strip():
            raise ValueError(
                "transaction_id ne doit pas être vide."
            )

        query = """
            SELECT
                id_mutation,
                date_mutation,
                code_postal,
                type_bien_principal,
                surface_reelle_bati,
                nombre_pieces_principales,
                valeur_fonciere,
                prix_m2
            FROM transactions
            WHERE id_mutation = ?
            ORDER BY
                type_bien_principal,
                surface_reelle_bati DESC NULLS LAST
        """

        with duckdb.connect(
            str(self.database_path),
            read_only=True,
        ) as connection:
            return connection.execute(
                query,
                [transaction_id],
            ).pl()

    def count_transactions(
        self,
        postal_code: str,
        property_type: str,
        start_date: date,
        end_date: date,
    ) -> int:
        """
        Count distinct real-estate mutations for a market segment.

        The count uses ``COUNT(DISTINCT id_mutation)`` because several
        property records can belong to the same mutation.

        Parameters
        ----------
        postal_code : str
            Postal code used to filter transactions.
        property_type : str
            Principal property type.
        start_date : date
            First transaction date included in the count.
        end_date : date
            Last transaction date included in the count.

        Returns
        -------
        int
            Number of distinct matching mutations. Zero is returned when
            no transaction matches the filters.

        Raises
        ------
        ValueError
            If ``start_date`` is later than ``end_date``.
        """
        self._validate_date_range(start_date, end_date)

        query = """
            SELECT
                COUNT(DISTINCT id_mutation) AS transaction_count
            FROM transactions
            WHERE code_postal = ?
              AND type_bien_principal = ?
              AND date_mutation BETWEEN ? AND ?
        """

        parameters = [
            postal_code,
            property_type,
            start_date,
            end_date,
        ]

        with duckdb.connect(
            str(self.database_path),
            read_only=True,
        ) as connection:
            row = connection.execute(
                query,
                parameters,
            ).fetchone()

        return int(row[0])


    @staticmethod
    def _validate_date_range(
        start_date: date,
        end_date: date,
    ) -> None:
        """
        Validate that a date range is chronologically consistent.

        Raises
        ------
        ValueError
            If ``start_date`` is later than ``end_date``.
        """
        if start_date > end_date:
            raise ValueError(
                "start_date doit être antérieure ou égale à end_date."
            )

    @staticmethod
    def _validate_surface_range(
        min_surface: float | None,
        max_surface: float | None,
    ) -> None:
        """
        Validate optional surface boundaries.

        Raises
        ------
        ValueError
            If a surface is negative or if the minimum surface is greater
            than the maximum surface.
        """
        if min_surface is not None and min_surface < 0:
            raise ValueError(
                "min_surface doit être positive ou nulle."
            )

        if max_surface is not None and max_surface < 0:
            raise ValueError(
                "max_surface doit être positive ou nulle."
            )

        if (
            min_surface is not None
            and max_surface is not None
            and min_surface > max_surface
        ):
            raise ValueError(
                "min_surface doit être inférieure ou égale à max_surface."
            )

    @staticmethod
    def _validate_limit(limit: int) -> None:
        """
        Validate the maximum number of rows returned.

        Raises
        ------
        ValueError
            If ``limit`` is not strictly positive.
        """
        if limit <= 0:
            raise ValueError(
                "limit doit être strictement positif."
            )