from pathlib import Path
from typing import Any

import duckdb


class LocationRepository:
    """
    Repository providing read access to aggregated location features.

    The ``location_features`` view is expected to contain one aggregated
    area profile per postal code.

    This repository is responsible only for querying DuckDB. Business rules,
    scoring logic and interpretation of location features must be implemented
    in the service layer.
    """

    def __init__(self, database_path: Path) -> None:
        """
        Initialize the location repository.

        Parameters
        ----------
        database_path : Path
            Path to the DuckDB database file.
        """
        self.database_path = database_path

    def get_area_profile(
        self,
        postal_code: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve the aggregated location profile of one area.

        The returned dictionary uses the column names from the
        ``location_features`` view as keys.

        Parameters
        ----------
        postal_code : str
            Postal code identifying the requested area.

        Returns
        -------
        dict[str, Any] | None
            Aggregated location features for the requested postal code,
            or ``None`` when no matching area exists.

        Raises
        ------
        ValueError
            If ``postal_code`` is empty or does not contain exactly
            five digits.
        RuntimeError
            If several rows exist for the same postal code, even though
            the view is expected to contain one profile per area.
        """
        normalized_postal_code = self._validate_postal_code(postal_code)

        query = """
            SELECT *
            FROM location_features
            WHERE code_postal = ?
        """

        with duckdb.connect(
            str(self.database_path),
            read_only=True,
        ) as connection:
            cursor = connection.execute(
                query,
                [normalized_postal_code],
            )

            rows = cursor.fetchall()
            column_names = [
                column_description[0]
                for column_description in cursor.description
            ]

        if not rows:
            return None

        if len(rows) > 1:
            raise RuntimeError(
                "Plusieurs profils ont été trouvés pour le code postal "
                f"{normalized_postal_code}. La vue location_features doit "
                "contenir une seule ligne par code postal."
            )

        return dict(zip(column_names, rows[0], strict=True))

    @staticmethod
    def _validate_postal_code(postal_code: str) -> str:
        """
        Validate and normalize a postal code.

        Parameters
        ----------
        postal_code : str
            Postal code to validate.

        Returns
        -------
        str
            Postal code without leading or trailing whitespace.

        Raises
        ------
        ValueError
            If the postal code is empty or is not composed of exactly
            five digits.
        """
        normalized_postal_code = postal_code.strip()

        if not normalized_postal_code:
            raise ValueError(
                "postal_code ne doit pas être vide."
            )

        if (
            len(normalized_postal_code) != 5
            or not normalized_postal_code.isdigit()
        ):
            raise ValueError(
                "postal_code doit contenir exactement cinq chiffres."
            )

        return normalized_postal_code