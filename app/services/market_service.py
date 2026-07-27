from collections.abc import Sequence
from typing import Any

import polars as pl

from app.data.repositories.market_repository import MarketRepository


class MarketService:
    """
    Service implementing real-estate market analysis use cases.

    The service retrieves aggregated data through ``MarketRepository`` and
    transforms it into structured results that can be exposed to an AI agent.
    """

    def __init__(self, market_repository: MarketRepository) -> None:
        """
        Initialize the market service.

        Parameters
        ----------
        market_repository : MarketRepository
            Repository providing access to aggregated market data.
        """
        self.market_repository = market_repository

    def get_market_overview(
        self,
        postal_code: str,
        property_type: str,
        year: int,
    ) -> dict[str, Any]:
        """
        Return the market overview of one area for a given year.

        Parameters
        ----------
        postal_code : str
            Postal code identifying the requested area.
        property_type : str
            Principal property type.
        year : int
            Year used to retrieve market metrics.

        Returns
        -------
        dict[str, Any]
            Structured market metrics, or a ``not_found`` result when no
            matching data exists.
        """
        metrics = self.market_repository.get_market_metrics(
            postal_code=postal_code,
            property_type=property_type,
            year=year,
        )

        if metrics is None:
            return self._not_found(
                "Aucune métrique trouvée pour cette demande."
            )

        price_iqr = None

        if (
            metrics.q25_price_m2 is not None
            and metrics.q75_price_m2 is not None
        ):
            price_iqr = (
                metrics.q75_price_m2
                - metrics.q25_price_m2
            )

        return {
            "status": "success",
            "code_postal": metrics.postal_code,
            "type_bien_principal": metrics.property_type,
            "annee": metrics.year,
            "nb_transactions": metrics.transaction_count,
            "prix_m2": {
                "moyenne": self._round_optional(
                    metrics.mean_price_m2
                ),
                "mediane": self._round_optional(
                    metrics.median_price_m2
                ),
                "q25": self._round_optional(
                    metrics.q25_price_m2
                ),
                "q75": self._round_optional(
                    metrics.q75_price_m2
                ),
                "ecart_interquartile": self._round_optional(
                    price_iqr
                ),
            },
            "valeur_fonciere_mediane": self._round_optional(
                metrics.median_vf
            ),
            "surface_totale_mediane": self._round_optional(
                metrics.median_surface,
                digits=1,
            ),
        }

    def get_price_evolution(
        self,
        postal_code: str,
        property_type: str,
        start_year: int,
        end_year: int,
    ) -> dict[str, Any]:
        """
        Calculate price evolution between the first and last available year.

        The calculation uses the first and last observations returned within
        the requested period. The actual years used are included in the result.

        Parameters
        ----------
        postal_code : str
            Postal code identifying the requested area.
        property_type : str
            Principal property type.
        start_year : int
            First requested year.
        end_year : int
            Last requested year.

        Returns
        -------
        dict[str, Any]
            Price evolution, source values and annual history.

        Raises
        ------
        ValueError
            If ``start_year`` is greater than or equal to ``end_year``.
        """
        if start_year >= end_year:
            raise ValueError(
                "start_year doit être strictement inférieur à end_year."
            )

        history = self.market_repository.get_price_history(
            postal_code=postal_code,
            property_type=property_type,
            start_year=start_year,
            end_year=end_year,
        )

        if history.is_empty():
            return self._not_found(
                "Aucune donnée disponible sur la période demandée."
            )

        if history.height < 2:
            return {
                "status": "insufficient_data",
                "message": (
                    "Au moins deux années sont nécessaires pour calculer "
                    "une évolution."
                ),
                "historique": history.to_dicts(),
            }

        first_row = history.row(0, named=True)
        last_row = history.row(history.height - 1, named=True)

        early_mean = first_row["prix_m2_mean"]
        early_median = first_row["prix_m2_median"]
        latest_mean = last_row["prix_m2_mean"]
        latest_median = last_row["prix_m2_median"]

        if early_median is None or latest_median is None:
            return {
                "status": "insufficient_data",
                "message": (
                    "Les prix médians nécessaires au calcul sont manquants."
                ),
                "historique": history.to_dicts(),
            }

        if early_median <= 0:
            return {
                "status": "invalid_data",
                "message": (
                    "Le prix médian initial doit être strictement positif."
                ),
            }

        evolution_median = (
            (latest_median / early_median) - 1
        ) * 100

        evolution_mean = None

        if (
            early_mean is not None
            and latest_mean is not None
            and early_mean > 0
        ):
            evolution_mean = (
                (latest_mean / early_mean) - 1
            ) * 100

        return {
            "status": "success",
            "code_postal": postal_code,
            "type_bien_principal": property_type,
            "annee_debut_effective": first_row["annee"],
            "annee_fin_effective": last_row["annee"],
            "prix_m2_initial": {
                "moyenne": self._round_optional(early_mean),
                "mediane": self._round_optional(early_median),
            },
            "prix_m2_final": {
                "moyenne": self._round_optional(latest_mean),
                "mediane": self._round_optional(latest_median),
            },
            "evolution_pourcentage": {
                "moyenne": self._round_optional(
                    evolution_mean,
                    digits=2,
                ),
                "mediane": round(evolution_median, 2),
            },
            "historique": history.to_dicts(),
        }

    def compare_areas(
        self,
        postal_codes: Sequence[str],
        property_type: str,
        year: int,
    ) -> dict[str, Any]:
        """
        Compare market metrics between several areas.

        Areas are primarily ranked by median price per square metre.

        Parameters
        ----------
        postal_codes : Sequence[str]
            Postal codes of the areas to compare.
        property_type : str
            Principal property type.
        year : int
            Year used for the comparison.

        Returns
        -------
        dict[str, Any]
            Full ranking, key leaders and missing requested areas.

        Raises
        ------
        ValueError
            If fewer than two postal codes are provided.
        """
        unique_postal_codes = list(dict.fromkeys(postal_codes))

        if len(unique_postal_codes) < 2:
            raise ValueError(
                "Au moins deux codes postaux sont nécessaires."
            )

        result = self.market_repository.get_metrics_for_areas(
            postal_codes=unique_postal_codes,
            property_type=property_type,
            year=year,
        )

        if result.is_empty():
            return self._not_found(
                "Aucune donnée disponible pour les zones demandées."
            )

        result = result.with_columns(
            (
                pl.col("prix_m2_q75")
                - pl.col("prix_m2_q25")
            ).alias("prix_m2_iqr")
        )

        price_ranking = (
            result
            .filter(pl.col("prix_m2_median").is_not_null())
            .sort("prix_m2_median", descending=True)
        )

        if price_ranking.is_empty():
            return {
                "status": "insufficient_data",
                "message": (
                    "Les prix médians sont absents pour les zones trouvées."
                ),
                "zones": result.to_dicts(),
            }

        most_expensive = price_ranking.row(0, named=True)
        least_expensive = price_ranking.row(
            price_ranking.height - 1,
            named=True,
        )

        returned_postal_codes = set(
            result["code_postal"].to_list()
        )

        missing_postal_codes = [
            postal_code
            for postal_code in unique_postal_codes
            if postal_code not in returned_postal_codes
        ]

        return {
            "status": "success",
            "annee": year,
            "type_bien_principal": property_type,
            "zone_plus_chere": {
                "code_postal": most_expensive["code_postal"],
                "prix_m2_median": most_expensive["prix_m2_median"],
            },
            "zone_moins_chere": {
                "code_postal": least_expensive["code_postal"],
                "prix_m2_median": least_expensive["prix_m2_median"],
            },
            "ecart_prix_m2": round(
                most_expensive["prix_m2_median"]
                - least_expensive["prix_m2_median"],
                2,
            ),
            "classement": price_ranking.to_dicts(),
            "codes_postaux_absents": missing_postal_codes,
        }

    @staticmethod
    def _not_found(message: str) -> dict[str, str]:
        """Build a standardized result for unavailable data."""
        return {
            "status": "not_found",
            "message": message,
        }

    @staticmethod
    def _round_optional(
        value: float | None,
        digits: int = 2,
    ) -> float | None:
        """Round an optional numeric value."""
        if value is None:
            return None

        return round(float(value), digits)