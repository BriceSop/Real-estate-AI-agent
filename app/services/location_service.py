from typing import Any

from app.data.repositories.location_repository import LocationRepository


class LocationService:
    """
    Service implementing location-related use cases.

    The service retrieves aggregated location data through
    ``LocationRepository`` and transforms it into structured results
    that can be exposed to an AI agent.
    """

    EQUIPMENT_COLUMNS = (
        "nb_sante",
        "nb_enseignement",
        "nb_services_publics",
        "nb_transports",
        "nb_commerces",
        "nb_sports_loisirs",
    )

    SMALL_AREA_THRESHOLD = 2.0
    LARGE_AREA_THRESHOLD = 4.3

    LOW_EQUIPMENT_SCORE_THRESHOLD = 30.0
    HIGH_EQUIPMENT_SCORE_THRESHOLD = 60.0

    def __init__(
        self,
        location_repository: LocationRepository,
    ) -> None:
        """
        Initialize the location service.

        Parameters
        ----------
        location_repository : LocationRepository
            Repository providing access to aggregated location data.
        """
        self.location_repository = location_repository

    def get_area_overview(
        self,
        postal_code: str,
    ) -> dict[str, Any]:
        """
        Return an overview of the location features of one area.

        Parameters
        ----------
        postal_code : str
            Postal code identifying the requested area.

        Returns
        -------
        dict[str, Any]
            Structured location overview, or a ``not_found`` result when
            no matching profile exists.
        """
        profile = self.location_repository.get_area_profile(
            postal_code=postal_code,
        )

        if profile is None:
            return {
                "status": "not_found",
                "message": (
                    "Aucun profil géographique trouvé pour cette zone."
                ),
            }

        surface_km2 = profile.get("surface_km2")
        equipment_score = profile.get("score_equipements")

        if surface_km2 is None or equipment_score is None:
            return {
                "status": "invalid_data",
                "message": (
                    "Le profil ne contient pas la superficie ou le score "
                    "d'équipements attendu."
                ),
            }

        equipment_counts = self._extract_equipment_counts(profile)

        top_equipment_categories = sorted(
            equipment_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]

        return {
            "status": "success",
            "code_postal": profile.get("code_postal", postal_code),
            "annee": profile.get("annee"),
            "principales_categories_equipements": [
                {
                    "categorie": category,
                    "valeur": value,
                }
                for category, value in top_equipment_categories
            ],
            "repartition_equipements": equipment_counts,
            "superficie": {
                "valeur_km2": round(float(surface_km2), 2),
                "categorie": self._evaluate_surface(
                    float(surface_km2)
                ),
            },
            "score_equipements": {
                "valeur": round(float(equipment_score), 2),
                "niveau": self._evaluate_equipment_score(
                    float(equipment_score)
                ),
            },
        }

    def _extract_equipment_counts(
        self,
        profile: dict[str, Any],
    ) -> dict[str, float]:
        """
        Extract valid equipment-category values from an area profile.

        Parameters
        ----------
        profile : dict[str, Any]
            Raw profile returned by the repository.

        Returns
        -------
        dict[str, float]
            Available numeric equipment counts.
        """
        equipment_counts: dict[str, float] = {}

        for column in self.EQUIPMENT_COLUMNS:
            value = profile.get(column)

            if isinstance(value, (int, float)):
                equipment_counts[column] = float(value)

        return equipment_counts

    @classmethod
    def _evaluate_surface(
        cls,
        surface_km2: float,
    ) -> str:
        """
        Categorize an area according to its surface.

        Parameters
        ----------
        surface_km2 : float
            Area surface in square kilometres.

        Returns
        -------
        str
            ``Petite``, ``Moyenne`` or ``Grande``.
        """
        if surface_km2 < cls.SMALL_AREA_THRESHOLD:
            return "Petite"

        if surface_km2 <= cls.LARGE_AREA_THRESHOLD:
            return "Moyenne"

        return "Grande"

    @classmethod
    def _evaluate_equipment_score(
        cls,
        equipment_score: float,
    ) -> str:
        """
        Categorize an equipment score.

        Parameters
        ----------
        equipment_score : float
            Equipment score expected to be between 0 and 100.

        Returns
        -------
        str
            ``Faible``, ``Moyen`` or ``Élevé``.
        """
        if equipment_score < cls.LOW_EQUIPMENT_SCORE_THRESHOLD:
            return "Faible"

        if equipment_score < cls.HIGH_EQUIPMENT_SCORE_THRESHOLD:
            return "Moyen"

        return "Élevé"
