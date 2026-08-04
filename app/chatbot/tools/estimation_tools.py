from datetime import date
from typing import Literal

from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.services.estimation_service import EstimationService


class EstimatePropertyInput(BaseModel):
    """Input for estimation queries"""
    postal_code: str = Field(
        pattern=r"^750(?:0[1-9]|1[0-9]|20)$",
        description="Code postal parisien du bien à estimer."
    )

    property_type: Literal["Appartement", "Maison"] = Field(
        description="Type de bien immobilier."
    )

    reference_date: date = Field(
        ge=date(2020, 7, 1),
        le=date(2025, 6, 30),
        description="Date de référence pour l'estimation, "
        "comprise entre le 1er juillet 2020 et le 30 juin 2025, "
        "au format YYYY-MM-DD."
    )

    surface: float = Field(
        gt=0,
        description="Surface du bien immobilier en mètres carrés."
    )
    rooms: int | None = Field(
        default=None,
        ge=1,
        description="Nombre de pièces du bien immobilier."
    )

def create_estimation_tools(estimation_service: EstimationService) -> list[BaseTool]:

    @tool(args_schema=EstimatePropertyInput)
    def estimate_property(
        postal_code: str,
        property_type: str,
        reference_date: date,
        surface: float,
        rooms: int | None = None
    ) -> dict:
        """Estime le prix d'un appartement ou d'une maison à partir de sa zone,
           de sa surface, de son nombre de pièces éventuel et d'une date de référence.

           Retourne le prix estimé, une borne basse, une borne haute et le nombre
           de transactions utilisées. L'estimation repose sur les transactions
           des trois mois précédant la date de référence.

           Utilise cet outil lorsqu'un utilisateur souhaite estimer la valeur
           d'un bien immobilier.
           """

        return estimation_service.estimate_property(
            postal_code=postal_code,
            property_type=property_type,
            reference_date=reference_date,
            surface=surface,
            rooms=rooms
        )

    return [estimate_property]