from typing import Literal

from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.services.market_service import MarketService


class MarketOverviewInput(BaseModel):
    """Input for market metrics queries"""
    postal_code: str = Field(
        pattern=r"^750(?:0[1-9]|1[0-9]|20)$",
        description="Code postal parisien du bien immobilier."
    )

    property_type: Literal["Appartement", "Maison"] = Field(
        description="Type de bien immobilier."
    )

    year: int = Field(
        ge=2020,
        le=2025,
        description="Année de référence utilisée pour filtrer les métriques immobilières,"
        "comprise entre 2020 et 2025, au format YYYY."
    )


class PriceEvolutionInput(BaseModel):
    """Input for price evolution queries"""
    postal_code: str = Field(
        pattern=r"^750(?:0[1-9]|1[0-9]|20)$",
        description="Code postal parisien du bien immobilier."
    )

    property_type: Literal["Appartement", "Maison"] = Field(
        description="Type de bien immobilier."
    )

    start_year: int = Field(
        ge=2020,
        le=2025,
        description="Année de départ pour le calcul d'évolution des prix, "
        "comprise entre 2020 et 2025, au format YYYY."
    )

    end_year: int = Field(
        ge=2020,
        le=2025,
        description="Année de fin pour le calcul d'évolution des prix, "
        "comprise entre 2020 et 2025, au format YYYY."
    )


class AreaComparisonInput(BaseModel):
    """Input for area comparison queries"""
    postal_codes: list[str] = Field(
        min_length=2,
        description="Liste de codes postaux parisiens."
    )

    property_type: Literal["Appartement", "Maison"] = Field(
        description="Type de bien immobilier."
    )

    year: int = Field(
        ge=2020,
        le=2025,
        description="Année de référence utilisée pour la comparaison,"
        "comprise entre 2020 et 2025, au format YYYY."
    )


def create_market_tools(market_service: MarketService) -> list[BaseTool]:

    @tool(args_schema=MarketOverviewInput)
    def get_market_overview(
        postal_code: str,
        property_type: str,
        year: int
    ) -> dict:
        """Retourne les principales métriques du marché immobilier pour une zone: 
           prix médian au m², quartiles, et nombre de transactions.

           Utilise cet outil pour répondre aux questions concernant les prix au m²,
           le volume de transactions ou la distribution des prix.
           """
        
        return market_service.get_market_overview(
            postal_code=postal_code,
            property_type=property_type,
            year=year
        )

    @tool(args_schema=PriceEvolutionInput)
    def get_price_evolution(
        postal_code: str,
        property_type: str,
        start_year: int,
        end_year: int
    ) -> dict:
        """Retourne les principales métriques d'évolution du prix au m² d'une zone 
           sur une certaine période: évolution du prix en %, prix au m² initial, 
           prix au m² final.
           
           Utilise cet outil pour répondre aux questions concernant l'évolution du
           prix au m².
           """
            
        return market_service.get_price_evolution(
            postal_code=postal_code,
            property_type=property_type,
            start_year=start_year,
            end_year=end_year
        )

    @tool(args_schema=AreaComparisonInput)
    def compare_areas(
        postal_codes: list[str],
        property_type: str,
        year: int
    ) -> dict:
        """Compare les principales métriques du marché immobilier entre plusieurs zones: 
           zone la plus chère, celle la moins chère, et écart de prix.
           
           Utilise cet outil pour répondre aux questions visant à comparer les prix
           entre plusieurs arrondissements.
           """
                
        return market_service.compare_areas(
            postal_codes=postal_codes,
            property_type=property_type,
            year=year
        )

    return [
        get_market_overview, 
        get_price_evolution, 
        compare_areas
    ]