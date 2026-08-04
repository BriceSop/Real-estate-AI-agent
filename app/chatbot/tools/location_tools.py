from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.services.location_service import LocationService


class AreaOverviewInput(BaseModel):
    postal_code: str = Field(
        pattern=r"^750(?:0[1-9]|1[0-9]|20)$",
        description=(
            "Code postal de l'arrondissement parisien à analyser, "
            "compris entre 75001 et 75020."
        )
    )

def create_location_tools(location_service: LocationService) -> list[BaseTool]:

    @tool(args_schema=AreaOverviewInput)
    def get_area_overview(
        postal_code: str,
    ) -> dict:
        """Retourne des informations concernant les équipements d'une zone: 
           équipements les plus représentés, un score d'équipement, 
           la répartition entre les différentes catégories, et 
           la superficie de la zone.
           
           Utilise cet outil pou répondre aux questions concernant la
           manière dont un arrondissement est équipé.
           """

        return location_service.get_area_overview(postal_code=postal_code)

    return [get_area_overview]