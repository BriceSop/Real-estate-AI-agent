from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Corps JSON attendu par l'endpoint de chat."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    message: str = Field(
        ...,
        min_length=3,
        max_length=1_000,
        description="Question immobilière envoyée à l'agent",
        examples=[
            "Quel est le prix médian au m² dans le 15e arrondissement en 2024 ?"
        ],
    )


class ChatResponse(BaseModel):
    """Réponse renvoyée par l'agent."""

    answer: str