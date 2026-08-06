from typing import Any

from fastapi import HTTPException, Request, status


def get_agent(request: Request) -> Any:
    """Récupère l'agent chargé au démarrage de l'application."""

    agent = getattr(request.app.state, "agent", None)

    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="L'agent immobilier est indisponible.",
        )

    return agent