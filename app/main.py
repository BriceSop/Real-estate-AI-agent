from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.chatbot.agent import build_agent

AgentFactory = Callable[[], Any]


def create_app(
    agent_factory: AgentFactory = build_agent,
) -> FastAPI:
    """Crée et configure l'application FastAPI."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Exécuté une seule fois au démarrage du serveur.
        app.state.agent = agent_factory()

        yield

        # Exécuté lors de l'arrêt du serveur.
        app.state.agent = None

    application = FastAPI(
        title="Real Estate AI Agent API",
        description="API d'interrogation d'un agent immobilier basé sur les données DVF.",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    application.include_router(
        chat_router,
        prefix="/api/v1",
    )

    @application.get(
        "/health",
        tags=["Health"],
        summary="Vérifier l'état de l'API",
    )
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()