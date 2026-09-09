import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_agent
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

AgentDependency = Annotated[Any, Depends(get_agent)]


def extract_answer(result: Any) -> str:
    """Extract an answer from Agent's result."""

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        # AgentExecutor LangChain classique
        for key in ("output", "answer", "response"):
            value = result.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        # Compatibilité éventuelle avec un agent basé sur des messages
        messages = result.get("messages")

        if isinstance(messages, list) and messages:
            content = getattr(messages[-1], "content", None)

            if isinstance(content, str) and content.strip():
                return content.strip()

    raise ValueError(
        "Le résultat retourné par l'agent ne contient pas de réponse textuelle."
    )


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Interroger l'agent immobilier",
)
def ask_agent(
    payload: ChatRequest,
    agent: AgentDependency,
) -> ChatResponse:
    """Send a question to the real estate Agent."""

    try:
        result = agent.invoke({"input": payload.message})
        answer = extract_answer(result)

        return ChatResponse(answer=answer)

    except ValueError as exc:
        logger.exception("Format de réponse invalide retourné par l'agent.")

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="L'agent a retourné une réponse invalide.",
        ) from exc

    except Exception as exc:
        logger.exception("Erreur pendant l'exécution de l'agent immobilier.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossible de générer une réponse pour le moment.",
        ) from exc