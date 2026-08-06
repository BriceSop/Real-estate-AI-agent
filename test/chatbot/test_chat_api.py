from typing import Any

from fastapi.testclient import TestClient

from app.main import create_app


class FakeAgent:
    """Agent simulé utilisé uniquement dans les tests."""

    def invoke(self, payload: dict[str, str]) -> dict[str, str]:
        question = payload["input"]

        return {
            "output": f"Réponse simulée à la question : {question}"
        }


def fake_agent_factory() -> Any:
    return FakeAgent()


def test_chat_returns_agent_answer() -> None:
    app = create_app(agent_factory=fake_agent_factory)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Quel est le prix médian dans le 15e arrondissement ?"
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": (
            "Réponse simulée à la question : "
            "Quel est le prix médian dans le 15e arrondissement ?"
        )
    }


def test_chat_rejects_empty_message() -> None:
    app = create_app(agent_factory=fake_agent_factory)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"message": "  "},
        )

    assert response.status_code == 422