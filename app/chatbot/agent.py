from typing import Any

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from app.chatbot.tools.estimation_tools import create_estimation_tools
from app.chatbot.tools.location_tools import create_location_tools
from app.chatbot.tools.market_tools import create_market_tools
from app.core.paths import LOCAL_DIR
from app.data.repositories.location_repository import LocationRepository
from app.data.repositories.market_repository import MarketRepository
from app.data.repositories.transaction_repository import TransactionRepository
from app.services.estimation_service import EstimationService
from app.services.location_service import LocationService
from app.services.market_service import MarketService


def build_agent() -> Any:
    """Construct and build the real estate AI Agent."""
    db_path = LOCAL_DIR / "analytics.duckdb"

    location_repository = LocationRepository(db_path)
    market_repository = MarketRepository(db_path)
    transaction_repository = TransactionRepository(db_path)

    estimation_service = EstimationService(transaction_repository)
    location_service = LocationService(location_repository)
    market_service = MarketService(market_repository)

    estimation_tools = create_estimation_tools(estimation_service)
    market_tools = create_market_tools(market_service)
    location_tools = create_location_tools(location_service)

    SYSTEM_PROMPT = "Tu peux répondre aux questions de l'utilisateur en utilisant les outils\n" \
                    "mis à ta disposition pour :\n" \
                    "- analyser le marché immobilier ;\n" \
                    "- consulter des statistiques de prix ;\n" \
                    "- rechercher des informations sur les transactions ;\n" \
                    "- fournir des informations géographiques ;\n" \
                    "- estimer la valeur d'un bien.\n" \
                    "N'invente jamais de prix ou de statistiques.\n" \
                    "Explique clairement les limites des données et des estimations.\n" \
                    "Réponds de manière concise et accessible.\n" \
                    "Ne demande les paramètres nécessaires à un outil que lorsque\n" \
                    "l'utilisateur exprime réellement une intention correspondant à cet outil.\n" \
                    "Pour une salutation ou une question générale sur tes capacités,\n" \
                    "réponds directement sans appeler d'outil."

    model = ChatOllama(
        model="qwen3:4b",
        temperature=0,
        num_ctx=4096,
    )

    tools = estimation_tools + market_tools + location_tools

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent

if __name__ == "__main__":
    agent = build_agent()
    example_query = (
        "Bonjour, que peux-tu faire ?"
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": example_query,
                }
            ]
        }
    )

    print(result["messages"][-1].content)
    print(result["messages"][-2].name)