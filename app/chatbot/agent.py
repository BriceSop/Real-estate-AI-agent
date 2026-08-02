from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from app.services.data.repositories.estimation_repository import EstimationRepository
from app.services.location_service import LocationService
from app.services.market_service import MarketService

from app.services.estimation_service import EstimationService
from app.services.location_service import LocationService
from app.services.market_service import MarketService

from app.chatbot.tools.estimation_tools import create_estimation_tools
from app.chatbot.tools.location_tools import create_location_tools
from app.chatbot.tools.market_tools import create_market_tools


model = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    num_ctx=4096,
)

tools = [
    *create_market_tools(market_service),
    *create_location_tools(location_service),
    *create_estimation_tools(estimation_service),
]

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)