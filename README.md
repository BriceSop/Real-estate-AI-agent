# 🏠 Real Estate AI Agent

AI-powered assistant for exploring the **Paris real-estate market** using French open data.

The application combines a data pipeline, an analytical database and a local LLM agent to answer natural-language questions about property prices, market trends, neighborhoods and property valuation.

## Features

* 📊 Real-estate market analysis by Paris arrondissement
* 📈 Price evolution and district comparison
* 🏠 Property estimation based on comparable transactions
* 📍 Neighborhood analysis using local amenities
* 💬 Conversational interface
* 🔒 Local LLM execution with Ollama

## Architecture

```text
DVF + BPE Open Data
        │
        ▼
 Bronze → Silver → Gold
        │
      Polars
        │
        ▼
      DuckDB
        │
        ▼
Repositories → Services
        │
        ▼
  LangChain Tools
        │
        ▼
 Qwen3 / Ollama
        │
        ▼
     FastAPI
        │
        ▼
    Streamlit
```

The LLM does not directly query the database.
Market calculations and property estimations are handled by deterministic tools built on top of the service and repository layers.

## Project structure

```text
real-estate-ai-agent/
│
├── app/
│   ├── chatbot/        # AI agent and LangChain tools
│   ├── core/           # Configuration, paths and logging
│   ├── data/           # Data processing, repositories and DuckDB
│   ├── frontend/       # Streamlit interface
│   ├── services/       # Business logic and analytics
│   └── main.py         # FastAPI application
│
├── configs/            # Data and application configuration
├── data/               # Bronze, Silver and Gold datasets
├── scripts/            # Data pipeline execution scripts
├── test/               # Unit and integration tests
│
├── pyproject.toml
└── README.md
```

## Data

The project uses French public datasets:

* **DVF (Demandes de valeurs foncières)** — historical real-estate transactions
* **BPE (Base Permanente des Équipements)** — local amenities and services

The current scope focuses on residential properties in Paris.

## Tech stack

**Python 3.12 · Polars · DuckDB · LangChain · Ollama · FastAPI · Streamlit · Pydantic · Pytest · uv**

## Run locally

Install dependencies:

```bash
uv sync
```

Install the local LLM:

```bash
ollama pull qwen3:4b
```

Prepare the datasets:

```bash
uv run python -m scripts.build_silver
uv run python -m scripts.build_gold
uv run python -m app.data.loaders.duckdb_loader
```

Start the API:

```bash
uv run uvicorn app.main:app --reload
```

Start the frontend:

```bash
uv run streamlit run app/frontend/app.py
```

## Tests

```bash
uv run pytest
```

## Status

🚧 **Work in progress**

The core data pipeline, analytical layer, AI agent and API are implemented.
The Streamlit user interface is currently under development.
