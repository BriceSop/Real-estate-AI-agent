from app.data.repositories.market_repository import MarketRepository
from app.services.market_service import MarketService


def test_market_overview_workflow_with_real_duckdb(duckdb_path):
    repository = MarketRepository(duckdb_path)
    service = MarketService(repository)

    result = service.get_market_overview(
        postal_code="75015",
        property_type="Appartement",
        year=2024,
    )

    assert result["status"] == "success"
    assert result["code_postal"] == "75015"
    assert result["annee"] == 2024
    assert result["nb_transactions"] == 4
    assert result["prix_m2"]["mediane"] == 9680.0
