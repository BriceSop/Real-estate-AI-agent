import duckdb
from app.core.paths import LOCAL_DIR

connection = duckdb.connect(LOCAL_DIR / "analytics.duckdb")

connection.execute("""
    CREATE OR REPLACE VIEW transactions AS
    SELECT *
    FROM read_csv('data/gold/transactions.csv')
""")

connection.execute("""
    CREATE OR REPLACE VIEW market_metrics AS
    SELECT *
    FROM read_csv('data/gold/market_metrics.csv')
""")

connection.execute("""
    CREATE OR REPLACE VIEW location_features AS
    SELECT *
    FROM read_csv('data/gold/location_features.csv')
""")

connection.close()