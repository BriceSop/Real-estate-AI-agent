from app.core.logging import setup_logging
from app.core.paths import CONFIGS_DIR, GOLD_DIR, SILVER_DIR
from app.data.transforming import Equipments, Transactions
from app.data.utils import (
                            build_polars_schema,
                            load_data_from_csv,
                            load_yaml,
                            save_data_to_csv,
)

setup_logging()

# Loading transforming config file
config = load_yaml(CONFIGS_DIR / "transforming.yaml")
print(config.keys())

# Preparing data schema
dvf_sch = build_polars_schema(config["TRANSACTIONS"]["SCHEMA"])
bpe_sch = build_polars_schema(config["EQUIPEMENTS"]["SCHEMA"])

# Loading silver data
dvf_df = load_data_from_csv(SILVER_DIR, 
                            config["TRANSACTIONS"]["SILVER_FILE"], 
                            sep=",", 
                            sch=dvf_sch)
bpe_df = load_data_from_csv(SILVER_DIR, 
                            config["EQUIPEMENTS"]["SILVER_FILE"], 
                            sep=",",
                            sch=bpe_sch)

# Building gold data
transac = Transactions(dvf_df, config)
gold_tr, gold_mm = transac.build_gold_transaction_df()

equip = Equipments(bpe_df, config)
gold_lf = equip.build_gold_equipement_df()

# Saving upgraded data to gold folder
save_data_to_csv(gold_tr, GOLD_DIR, transac._cfg["GOLD_FILE_TR"])
save_data_to_csv(gold_mm, GOLD_DIR, transac._cfg["GOLD_FILE_MM"])
save_data_to_csv(gold_lf, GOLD_DIR, equip._cfg["GOLD_FILE_LF"])