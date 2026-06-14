from app.core.paths import BRONZE_DIR, CONFIGS_DIR, SILVER_DIR
from app.core.logging import setup_logging
from app.data.utils import load_yaml, load_data_from_csv, save_data_to_csv
from app.data.cleaning import Bpe, Dvf

setup_logging()

# Loading cleaning config file
config = load_yaml(CONFIGS_DIR / "cleaning.yaml")
print(config.keys())

# Loading raw data
dvf_df = load_data_from_csv(BRONZE_DIR, config["DVF"]["FILENAME"], sep=",")
bpe_df = load_data_from_csv(BRONZE_DIR, config["BPE"]["FILENAME"], sep=";")
dom_codes_df = load_data_from_csv(BRONZE_DIR, config["BPE"]["DOM_CODES"]["FILENAME"], sep=";", utf=False)
sdom_codes_df = load_data_from_csv(BRONZE_DIR, config["BPE"]["SDOM_CODES"]["FILENAME"], sep=";", utf=False)
type_codes_df = load_data_from_csv(BRONZE_DIR, config["BPE"]["TYPES_CODES"]["FILENAME"], sep=";", utf=False)

# DVF data cleaning
dvf = Dvf(dvf_df, config)
dvf.run_dvf_to_silver()

# BPE data cleaning
bpe = Bpe(bpe_df, config)
bpe.run_bpe_to_silver(dom_codes_df, sdom_codes_df, type_codes_df)

# Saving upgraded data to silver folder
save_data_to_csv(dvf.df, SILVER_DIR, dvf._cfg["SILVER_FILE"])
save_data_to_csv(bpe.df, SILVER_DIR, bpe._cfg["SILVER_FILE"])