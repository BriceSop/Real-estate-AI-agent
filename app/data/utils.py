from loguru import logger
from pathlib import Path
import polars as pl
import yaml

def load_yaml(filepath: str) -> dict:
    """
    Function that load a yaml file as a dictionary.
    
    Arguments
    ----------
    filepath : str
        Path of the file.

    Raises
    ------
    Logger exception.

    Returns
    -------
    config : dict
        Yaml as a dictionary.
    """
    try:
        logger.info("Lecture du fichier yaml : {}", filepath)
        with open(filepath, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            logger.debug("Fichier lu avec succès")
            return config
        
    except Exception:
        logger.exception("Erreur lors du chargement du fichier {}", filepath)
        raise

def load_data_from_csv(dir_name: Path = None, file_name: str = None, sep: str = None, utf: bool = True, sch: dict = None) -> pl.DataFrame:
    """
    Function used to load data from a csv file.
    
    Arguments
    ----------
    dir_name : str
        Directory name of the file.
    file_name : str
        Name of the file.
    sep : str
        Separator of the csv file.
    utf : bool
        Indicate the csv is in utf-8 format.
    sch : dict
        Contains the data schema.

    Raises
    ------
    Logger exception.

    Returns
    -------
    df: pl.DataFrame
        Data loaded as a Dataframe.
    """
    try:
        path = dir_name / file_name
        if utf:
            if sch is None:
                logger.info("Lecture du fichier CSV : {}", path)
                df = pl.read_csv(path,
                                separator=sep,
                                infer_schema_length=0
                                )
                logger.debug("Fichier lu avec succès : {} lignes, {} colonnes", df.height, df.width)
                return df
            
            else:
                logger.info("Lecture du fichier CSV : {}", path)
                df = pl.read_csv(path,
                                separator=sep,
                                schema=sch
                                )
                logger.debug("Fichier lu avec succès : {} lignes, {} colonnes", df.height, df.width)
                return df
        
        else:
            logger.info("Lecture du fichier CSV : {}", path)
            df = pl.read_csv(path,
                             separator=sep,
                             infer_schema_length=0,
                             encoding="windows-1252"
                            )
            logger.debug("Fichier lu avec succès : {} lignes, {} colonnes", df.height, df.width)
            return df

    except Exception:
        logger.exception("Erreur lors du chargement du fichier {}", path)
        raise

def save_data_to_csv(data: pl.DataFrame, dir_path: Path, filename: str):
    """
    Function used to save data as a csv file.
    
    Arguments
    ----------
    data : pl.DataFrame
        Data to save.
    dir_path : str
        Location of the directory used to store the csv file.
    filename : str
        Name of the file.

    Raises
    ------
    e : Exception
    """
    try:
        path = dir_path / filename
        logger.info("Sauvegarde des données en fichier CSV : {}", path)
        data.write_csv(path)
        logger.debug("{} Data successfully saved at {}", filename, dir_path)
    
    except Exception:
        logger.exception("Erreur lors de la sauvegarde du fichier {}", path)
        raise

def build_polars_schema(schema_config: dict[str, str]) -> pl.Schema:
    """
    Function used to create a data schema compatible with polars.
    
    Arguments
    ----------
    schema_config : dict
        Schema configuration to apply.

    Return
    ------
    pl.Schema
    """
    POLARS_DTYPES = {
    "String": pl.String,
    "Int8": pl.Int8,
    "Int16": pl.Int16,
    "Int32": pl.Int32,
    "Int64": pl.Int64,
    "Float32": pl.Float32,
    "Float64": pl.Float64,
    "Boolean": pl.Boolean,
    "Date": pl.Date,
    "Datetime": pl.Datetime,
    }

    unknown_types = {
        dtype
        for dtype in schema_config.values()
        if dtype not in POLARS_DTYPES
    }

    if unknown_types:
        raise ValueError(
            f"Types Polars non supportés dans le YAML : {unknown_types}"
        )

    return pl.Schema({
        column: POLARS_DTYPES[dtype]
        for column, dtype in schema_config.items()
    })