from loguru import logger
from pathlib import Path
import polars as pl

def load_data_from_csv(dir_name: Path = None, file_name: str = None, sep: str = None, utf: bool = True) -> pl.DataFrame:
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
                             infer_schema_length=0,
                             encoding="windows-1252"
                            )
            logger.debug("Fichier lu avec succès : {} lignes, {} colonnes", df.height, df.width)
            return df

    except Exception:
        logger.exception("Erreur lors du chargement du fichier {}", path)
        raise
