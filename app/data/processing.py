import polars as pl

class Cleaning:
    """
    Cleaning pipeline, to pass data quality from bronze to silver.

    All transformations are applied in place.
    Treatments are separated considering the source data, using child classes: Dvf and Bpe.
    """
    def __init__(self, df: pl.DataFrame, cfg: dict):
        """
        Initialization function of the Cleaning class.
        
        Arguments
        ----------
        df : pl.DataFrame
            Raw data to preprocess.
        cfg : dict
            Cleaning configuration.
        """
        self.df = df.clone()
        self.cfg = cfg

    
class Dvf(Cleaning):
    """
    Class containing special treatments for the raw DVF dataframe.

    Inheriting from the Cleaning class.
    """
    def __init__(self, df: pl.DataFrame, cfg: dict):
        """
        Initialization function of the Dvf class.
        
        Arguments
        ----------
        df : pl.DataFrame
            Raw data to preprocess.
        cfg : dict
            Cleaning configuration.
        """
        super().__init__(df, cfg)
        self.dvf_cfg = cfg["DVF"]

    def remove_header_rows(self) -> "Dvf":
        """
        Function that remove rows containing duplicates of the columns' header.

        Returns
        -------
        self : Dvf
            The same object, with self.df updated.
        """
        # Remove duplicate header rows
        self.df = self.df.remove(self.df["date_mutation"].str.contains("[a-zA-Z]"))

        return self


class Bpe(Cleaning):
    """
    Class containing special treatments for the raw BPE dataframe.

    Inheriting from the Cleaning class.
    """
    def __init__(self, df: pl.DataFrame, cfg: dict):
        """
        Initialization function of the Bpe class.
        
        Arguments
        ----------
        df : pl.DataFrame
            Raw data to preprocess.
        cfg : dict
            Cleaning configuration.
        """
        super().__init__(df, cfg)
        self.bpe_cfg = cfg["BPE"]