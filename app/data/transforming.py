import polars as pl
from loguru import logger

class Transforming:
    """
    Transforming pipeline, to pass data quality from silver to gold.

    Treatments are separated considering the source data, using child classes: Transactions and Equipments.
    """
    def __init__(self, df: pl.DataFrame, cfg: dict):
        """
        Initialization function of the Transforming class.
        
        Arguments
        ----------
        df : pl.DataFrame
            Raw data to preprocess.
        cfg : dict
            Cleaning configuration.
        """
        self.df = df.clone()
        self.cfg = cfg

    
class Transactions(Transforming):
    """
    Class containing methods to obtain gold data from the silver dvf data.

    Inheriting from the Transforming class.
    """
    def __init__(self, df: pl.DataFrame, cfg: dict):
        """
        Initialization function of the Transactions class.
        
        Arguments
        ----------
        df : pl.DataFrame
            Raw data to preprocess.
        cfg : dict
            Transforming configuration.
        """
        super().__init__(df, cfg)
        self._cfg = cfg["TRANSACTIONS"]

    def set_date_column_format(self) -> "Transactions":
        """
        Function that set the right format for the date column in the silver dvf.

        Returns
        -------
        self : Transactions
            The same object, with self.df updated.
        """
        self.df = self.df.with_columns(
            pl.col("date_mutation")
            .str.to_date(format="%Y-%m-%d", strict=True)
            .alias("date_mutation")
        )

        return self
    
    def prepare_prix_m2_df(self, cfg: dict) -> pl.DataFrame:
        """
        Function that build an intermediate dataframe to get the price per m².

        Arguments
        -------
        cfg : dict
            Configuration dictionary.

        Returns
        -------
        prix_m2_df : pl.DataFrame
            Intermediate DataFrame used to create the price per m² column.
        """
        prix_m2_df = (
            self.df
            .group_by(self._cfg["KEY"], 
                       maintain_order=True
            )
            .agg(pl.col(cfg["VF"]),
                 pl.sum(cfg["SR"])
                 .alias(cfg["ST"])
            )
            .with_columns(pl.col(cfg["VF"])
                          .list.unique()
                          .list.item()
                          .alias(cfg["VF"])
            )
        )
        
        return prix_m2_df
    
    def build_prix_m2_column(self, prix_m2_df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
        """
        Function that build the price per m² column.

        Arguments
        -------
        prix_m2_df : pl.DataFrame
            Intermediate DataFrame used to create the price per m² column.
        cfg : dict
            Configuration dictionary.

        Returns
        -------
        df_to_join : pl.DataFrame
            DataFrame containing the price per m² column, ready to be merged.
        """
        df_to_join = prix_m2_df.with_columns(
            pl.when(pl.col(cfg["ST"]) != 0)
            .then(pl.col(cfg["VF"])/pl.col(cfg["ST"]))
            .otherwise(None)
            .alias(cfg["NEW_COL"])
        )

        return df_to_join
    
    def prepare_type_biens_df(self, cfg: dict) -> pl.DataFrame:
        """
        Function that build an intermediate dataframe to get the principal type of property.

        Arguments
        -------
        cfg : dict
            Configuration dictionary.

        Returns
        -------
        type_biens_df : pl.DataFrame
            Intermediate DataFrame used to create new columns related to the property type.
        """
        type_biens_df = (
            self.df
            .group_by(self._cfg["KEY"], 
                       maintain_order=True
            )
            .agg(pl.col(cfg["TYPE"])
            )
            .with_columns(pl.col(cfg["TYPE"])
                          .list.len()
                          .alias(cfg["NOMBRE"])
                          ,
                          pl.col(cfg["TYPE"])
                          .list.unique()
                          .alias(cfg["TYPE_UNIQUE"])
            )
        )
        
        return type_biens_df
    
    def create_flag_for_type_biens(self, type_biens_df : pl.DataFrame, cfg : dict) -> pl.DataFrame:
        """
        Function that build flag columns to indicate the type of property.

        Arguments
        -------
        type_biens_df : pl.DataFrame
            Intermediate DataFrame used to create new columns related to the property type.
        cfg : dict
            Configuration dictionary.
        
        Returns
        -------
        df : pl.DataFrame
            DataFrame containing the new flag columns.
        """
        df = type_biens_df.with_columns(
            pl.when(pl.col(val[0]).list.contains(val[1]))
            .then(True)
            .otherwise(False)
            .alias(col)
            for col, val in cfg.items()
            )
        
        return df
    
    def create_type_bien_princ(self, type_biens_df : pl.DataFrame, cfg : dict) -> pl.DataFrame:
        """
        Function that build a column to indicate the principal type of property for each transactions.

        Arguments
        -------
        type_biens_df : pl.DataFrame
            Intermediate DataFrame used to create new columns related to the property type.
        cfg : dict
            Configuration dictionary.
        
        Returns
        -------
        df : pl.DataFrame
            DataFrame containing the new column.
        """
        df = type_biens_df.with_columns(
                pl.when(
                    (pl.col(cfg["TYPE_UNIQUE"]).list.contains(cfg["APPT"]["KEY"])) 
                    &
                    (pl.col(cfg["TYPE_UNIQUE"]).list.eval(pl.element().is_in(cfg["APPT"]["VAL"])).list.all())

                )
                .then(pl.lit("Appartement"))

                .when(
                    (pl.col(cfg["TYPE_UNIQUE"]).list.contains(cfg["MAI"]["KEY"]))
                    &
                    (pl.col(cfg["TYPE_UNIQUE"]).list.eval(pl.element().is_in(cfg["MAI"]["VAL"])).list.all())
                )
                .then(pl.lit("Maison"))

                .when(
                    (pl.col(cfg["TYPE_UNIQUE"]).list.contains(cfg["APPT"]["KEY"]))
                    &
                    (pl.col(cfg["TYPE_UNIQUE"]).list.contains(cfg["MAI"]["KEY"]))
                    &
                    (pl.col(cfg["TYPE_UNIQUE"]).list.eval(pl.element().is_in(cfg["MIX"]["VAL"])).list.all())
                )
                .then(pl.lit("Mixte"))

                .when(
                    (pl.col(cfg["TYPE_UNIQUE"]).list.contains(cfg["CPLX"]["KEY"]))
                )
                .then(pl.lit(cfg["CPLX"]["VAL"]))

                .otherwise(pl.lit("Autre"))
                .alias(cfg["TYPE_PRINC"])
            )
        
        return df
    
    def create_year_column(self) -> "Transactions":
        """
        Function that build a column to indicate the year of each transactions.
        
        Returns
        -------
        df : Transactions
            The same object, with self.df updated.
        """
        self.df = self.df.with_columns(pl.col("date_mutation")
                             .dt.year()
                             .cast(pl.Int64)
                             .alias("annee")
                            )
        
        return self
    
    def join_to_initial_df(self, l_df: pl.DataFrame, r_df: pl.DataFrame, l_key: list, r_key: list) -> pl.DataFrame:
        """
        Function that join a dataframe to the original data.

        Arguments
        -------
        l_df : pl.DataFrame
            Left dataframe.
        r_df : pl.DataFrame
            Right dataframe.
        l_key : list
            Left key(s) for jointure.
        r_key : list
            Right key(s) for jointure.

        Returns
        -------
        merged_df : pl.DataFrame
            Final dataframe with new columns added.
        """
        merged_df = l_df.join(r_df,
                              left_on=l_key,
                              right_on=r_key,
                              how="left",
                              validate="m:1")
        
        return merged_df
    
    def arrange_transactions_df(self, transaction_df: pl.DataFrame) -> pl.DataFrame:
        """
        Function that arrange the transaction data into its final format.

        Arguments
        -------
        transaction_df : pl.DataFrame
            Transaction dataframe.
    
        Returns
        -------
        final_df : pl.DataFrame
            Final dataframe with the right configuration.
        """
        # Filtering the data
        final_df = transaction_df.filter(
        (pl.col("is_valeur_fonciere_missing") == False) 
        & 
        (pl.col("is_surface_missing") == False)
        &
        (pl.col("is_transaction_atypique_agent") == False)
        )

        # Keeping relevant columns only
        final_df = final_df[self._cfg["COLS_TO_KEEP"]]

        #Dropping duplicates
        final_df = final_df.unique(subset=self._cfg["KEY"], 
                                   maintain_order=True)
        
        return final_df
    
    def build_metrics_df(self, transaction_df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
        """
        Function that build the market metrics dataframe.

        Arguments
        -------
        transaction_df : pl.DataFrame
            Transaction dataframe.
        cfg : dict
            Configuration dictionary.

        Returns
        -------
        market_df : pl.DataFrame
            Dataframe containing metrics related to the real estate market.
        """
        market_df = (
            transaction_df
            .group_by(["code_postal","type_bien_principal","annee"])
            .agg(
                *[pl.count(col).alias(new_col)
                 for col, new_col in cfg["COUNT"].items()]
                ,
                *[pl.mean(col).alias(new_col) 
                 for col, new_col in cfg["MEAN"].items()]
                ,
                *[pl.median(col).alias(new_col)
                 for col, new_col in cfg["MEDIAN"].items()]
                ,
                *[pl.quantile(val[1], val[0]).alias(new_col)
                 for new_col, val in cfg["QUANTILE"].items()]
                )
            )

        return market_df

    def build_gold_transaction_df(self) -> pl.DataFrame:
        """
        Function that build the 2 gold dataframes based on the dvf transactions.

        Returns
        -------
        self.gold_tr : pl.DataFrame
            Gold dataframe containing informations about transactions.
        self.gold_mm : pl.DataFrame
            Gold dataframe containing market metrics.
        """
        logger.info("Démarrage de l'upgrade dvf vers gold")


        # Setting the right date format and creating year column
        logger.info("Gestion des colonnes de dates")
        self.set_date_column_format()
        self.create_year_column()

        # Preparing the dataframe containing price per m²
        logger.info("Préparation du dataframe de prix par m²")
        prix_m2_df = self.prepare_prix_m2_df(self._cfg["PRIX_M2"])
        prix_m2_df = self.build_prix_m2_column(prix_m2_df, self._cfg["PRIX_M2"])

        # Preparing the dataframe containing columns related to the type of property
        logger.info("Préparation du dataframe de type de biens")
        biens_df = self.prepare_type_biens_df(self._cfg["TYPE_BIENS"])
        biens_df = self.create_flag_for_type_biens(biens_df, self._cfg["TYPE_BIENS"]["FLAG_TYPE_BIENS"])
        biens_df = self.create_type_bien_princ(biens_df, self._cfg["TYPE_BIENS"])

        # Adding price per m² to the initial data
        logger.info("Jointure pour ajouter la colonne de prix au m²")
        drop_cols = self._cfg["PRIX_M2"]["VF"]
        self.gold_tr = self.join_to_initial_df(self.df, 
                                               prix_m2_df.drop(drop_cols), 
                                               self._cfg["KEY"],
                                               self._cfg["KEY"])
        
        # Adding new columns linked to type of property to the initial data
        logger.info("Jointure pour ajouter les colonnes en lien avec le type de bien")
        drop_cols = [self._cfg["TYPE_BIENS"]["TYPE"], self._cfg["TYPE_BIENS"]["TYPE_UNIQUE"]]
        self.gold_tr = self.join_to_initial_df(self.gold_tr, 
                                               biens_df.drop(drop_cols), 
                                               self._cfg["KEY"],
                                               self._cfg["KEY"])
        
        logger.info("Construction des dataframes gold issus de Dvf")
        # Building gold transactions data
        self.gold_tr = self.arrange_transactions_df(self.gold_tr)

        # Building gold market metrics data
        self.gold_mm = self.build_metrics_df(self.gold_tr, self._cfg["MARKET_METRICS"])

        logger.info("Fin de l'upgrade dvf vers gold")

        return self.gold_tr, self.gold_mm




class Equipments(Transforming):
    """
    Class containing methods to obtain gold data from the silver bpe data.

    Inheriting from the Transforming class.
    """
    def __init__(self, df: pl.DataFrame, cfg: dict):
        """
        Initialization function of the Equipments class.
        
        Arguments
        ----------
        df : pl.DataFrame
            Raw data to preprocess.
        cfg : dict
            Transforming configuration.
        """
        super().__init__(df, cfg)
        self._cfg = cfg["EQUIPEMENTS"]

    def create_equipement_count(self, cfg: dict) -> pl.DataFrame:
        """
        Function that build columns about the number of equipments for certain type.

        Arguments
        -------
        cfg : dict
            Configuration dictionary.

        Returns
        -------
        equipement_df : pl.DataFrame
            Dataframe containing new columns.
        """
        equipement_df = self.df.with_columns(
            *[pl.when((pl.col(cfg["CN1"]) == val) 
                    &
                    (pl.col(cfg["CN2"]) != cfg["FIXED_VAL"])
                    &
                    (pl.col(cfg["CN3"]) != cfg["FIXED_VAL"])
                    )
            .then(pl.col(cfg["INIT_COL"]))
            .otherwise(pl.lit(0))
            .alias(col)
            for col, val in cfg["NEW_COLS"].items()]
            ,
            *[pl.when((pl.col(cfg["CN2"]) == val)
                    &
                    (pl.col(cfg["CN3"]) != cfg["FIXED_VAL"])
                    )
            .then(cfg["INIT_COL"])
            .otherwise(pl.lit(0))
            .alias(col)
            for col, val in cfg["EXCEPTION"].items()]
        )

        return equipement_df
    
    def aggregate_equipement_count(self, equipement_df: pl.DataFrame, cfg: dict):
        """
        Function that aggregate the number of equipement for the selected type.

        Arguments
        -------
        equipement_df : pl.DataFrame
            Equipment dataframe.
        cfg : dict
            Configuration dictionary.

        Returns
        -------
        equipement_df : pl.DataFrame
            Dataframe updated.
        """
        df = (equipement_df
              .group_by(cfg["KEY"])
              .agg(
                  pl.sum(col)
                  for col in cfg["EQUIP_COLS"]
                  )
            )
        
        return df
    
    def create_equipement_score(self, equipement_df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
        """
        Function that creates a column giving a score depending of the equipment repartition in each zone.

        Arguments
        -------
        equipement_df : pl.DataFrame
            Equipment dataframe.
        cfg : dict
            Configuration dictionary.

        Returns
        -------
        df : pl.DataFrame
            Dataframe updated.
        """
        df = (equipement_df
              .with_columns([
                  (
                      (pl.col(col) - pl.col(col).min().over("annee"))
                      / (pl.col(col).max().over("annee") - pl.col(col).min().over("annee"))
                  )
              .fill_nan(0)
              .alias(f"{col}_score")
              for col in cfg["EQUIP_COLS"]
            ])
              .with_columns(
                (
                    100 * sum(
                        pl.col(f"{col}_score") * weight
                        for col, weight in cfg["SCORE_WEIGHTS"].items()
                    )
                )
                .round(2)
                .alias("score_equipements")
            )
              .drop([f"{col}_score" for col in cfg["EQUIP_COLS"]])
        )

        return df
    
    def build_gold_equipement_df(self) -> pl.DataFrame:
         """
        Function that build the gold dataframe based on the Bpe transactions.

        Returns
        -------
        self.gold_lf : pl.DataFrame
            Gold dataframe containing informations about equipments.
        """
         logger.info("Démarrage de l'upgrade bpe vers gold")

         # Creating aggregation columns related to the number of certain types of equipments
         logger.info("Création de colonnes calculant le nombre d'équipement pour certains services")
         equipement_df = self.create_equipement_count(self._cfg["COUNT"])
         equipement_df = self.aggregate_equipement_count(equipement_df, self._cfg)

         # Adding an equipment score column
         logger.info("Ajout d'une colonne de score d'équipement")
         self.gold_lf = self.create_equipement_score(equipement_df, self._cfg)

         logger.info("Fin de l'upgrade bpe vers gold")

         return self.gold_lf