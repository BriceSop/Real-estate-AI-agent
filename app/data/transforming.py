import polars as pl
from loguru import logger

class Transforming:
    """
    Transforming pipeline, to pass data quality from silver to gold.

    All transformations are applied in place.
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
    
    def prepare_prix_m2_df(self) -> pl.DataFrame:
        """
        Function that build an intermediate dataframe to get the price per m².

        Returns
        -------
        prix_m2_df : pl.DataFrame
            Intermediate DataFrame used to create the price per m² column.
        """
        prix_m2_df = (
            self.df
            .group_by(["id_mutation",
                       "numero_disposition"], 
                       maintain_order=True
            )
            .agg(pl.col("valeur_fonciere"),
                 pl.sum("surface_reelle_bati")
                 .alias("surface_totale")
            )
            .with_columns(pl.col("valeur_fonciere")
                          .list.unique()
                          .list.item()
                          .alias("valeur_fonciere")
            )
        )
        
        return prix_m2_df
    
    def build_prix_m2_column(self, prix_m2_df: pl.DataFrame) -> pl.DataFrame:
        """
        Function that build the price per m² column.

        Arguments
        -------
        prix_m2_df : pl.DataFrame
            Intermediate DataFrame used to create the price per m² column.

        Returns
        -------
        df_to_join : pl.DataFrame
            DataFrame containing the price per m² column, ready to be merged.
        """
        df_to_join = prix_m2_df.with_columns(
            pl.when(pl.col("surface_totale") != 0)
            .then(pl.col("valeur_fonciere")/pl.col("surface_totale"))
            .otherwise(None)
            .alias("prix_m2")
        )

        return df_to_join
    
    def prepare_type_biens_df(self) -> pl.DataFrame:
        """
        Function that build an intermediate dataframe to get the principal type of property.

        Returns
        -------
        type_biens_df : pl.DataFrame
            Intermediate DataFrame used to create new columns related to the property type.
        """
        type_biens_df = (
            self.df
            .group_by(["id_mutation",
                       "numero_disposition"], 
                       maintain_order=True
            )
            .agg(pl.col("type_local")
            )
            .with_columns(pl.col("type_local")
                          .list.len()
                          .alias("nb_biens")
                          ,
                          pl.col("type_local")
                          .list.unique()
                          .alias("type_biens_unique")
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
                    (pl.col("type_biens_unique").list.contains("Appartement")) 
                    &
                    (pl.col("type_biens_unique").list.eval(pl.element().is_in(cfg["APPARTEMENT"])).list.all())

                )
                .then(pl.lit("Appartement"))

                .when(
                    (pl.col("type_biens_unique").list.contains("Maison"))
                    &
                    (pl.col("type_biens_unique").list.eval(pl.element().is_in(cfg["MAISON"])).list.all())
                )
                .then(pl.lit("Maison"))

                .when(
                    (pl.col("type_biens_unique").list.contains("Appartement"))
                    &
                    (pl.col("type_biens_unique").list.contains("Maison"))
                    &
                    (pl.col("type_biens_unique").list.eval(pl.element().is_in(cfg["MIXTE"])).list.all())
                )
                .then(pl.lit("Mixte"))

                .when(
                    (pl.col("type_biens_unique").list.contains("Indisponible"))
                )
                .then(pl.lit("Complex"))

                .otherwise(pl.lit("Autre"))
                .alias("type_bien_principal")
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
    
    def arrange_transactions_df(self, transaction_df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
        """
        Function that arrange the transaction data into its final format.

        Arguments
        -------
        transaction_df : pl.DataFrame
            Transaction dataframe.
        cfg : dict
            Configuration dictionary.

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
        final_df = final_df[cfg["COLS_TO_KEEP"]]

        #Dropping duplicates
        final_df = final_df.unique(subset=["id_mutation","numero_disposition"], 
                                   maintain_order=True)
        
        return final_df
    
    def build_metrics_df(self, transaction_df: pl.DataFrame) -> pl.DataFrame:
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
                pl.count("id_mutation").alias("nb_transactions"),
                pl.mean("prix_m2").alias("prix_m2_mean"),
                pl.median("prix_m2").alias("prix_m2_median"),
                pl.quantile("prix_m2", 0.25).alias("prix_m2_q25"),
                pl.quantile("prix_m2", 0.75).alias("prix_m2_q75"),
                pl.median("valeur_fonciere").alias("valeur_fonciere_median"),
                pl.mean("surface_totale").alias("surface_mean"),
                pl.median("surface_totale").alias("surface_median")
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
        prix_m2_df = self.prepare_prix_m2_df()
        prix_m2_df = self.build_prix_m2_column(prix_m2_df)

        # Preparing the dataframe containing columns related to the type of property
        logger.info("Préparation du dataframe de type de biens")
        biens_df = self.prepare_type_biens_df()
        biens_df = self.create_flag_for_type_biens(biens_df, cfg = self.cfg["FLAG_TYPE_BIENS"])
        biens_df = self.create_type_bien_princ(biens_df, cfg = self.cfg["BIEN_PRINC"])

        # Adding price per m² to the initial data
        logger.info("Jointure pour ajouter la colonne de prix au m²")
        join_cfg = self.cfg["PRIX_M2_JOIN"]
        self.gold_tr = self.join_to_initial_df(self.df, 
                                               prix_m2_df.drop(join_cfg["DROP_COLS"]), 
                                               join_cfg["L_KEY"],
                                               join_cfg["R_KEY"])
        
        # Adding new columns linked to type of property to the initial data
        logger.info("Jointure pour ajouter les colonnes en lien avec le type de bien")
        join_cfg = self.cfg["TYPE_BIEN_JOIN"]
        self.gold_tr = self.join_to_initial_df(self.df, 
                                               biens_df.drop(join_cfg["DROP_COLS"]), 
                                               join_cfg["L_KEY"],
                                               join_cfg["R_KEY"])
        
        logger.info("Construction des dataframes gold issus de Dvf")
        # Building gold transactions data
        self.gold_tr = self.arrange_transactions_df(self.gold_tr, self.cfg["GOLD_TRANSACTIONS"])

        # Building gold market metrics data
        self.gold_mm = self.build_metrics_df(self.gold_tr)

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
        self._cfg = cfg["EQUIPMENTS"]

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
            [pl.when((pl.col("classification_n1") == col[0]) 
                    &
                    (pl.col("classification_n2") != cfg["VALUE"])
                    &
                    (pl.col("classification_n3") != cfg["VALUE"])
                    )
            .then(pl.col("nombre_equipements"))
            .otherwise(pl.lit(0))
            .alias(col[1])
            for col in cfg["COLS"]
            ],
            pl.when((pl.col("classification_n2") == cfg["EXCEPT"][0])
                    &
                    (pl.col("classification_n3") != cfg["VALUE"])
                    )
            .then(pl.col("nombre_equipements"))
            .otherwise(pl.lit(0))
            .alias(cfg["EXCEPT"][1])
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
              .group_by(cfg["GROUP_KEY"])
              .agg(
                  pl.sum(col)
                  for col in cfg["EQUIPEMENT_COLS"]
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
              for col in cfg.keys()
            ])
              .with_columns(
                (
                    100 * sum(
                        pl.col(f"{col}_score") * weight
                        for col, weight in cfg.items()
                    )
                )
                .round(2)
                .alias("score_equipements")
            )
              .drop([f"{col}_score" for col in cfg.keys()])
        )

        return df
    
    def buils_gold_equipement_df(self) -> pl.DataFrame:
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
         equipement_df = self.create_equipement_count(self.cfg["EQUIPEMENT_COUNT"])
         equipement_df = self.aggregate_equipement_count(equipement_df, self.cfg["EQUIPEMENT_COUNT"])

         # Adding an equipment score column
         logger.info("Ajout d'une colonne de score d'équipement")
         self.gold_lf = self.create_equipement_score(equipement_df, self.cfg["EQUIPEMENT_SCORE"])

         logger.info("Fin de l'upgrade bpe vers gold")

         return self.gold_lf