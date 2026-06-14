import polars as pl
from loguru import logger
import requests
import time
import re
import unicodedata


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
    
    def casting_columns_types(self) -> "Dvf":
        """
        Function that cast columns in the right type.

        Returns
        -------
        self : Dvf
            The same object, with self.df updated.
        """
        # Casting the right type for each columns
        self.df = self.df.with_columns(
            pl.col(self.dvf_cfg["DATE_COLS"]).str.to_date("%Y-%m-%d"),
            pl.col(self.dvf_cfg["INT_COLS"]).cast(pl.Int64),
            pl.col(self.dvf_cfg["FLOAT_COLS"]).str.replace_all(",", ".").cast(pl.Float64),
        )

        return self

    def imputing_postal_code_null(self) -> "Dvf":
        """
        Function that impute null values for the postal code column.

        Returns
        -------
        self : Dvf
            The same object, with self.df updated.
        """
        # Using code_commune (no null values) to impute
        self.df = self.df.with_columns(
            pl.when(
                pl.col("code_postal").is_null()
                & pl.col("code_commune").is_not_null()
            )
            .then(
                pl.lit("750") + pl.col("code_commune").str.slice(-2)
            )
            .otherwise(pl.col("code_postal"))
            .alias("code_postal")
        )

        return self
    
    def reverse_geocode_fr(self, lon: float, lat: float, postcode: str = None, citycode: str = None) -> dict:
        """
        Function that uses a Geo API to return address informations using lontitude and latitude.

        Arguments
        ----------
        lon : float
            Longitude of the address.
        lat : float
            Latitude of the address.
        postcode : str
            Postal code of the address.
        citycode : str
            City code of the address.

        Returns
        -------
        dict
            Object containing address infos.
        """
        # Initializing parameters
        params = {
            "index": "address",
            "lon": lon,
            "lat": lat,
            "limit": 1,
        }

        if postcode is not None:
            params["postcode"] = str(postcode)
        if citycode is not None:
            params["citycode"] = str(citycode)

        # API call
        r = requests.get(self.dvf_cfg["API_URL"], params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        features = data.get("features", [])
        if not features:
            return None

        # Storing the properties needed
        props = features[0]["properties"]

        return {
            "full_address_geocoded": props.get("label"),
            "street_geocoded": props.get("street"),
            "housenumber_geocoded": props.get("housenumber"),
            "postcode_geocoded": props.get("postcode"),
            "city_geocoded": props.get("city"),
            "citycode_geocoded": props.get("citycode"),
            "score_geocoded": props.get("score"),
        }

    def get_address_imputation(self) -> dict:
        """
        Function that generate imputation of address infos for the selected rows.

        Returns
        -------
        results : dict
            Dictionary containing all rows to impute with adress infos.
        """
        # Preparing the output dict and rows to impute
        results = []
        rows_to_impute = (self.df
                          .filter(pl.col("adresse_numero").is_null() 
                                  & pl.col("longitude").is_not_null() 
                                  & pl.col("latitude").is_not_null()
                                  )
                                  .select(["id_mutation", 
                                           "longitude", 
                                           "latitude", 
                                           "code_postal", 
                                           "code_commune"])
                        )

        # Iterating over the rows to get address infos
        for row in rows_to_impute.iter_rows(named=True):
            try:
                geo = Dvf.reverse_geocode_fr(
                    lon=row["longitude"],
                    lat=row["latitude"],
                    postcode=row["code_postal"],
                )

                if geo is None:
                    results.append({
                        "id_mutation": row["id_mutation"],
                        "street_geocoded": None,
                        "postcode_geocoded": None,
                        "citycode_geocoded": None,
                        "score_geocoded": None,
                        "address_inferred_from_coords": False,
                        "lon": row["longitude"],
                        "lat": row["latitude"]
                    })
                    continue

                results.append({
                    "id_mutation": row["id_mutation"],
                    **geo,
                    "address_inferred_from_coords": True,
                    "lon": row["longitude"],
                    "lat": row["latitude"]
                })

                time.sleep(0.05)
            except Exception:
                results.append({
                    "id_mutation": row["id_mutation"],
                    "street_geocoded": None,
                    "postcode_geocoded": None,
                    "citycode_geocoded": None,
                    "score_geocoded": None,
                    "address_inferred_from_coords": False,
                    "lon": row["longitude"],
                    "lat": row["latitude"]
                })
        
        return results
    
    def prepare_imputation_df(self) -> pl.DataFrame:
        """
        Function that prepare the dataframe containing the imputation for address infos.

        Returns
        -------
        df : pl.DataFrame
            DataFrame to join with dvf data.
        """
        # Dictionary with address infos
        imputed_rows = Dvf.get_address_imputation(self)

        # DataFrame containing address infos
        df = pl.DataFrame(imputed_rows)

        # Cleaning address number
        df = df.with_columns(
            pl.when(pl.col("housenumber_geocoded").str.contains("[a-zA-Z]"))
            .then(pl.col("housenumber_geocoded").str.extract(r"^(\d+)", group_index=1))
            .otherwise(pl.col("housenumber_geocoded"))
            .alias("adresse_numero_clean")
            )
        
        # Dropping duplicates
        df = df.unique(subset=["id_mutation","lon","lat"], maintain_order=True)

        # Dropping unnecessary columns
        df = df.drop(["full_address_geocoded", "housenumber_geocoded", "city_geocoded", "citycode_geocoded"])

        return df
    
    def join_impute_df(self) -> "Dvf":
        """
        Function that join the dataframe containing imputation about address infos with the dvf dataframe.

        Returns
        -------
        self : Dvf
            The same object, with self.df updated.
        """
        geo_df = Dvf.prepare_imputation_df(self)
        self.df = self.df.join(geo_df,
                               left_on=["id_mutation","longitude","latitude"],
                               ight_on=["id_mutation","lon","lat"],
                               how="left",
                               validate="m:1"
                                )

        return self
    
    def impute_address_infos(self) -> "Dvf":
        """
        Function that impute null values for "adresse_nom_voie" and "adresse_numero".

        Returns
        -------
        self : Dvf
            The same object, with self.df updated.
        """
        # Replacing address infos null values by its imputation if possible
        self.df = self.df.with_columns(
            pl.when(pl.col("address_inferred_from_coords").is_null())
            .then(False)
            .otherwise(True)
            .alias("address_inferred_from_coords")
            ,
            pl.when(pl.col("address_inferred_from_coords") & 
                    (pl.col("score_geocoded") > 0.9))
            .then(pl.col("adresse_numero_clean"))
            .otherwise(pl.col("adresse_numero"))
            .alias("adresse_numero")
            ,
            pl.when(pl.col("address_inferred_from_coords") &
                    (pl.col("score_geocoded") > 0.9))
            .then(pl.col("street_geocoded"))
            .otherwise(pl.col("adresse_nom_voie"))
            .alias("adresse_nom_voie")
        )

        # Setting remaining null values with "Indisponible"
        self.df = self.df.with_columns(
            pl.when(pl.col("adresse_numero").is_null())
            .then(pl.lit("Indisponible"))
            .otherwise(pl.col("adresse_numero"))
            .alias("adresse_numero")
            ,
            pl.when(pl.col("adresse_nom_voie").is_null())
            .then(pl.lit("Indisponible"))
            .otherwise(pl.col("adresse_nom_voie"))
            .alias("adresse_nom_voie")
            ,
            pl.when(pl.col("type_local").is_null())
            .then(pl.lit("Indisponible"))
            .otherwise(pl.col("type_local"))
            .alias("type_local")
        )

        return self
    
    def normalize_string(self, s: str) -> str:
        """
        Function used to normalize the format of string values.

        Arguments
        -------
        s : str
            A string value.

        Returns
        -------
        s : str
            String normalized.
        """
        if s is None:
            return None

        # Uppercase
        s = s.upper()

        # Suppressing accents
        s = "".join(
            c for c in unicodedata.normalize("NFD", s)
            if not unicodedata.combining(c)
        )

        # Converting certain character to spaces
        s = re.sub(r"['’`´\-_/.,;:()]", " ", s)

        # Abreviations
        for mot, abbr in self.dvf_cfg["ABBREVIATIONS"].items():
            s = re.sub(rf"\b{mot}\b", abbr, s)

        # Keeping only letters, numbers, and spaces
        s = re.sub(r"[^A-Z0-9 ]", " ", s)

        # Spaces normalization
        s = re.sub(r"\s+", " ", s).strip()

        return s
    
    def normalize_address(self) -> "Dvf":
        """
        Function that normalize string format for the address column.

        Returns
        -------
        self : Dvf
            The same object, with self.df updated.
        """
        self.df = self.df.with_columns(
            pl.col("adresse_nom_voie")
            .map_elements(Dvf.normalize_string, return_dtype=pl.String)
            .alias("adresse_nom_voie")
        )

        return self
    
    def create_flag_for_null_val(self) -> "Dvf":
        """
        Function that creates columns to flag remaining null values that can't be imputed with coinfidence,
        for several columns.

        Returns
        -------
        self : Dvf
            The same object, with self.df updated.
        """
        self.df = self.df.with_columns(
            pl.col("valeur_fonciere").is_null().alias("is_valeur_fonciere_missing"),
            pl.col("surface_reelle_bati").is_null().alias("is_surface_missing"),
            pl.col("nombre_pieces_principales").is_null().alias("is_nombre_pieces_missing")
        )

        return self
    
    def create_flag_for_atypical_val(self) -> "Dvf":
        """
        Function that creates columns to flag atypical/invalid values that can't be imputed with coinfidence,
        for several columns.

        Returns
        -------
        self : Dvf
            The same object, with self.df updated.
        """
        # Flag for suspect values
        self.df = self.df.with_columns([
            pl.when((~pl.col("is_surface_missing")) 
                    & (pl.col("surface_reelle_bati") <= 5)
            )
            .then(True)
            .otherwise(False)
            .alias("is_surface_tres_faible"),

            pl.when((~pl.col("is_valeur_fonciere_missing"))
                    & (pl.col("valeur_fonciere") < 10_000)
            )
            .then(True)
            .otherwise(False)
            .alias("is_valeur_fonciere_tres_basse"),

            (
                (pl.col("type_local") == "Appartement") &
                (pl.col("nombre_lots") == 0)
            ).alias("is_appartement_sans_lot")])
        
        # Flag for atypical values
        self.df = self.df.with_columns([(
            (pl.col("is_surface_tres_faible")) |
            (pl.col("is_valeur_fonciere_tres_basse")) |
            (
                (pl.col("type_local") == "Appartement") &
                (pl.col("nombre_lots") == 0)
            )
        ).alias("is_transaction_atypique_agent")
        ])

        return self

    def drop_duplicated_rows(self):
        """
        Function that drops duplicated rows.

        Returns
        -------
        self : Dvf
            The same object, with self.df updated.
        """
        self.df = self.df[self.dvf_cfg["COLS_TO_KEEP"]].unique(maintain_order=True)

        return self
    
    def run_dvf_to_silver(self):
        """
        Function used to compile all the transformation needed to upgrade the data to silver level.

        Returns
        -------
        self : Dvf
            The same object, with self.df updated.
        """
        logger.info("Démarrage de l'upgrade dvf vers silver")

        # Removing header rows
        logger.info("Suppression des lignes en-têtes")
        self.remove_header_rows()

        # Casting columns type
        logger.info("Typage des colonnes")
        self.casting_columns_types()

        # Imputing null values
        logger.info("Imputation des valeurs null pour code_postal")
        self.imputing_postal_code_null()

        logger.info("Imputation des valeurs null pour adresse_numero et adresse_nom_voie")
        self.join_impute_df()
        self.impute_address_infos()

        # Normalizing address
        logger.info("Normalisation du format des adresses")
        self.normalize_address()

        # Flag columns
        logger.info("Création de colonnes flag de valeurs null (pour valeur_fonciere,...)")
        self.create_flag_for_null_val()

        logger.info("Création de colonne flag de valeurs atypiques/invalides")
        self.create_flag_for_atypical_val()

        # Dropping duplicates
        logger.info("Drop des lignes dupliquées")
        self.drop_duplicated_rows()

        logger.info("Fin de l'upgrade dvf vers silver")

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

    def casting_columns_types(self) -> "Bpe":
        """
        Function that cast columns in the right type.

        Returns
        -------
        self : Bpe
            The same object, with self.df updated.
        """
        # Casting the right type for each columns
        self.df = self.df.with_columns(
            pl.col(self.bpe_cfg["DATE_COLS"]).str.to_date("%Y", strict=True).dt.year().cast(pl.Int64).alias("TIME_PERIOD"),
            pl.col(self.bpe_cfg["INT_COLS"]).cast(pl.Int64, strict=True)
        )

        return self
    
    def slicing_by_code_arr(self) -> "Bpe":
        """
        Function that slice bpe data by selected districts.

        Returns
        -------
        self : Bpe
            The same object, with self.df updated.
        """
        self.df = self.df.filter(pl.col("GEO").is_in(self.bpe_cfg["CODE_ARR"]))

        return self
    
    def join_code_nom(self, code_df: pl.DataFrame, code_cfg: dict) -> "Bpe":
        """
        Function that join bpe dataframe to obtain the name associated to each code.

        Arguments
        -------
        code_df : pl.DataFrame
            Dataframe containing code and their associated name.
        code_cfg : dict
            Dictionary containing convention for the jointure.
        
        Returns
        -------
        self : Bpe
            The same object, with self.df updated.
        """
        code_df = code_df.rename(code_cfg["RENAME"])
        self.df = self.df.join(code_df,
                               left_on=[code_cfg["KEY"]],
                               right_on=["code"],
                               how="left",
                               validate="m:1"
                            )
        
        return self
        
    def creating_postal_code(self) -> "Bpe":
        """
        Function that creates a postal code column.

        Returns
        -------
        self : Bpe
            The same object, with self.df updated.
        """
        self.df = self.df.with_columns((pl.lit("750") + pl.col("GEO")
                                     .str.slice(-2))
                                     .alias("code_postal")
                                     )
        
        return self
    
    def renaming_columns(self) -> "Bpe":
        """
        Function that rename and keep relevant columns.

        Returns
        -------
        self : Bpe
            The same object, with self.df updated.
        """
        self.df = self.df.rename(self.bpe_cfg["RENAME_COLS"])
        self.df = self.df[self.bpe_cfg["COLS_TO_KEEP"]]

    def run_bpe_to_silver(self, codes_dom_df: pl.DataFrame, codes_sdom_df: pl.DataFrame, codes_type_df: pl.DataFrame) -> "Bpe":
        """
        Function used to compile all the transformation needed to upgrade the data to silver level.

        Arguments
        -------
        codes_dom_df : pl.DataFrame
            Dataframe used to get facility domain code into name.
        codes_sdom_df : pl.DataFrame
            Dataframe used to get facility sub-domain code into name.
        codes_type_df : pl.DataFrame
            Dataframe used to get facility type code into name.
        
        Returns
        -------
        self : Bpe
            The same object, with self.df updated.
        """
        logger.info("Démarrage de l'upgrade bpe vers silver")

        # Casting columns types
        logger.info("Typage des colonnes")
        self.casting_columns_types()

        # Slicing the data
        logger.info("Slicing des données")
        self.slicing_by_code_arr()

        # Converting codes columns into names
        logger.info("Jointure des noms de domaines")
        self.join_code_nom(codes_dom_df, self.bpe_cfg["DOM_CODES"])

        logger.info("Jointure des noms de sous-domaines")
        self.join_code_nom(codes_sdom_df, self.bpe_cfg["SDOM_CODES"])

        logger.info("Jointure des noms de types d'équipements")
        self.join_code_nom(codes_type_df, self.bpe_cfg["TYPES_CODES"])

        # Creating postal code column
        logger.info("Création d'une colonne code_postal")
        self.creating_postal_code()

        # Renaming columns
        logger.info("Renommage des colonnes")
        self.renaming_columns()

        logger.info("Fin de l'upgrade bpe vers silver")

        return self