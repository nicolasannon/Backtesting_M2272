import pandas as pd
from pathlib import Path
from datetime import datetime
import openpyxl
import re


class ParquetStockage:
    """
    Classe simple pour sauvegarder et lire des DataFrames en format parquet.
    Stockage automatique dans BACKTESTING_M2272/data/initial_data
    """

    def __init__(self):
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        self.dossier = project_root / "data" / "initial_data"
        self.dossier.mkdir(parents=True, exist_ok=True)

    def sauvegarder(self, df: pd.DataFrame, nom_fichier: str, ticker: str):
        """
        Sauvegarde un DataFrame en format :
        Date | Ticker
        """

        # Mettre Date en colonne
        df_clean = df.reset_index()

        # Renommer colonne prix avec le nom du ticker
        df_clean.columns = ["Date", ticker]

        chemin = self.dossier / f"{nom_fichier}.parquet"
        df_clean.to_parquet(chemin)

    def charger(self, nom_fichier: str) -> pd.DataFrame:
        chemin = self.dossier / f"{nom_fichier}.parquet"
        return pd.read_parquet(chemin)


class ExportData:

    def __init__(self, folder_type: str = "verification"):
        """
        Parameters
        ----------
        folder_type : str
            Choix parmi : "verification", "initial_data", "stockage"
        """
        allowed_folders = ["verification", "initial_data", "stockage"]

        if folder_type not in allowed_folders:
            raise ValueError(f"folder_type doit être dans {allowed_folders}")

        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent

        self.dossier = project_root / "data" / folder_type
        self.dossier.mkdir(parents=True, exist_ok=True)

    def export_df(self, df: pd.DataFrame, filename: str, with_timestamp: bool = True):
        """
        Exporte un DataFrame en Excel.

        Parameters
        ----------
        df : pd.DataFrame
        filename : str
        with_timestamp : bool
        """
        if with_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"

        file_path = self.dossier / f"{filename}.xlsx"

        df.to_excel(file_path, index=True)

        return file_path
    

class MonthEndExtractor:
    
    def __init__(self, df: pd.DataFrame, date_column: str = "Date"):
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contenant une colonne de dates
        date_column : str
            Nom de la colonne date
        """
        self.df = df.copy()
        self.date_column = date_column
        
        self._prepare_dates()
    
    
    def _prepare_dates(self):
        """Convertit la colonne en datetime et trie les données"""
        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
        self.df = self.df.sort_values(self.date_column)
    
    
    def get_month_end_dates(self) -> pd.Series:
        """
        Retourne la dernière date observée pour chaque mois.
        """
        month_end_dates = (
            self.df
            .groupby(self.df[self.date_column].dt.to_period("M"))[self.date_column]
            .max()
            .sort_values()
        )
        
        return month_end_dates
    
class BloombergDateFormatter:
    
    def __init__(self, date_list):
        """
        Parameters
        ----------
        date_list : list of pd.Timestamp / datetime
        """
        self.date_list = date_list

    def to_yyyymmdd(self):
        """
        Retourne une liste au format Bloomberg YYYYMMDD (int)
        """
        return [int(d.strftime("%Y%m%d")) for d in self.date_list]



class IndexMembersParquetMerger:
    """
    Merge tous les fichiers: data/initial_data/index_memb_*.parquet
    en un seul dataset, sans supprimer de lignes.
    """

    def __init__(self, verbose: bool = True):
        self.project_root = self._find_project_root()
        self.folder = self.project_root / "data" / "initial_data"
        self.pattern = "index_memb_*.parquet"
        self.verbose = verbose

        if not self.folder.exists():
            raise FileNotFoundError(f"Dossier introuvable: {self.folder}")

        if self.verbose:
            print(f"[INFO] Project root détecté : {self.project_root}")
            print(f"[INFO] Dossier cible        : {self.folder}")

    def _find_project_root(self) -> Path:
        """
        Détection robuste:
        - si exécuté depuis un .py : part de __file__
        - sinon (notebook) : part de Path.cwd()
        Puis remonte jusqu'à trouver data/initial_data.
        """
        start_paths = []

        try:
            start_paths.append(Path(__file__).resolve())
        except NameError:
            pass

        start_paths.append(Path.cwd().resolve())

        for start in start_paths:
            for parent in [start] + list(start.parents):
                if (parent / "data" / "initial_data").exists():
                    return parent

        raise FileNotFoundError("Impossible de détecter le project_root (data/initial_data introuvable).")

    def merge(self) -> pd.DataFrame:
        files = sorted(self.folder.glob(self.pattern))
        if not files:
            raise FileNotFoundError(f"Aucun fichier trouvé dans {self.folder} avec {self.pattern}")

        merged = pd.concat((pd.read_parquet(fp) for fp in files), ignore_index=True, sort=False)

        if self.verbose:
            print(f"[INFO] {len(files)} fichiers fusionnés")
            print(f"[INFO] Shape final : {merged.shape}")

        return merged

    def merge_to_parquet(self, output_name: str = "historical_index_memb.parquet") -> Path:
        out = (self.folder / output_name).resolve()
        if self.verbose:
            print(f"[INFO] Sortie attendue : {out}")

        df = self.merge()
        df.to_parquet(out, index=False)

        if not out.exists():
            raise FileNotFoundError(f"[ERROR] Échec écriture fichier : {out}")

        if self.verbose:
            size_mb = out.stat().st_size / (1024 * 1024)
            print(f"[SUCCESS] Fichier créé : {out} ({size_mb:.2f} MB)")

        return out
    
    
class DataManagement:
    """
    Classe utilitaire pour transformer les données historiques.
    """

    @staticmethod
    def format_historical_price(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme un DataFrame long (Date, Ticker, Price)
        en format wide (Date | Ticker1 | Ticker2 | ...)

        Parameters
        ----------
        df : DataFrame avec colonnes ['Date', 'Ticker', 'Price']

        Returns
        -------
        DataFrame pivoté
        """

        # Sécurité minimale
        df = df.copy()

        # Assure que Date est datetime
        df["Date"] = pd.to_datetime(df["Date"])

        # Pivot
        df_wide = df.pivot(index="Date", columns="Ticker", values="Price")

        # Tri par date
        df_wide = df_wide.sort_index()

        # Optionnel : reset index si tu veux Date en colonne
        df_wide = df_wide.reset_index()

        return df_wide