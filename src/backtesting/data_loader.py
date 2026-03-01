
# Backtesting_M2272/src/backtesting/data_loader.py

# ------------------------------------------------------------------
# Librairies
# ------------------------------------------------------------------

# externes
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Présenatations
# ------------------------------------------------------------------
"""
Charge et structure toutes les données nécessaires au backtesting.

Données chargées
----------------
- historical_price_memb_index.parquet  : prix quotidiens (long -> wide)
- ESTRON Index_PX_LAST.parquet         : taux ESTER annualisé (%)
- informations_hist_memb_index.parquet : métadonnées par ticker
- index_memb_YYYYMMDD.parquet          : compositions mensuelles

"""


class DataLoader:
    """
    Charge et aligne toutes les données pour le backtesting long/short
    sur le STOXX Europe 600.

    Attributs publics
    -----------------
    prices : pd.DataFrame
        Prix quotidiens wide (index=Date, colonnes=Ticker).
    ester : pd.Series
        Taux ESTER annualisé (%) indexé par Date.
    informations : pd.DataFrame
        Métadonnées (Name, Country, Currency, Sector, Industry) indexées par Ticker.
    compositions : dict[pd.Timestamp, list[str]]
        Composition de l'indice à chaque date de rebalancement.
    rebalancing_dates : list[pd.Timestamp]
        Dates de rebalancement dans la période de backtesting.
    """

    _DATA_ROOT = (Path(__file__).resolve().parent.parent.parent / "data" / "initial_data")

    def __init__(self, start_date: str = "2016-12-30", end_date: str = "2026-02-20"):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)

        self.prices: pd.DataFrame = pd.DataFrame()
        self.ester: pd.Series = pd.Series(dtype=float)
        self.informations: pd.DataFrame = pd.DataFrame()
        self.compositions: dict = {}
        self.rebalancing_dates: list = []

        self._load_all()

    def _load_all(self):
        print("[DataLoader] Chargement des données en cours...")
        self._load_prices()
        self._load_ester()
        self._load_informations()
        self._load_compositions()
        print(f"[DataLoader] Prêt : {len(self.rebalancing_dates)} dates de rebalancement "
            f"({self.start_date.date()} → {self.end_date.date()}).")

    def _load_prices(self):
        df = pd.read_parquet(self._DATA_ROOT / "historical_price_memb_index.parquet")
        
        df["Date"] = pd.to_datetime(df["Date"])
        self.prices = (df.pivot(index="Date", columns="Ticker", values="Price").sort_index())

    def _load_ester(self):
        """Taux ESTER annualisé en % (Series indexée par Date)"""
        df = pd.read_parquet(self._DATA_ROOT / "ESTRON Index_PX_LAST.parquet")
        df["Date"] = pd.to_datetime(df["Date"])
        self.ester = df.set_index("Date")["ESTRON Index"]

    def _load_informations(self):
        """
        Informations descriptives par ticker
        Index = Ticker au format "XXXX Equity" (aligné avec les prix)
        """
        df = pd.read_parquet(self._DATA_ROOT / "informations_hist_memb_index.parquet")
        self.informations = df.set_index("Ticker")

    def _load_compositions(self):
        """
        Charge les compositions mensuelles depuis index_memb_YYYYMMDD.parquet
        - La date est extraite du nom du fichier
        - Le suffix ' Equity' est ajouté aux tickers pour s'aligner avec les prix
        - Seules les dates dans [start_date, end_date] sont conservées
        """
        pattern = str(self._DATA_ROOT / "index_memb_*.parquet")
        for fp in sorted(glob.glob(pattern)):
            match = re.search(r"index_memb_(\d{8})\.parquet", fp)
            if not match:
                continue
            ds = match.group(1)
            date = pd.Timestamp(f"{ds[:4]}-{ds[4:6]}-{ds[6:]}")
            if date < self.start_date or date > self.end_date:
                continue
            df = pd.read_parquet(fp)
            tickers = [t + " Equity" for t in df["Ticker"].tolist()]
            self.compositions[date] = tickers

        self.rebalancing_dates = sorted(self.compositions.keys())

    def get_universe(self, date: pd.Timestamp) -> list:
        """
        Retourne la liste des tickers investissables à la date donnée
        Filtre les tickers sans prix disponible à cette date
        """
        tickers = self.compositions.get(date, [])
        available_dates = self.prices.index[self.prices.index <= date]
        if len(available_dates) == 0:
            return []
        last_date = available_dates[-1]
        valid_cols = [t for t in tickers if t in self.prices.columns]
        if not valid_cols:
            return []
        has_price = self.prices.loc[last_date, valid_cols].notna()
        return has_price[has_price].index.tolist()

    def get_price_at(self, date: pd.Timestamp, tickers: list) -> pd.Series:
        """Prix au plus tard à `date` pour les tickers donnés"""
        available = self.prices.index[self.prices.index <= date]
        if len(available) == 0:
            return pd.Series(dtype=float)
        cols = [t for t in tickers if t in self.prices.columns]
        return self.prices.loc[available[-1], cols]

    def get_price_n_months_before(self, date: pd.Timestamp, n_months: int, tickers: list) -> pd.Series:
        """Prix au plus tard à `date - n_months` pour les tickers donnés"""
        target = date - pd.DateOffset(months=n_months)
        available = self.prices.index[self.prices.index <= target]
        if len(available) == 0:
            return pd.Series(dtype=float)
        cols = [t for t in tickers if t in self.prices.columns]
        return self.prices.loc[available[-1], cols]

    def get_daily_returns(self, start: pd.Timestamp, end: pd.Timestamp, tickers: list) -> pd.DataFrame:
        """
        Retours journaliers simples entre start et end
        Forward-fill des données manquantes avant calcul
        """
        cols = [t for t in tickers if t in self.prices.columns]
        mask = (self.prices.index >= start) & (self.prices.index <= end)
        sub = self.prices.loc[mask, cols].ffill()
        returns = sub.pct_change()
        return returns.iloc[1:]

    def get_ester_daily_return(self, date: pd.Timestamp) -> float:
        available = self.ester.index[self.ester.index <= date]
        if len(available) == 0:
            return 0.0
        return float(self.ester.loc[available[-1]]) / 100 / 252

    def get_ester_returns_series(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """
        Série de retours journaliers ESTER sur [start, end]
        """
        date_range = pd.date_range(start, end, freq="B")
        ester_aligned = self.ester.reindex(date_range).ffill().bfill().fillna(0)
        return ester_aligned / 100 / 252

    def get_sector(self, ticker: str) -> str:
        """Secteur GICS du ticker. Retourne 'Unknown' si absent"""
        if ticker in self.informations.index:
            return self.informations.loc[ticker, "Sector"] or "Unknown"
        return "Unknown"
