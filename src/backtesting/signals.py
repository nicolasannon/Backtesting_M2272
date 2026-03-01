# Backtesting_M2272/src/backtesting/signals.py

# ------------------------------------------------------------------
# Librairies
# ------------------------------------------------------------------

# externes
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Présenatations
# ------------------------------------------------------------------
"""
Calcul du signal Momentum Dual avec neutralisation sectorielle obligatoire.

Signal implémenté
-----------------
Momentum Dual = 0.5 × Momentum(12-1) + 0.5 × Momentum(6-1)

Avec :
  - Momentum(12-1) = P(t-1M) / P(t-12M) - 1
  - Momentum(6-1)  = P(t-1M) / P(t-6M)  - 1
  - t-1M  : dernier prix disponible 1 mois avant la date de rebalancement
  - t-nM  : dernier prix disponible n mois avant la date de rebalancement

Le signal est ensuite standardisé (z-score) intra-sectoriel :
  score_i = (raw_i - mean_secteur) / std_secteur

Contraintes
-----------
- Aucune donnée future utilisée (no look-ahead bias).
- Tickers sans historique suffisant exclus (signal = NaN → ignorés)
"""

class SignalEngine:
    """
    Calcule le signal Momentum Dual sectoriellement neutralisé

    Paramètres
    ----------
    data_loader : DataLoader
        Instance du DataLoader contenant prix et informations
    """

    def __init__(self, data_loader):
        self.loader = data_loader

    def compute(self, date: pd.Timestamp, universe: list) -> pd.Series:
        """
        Calcule le signal Momentum Dual standardisé pour tous les tickers
        de l'univers à la date de rebalancement donnée

        Paramètres
        ----------
        date : date de rebalancement
        universe: liste des tickers investissables

        Retourne
        --------
        pd.Series
            Scores standardisés par secteur (index = Ticker)
            Tickers sans signal suffisant exclus
        """
        mom12 = self._momentum(date, universe, n_months=12)
        mom6 = self._momentum(date, universe, n_months=6)
        common = mom12.index.intersection(mom6.index)
        if common.empty:
            return pd.Series(dtype=float)

        dual_mom = 0.5 * mom12.loc[common] + 0.5 * mom6.loc[common]
        dual_mom = dual_mom.dropna()

        if dual_mom.empty:
            return pd.Series(dtype=float)

        return self._sector_standardize(dual_mom)

    def _momentum(self, date: pd.Timestamp, universe: list, n_months: int) -> pd.Series:
        """
        Exclusion du dernier mois (skip 1 mois) pour éviter le biais
        de mean-reversionà court terme
        """
        price_1m = self.loader.get_price_n_months_before(date, 1, universe)
        price_nm = self.loader.get_price_n_months_before(date, n_months, universe)

        common = price_1m.index.intersection(price_nm.index)
        if common.empty:
            return pd.Series(dtype=float)

        p1 = price_1m.loc[common]
        pn = price_nm.loc[common]

        valid = p1.notna() & pn.notna() & (pn > 0) & (p1 > 0)
        if not valid.any():
            return pd.Series(dtype=float)

        return (p1[valid] / pn[valid]) - 1.0

    def _sector_standardize(self, signals: pd.Series) -> pd.Series:
        """
        Z-score intra-sectoriel : pour chaque secteur, soustrait la
        moyenne et divise par l'écart-type des signaux bruts du secteur

        Si un secteur contient un seul ticker, son score reste le score
        brut (non standardisé) car la variance est nulle
        """
        info = self.loader.informations
        result = pd.Series(index=signals.index, dtype=float)

        tickers_with_signal = signals.index.tolist()
        info_subset = info.loc[info.index.intersection(tickers_with_signal), "Sector"].dropna()

        for sector, group in info_subset.groupby(info_subset):
            sector_tickers = group.index.intersection(signals.index)
            vals = signals.loc[sector_tickers]

            if len(vals) < 2:
                result.loc[sector_tickers] = vals
                continue

            mu = vals.mean()
            sigma = vals.std()

            if sigma < 1e-8:
                result.loc[sector_tickers] = 0.0
            else:
                result.loc[sector_tickers] = (vals - mu) / sigma

        no_sector = signals.index.difference(info_subset.index)
        if not no_sector.empty:
            vals_no_sector = signals.loc[no_sector]
            mu = vals_no_sector.mean()
            sigma = vals_no_sector.std()
            if sigma < 1e-8:
                result.loc[no_sector] = 0.0
            else:
                result.loc[no_sector] = (vals_no_sector - mu) / sigma

        return result.dropna()
