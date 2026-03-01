
# Backtesting_M2272/src/backtesting/__init__.py

# ------------------------------------------------------------------
# Librairies
# ------------------------------------------------------------------

# exter,es
from pathlib import Path
from datetime import datetime
from typing import Optional
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ------------------------------------------------------------------
# Présenatations
# ------------------------------------------------------------------

"""
Reporting à chaque date de rebalancement
-------------------------------------------------
1. Performance   : rendements cumulés, annualisés, CAGR sur multiples horizons
2. Risque        : Sharpe, Sortino, Tracking Error, Information Ratio, Calmar,
                   VaR, CVaR, Volatilité, Max Drawdown
3. Benchmark     : Corrélations et rendements ESTER
4. Coûts         : TC cumulés en % et en €
5. Visualisations: graphiques Plotly (cumul returns, drawdowns, vol, corrél.)
6. Composition   : barcharts secteur/pays/devise/industrie, top 10 poids,
                   top 10 contributions au performance et au risque
"""

# Associationnde chaque méthode à un label
METHODS_LABELS = {
    "equal_weight":    "EqualWeight",
    "risk_parity":     "RiskParity",
    "min_variance":    "MinVariance",
    "signal_weight":   "SignalWeight",
}

# Constantes, annualisation, seuils, affichage des NaN dans le reporting
ANNUALIZATION = 252
CONF_LEVEL = 0.05        # 95% VaR / CVaR
_NA = "--"               # Affichage quand données insuffisantes

# Palette institutionnelle type Amundi
AMUNDI_DARK = "#101828"
AMUNDI_NAVY = "#0E2A47"
AMUNDI_TEAL = "#00A3AD"
AMUNDI_RED = "#E30613"
AMUNDI_LIGHT_BG = "#F7F9FC"
AMUNDI_GRID = "#D9E2EC"
AMUNDI_TEXT = "#1A1A1A"


class ReportingEngine:
    """
    --> Calcule des métriques de performances, de risk, du benchmark, des couts
    --> Génère visualisation
    --> Export des résultats vers le folder de stockage.

    Paramètres
    ----------
    result      : BacktestResult — résultats du moteur de backtesting.
    data_loader : DataLoader    — accès aux prix et métadonnées.
    """

    # Racine stockage
    _STOCKAGE_ROOT = (
        Path(__file__).resolve().parent.parent.parent / "data" / "stockage"
    )

    def __init__(self, result, data_loader):
        # Récup résultats de backtesting et du data loader des données
        self.result = result
        self.loader = data_loader

        # Séries journalières de référence
        self._dr = result.daily_returns          # Rndt journaliers
        self._nav = result.nav                    # NAV quotidienne

        # Rendement ESTER alignés sur ceux du portefeuille
        if not self._dr.empty:
            self._ester_dr = self.loader.get_ester_returns_series(
                self._dr.index[0], self._dr.index[-1]
            ).reindex(self._dr.index).fillna(0.0)
        else:
            self._ester_dr = pd.Series(dtype=float)
        
        # niveau de l'ester : on a opéré deux noms différents pour ne pas se mélanger avec les rndts
        self._estron_levels = self._load_estron_levels()

    # ------------------------------------------------------------------
    # Calcul des métriques
    # ------------------------------------------------------------------

    def compute_metrics(self, as_of_date: pd.Timestamp) -> dict:
        """
        Calcule l'ensemble des métriques de performance à chaque date de rebalancement.
        
        Retourne un dictionnaire {metric_name: value}.
        """
        # filtre des rendemtns journailiers de la date de début du backtest à la date de reporting
        dr = self._dr[self._dr.index <= as_of_date]
        
        # rendements journaliers de l'ester avec rendements portefeuille
        ester = self._ester_dr.reindex(dr.index).fillna(0.0)

        # si une equity ou bench non pas assez de données, retourne NaN
        if dr.empty:
            return self._empty_metrics()
        
        # récupération de la NAV
        nav_slice = self._nav[self._nav.index <= as_of_date]
        
        # fixe la nav initiale et finale
        start_nav = float(self.result.initial_capital)
        end_nav = float(nav_slice.iloc[-1]) if not nav_slice.empty else start_nav

        # création d'un dictionnaire de stockage
        m = {}

        # Allocation des résultats des calculs de métriques à chaque métrique correspondant
        # ---- A - Performance ----
        m["Rendement cumulé (période)"] = self._cum_return(dr)
        m["Rendement cumulé 1 an"] = self._cum_return_nav_window(as_of_date, months=12)
        m["Rendement cumulé 2 ans"] = self._cum_return_nav_window(as_of_date, months=24)
        m["Rendement cumulé YTD"] = self._cum_return_ytd(dr, as_of_date)
        m["Rendement cumulé MTD"] = self._cum_return_mtd(dr, as_of_date)

        m["Rendement annualisé (période)"] = self._ann_return(dr)
        m["CAGR (période)"] = self._cagr(start_nav, end_nav, dr)
        m["CAGR 1 an"] = self._cagr_window(dr, as_of_date, months=12)
        m["CAGR 2 ans"] = self._cagr_window(dr, as_of_date, months=24)
        m["CAGR YTD"] = self._cagr_ytd(dr, as_of_date)
        m["CAGR MTD"] = self._cagr_mtd(dr, as_of_date)

        # ---- B - Risque & Performance ajustée ----
        rf_annual = float(ester.mean() * ANNUALIZATION)
        m["Sharpe ratio (période)"] = self._sharpe(dr, ester)
        m["Sharpe ratio 1 an"] = self._sharpe_window(dr, ester, as_of_date, months=12)
        m["Sortino ratio (période)"] = self._sortino(dr, ester)
        m["Sortino ratio 1 an"] = self._sortino_window(dr, ester, as_of_date, months=12)

        m["Tracking Error (période)"] = self._tracking_error(dr, ester)
        m["Tracking Error 1 an"] = self._tracking_error_window(dr, ester, as_of_date, months=12)
        m["Tracking Error 3 ans"] = self._tracking_error_window(dr, ester, as_of_date, months=36)

        m["Information Ratio (période)"] = self._info_ratio(dr, ester)
        m["Information Ratio 1 an"] = self._info_ratio_window(dr, ester, as_of_date, months=12)
        m["Information Ratio 2 ans"] = self._info_ratio_window(dr, ester, as_of_date, months=24)

        m["Calmar ratio (période)"] = self._calmar(dr)

        m["VaR 95% (période)"] = self._var(dr)
        m["VaR 95% 1 an"] = self._var_window(dr, as_of_date, months=12)
        m["CVaR 95% (période)"] = self._cvar(dr)
        m["CVaR 95% 1 an"] = self._cvar_window(dr, as_of_date, months=12)

        m["Volatilité (période)"] = self._volatility(dr)
        m["Volatilité 1 an"] = self._vol_window(dr, as_of_date, months=12)
        m["Volatilité 2 ans"] = self._vol_window(dr, as_of_date, months=24)

        m["Max Drawdown (période)"] = self._max_drawdown(dr)
        m["Max Drawdown 1 an"] = self._mdd_window(dr, as_of_date, months=12)
        m["Max Drawdown 2 ans"] = self._mdd_window(dr, as_of_date, months=24)
        m["Max Drawdown YTD"] = self._mdd_ytd(dr, as_of_date)
        m["Max Drawdown MTD"] = self._mdd_mtd(dr, as_of_date)

        # ---- C- Benchmark (ESTER) ----
        m["Corrélation avec benchmark (période)"] = self._corr(dr, ester)
        m["Benchmark - Rendement cumulé (période)"] = self._benchmark_cum_return_period(as_of_date)
        m["Benchmark - Rendement cumulé 1 an"] = self._benchmark_cum_return_window(as_of_date, months=12)
        m["Benchmark - Rendement cumulé 2 ans"] = self._benchmark_cum_return_window(as_of_date, months=24)
        m["Benchmark - Rendement cumulé YTD"] = self._benchmark_cum_return_period_type(as_of_date, "YTD")
        m["Benchmark - Rendement cumulé MTD"] = self._benchmark_cum_return_period_type(as_of_date, "MTD")
        m["Benchmark - Rendement annualisé (période)"] = self._benchmark_ann_return_period(as_of_date)

        # ---- D - Coûts de transaction ----
        
        # cout de transaction en euros
        tc_eur_all = self.result.transaction_costs_eur
        # cumul des couts jusquà date de reporting (=cout total en €)
        tc_eur_total = sum(
            v for k, v in tc_eur_all.items() if k <= as_of_date
        )
        # cout total en %
        tc_pct_total = tc_eur_total / self.result.initial_capital
        # implémente
        m["TC total (€)"] = tc_eur_total
        m["TC total (%)"] = tc_pct_total

        # ---- E- PnL (€) ----
        # calcul du PnL en euros
        m["PnL (€)"] = end_nav - start_nav

        return m

    def compute_all_metrics(self) -> pd.DataFrame:
        """
        Retourne un DataFrame (index=Date, colonnes=métriques).
        """
        # récupère date de rebalancement
        rebal_dates = sorted(self.result.weights.keys())
        # dictionnaire de stockage des métrics par date
        rows = {}
        # pour chaque date on fait le calcul
        for date in rebal_dates:
            rows[date] = self.compute_metrics(date)
        # mise en DataFrame
        df = pd.DataFrame(rows).T
        df.index.name = "Date"
        return df

    # ------------------------------------------------------------------
    # Fonctions internes de calcil des métriques
    # ------------------------------------------------------------------

    def _slice(self, dr: pd.Series, start: pd.Timestamp) -> pd.Series:
        """Retourne les retours à partir de start (date la plus proche supérieure)"""
        # car on fait date reporting - 12 mois par exemple et peut ne pas correspondre à un jour de bourse
        available = dr.index[dr.index >= start]
        
        # retourne vide si rien
        if available.empty:
            return pd.Series(dtype=float)
        # je stoppe la recherche dès que j'ai trouvé une date ok
        return dr.loc[available[0]:]

    def _closest_before(self, dr: pd.Series, target: pd.Timestamp) -> pd.Timestamp:
        """Date la plus proche <= target dans l'index"""
        # même principe qu'au dessus mais l'inverse, date inférieure la plus proche
        candidates = dr.index[dr.index <= target]
        # stop la recherche si date ok et stock la date ok
        return candidates[-1] if not candidates.empty else None

    def _window(self, dr: pd.Series, as_of: pd.Timestamp, months: int) -> pd.Series:
        # création de la fenetre glissantes : date - temps en moins (ex: 12 mois)
        start_target = as_of - pd.DateOffset(months=months)
        # cherche la date ok la plus proche (inférieure ou égale)
        start = self._closest_before(dr, start_target)
        if start is None:
            return pd.Series(dtype=float)
        return dr.loc[start:as_of]

    def _load_estron_levels(self) -> pd.Series:
        """ charge le nibeau de estron depuis son parquet stocker dans le chemin de stockagae"""
        path = Path(__file__).resolve().parent.parent.parent / "data" / "initial_data" / "ESTRON Index_PX_LAST.parquet"
        try:
            df = pd.read_parquet(path)
        except Exception:
            # retourne une pandas series
            return pd.Series(dtype=float)
        # si date colonne  = convertion en bon format
        if "Date" in df.columns:
            dt = pd.to_datetime(df["Date"])
            values_df = df.drop(columns=["Date"])
        else:
            # sinon je prends l'index comme date si date deja en indice
            dt = pd.to_datetime(df.index)
            values_df = df
        # récupération prix close
        if "PX_LAST" in values_df.columns:
            s = values_df["PX_LAST"]
        else:
            # Si pas de prix, je prends la première colonne numérique disponible
            num_cols = values_df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                return pd.Series(dtype=float)
            s = values_df[num_cols[0]]

        # Assurance d'une serie série propre avec dates en index et valeur floaat
        out = pd.Series(s.values, index=dt).sort_index()
        # supprime les dates dupliquées et les valeurs manquantes
        out = out[~out.index.duplicated(keep="last")].dropna()
        return out.astype(float)

    def _benchmark_bounds_window(self, as_of: pd.Timestamp, months: int):
        """
        Je récupère les bornes (start, end) du benchmark sur une fenêtre glissante
        """
        levels = self._estron_levels

        # Si pas de données benchmark, je retourne vide
        if levels.empty:
            return None, None, None, None
        # covertit date
        as_of = pd.Timestamp(as_of)

        # cherche la dernière date disponible <= as_of
        end_candidates = levels.index[levels.index <= as_of]

        # Je définis la date cible de début
        start_target = as_of - pd.DateOffset(months=months)
        start_candidates = levels.index[levels.index <= start_target]

        # retourne None pour pas bloqué si pas dispo
        if end_candidates.empty or start_candidates.empty:
            return None, None, None, None

        s = start_candidates[-1]
        e = end_candidates[-1]

        # date de début, date de fin, niveau du bench à la date début, niveau du bench à la date de fin
        return s, e, float(levels.loc[s]), float(levels.loc[e])


    def _benchmark_daily_returns(self) -> pd.Series:
        """
        Taux annualisé ESTRON Index (en %) en rendement journalier
        """
        # prends valeur
        levels = self._estron_levels

        if levels.empty:
            return pd.Series(dtype=float)

        # décimalise
        rate = levels.astype(float) / 100.0

        # Nombre de jours entre chaque observation
        delta_days = rate.index.to_series().diff().dt.days.fillna(1).clip(lower=1)

        # Calcul
        dr = rate * (delta_days.values / 360.0)

        return dr.astype(float)


    def _benchmark_bounds_period(self, as_of: pd.Timestamp):
        """
        Récupèration les bornes début à date de reporting du benchmark
        """
        # level
        levels = self._estron_levels

        if levels.empty:
            return None, None, None, None

        as_of = pd.Timestamp(as_of)

        eligible = levels.index[levels.index <= as_of]
        if eligible.empty:
            return None, None, None, None

        # implemente
        s = eligible[0]
        e = eligible[-1]

        return s, e, float(levels.loc[s]), float(levels.loc[e])


    def _benchmark_bounds_period_type(self, as_of: pd.Timestamp, period: str):
        """
        Bornes benchmark pour calcul YTD ou MTD
        """
        levels = self._estron_levels

        if levels.empty:
            return None, None, None, None

        as_of = pd.Timestamp(as_of)
        idx = levels.index
        # dernier jours de bourse de l'année précédente du jour actuel
        if period == "YTD":
            prev_mask = idx.year == (as_of.year - 1)
            curr_mask = (idx.year == as_of.year) & (idx <= as_of)
        # dernier jour de bourse du mois précédent du jour actuel
        elif period == "MTD":
            prev_month = as_of.to_period("M") - 1
            prev_mask = (idx.year == prev_month.year) & (idx.month == prev_month.month)
            curr_mask = (idx.year == as_of.year) & (idx.month == as_of.month) & (idx <= as_of)

        else:
            return None, None, None, None

        # implemente
        prev_idx = idx[prev_mask]
        curr_idx = idx[curr_mask]

        if prev_idx.empty or curr_idx.empty:
            return None, None, None, None

        s = prev_idx[-1]
        e = curr_idx[-1]

        return s, e, float(levels.loc[s]), float(levels.loc[e])


    def _benchmark_cum_return_period(self, as_of: pd.Timestamp) -> object:
        """
        Rendement cumulé benchmark de date de début à date de reporting = rendement période
        """
        s, e, _, _ = self._benchmark_bounds_period(as_of)

        if s is None or e is None:
            return _NA

        dr = self._benchmark_daily_returns()
        sub = dr[(dr.index > s) & (dr.index <= e)]

        if len(sub) < 1:
            return _NA

        return float((1.0 + sub).prod() - 1.0)


    def _benchmark_cum_return_window(self, as_of: pd.Timestamp, months: int) -> object:
        """
        Rendement cumulé benchmark sur une fenêtre glissante
        """
        s, e, _, _ = self._benchmark_bounds_window(as_of, months)

        if s is None or e is None:
            return _NA

        dr = self._benchmark_daily_returns()
        sub = dr[(dr.index > s) & (dr.index <= e)]

        if len(sub) < 1:
            return _NA

        return float((1.0 + sub).prod() - 1.0)


    def _benchmark_cum_return_period_type(self, as_of: pd.Timestamp, period: str) -> object:
        """
        Rendement cumulé benchmark en YTD ou MTD
        """
        s, e, _, _ = self._benchmark_bounds_period_type(as_of, period)

        if s is None or e is None:
            return _NA

        dr = self._benchmark_daily_returns()
        sub = dr[(dr.index > s) & (dr.index <= e)]

        if len(sub) < 1:
            return _NA

        return float((1.0 + sub).prod() - 1.0)


    def _benchmark_ann_return_period(self, as_of: pd.Timestamp) -> object:
        """
        Rendement annualisé benchmark date de début du backtest à date de reporting
        """
        s, e, _, _ = self._benchmark_bounds_period(as_of)

        if s is None or e is None:
            return _NA

        dr = self._benchmark_daily_returns()
        sub = dr[(dr.index > s) & (dr.index <= e)]

        if len(sub) < 2:
            return _NA

        return float(sub.mean() * ANNUALIZATION)

    def _period_nav_bounds(self, as_of: pd.Timestamp, period: str):
        """
        Retourne (start_date, end_date, start_nav, end_nav) pour YTD/MTD.
        - start_date : dernier point disponible de l'année/mois précédent
        - end_date   : dernier point disponible de la periode courante <= as_of = proche date reporting si pas présent.
        """
        nav = self._nav.dropna().sort_index()
        if nav.empty:
            return None, None, None, None

        as_of = pd.Timestamp(as_of)
        idx = nav.index

        if period == "YTD":
            prev_mask = idx.year == (as_of.year - 1)
            curr_mask = (idx.year == as_of.year) & (idx <= as_of)
        elif period == "MTD":
            prev_month_cal = as_of.to_period("M") - 1
            prev_mask = (idx.year == prev_month_cal.year) & (idx.month == prev_month_cal.month)
            curr_mask = (idx.year == as_of.year) & (idx.month == as_of.month) & (idx <= as_of)
        else:
            return None, None, None, None

        prev_idx = idx[prev_mask]
        curr_idx = idx[curr_mask]
        if prev_idx.empty or curr_idx.empty:
            return None, None, None, None

        start_date = prev_idx[-1]
        end_date = curr_idx[-1]
        start_nav = float(nav.loc[start_date])
        end_nav = float(nav.loc[end_date])

        if os.getenv("BACKTEST_YTD_MTD_DEBUG", "0") == "1":
            if not hasattr(self, "_ytd_mtd_debug_seen"):
                self._ytd_mtd_debug_seen = set()
            key = (period, pd.Timestamp(as_of), start_date, end_date)
            if key not in self._ytd_mtd_debug_seen:
                self._ytd_mtd_debug_seen.add(key)
                print(
                    f"[YTD/MTD DEBUG] {period} as_of={as_of.date()} "
                    f"start={start_date.date()} ({start_nav:.2f}) "
                    f"end={end_date.date()} ({end_nav:.2f})"
                )

        return start_date, end_date, start_nav, end_nav

    def _ytd_start(self, as_of: pd.Timestamp) -> pd.Timestamp:
        """
        Dernier jour de bourse de l'année précédente
        Cherche la NAV (inclut les dates de rebalancement)
        """
        s, _, _, _ = self._period_nav_bounds(as_of, "YTD")
        return s

    def _mtd_start(self, as_of: pd.Timestamp) -> pd.Timestamp:
        """
        Dernier jour de bourse du mois précédent
        Cherche la NAV (inclut les dates de rebalancement)
        """
        s, _, _, _ = self._period_nav_bounds(as_of, "MTD")
        return s

    # ---- Rendements cumulés ----

    def _cum_return(self, dr: pd.Series) -> object:
        # cumulé total
        if len(dr) < 2:
            return _NA
        return float((1 + dr).prod() - 1)

    def _cum_return_window(self, dr, as_of, months) -> object:
        # cumulé sur fenetre glissante ex : 12 mois 
        sub = self._window(dr, as_of, months)
        return self._cum_return(sub) if len(sub) >= 2 else _NA

    def _cum_return_nav_window(self, as_of: pd.Timestamp, months: int) -> object:
        # rendement cumulé sur nav sur fenetre glissante
        nav = self._nav.dropna().sort_index()
        if nav.empty:
            return _NA

        start_target = pd.Timestamp(as_of) - pd.DateOffset(months=months)
        start_candidates = nav.index[nav.index <= start_target]
        end_candidates = nav.index[nav.index <= pd.Timestamp(as_of)]
        if start_candidates.empty or end_candidates.empty:
            return _NA

        nav_start = float(nav.loc[start_candidates[-1]])
        nav_end = float(nav.loc[end_candidates[-1]])
        return float(nav_end / nav_start - 1.0)

    def _cum_return_ytd(self, dr, as_of) -> object:
        # rendement cumulé spécifique pour YTD
        _, _, nav_s, nav_e = self._period_nav_bounds(as_of, "YTD")
        if nav_s is None or nav_e is None:
            return _NA
        return float(nav_e / nav_s - 1.0)

    def _cum_return_mtd(self, dr, as_of) -> object:
        # rendment cumulé spécifique pour MTD
        _, _, nav_s, nav_e = self._period_nav_bounds(as_of, "MTD")
        if nav_s is None or nav_e is None:
            return _NA
        return float(nav_e / nav_s - 1.0)

    # ---- Rendements annualisés ----
    
    def _ann_return(self, dr: pd.Series) -> object:
        # au moins avoir 2 returns pour calculer un rendement annualisé
        if len(dr) < 2:
            return _NA

        cumulative = (1 + dr).prod()
        n_years = len(dr) / ANNUALIZATION

        return float(cumulative ** (1 / n_years) - 1)

    """"
    def _ann_return(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        return float(dr.mean() * ANNUALIZATION)
    """
    
    # ---- CAGR ----

    def _cagr(self, start_nav, end_nav, dr) -> object:
        if len(dr) < 2:
            return _NA
        n_days = len(dr)
        n_years = n_days / ANNUALIZATION
        if n_years <= 0 or start_nav <= 0:
            return _NA
        return float((end_nav / start_nav) ** (1 / n_years) - 1)

    def _cagr_from_returns(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        cum = float((1 + dr).prod())
        n_years = len(dr) / ANNUALIZATION
        if n_years <= 0:
            return _NA
        return float(cum ** (1 / n_years) - 1)

    def _cagr_window(self, dr, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        return self._cagr_from_returns(sub)

    def _cagr_ytd(self, dr, as_of) -> object:
        cr = self._cum_return_ytd(dr, as_of)
        if cr == _NA:
            return _NA
        s, e, _, _ = self._period_nav_bounds(as_of, "YTD")
        if s is None or e is None:
            return _NA
        n_years = (e - s).days / 365.25
        if n_years <= 0:
            return _NA
        return float((1 + float(cr)) ** (1 / n_years) - 1)

    def _cagr_mtd(self, dr, as_of) -> object:
        cr = self._cum_return_mtd(dr, as_of)
        if cr == _NA:
            return _NA
        s, e, _, _ = self._period_nav_bounds(as_of, "MTD")
        if s is None or e is None:
            return _NA
        n_years = (e - s).days / 365.25
        if n_years <= 0:
            return _NA
        return float((1 + float(cr)) ** (1 / n_years) - 1)

    # ---- Sharpe ----

    def _sharpe(self, dr: pd.Series, ester: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        excess = dr - ester.reindex(dr.index).fillna(0)
        vol = excess.std()
        if vol < 1e-10:
            return _NA
        return float(excess.mean() / vol * np.sqrt(ANNUALIZATION))

    def _sharpe_window(self, dr, ester, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        if len(sub) < 2:
            return _NA
        return self._sharpe(sub, ester)

    # ---- Sortino ----

    def _sortino(self, dr: pd.Series, ester: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        excess = dr - ester.reindex(dr.index).fillna(0)
        downside = excess[excess < 0].std()
        if downside < 1e-10:
            return _NA
        return float(excess.mean() / downside * np.sqrt(ANNUALIZATION))

    def _sortino_window(self, dr, ester, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        if len(sub) < 2:
            return _NA
        return self._sortino(sub, ester)

    # ---- Tracking Error ----

    def _tracking_error(self, dr: pd.Series, ester: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        active = dr - ester.reindex(dr.index).fillna(0)
        return float(active.std() * np.sqrt(ANNUALIZATION))

    def _tracking_error_window(self, dr, ester, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        if len(sub) < 2:
            return _NA
        return self._tracking_error(sub, ester)

    # ---- Information Ratio ----

    def _info_ratio(self, dr: pd.Series, ester: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        active = dr - ester.reindex(dr.index).fillna(0)
        te = active.std()
        if te < 1e-10:
            return _NA
        return float(active.mean() / te * np.sqrt(ANNUALIZATION))

    def _info_ratio_window(self, dr, ester, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        if len(sub) < 2:
            return _NA
        return self._info_ratio(sub, ester)

    # ---- Calmar ----

    def _calmar(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        ann = self._ann_return(dr)
        if ann == _NA:
            return _NA
        mdd = self._max_drawdown(dr)
        if mdd == _NA or abs(float(mdd)) < 1e-10:
            return _NA
        return float(ann) / abs(float(mdd))

    # ---- VaR / CVaR ----

    def _var(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        return float(-np.percentile(dr.dropna(), CONF_LEVEL * 100))

    def _var_window(self, dr, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        return self._var(sub)

    def _cvar(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        threshold = np.percentile(dr.dropna(), CONF_LEVEL * 100)
        tail = dr[dr <= threshold]
        return float(-tail.mean()) if len(tail) > 0 else _NA

    def _cvar_window(self, dr, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        return self._cvar(sub)

    # ---- Volatilité ----

    def _volatility(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        return float(dr.std() * np.sqrt(ANNUALIZATION))

    def _vol_window(self, dr, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        return self._volatility(sub)

    # ---- Max Drawdown ----

    def _max_drawdown(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        cum = (1 + dr).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd.min())

    def _mdd_window(self, dr, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        return self._max_drawdown(sub)

    def _mdd_ytd(self, dr, as_of) -> object:
        s, e, _, _ = self._period_nav_bounds(as_of, "YTD")
        if s is None or e is None:
            return _NA
        sub = dr[(dr.index > s) & (dr.index <= e)]
        return self._max_drawdown(sub)

    def _mdd_mtd(self, dr, as_of) -> object:
        s, e, _, _ = self._period_nav_bounds(as_of, "MTD")
        if s is None or e is None:
            return _NA
        sub = dr[(dr.index > s) & (dr.index <= e)]
        return self._max_drawdown(sub)

    # ---- Corrélation ----

    def _corr(self, dr: pd.Series, ester: pd.Series) -> object:
        df = pd.concat([dr, ester], axis=1, join="inner").dropna()
        if df.shape[0] < 30:
            return _NA
        if df.iloc[:, 1].std() <= 1e-12:
            return _NA
        return float(df.iloc[:, 0].corr(df.iloc[:, 1]))

    def _empty_metrics(self) -> dict:
        """Retourne un dictionnaire de métriques vides (_NA)."""
        return {
            k: _NA
            for k in [
                "Rendement cumulé (période)", "Rendement cumulé 1 an",
                "Rendement cumulé 2 ans", "Rendement cumulé YTD",
                "Rendement cumulé MTD", "Rendement annualisé (période)",
                "CAGR (période)",
                "CAGR 1 an", "CAGR 2 ans", "CAGR YTD", "CAGR MTD",
                "Sharpe ratio (période)", "Sharpe ratio 1 an",
                "Sortino ratio (période)", "Sortino ratio 1 an",
                "Tracking Error (période)", "Tracking Error 1 an",
                "Tracking Error 3 ans", "Information Ratio (période)",
                "Information Ratio 1 an", "Information Ratio 2 ans",
                "Calmar ratio (période)", "VaR 95% (période)", "VaR 95% 1 an",
                "CVaR 95% (période)", "CVaR 95% 1 an", "Volatilité (période)",
                "Volatilité 1 an", "Volatilité 2 ans", "Max Drawdown (période)",
                "Max Drawdown 1 an", "Max Drawdown 2 ans", "Max Drawdown YTD",
                "Max Drawdown MTD", "Corrélation avec benchmark (période)",
                "Benchmark - Rendement cumulé (période)",
                "Benchmark - Rendement cumulé 1 an",
                "Benchmark - Rendement cumulé 2 ans",
                "Benchmark - Rendement cumulé YTD",
                "Benchmark - Rendement cumulé MTD",
                "Benchmark - Rendement annualisé (période)",
                "TC total (€)", "TC total (%)",
                "PnL (€)",
            ]
        }

    # ------------------------------------------------------------------
    # Analyses de composition du portefeuille
    # ------------------------------------------------------------------

    def get_portfolio_composition(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Retourne la composition détaillée du portefeuille à une date donnée.
        Colonnes : Ticker, Name, Country, Currency, Sector, Industry, Price, Weight.
        """
        if date not in self.result.weights:
            rebal_dates = sorted(self.result.weights.keys())
            eligible = [d for d in rebal_dates if d <= date]
            if not eligible:
                return pd.DataFrame()
            date = eligible[-1]

        weights = self.result.weights[date]
        info = self.loader.informations
        prices = self.loader.get_price_at(date, weights.index.tolist())

        rows = []
        for ticker, w in weights.items():
            row = {
                "Ticker": ticker,
                "Weight": w,
                "Price": float(prices.get(ticker, np.nan)),
            }
            if ticker in info.index:
                row["Name"] = info.loc[ticker, "Name"]
                row["Country"] = info.loc[ticker, "Country"]
                row["Currency"] = info.loc[ticker, "Currency"]
                row["Sector"] = info.loc[ticker, "Sector"]
                row["Industry"] = info.loc[ticker, "Industry"]
            else:
                row.update({"Name": "", "Country": "", "Currency": "", "Sector": "", "Industry": ""})
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df[["Ticker", "Name", "Country", "Currency", "Sector", "Industry", "Price", "Weight"]]
        return df.sort_values("Weight", ascending=False).reset_index(drop=True)

    def get_top_10_weights(self, date: pd.Timestamp) -> pd.DataFrame:
        """Top 10 positions par poids absolu."""
        comp = self.get_portfolio_composition(date)
        if comp.empty:
            return pd.DataFrame()
        comp["Abs Weight"] = comp["Weight"].abs()
        return comp.nlargest(10, "Abs Weight")[
            ["Ticker", "Name", "Sector", "Country", "Weight"]
        ].reset_index(drop=True)

    def get_top_10_return_contribution(self, date: pd.Timestamp):
        """
        Retourne deux tableaux :
        1) Top 5 contributeurs positifs (Contribution > 0), tri décroissant
        2) Top 5 contributeurs négatifs (Contribution < 0), tri croissant
        """

        rebal_dates = sorted(self.result.weights.keys())

        if date not in self.result.weights:
            eligible = [d for d in rebal_dates if d <= date]
            if not eligible:
                return pd.DataFrame(), pd.DataFrame()
            date = eligible[-1]

        if date not in rebal_dates:
            return pd.DataFrame(), pd.DataFrame()

        idx = rebal_dates.index(date)
        if idx == 0:
            return pd.DataFrame(), pd.DataFrame()

        prev_date = rebal_dates[idx - 1]
        prev_weights = self.result.weights[prev_date]
        all_tickers = prev_weights.index.tolist()

        daily_ret = self.loader.get_daily_returns(prev_date, date, all_tickers)
        if daily_ret.empty:
            return pd.DataFrame(), pd.DataFrame()

        period_ret = (1 + daily_ret.fillna(0)).prod() - 1
        ctr = prev_weights * period_ret.reindex(prev_weights.index).fillna(0)

        info = self.loader.informations

        df = pd.DataFrame({
            "Ticker": ctr.index,
            "Contribution": ctr.values})

        # Ajout des métadonnées (Name, Sector) si dispo
        if info is not None and not info.empty:
            df = df.merge(info[["Name", "Sector"]], left_on="Ticker", right_index=True, how="left")
        else:
            df["Name"] = ""
            df["Sector"] = ""

        # Top 5 positifs : tri décroissant
        top_pos = (df[df["Contribution"] > 0].sort_values("Contribution", ascending=False).head(5).reset_index(drop=True))

        # Top 5 négatifs : tri croissant
        top_neg = (df[df["Contribution"] < 0].sort_values("Contribution", ascending=True).head(5).reset_index(drop=True))
        return top_pos, top_neg

    def get_top_10_risk_contribution(self, date: pd.Timestamp):
        """
        Retourne deux tableaux :
        1) Top 5 contributeurs POSITIFS au risque (Risk Contribution > 0), tri décroissant
        2) Top 5 contributeurs NÉGATIFS au risque (Risk Contribution < 0), tri croissant
        """

        if date not in self.result.weights:
            rebal_dates = sorted(self.result.weights.keys())
            eligible = [d for d in rebal_dates if d <= date]
            if not eligible:
                return pd.DataFrame(), pd.DataFrame()
            date = eligible[-1]

        weights = self.result.weights[date]
        tickers = weights.index.tolist()

        # Lookback ~ 1 an (calendrier jour pour jour), covariance annualisée
        start_lookback = date - pd.DateOffset(days=365)
        daily_ret = self.loader.get_daily_returns(start_lookback, date, tickers)

        if daily_ret.empty or daily_ret.shape[0] < 20:
            return pd.DataFrame(), pd.DataFrame()

        w = weights.reindex(daily_ret.columns).fillna(0).values

        # Covariance annualisée
        cov = daily_ret.fillna(0).cov().values * ANNUALIZATION

        portfolio_var = float(w @ cov @ w)
        if portfolio_var < 1e-12:
            return pd.DataFrame(), pd.DataFrame()

        sigma_p = float(np.sqrt(portfolio_var))
        rc = w * (cov @ w) / sigma_p  # Risk Contributions (Euler / MCTR)

        df = pd.DataFrame({"Ticker": daily_ret.columns, "Risk Contribution": rc,
            "Weight": [float(weights.get(t, 0.0)) for t in daily_ret.columns]})

        # Ajout metadata
        info = self.loader.informations
        if info is not None and not info.empty:
            df = df.merge(info[["Name", "Sector"]], left_on="Ticker", right_index=True, how="left")
        else:
            df["Name"] = ""
            df["Sector"] = ""

        # Top 5 positifs : tri décroissant
        top_pos = (df[df["Risk Contribution"] > 0].sort_values("Risk Contribution", ascending=False).head(5).reset_index(drop=True))

        # Top 5 négatifs : tri croissant (plus négatif en premier)
        top_neg = (df[df["Risk Contribution"] < 0].sort_values("Risk Contribution", ascending=True).head(5).reset_index(drop=True))
        return top_pos, top_neg

    # ------------------------------------------------------------------
    # Visualisations Plotly - graphiques
    # ------------------------------------------------------------------

    def plot_cumulative_returns(self) -> go.Figure:
        """Courbes des rendements cumulés : portefeuille vs ESTER"""
        dr = self._dr
        if dr.empty:
            return go.Figure()

        nav_norm = (self._nav / self.result.initial_capital) * 100
        ester_cum = (1 + self._ester_dr.reindex(dr.index).fillna(0)).cumprod() * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nav_norm.index, y=nav_norm.values,
                name=f"Portefeuille ({METHODS_LABELS.get(self.result.method, self.result.method)})",
                line=dict(color=AMUNDI_TEAL, width=2)))
        fig.add_trace(go.Scatter(x=ester_cum.index, y=ester_cum.values,
                name="ESTER (benchmark)",
                line=dict(color=AMUNDI_NAVY, width=1.5, dash="dash")))
        fig.update_layout(
            title="Rendements cumulés : Portefeuille vs ESTER",
            xaxis_title="Date",
            yaxis_title="Valeur normalisée (base 100)",
            legend=dict(x=0.01, y=0.99, font=dict(color=AMUNDI_TEXT)),
            paper_bgcolor="white",
            plot_bgcolor=AMUNDI_LIGHT_BG,
            font=dict(color=AMUNDI_TEXT),
            xaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID, linecolor=AMUNDI_GRID),
            yaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID, linecolor=AMUNDI_GRID),
            template="plotly_white")
        return fig

    def plot_drawdowns(self) -> go.Figure:
        """Courbe des drawdowns du portefeuille sur toute la période"""
        dr = self._dr
        if dr.empty:
            return go.Figure()

        cum = (1 + dr).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values * 100, fill="tozeroy",
                name="Drawdown (%)",
                line=dict(color=AMUNDI_RED),
                fillcolor="rgba(227, 6, 19, 0.18)"))
        fig.update_layout(
            title="Drawdowns du portefeuille",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            paper_bgcolor="white",
            plot_bgcolor=AMUNDI_LIGHT_BG,
            font=dict(color=AMUNDI_TEXT),
            xaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID, linecolor=AMUNDI_GRID),
            yaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID, linecolor=AMUNDI_GRID),
            template="plotly_white")
        return fig

    def plot_historical_volatility(self) -> go.Figure:
        """Volatilité historique annualisée (de première date à date reporting) du portefeuille"""

        dr = self._dr.dropna().sort_index()
        if dr.empty:
            return go.Figure()

        # Volatilité historique (annualisée)
        vol_hist = dr.expanding(min_periods=2).std() * np.sqrt(ANNUALIZATION) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vol_hist.index, y=vol_hist.values,
                name="Volatilité historique (%)",
                line=dict(color=AMUNDI_NAVY, width=2)))
        fig.update_layout(
            title="Volatilité historique annualisée",
            xaxis_title="Date",
            yaxis_title="Volatilité (%)",
            paper_bgcolor="white",
            plot_bgcolor=AMUNDI_LIGHT_BG,
            font=dict(color=AMUNDI_TEXT),
            xaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID, linecolor=AMUNDI_GRID),
            yaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID, linecolor=AMUNDI_GRID),
            template="plotly_white")
        return fig

    def plot_historical_correlation(self) -> go.Figure:
        """Corrélation historique entre le portefeuille et l'ESTRON Index"""

        dr = self._dr.dropna().sort_index()
        if dr.empty:
            return go.Figure()

        # Rendements benchmark (ESTRON)
        bench = self._benchmark_daily_returns().dropna().sort_index()

        # Alignement strict sur dates communes
        df = pd.concat([dr.rename("portfolio"), bench.rename("benchmark")], axis=1, join="inner").dropna()

        if df.shape[0] < 30:
            return go.Figure()

        # Corrélation historique
        corr_hist = df["portfolio"].expanding(min_periods=30).corr(df["benchmark"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=corr_hist.index, y=corr_hist.values,
                name="Corrélation historique",
                line=dict(color=AMUNDI_TEAL, width=2)))

        fig.add_hline(y=0, line_dash="dash", line_color=AMUNDI_GRID)

        fig.update_layout(
            title="Corrélation historique Portefeuille vs ESTRON",
            xaxis_title="Date",
            yaxis_title="Corrélation",
            paper_bgcolor="white",
            plot_bgcolor=AMUNDI_LIGHT_BG,
            font=dict(color=AMUNDI_TEXT),
            xaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID, linecolor=AMUNDI_GRID),
            yaxis=dict(range=[-1, 1], gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID, linecolor=AMUNDI_GRID),
            template="plotly_white")
        return fig

    def plot_composition_barcharts(self, date: pd.Timestamp) -> dict:
        """
        Bar charts de la composition du portefeuille à une date donnée.
        Dimensions : 'Sector', 'Country', 'Industry', 'Currency'.
        """
        comp = self.get_portfolio_composition(date)
        if comp.empty:
            return {}

        figures = {}
        for dim in ["Sector", "Country", "Industry", "Currency"]:
            grouped = comp.groupby(dim)["Weight"].sum().sort_values()
            fig = go.Figure(go.Bar(x=grouped.values * 100, y=grouped.index,
                    orientation="h",
                    marker_color=[
                        AMUNDI_TEAL if v >= 0 else AMUNDI_RED
                        for v in grouped.values]))
            
            fig.update_layout(
                title=f"Poids par {dim} — {date.date()}",
                xaxis_title="Poids (%)",
                yaxis_title=dim,
                paper_bgcolor="white",
                plot_bgcolor=AMUNDI_LIGHT_BG,
                font=dict(color=AMUNDI_TEXT),
                xaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID, linecolor=AMUNDI_GRID),
                yaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID, linecolor=AMUNDI_GRID),
                template="plotly_white")
            
            figures[dim] = fig
        return figures

    def plot_pnl(self) -> go.Figure:
        """PnL cumulé du portefeuille"""
        
        dr = self._dr
        if dr.empty:
            return go.Figure()

        pnl = self._nav - self.result.initial_capital

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pnl.index, y=pnl.values,
                name="PnL cumulé (€)",
                line=dict(color=AMUNDI_TEAL, width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 163, 173, 0.12)"))
        fig.add_hline(y=0, line_dash="dash", line_color=AMUNDI_GRID, line_width=1)
        fig.update_layout(
            title=f"Portfolio PnL — {METHODS_LABELS.get(self.result.method, self.result.method)} (€)",
            xaxis_title="Date",
            yaxis_title="PnL cumulé (€)",
            paper_bgcolor="white",
            plot_bgcolor=AMUNDI_LIGHT_BG,
            font=dict(color=AMUNDI_TEXT),
            xaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID, linecolor=AMUNDI_GRID),
            yaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID, linecolor=AMUNDI_GRID),
            template="plotly_white")
        return fig

    def _calendar_returns_from_levels(self, levels: pd.Series) -> pd.Series:
        """
        Rendement annuel calendrier sur le dernier point disponible de chaque année
        """
        if levels.empty:
            return pd.Series(dtype=float)

        s = levels.dropna().sort_index()
        if s.empty:
            return pd.Series(dtype=float)

        year_end = s.groupby(s.index.year).last()
        calendar = year_end / year_end.shift(1) - 1.0
        calendar.index = calendar.index.astype(int)
        return calendar

    def compute_calendar_returns(self) -> pd.DataFrame:
        """Calcule les calendar returns annuels du portefeuille"""
        portfolio_calendar = self._calendar_returns_from_levels(self._nav)
        years = sorted(portfolio_calendar.index.tolist())
        calendar_returns = pd.DataFrame(index=["Portfolio"], columns=years, dtype=float)
        if len(years) == 0:
            return calendar_returns

        calendar_returns.loc["Portfolio", portfolio_calendar.index] = portfolio_calendar.values
        return calendar_returns

    def plot_calendar_returns_heatmap(self) -> go.Figure:
        """Heatmap des rendements annuels calendrier"""

        calendar_returns = self.compute_calendar_returns()

        if calendar_returns.empty or calendar_returns.shape[1] == 0:
            return go.Figure()

        # supp les années sans aucune valeur (colonne full NaN)
        calendar_returns = calendar_returns.dropna(axis=1, how="all")

        if calendar_returns.shape[1] == 0:
            return go.Figure()

        # afficher en %
        z = calendar_returns.values * 100.0

        zmax = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
        if zmax == 0:
            zmax = 1.0

        text = np.empty_like(z, dtype=object)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                text[i, j] = "" if pd.isna(z[i, j]) else f"{z[i, j]:.2f}%"

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=calendar_returns.columns.tolist(),
                y=calendar_returns.index.tolist(),
                text=text,
                texttemplate="%{text}",
                textfont={"size": 12, "color": AMUNDI_DARK},
                colorscale=[[0.0, AMUNDI_RED],[0.5, "#FFFFFF"],[1.0, AMUNDI_TEAL],],
                zmin=-zmax,
                zmax=zmax,
                colorbar=dict(
                    title=dict(text="Return (%)", font=dict(color=AMUNDI_TEXT)),
                    tickfont=dict(color=AMUNDI_TEXT),
                    ),
                hovertemplate="%{y} | %{x}: %{z:.2f}%<extra></extra>"))

        fig.update_layout(
            title="Calendar Returns (Annual) – Portfolio",
            xaxis_title="Year",
            yaxis_title="",
            xaxis=dict(
                tickmode="array",
                tickvals=calendar_returns.columns.tolist(),
                ticktext=[str(y) for y in calendar_returns.columns.tolist()],
                gridcolor=AMUNDI_GRID,
                zerolinecolor=AMUNDI_GRID,
                linecolor=AMUNDI_GRID,),
            paper_bgcolor="white",
            plot_bgcolor=AMUNDI_LIGHT_BG,
            template="plotly_white",
            font=dict(family="Arial", size=12, color=AMUNDI_TEXT),
            yaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID, linecolor=AMUNDI_GRID))

        return fig

    # ------------------------------------------------------------------
    # Exports CSV à modifier pour diminuer la taille
    # ------------------------------------------------------------------

    def _get_output_dir(self) -> Path:
        """Crée et retourne le sous-dossier de stockage pour cette stratégie"""
        label = METHODS_LABELS.get(self.result.method, self.result.method)
        rebal = sorted(self.result.weights.keys())
        start_str = rebal[0].strftime("%Y%m%d") if rebal else "NA"
        end_str = rebal[-1].strftime("%Y%m%d") if rebal else "NA"
        folder_name = f"MomentumDual_{label}_{start_str}_{end_str}"
        out_dir = self._STOCKAGE_ROOT / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def export_bbu_csv(self, output_dir: Optional[Path] = None) -> Path:
        """
        Exporte le fichier CSV Bloomberg BBU (Date, Ticker, Weights)
        Format ok pour l'upload dans BBU.
        """
        if output_dir is None:
            output_dir = self._get_output_dir()

        label = METHODS_LABELS.get(self.result.method, self.result.method)
        rows = []
        for date, weights in self.result.weights.items():
            for ticker, w in weights.items():
                rows.append({"Date": date.strftime("%Y-%m-%d"), "Ticker": ticker, "Weights": round(w, 6)})

        df = pd.DataFrame(rows)[["Date", "Ticker", "Weights"]]
        path = output_dir / f"bbu_MomentumDual_{label}.csv"
        df.to_csv(path, index=False)
        print(f"[Report] BBU CSV exporté : {path}")
        return path

    def export_detailed_parquet(self, output_dir: Optional[Path] = None) -> Path:
        """
        Exporte la composition enrichie en format Parquet le fichier des positions détailllées du portefeuille 
        à chq date reporting = (Date, Ticker, Name, Country, Currency, Sector, Industry, Price, Weight)
        """
        if output_dir is None:
            output_dir = self._get_output_dir()

        label = METHODS_LABELS.get(self.result.method, self.result.method)
        all_rows = []
        for date in sorted(self.result.weights.keys()):
            comp = self.get_portfolio_composition(date)
            if not comp.empty:
                comp.insert(0, "Date", date.strftime("%Y-%m-%d"))
                all_rows.append(comp)

        if not all_rows:
            return output_dir / "empty.parquet"

        df = pd.concat(all_rows, ignore_index=True)
        path = output_dir / f"composition_detaillee_MomentumDual_{label}.parquet"
        df.to_parquet(path, index=False)
        print(f"[Report] Composition détaillée exportée : {path}")
        return path

    def export_metrics_parquet(self, metrics_df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
        """Exporte le tableau des métriques de performance en Parquet"""
        if output_dir is None:
            output_dir = self._get_output_dir()

        df = metrics_df.copy()
        df = df.replace(_NA, np.nan)  # _NA == "--"  --> remplacer "--" par NaN

        # Convertit chaque colonne en numérique
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        label = METHODS_LABELS.get(self.result.method, self.result.method)
        path = output_dir / f"metriques_MomentumDual_{label}.parquet"
        df.to_parquet(path)  # index conservé comme avant
        print(f"[Report] Métriques exportées : {path}")
        return path

    def export_nav_parquet(self, output_dir: Optional[Path] = None) -> Path:
        """Exporte la série de NAV journalière en Parquet"""
        if output_dir is None:
            output_dir = self._get_output_dir()

        label = METHODS_LABELS.get(self.result.method, self.result.method)
        nav_df = self._nav.rename("NAV").to_frame()
        nav_df.index.name = "Date"
        path = output_dir / f"nav_MomentumDual_{label}.parquet"
        nav_df.to_parquet(path)
        print(f"[Report] NAV exportée : {path}")
        return path

    # ------------------------------------------------------------------
    # Run complet du reporting
    # ------------------------------------------------------------------

    def run_full_report(self, output_dir: Optional[Path] = None) -> str:
        """
        Génère l'ensemble du reporting et sauvegarde tous les fichiers.

        les 4 fichiers souhaités
        ---------------
        - CSV BBU
        - Parquet composition enrichie
        - Parquet métriques
        - Parquet NAV

        Retourne
        --------
        str : chemin du dossier de sortie
        """
        if output_dir is None:
            output_dir = self._get_output_dir()

        print(f"\n[ReportingEngine] Génération du reporting dans : {output_dir}")

        # 1. Métriques
        metrics_df = self.compute_all_metrics()
        self.export_metrics_parquet(metrics_df, output_dir)

        # 2. Exports
        self.export_bbu_csv(output_dir)
        self.export_detailed_parquet(output_dir)
        self.export_nav_parquet(output_dir)

        print(f"[ReportingEngine] Reporting complet généré dans : {output_dir}\n")
        return str(output_dir)