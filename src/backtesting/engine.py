# Backtesting_M2272/src/backtesting/engine.py

# ------------------------------------------------------------------
# Librairies
# ------------------------------------------------------------------

# externes
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .signals import SignalEngine
from .allocation import AllocationEngine

# ------------------------------------------------------------------
# Présenatations
# ------------------------------------------------------------------
"""
Logique de rebalancement

-------------------------
À chaque date de rebalancement (dernier jour de bourse du mois) :
  1. Univers dynamique : tickers présents dans l'indice à cette date
  2. Signal : Momentum Dual sectoriellement neutralisé
  3. Sélection : top 40% long, bottom 40% short (intra-sectoriel)
  4. Allocation : méthode choisie (EW, ERC, MinVar, SignalWeight)
  5. Contraintes : UCITS 5/10/40, frais de transaction
  6. Retours : calculés pour chaque jour jusqu'au prochain rebalancement

Capital : paramétrable (défaut 1 000 000 €)
Frais   : appliqués sur le turnover à chaque rebalancement
"""


@dataclass
class BacktestResult:
    """Résultats complets d'un run de backtesting"""

    method: str
    nav: pd.Series                    # NAV journalière indexée par Date (€)
    nav_gross: pd.Series              # NAV brute (sans TC) indexée par Date (€)
    weights: dict                     # {date: pd.Series} poids à chaque rebalancement
    daily_returns: pd.Series          # Retours journaliers nets du portefeuille
    daily_returns_gross: pd.Series    # Retours journaliers bruts (sans TC)
    transaction_costs_pct: dict       # date: TC en % du NAV
    transaction_costs_eur: dict       # date: TC en €
    long_tickers: dict                # date: list[str]
    short_tickers: dict               # date: list[str]
    initial_capital: float


class BacktestEngine:
    """
    Moteur de backtesting pour la stratégie long/short

    Paramètres
    ----------
    data_loader : DataLoader
    initial_capital : float
        Capital initial en € (défaut : 1 000 000 €).
    transaction_cost : float
        Frais de transaction en décimal (défaut : 5 bps = 0.0005).
    selection_quantile : float
        Quantile de sélection intra-sectoriel (défaut : 0.40 = top/bottom 40%).
    sector_min : dict | None
        Poids minimum par secteur {secteur: float}. None = pas de contrainte.
    sector_max : dict | None
        Poids maximum par secteur {secteur: float}. None = pas de contrainte.
    """

    def __init__(self, data_loader, initial_capital: float = 1_000_000,
        transaction_cost: float = 0.0005,
        selection_quantile: float = 0.40,
        sector_min: Optional[dict] = None,
        sector_max: Optional[dict] = None):
        
        self.loader = data_loader
        self.initial_capital = initial_capital
        self.tc_rate = transaction_cost
        self.quantile = selection_quantile
        self.sector_min = sector_min or {}
        self.sector_max = sector_max or {}

        self.signal_engine = SignalEngine(data_loader)
        self.allocation_engine = AllocationEngine(data_loader, transaction_cost)

    def run(self, method: str = "equal_weight") -> BacktestResult:
        """
        Lance le backtesting pour la méthode d'allocation spécifiée

        Méthodes disponibles :
          'equal_weight', 'risk_parity', 'min_variance', 'signal_weight'
        """
        print(f"\n[BacktestEngine] Démarrage — méthode : {method}")

        rebal_dates = self.loader.rebalancing_dates
        nav = float(self.initial_capital)
        nav_gross = float(self.initial_capital)

        all_nav: dict = {}
        all_nav_gross: dict = {}
        all_weights: dict = {}
        all_long: dict = {}
        all_short: dict = {}
        all_tc_pct: dict = {}
        all_tc_eur: dict = {}
        all_daily_ret: dict = {}
        all_daily_ret_gross: dict = {}

        prev_weights = pd.Series(dtype=float)

        for i, date in enumerate(rebal_dates):
            print(f"  [{i+1:>3}/{len(rebal_dates)}] {date.date()} ", end="", flush=True,)

            universe = self.loader.get_universe(date)
            if len(universe) < 10:
                print("skip (univers insuffisant)")
                continue

            signals = self.signal_engine.compute(date, universe)
            if signals.empty:
                print("skip (pas de signal)")
                continue

            long_tickers, short_tickers = self._select_long_short(
                universe, signals
            )
            if not long_tickers or not short_tickers:
                print("skip (sélection insuffisante)")
                continue

            new_weights = self.allocation_engine.compute_weights(
                method=method,
                date=date,
                long_tickers=long_tickers,
                short_tickers=short_tickers,
                signals=signals)
            
            if new_weights.empty:
                print("skip (pondération vide)")
                continue

            total_weight = float(new_weights.sum())
            tol = 1e-10
            if abs(total_weight) > tol:
                long_mask = new_weights > 0
                short_mask = new_weights < 0
                sum_long = float(new_weights[long_mask].sum())
                sum_short = float(new_weights[short_mask].sum())

                if sum_long > 0 and sum_short < 0:
                    gross = sum_long + abs(sum_short)
                    target_long = gross / 2.0
                    target_short = -gross / 2.0
                    new_weights.loc[long_mask] *= target_long / sum_long
                    new_weights.loc[short_mask] *= target_short / sum_short

            assert abs(float(new_weights.sum())) < 1e-12

            if prev_weights.empty:
                drifted = pd.Series(0.0, index=new_weights.index)
            else:
                drifted = prev_weights.reindex(new_weights.index).fillna(0.0)

            turnover = (new_weights - drifted).abs().sum()
            tc_pct = self.tc_rate * turnover
            tc_eur = tc_pct * nav
            nav -= tc_eur

            all_tc_pct[date] = tc_pct
            all_tc_eur[date] = tc_eur
            all_weights[date] = new_weights
            all_long[date] = long_tickers
            all_short[date] = short_tickers

            all_nav[date] = nav
            all_nav_gross[date] = nav_gross

            next_date = (rebal_dates[i + 1] if i + 1 < len(rebal_dates) else None)
            
            if next_date is not None and next_date > date:
                all_tickers = long_tickers + short_tickers
                daily_ret = self._compute_daily_portfolio_returns(new_weights, date, next_date, all_tickers)
                
                for dt, r in daily_ret.items():
                    nav *= (1.0 + r)
                    nav_gross *= (1.0 + r)
                    all_daily_ret[dt] = r
                    all_daily_ret_gross[dt] = r
                    all_nav[dt] = nav
                    all_nav_gross[dt] = nav_gross

                prev_weights = self._compute_drifted_weights(new_weights, date, next_date, all_tickers)
            else:
                prev_weights = new_weights.copy()

            print(f"NAV={nav:>12,.0f}€  Long={len(long_tickers):>3}  Short={len(short_tickers):>3}")

        nav_series = pd.Series(all_nav, dtype=float).sort_index()
        nav_gross_series = pd.Series(all_nav_gross, dtype=float).sort_index()
        dr_series = pd.Series(all_daily_ret, dtype=float).sort_index()
        dr_gross_series = pd.Series(all_daily_ret_gross, dtype=float).sort_index()

        print(f"\n[BacktestEngine] Terminé — NAV finale : {nav:,.2f} €\n")

        return BacktestResult(
            method=method,
            nav=nav_series,
            nav_gross=nav_gross_series,
            weights=all_weights,
            daily_returns=dr_series,
            daily_returns_gross=dr_gross_series,
            transaction_costs_pct=all_tc_pct,
            transaction_costs_eur=all_tc_eur,
            long_tickers=all_long,
            short_tickers=all_short,
            initial_capital=self.initial_capital)

    def run_all(self) -> dict:
        """
        Lance le backtesting pour les méthodes d'allocation

        Retourne
        --------
        dict : {method_name: BacktestResult}
        """
        methods = ["equal_weight", "risk_parity", "min_variance", "signal_weight"]
        
        results = {}
        for m in methods:
            results[m] = self.run(m)
        return results

    def _select_long_short(self, universe: list, signals: pd.Series) -> tuple:
        """
        Sélection intra-sectorielle : top 40% long, bottom 40% short

        Neutralité sectorielle structurelle :
        - Pour chaque secteur, même nombre de longs et de shorts
        - L'exposition nette par secteur est proche de zéro
        """
        info = self.loader.informations
        long_tickers = []
        short_tickers = []

        sig_universe = signals.reindex(universe).dropna()
        if sig_universe.empty:
            return [], []

        tickers_with_info = info.index.intersection(sig_universe.index)
        sectors = info.loc[tickers_with_info, "Sector"].dropna()

        for sector, group in sectors.groupby(sectors):
            sector_tickers = group.index.intersection(sig_universe.index)
            sector_signals = sig_universe.loc[sector_tickers].sort_values(ascending=False)
            n = len(sector_signals)

            if n < 3:
                continue 

            n_selected = max(1, int(np.floor(n * self.quantile)))
            long_tickers.extend(sector_signals.iloc[:n_selected].index.tolist())
            short_tickers.extend(sector_signals.iloc[-n_selected:].index.tolist())

        return long_tickers, short_tickers

    def _compute_daily_portfolio_returns(self, weights: pd.Series, date_start: pd.Timestamp, date_end: pd.Timestamp, tickers: list) -> pd.Series:
        
        if date_start >= date_end:
            return pd.Series(dtype=float)

        daily_ret = self.loader.get_daily_returns(date_start, date_end, tickers)
        if daily_ret.empty:
            return pd.Series(dtype=float)

        w = weights.reindex(daily_ret.columns).fillna(0.0)
        portfolio_returns = (daily_ret.fillna(0.0) * w).sum(axis=1)
        return portfolio_returns.to_dict()

    def _compute_drifted_weights(self, weights: pd.Series, date_start: pd.Timestamp, date_end: pd.Timestamp, tickers: list) -> pd.Series:
        
        daily_ret = self.loader.get_daily_returns(date_start, date_end, tickers)
        if daily_ret.empty:
            return weights.copy()

        stock_cum_ret = (1.0 + daily_ret.fillna(0.0)).prod() - 1.0
        w_aligned = weights.reindex(daily_ret.columns).fillna(0.0)
        period_ret = float((daily_ret.fillna(0.0) * w_aligned).sum(axis=1).add(1).prod() - 1)

        denom = 1.0 + period_ret if abs(1.0 + period_ret) > 1e-10 else 1.0
        drifted = weights * (1.0 + stock_cum_ret.reindex(weights.index).fillna(0.0)) / denom
        return drifted
