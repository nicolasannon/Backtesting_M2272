# Backtesting_M2272/src/backtesting/signals.py

# ------------------------------------------------------------------
# Librairies
# ------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ------------------------------------------------------------------
# Présenatations
# ------------------------------------------------------------------
"""
Quatre méthodes d'allocation pour le portefeuille long/short.

Convention de poids
-------------------
- Long  : poids positifs, somme = +1.0
- Short : poids négatifs, somme = -1.0
- Net   : 0.0 (neutralité de marché)

Méthodes implémentées
---------------------
(a) equal_weight    : équipondéré
(b) risk_parity     : ERC 
(c) min_variance    : Markowitz — minimise la variance du portefeuille
(d) signal_weight   : pondération par la qualité du signal

Contraintes UCITS 5/10/40
"""


class AllocationEngine:
    """
    Calcule les poids du portefeuille long/short selon différentes méthodes

    Paramètres
    ----------
    data_loader : DataLoader
    transaction_cost : float
        Frais de transaction (défaut : 5 bps = 0.0005)
    cov_lookback : int
        Nombre de jours pour l'estimation de la covariance (défaut : 252)
    """

    _MAX_WEIGHT = 0.10  

    def __init__(self, data_loader, transaction_cost: float = 0.0005, cov_lookback: int = 252):
        self.loader = data_loader
        self.tc = transaction_cost
        self.cov_lookback = cov_lookback

    def compute_weights(self, method: str, date: pd.Timestamp, long_tickers: list, short_tickers: list, signals: pd.Series = None) -> pd.Series:
        """
        Calcule les poids du portefeuille selon la méthode choisie,
        puis applique les contraintes UCITS 5/10/40

        Paramètres
        ----------
        method        : 'equal_weight' | 'risk_parity' | 'min_variance' | 'signal_weight'
        date          : date de rebalancement
        long_tickers  : tickers en position longue
        short_tickers : tickers en position courte
        signals       : scores standardisés (requis pour signal_weight)

        Retourne
        --------
        pd.Series : poids (long > 0, short < 0)
        """
        if not long_tickers or not short_tickers:
            return pd.Series(dtype=float)

        dispatch = {
            "equal_weight": self._equal_weight,
            "risk_parity": self._risk_parity,
            "min_variance": self._min_variance,
            "signal_weight": self._signal_weight}
        
        if method not in dispatch:
            raise ValueError(f"Méthode '{method}' inconnue. Options : {list(dispatch)}")

        if method == "equal_weight":
            weights = self._equal_weight(long_tickers, short_tickers)
        elif method == "signal_weight":
            weights = self._signal_weight(long_tickers, short_tickers, signals)
        else:
            weights = dispatch[method](long_tickers, short_tickers, date)

        return self._apply_ucits(weights)

    def _equal_weight(self, long_tickers: list, short_tickers: list) -> pd.Series:
        weights = {}
        for t in long_tickers:
            weights[t] = 1.0 / len(long_tickers)
        for t in short_tickers:
            weights[t] = -1.0 / len(short_tickers)
        return pd.Series(weights)


    def _risk_parity(self, long_tickers: list, short_tickers: list, date: pd.Timestamp) -> pd.Series:
        """
        Pondération inversement proportionnelle à la volatilité
        Chaque titre contribue équitablement au risque du portefeuille
        """
        weights = {}
        for tickers, sign in [(long_tickers, 1), (short_tickers, -1)]:
            vols = self._get_volatilities(date, tickers)
            if vols.empty or (vols <= 0).all():
                for t in tickers:
                    weights[t] = sign / len(tickers)
                continue
            inv_vol = 1.0 / vols.clip(lower=1e-8)
            w = inv_vol / inv_vol.sum()
            for t in w.index:
                weights[t] = sign * float(w[t])
        return pd.Series(weights)

    def _min_variance(self, long_tickers: list, short_tickers: list, date: pd.Timestamp) -> pd.Series:
        
        """Minimise la variance"""
        weights = {}
        for tickers, sign in [(long_tickers, 1), (short_tickers, -1)]:
            w = self._solve_min_var(date, tickers)
            for t, v in w.items():
                weights[t] = sign * v
        return pd.Series(weights)

    def _solve_min_var(self, date: pd.Timestamp, tickers: list) -> dict:
        cov = self._get_cov_matrix(date, tickers)
        if cov.empty:
            return {t: 1 / len(tickers) for t in tickers}

        Sigma = cov.values
        n = len(Sigma)
        w0 = np.ones(n) / n
        bounds = [(0.0, self._MAX_WEIGHT)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

        res = minimize(
            lambda w: float(w @ Sigma @ w),
            w0,
            jac=lambda w: 2.0 * Sigma @ w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if res.success:
            return dict(zip(cov.index, res.x))
        return {t: 1 / n for t in cov.index}

    def _signal_weight(self, long_tickers: list, short_tickers: list, signals: pd.Series) -> pd.Series:
        """Poids proportionnels au score du signal (|z-score|)"""
        weights = {}

        long_scores = (
            signals.reindex(long_tickers).fillna(0).clip(lower=0)
            if signals is not None
            else pd.Series(0.0, index=long_tickers)
        )
        total_long = long_scores.sum()
        if total_long > 0:
            for t in long_tickers:
                weights[t] = float(long_scores.get(t, 0)) / total_long
        else:
            for t in long_tickers:
                weights[t] = 1.0 / len(long_tickers)

        short_scores = (signals.reindex(short_tickers).fillna(0).clip(upper=0).abs()
            if signals is not None
            else pd.Series(0.0, index=short_tickers))
        
        total_short = short_scores.sum()
        if total_short > 0:
            for t in short_tickers:
                weights[t] = -float(short_scores.get(t, 0)) / total_short
        else:
            for t in short_tickers:
                weights[t] = -1.0 / len(short_tickers)

        return pd.Series(weights)

    def _apply_ucits(self, weights: pd.Series) -> pd.Series:
        """
        Applique la règle UCITS 5/10/40 :
        """
        if weights.empty:
            return weights

        w = weights.copy()

        for sign in [1, -1]:
            idx = w[w * sign > 0].index
            if len(idx) == 0:
                continue
            if sign == 1:
                w[idx] = w[idx].clip(upper=self._MAX_WEIGHT)
            else:
                w[idx] = w[idx].clip(lower=-self._MAX_WEIGHT)
            total = w[idx].abs().sum()
            if total > 0:
                w[idx] = w[idx] / total

        abs_w = w.abs()
        big_mask = abs_w > 0.05
        if big_mask.any():
            total_big = abs_w[big_mask].sum()
            if total_big > 0.40:
                scale = 0.40 / total_big
                w[big_mask] = w[big_mask] * scale
                for sign in [1, -1]:
                    idx = w[w * sign > 0].index
                    total = w[idx].abs().sum()
                    if total > 0:
                        w[idx] = w[idx] / total

        return w

    def _get_cov_matrix(self, date: pd.Timestamp, tickers: list) -> pd.DataFrame:
        """
        Matrice de covariance annualisée sur les `cov_lookback` derniers jours
        Shrinkage de Ledoit-Wolf simplifié pour stabilité numérique
        """
        available = self.loader.prices.index[self.loader.prices.index <= date]
        if len(available) == 0:
            return pd.DataFrame()

        end_idx = available[-1]
        pos = self.loader.prices.index.get_loc(end_idx)
        start_pos = max(0, pos - self.cov_lookback)
        start_idx = self.loader.prices.index[start_pos]

        cols = [t for t in tickers if t in self.loader.prices.columns]
        if not cols:
            return pd.DataFrame()

        sub = self.loader.prices.loc[start_idx:end_idx, cols].ffill()
        daily_ret = sub.pct_change().dropna(how="all")


        daily_ret = daily_ret.dropna(axis=1, thresh=max(10, int(0.5 * len(daily_ret))))

        if daily_ret.shape[0] < 20 or daily_ret.shape[1] < 2:
            n = len(cols)
            return pd.DataFrame(np.eye(n) * 0.04, index=cols, columns=cols)

        cov = daily_ret.cov() * 252
        alpha = 0.1
        D = np.diag(np.diag(cov.values))
        cov_shrunk = (1 - alpha) * cov.values + alpha * D
        return pd.DataFrame(cov_shrunk, index=cov.index, columns=cov.columns)

    def _get_volatilities(self, date: pd.Timestamp, tickers: list) -> pd.Series:
        cov = self._get_cov_matrix(date, tickers)
        if cov.empty:
            return pd.Series(0.20, index=tickers)
        return pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index)
