# Backtesting_M2272/src/request/request_historical_data.py

# ------------------------------------------------------------------
# Librairies
# ------------------------------------------------------------------

import blpapi
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Présenatations
# ------------------------------------------------------------------
"""
- Téléchargment groupé des historiques de prix "PX_LAST" pour une liste de tickers donnée, via l'API Bloomberg (blpapi).
"""

class BloombergPXLastHistory:
    """
    Télécharge PX_LAST (historique daily) pour une liste de tickers,
    renvoie un DataFrame long: Date | Ticker | Price,
    et sauvegarde en parquet
    """

    def __init__(self, host="localhost", port=8194, verbose=True):
        self.host = host
        self.port = port
        self.verbose = verbose

        current_file = Path(__file__).resolve()
        self.project_root = current_file.parent.parent.parent

        self.folder = self.project_root / "data" / "initial_data"
        self.folder.mkdir(parents=True, exist_ok=True)

        self.pattern = "historical_price_memb_index.parquet"

    def _start_session(self) -> blpapi.Session:
        options = blpapi.SessionOptions()
        options.setServerHost(self.host)
        options.setServerPort(self.port)

        session = blpapi.Session(options)
        if not session.start():
            raise ConnectionError("Impossible de démarrer la session Bloomberg.")
        if not session.openService("//blp/refdata"):
            raise ConnectionError("Impossible d'ouvrir le service //blp/refdata.")
        return session

    @staticmethod
    def _to_yyyymmdd(date_str: str) -> str:
        # accepte "YYYY-MM-DD" et renvoie "YYYYMMDD"
        return date_str.replace("-", "")

    def fetch(self, tickers: list[str], start: str, end: str, frequency: str = "DAILY") -> pd.DataFrame:
        """
        Parameters
        ----------
        tickers : list[str]
        start : str "YYYY-MM-DD" ou "YYYYMMDD"
        end : str "YYYY-MM-DD" ou "YYYYMMDD"
        frequency : str "DAILY" (default)

        Returns
        -------
        pd.DataFrame : colonnes Date, Ticker, Price (dropna appliqué)
        """

        start_ = self._to_yyyymmdd(start)
        end_ = self._to_yyyymmdd(end)

        session = self._start_session()
        service = session.getService("//blp/refdata")

        all_rows = []

        # Simple et robuste : 1 ticker par requête (évite les réponses trop grosses)
        for t in tickers:
            request = service.createRequest("HistoricalDataRequest")
            request.getElement("securities").appendValue(t)
            request.getElement("fields").appendValue("PX_LAST")

            request.set("startDate", start_)
            request.set("endDate", end_)
            request.set("periodicitySelection", frequency)

            session.sendRequest(request)

            while True:
                event = session.nextEvent()

                for msg in event:
                    if not msg.hasElement("securityData"):
                        continue

                    sec_data = msg.getElement("securityData")
                    security_name = sec_data.getElementAsString("security")

                    field_data = sec_data.getElement("fieldData")

                    for i in range(field_data.numValues()):
                        row = field_data.getValueAsElement(i)

                        # Date
                        dt = row.getElementAsDatetime("date")
                        dt_str = pd.to_datetime(dt).strftime("%Y-%m-%d")

                        # PX_LAST peut être absent certains jours => on skip (= dropna)
                        if row.hasElement("PX_LAST"):
                            try:
                                px = row.getElementAsFloat("PX_LAST")
                            except Exception:
                                # fallback si type différent
                                px = float(row.getElement("PX_LAST").getValue())

                            all_rows.append([dt_str, security_name, px])

                if event.eventType() == blpapi.Event.RESPONSE:
                    break

            if self.verbose:
                print(f"OK: {t} -> {sum(1 for r in all_rows if r[1] == security_name)} points")

        session.stop()

        df = pd.DataFrame(all_rows, columns=["Date", "Ticker", "Price"])

        df = df.dropna(subset=["Price"])

        return df

    def save_parquet(self, df: pd.DataFrame) -> Path:
        """
        Sauvegarde le DataFrame dans data/initial_data/<pattern>
        """
        path = self.folder / self.pattern
        df.to_parquet(path, index=False)

        if self.verbose:
            print(f"Parquet sauvegardé : {path}")

        return path

    def fetch_and_save(self, tickers: list[str], start: str, end: str, frequency: str = "DAILY") -> Path:
        df = self.fetch(tickers=tickers, start=start, end=end, frequency=frequency)
        return self.save_parquet(df)