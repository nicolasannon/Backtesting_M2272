# Backtesting_M2272/src/request/request_informations_memebers.py

# ------------------------------------------------------------------
# Librairies
# ------------------------------------------------------------------

import blpapi
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Présenatations
# ------------------------------------------------------------------
""""
- Récupère des informations de référence Bloomberg (BDP) pour une liste de tickers donnée
- Nom, Pays, Devise, Secteur, Industrie
- Sauvegarde le résultat en parquet en groupe entier comme ca on peut rappeler ces données si un ticker est présent dans plusieurs compositions mensuelles
"""
class BloombergTickerInfo:
    """
    Récupère des informations de référence Bloomberg (BDP) pour une liste de tickers
    et sauvegarde le résultat en parquet
    """

    def __init__(self, host="localhost", port=8194, verbose=True):
        self.host = host
        self.port = port
        self.verbose = verbose

        current_file = Path(__file__).resolve()
        self.project_root = current_file.parent.parent.parent

        self.folder = self.project_root / "data" / "initial_data"
        self.folder.mkdir(parents=True, exist_ok=True)

        self.pattern = "informations_hist_memb_index.parquet"

        # mapping champs Bloomberg -> colonnes output
        self._field_map = {
            "NAME": "Name",
            "CRNCY": "Currency",
            "GICS_SECTOR_NAME": "Sector",
            "GICS_INDUSTRY_NAME": "Industry",
            "COUNTRY_FULL_NAME": "Country"}

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

    def fetch(self, tickers: list[str]) -> pd.DataFrame:
        """
        Retourne un DataFrame avec colonnes:
        Ticker, Name, Currency, Sector, Industry, Country
        """
        session = self._start_session()
        service = session.getService("//blp/refdata")

        request = service.createRequest("ReferenceDataRequest")
        for t in tickers:
            request.getElement("securities").appendValue(t)

        for f in self._field_map.keys():
            request.getElement("fields").appendValue(f)

        session.sendRequest(request)

        rows = []

        while True:
            event = session.nextEvent()
            for msg in event:
                if not msg.hasElement("securityData"):
                    continue

                sec_array = msg.getElement("securityData")

                for i in range(sec_array.numValues()):
                    sec = sec_array.getValueAsElement(i)

                    ticker = sec.getElementAsString("security") if sec.hasElement("security") else None

                    # valeurs par défaut
                    out = {
                        "Ticker": ticker,
                        "Name": None,
                        "Currency": None,
                        "Sector": None,
                        "Industry": None,
                        "Country": None}

                    # field exceptions -> on laisse None si pas possible de téléhcarger
                    if sec.hasElement("fieldData"):
                        fd = sec.getElement("fieldData")
                        for bb_field, out_col in self._field_map.items():
                            if fd.hasElement(bb_field):
                                try:
                                    out[out_col] = fd.getElementAsString(bb_field)
                                except Exception:
                                    try:
                                        out[out_col] = str(fd.getElement(bb_field).getValue())
                                    except Exception:
                                        out[out_col] = None

                    rows.append(out)

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        session.stop()

        # Mise en format correct
        df = pd.DataFrame(rows, columns=["Ticker", "Name", "Currency", "Sector", "Industry", "Country"])
        return df

    def save_parquet(self, df: pd.DataFrame) -> Path:
        """
        Sauvegarde le DataFrame dans data/initial_data/informations_hist_memb_index.parquet
        """
        path = self.folder / self.pattern
        df.to_parquet(path, index=False)

        if self.verbose:
            print(f"Parquet sauvegardé : {path}")

        return path

    def fetch_and_save(self, tickers: list[str]) -> Path:
        """
        Helper : fetch + save
        """
        df = self.fetch(tickers)
        return self.save_parquet(df)