# Backtesting_M2272/src/request/request_index_members.py

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
- Télécharge les composants historiques d'un indice Bloomberg

Index = stoxx 600 Europe
"""

class IndexMembersDownloader:
    """
    Télécharge les composants historiques d'un indice Bloomberg
    et sauvegarde chaque date de téléchargement dans un fichier parquet séparé
    
    Objectifs : obtenir la composition de l'indice sans le biais des survivants pour le bakctesting
    """

    def __init__(self, host="localhost", port=8194):
        self.host = host
        self.port = port

        # Détection racine projet
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent

        self.dossier = project_root / "data" / "initial_data"
        self.dossier.mkdir(parents=True, exist_ok=True)

    def _start_session(self):
        options = blpapi.SessionOptions()
        options.setServerHost(self.host)
        options.setServerPort(self.port)

        session = blpapi.Session(options)

        if not session.start():
            raise ConnectionError("Impossible de démarrer la session Bloomberg.")

        if not session.openService("//blp/refdata"):
            raise ConnectionError("Impossible d'ouvrir le service refdata.")

        return session

    def download_members(self, index_ticker: str, dates: list):

        session = self._start_session()
        service = session.getService("//blp/refdata")

        for date in dates:

            request = service.createRequest("ReferenceDataRequest")
            request.getElement("securities").appendValue(index_ticker)
            request.getElement("fields").appendValue("INDX_MWEIGHT_HIST")

            overrides = request.getElement("overrides")
            override = overrides.appendElement()
            override.setElement("fieldId", "END_DATE_OVERRIDE")
            override.setElement("value", date)  # format YYYYMMDD

            session.sendRequest(request)

            data = []

            while True:
                event = session.nextEvent()

                for msg in event:

                    if msg.hasElement("securityData"):
                        sec_array = msg.getElement("securityData")

                        for i in range(sec_array.numValues()):
                            security = sec_array.getValueAsElement(i)

                            if security.hasElement("fieldData"):
                                field_data = security.getElement("fieldData")

                                if field_data.hasElement("INDX_MWEIGHT_HIST"):
                                    members = field_data.getElement("INDX_MWEIGHT_HIST")

                                    for j in range(members.numValues()):
                                        member = members.getValueAsElement(j)

                                        values = []
                                        for k in range(member.numElements()):
                                            elem = member.getElement(k)
                                            try:
                                                values.append(elem.getValue())
                                            except:
                                                values.append(None)

                                        if len(values) >= 2:
                                            data.append([date, values[0], values[1]])

                if event.eventType() == blpapi.Event.RESPONSE:
                    break

            # Création DataFrame
            df = pd.DataFrame(data, columns=["Date", "Ticker", "Weight"])

            # Format Date en YYYY-MM-DD
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

            # Suppression colonne Weight car NaN pas l'accès
            df = df.drop(columns=["Weight"])

            # Sauvegarde parquet
            file_name = f"index_memb_{date}"
            chemin = self.dossier / f"{file_name}.parquet"
            df.to_parquet(chemin)

            print(f"Sauvegardé : {chemin}")

        session.stop()