# Backtesting_M2272/src/request/request_unique.py

# ------------------------------------------------------------------
# Librairies
# ------------------------------------------------------------------

import blpapi
import pandas as pd
from datetime import datetime

# ------------------------------------------------------------------
# Présenatations
# ------------------------------------------------------------------
"""
- juste pour télcharger le benchmark et les valeurs de l'indice de référence (STOXX Europe 600) pour obtenir les dates où il est ouvert.
"""

class BloombergHistoricalData:
    """
    Classe simple pour télécharger un historique Bloomberg via blpapi
    """

    def __init__(self, host="localhost", port=8194):
        self.host = host
        self.port = port

    def get_history(self, ticker, field, start_date, end_date, frequency="DAILY"):
        """

        Parameters
        ----------
        ticker : str
            Exemple : "SXXP Index"
        field : str
            Exemple : "PX_LAST"
        start_date : str
            Format "YYYYMMDD"
        end_date : str
            Format "YYYYMMDD"
        frequency : str
            DAILY 

        Returns
        -------
        pandas.DataFrame
        """

        # Session setup
        options = blpapi.SessionOptions()
        options.setServerHost(self.host)
        options.setServerPort(self.port)

        session = blpapi.Session(options)
        if not session.start():
            raise ConnectionError("Impossible de démarrer la session Bloomberg.")

        if not session.openService("//blp/refdata"):
            raise ConnectionError("Impossible d'ouvrir le service refdata.")

        service = session.getService("//blp/refdata")
        request = service.createRequest("HistoricalDataRequest")

        request.getElement("securities").appendValue(ticker)
        request.getElement("fields").appendValue(field)

        request.set("startDate", start_date)
        request.set("endDate", end_date)
        request.set("periodicitySelection", frequency)

        session.sendRequest(request)

        data = []

        while True:
            event = session.nextEvent()
            for msg in event:
                if msg.hasElement("securityData"):
                    security_data = msg.getElement("securityData")
                    field_data = security_data.getElement("fieldData")

                    for i in range(field_data.numValues()):
                        row = field_data.getValueAsElement(i)
                        date = row.getElementAsDatetime("date")
                        value = row.getElementAsFloat(field)
                        data.append([date, value])

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        session.stop()

        df = pd.DataFrame(data, columns=["Date", field])
        df.set_index("Date", inplace=True)

        return df