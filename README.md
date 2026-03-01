# Backtesting_M2272

Backtesting d'une stratégie **long/short market-neutral** sur l'univers **STOXX Europe 600**, avec:
- signal **Momentum Dual** (12-1 et 6-1) neutralisé par secteur,
- sélection intra-sectorielle long/short,
- plusieurs méthodes d'allocation,
- reporting complet et exports (Valeur de portefeuilles, métriques (performance, risque, benchmark, couts de transation), composition, export csv format BBU pour implémenter dans Bloomberg).

## 1. Structure du projet

```text
Backtesting_M2272/
├── src/
│   ├── backtesting/
│   │   ├── data_loader.py
│   │   ├── signals.py
│   │   ├── allocation.py
│   │   ├── engine.py
│   │   └── reporting.py
│   └── request/
│       ├── request_unique.py
│       ├── request_index_members.py
│       ├── request_informations_members.py
│       ├── request_historical_data.py
│       └── utils.py
├── examples/
│   ├── request.ipynb
│   ├── MomentumDual_EqualWeight.ipynb
│   ├── MomentumDual_RiskParity.ipynb
│   ├── MomentumDual_MinVar.ipynb
│   └── MomentumDual_SignalWeight.ipynb
└── data/
    ├── initial_data/
    ├── verification/
    └── stockage/
```
### 1a. Workflow de collecte Bloomberg

- `Backtesting_M2272/src` : 
        - `Backtesting_M2272/src/request` : l'ensemble des classes et méthodes pour télécharger les données historiques via blpapi (Bloomberg), 
        - `Backtesting_M2272/src/backtesting` : gestion de la donnée pour le backtesting, méthode d'allocation, signaux, reporting type des différentes stratégies avec analyse du portefeuille.

- `Backtesting_M2272/data` : `Backtesting_M2272/data/initial_data` stockage des données téléchargées, `Backtesting_M2272/data/verification` verification excel/csv des données pour une meilleure fléxibilité, `Backtesting_M2272/data/stockage` stockage des résultats et des compositions de chaque stratégie.

- `Backtesting_M2272/examples` : jupyter notebook des requetes bloomberg, et des différentes stratégies.


### 1b. `Backtesting_M2272/data`

Ce folder sert à stocker l'ensemble de données input et ouput. 

- `Backtesting_M2272/data/initial_data` stocke les données télécharger issues de Bloomberg. 
  - `historical_price_memb_index.parquet` : Historique de prix des membres du stoxx 600 Europe sur la période
  - `ESTRON Index_PX_LAST.parquet` : Historique du Benchmark ESTER (également le taux sans risque pour le calcul des métriques)
  - `informations_hist_memb_index.parquet` : Noms, Devises, Pays, Industries, Secteurs des composants historiques du stoxx 600 Europe au cours de la periode de backtesting
  - `index_memb_YYYYMMDD.parquet` (un fichier par date de composition) : composition à chaque date de rebalancement du stoxx 600 Europe.
  - Les notebooks de `examples/request.ipynb` montrent comment les reconstruire avec Bloomberg.

- `Backtesting_M2272/data/stockage` : pour chaque stratégie, par exemple pour une stratégie : 
    - `bbu_MomentumDual_SignalWeight.csv` : stocke l'historique des poids des valeurs en portefeuille à chaque date de rebalancement.
    - `composition_detaillee_MomentumDual_SignalWeight.parquet`: la composition détaillée avec informations des valeurs en portefeuilles sur la periode de backtesting.
    - `metriques_MomentumDual_SignalWeight.parquet`: historiques des métriques calculées sur le portefeuille au cours de la période de backtesting.
    - `nav_MomentumDual_SignalWeight.parquet` : historique de la valorisation du portefeuille au cours de la periode de backtesting.

### 1c. `Backtesting_M2272/src`

- `Backtesting_M2272/src/backtesting` : contient l'ensembles des fichiers .py permettant de réaliser le backtesting et le reporting
    - `allocation.py` : méthode d'allocation des valeurs dans le portefeuille
    - `data_loader.py`: gère le chargement, le nettoyage et la préparation des données de marché nécessaires au backtest.
    - `èngine.py`: orchestre la logique du backtesting
    - `reporting.py`: produit les métriques de performance, statistiques de risque et visualisations à chaque date de rebalancement.
    - `signals.py`: définit et calcule les signaux quantitatifs utilisés pour sélectionner et classer les actifs.

- `Backtesting_M2272/src/request`: contient l'ensemble des fichiers .py permettant de réaliser la requete Bloomberg et stocker les données pour le bactesting
    - `request_historical_data.py`: requete des prix historiques des composants de l'indice Univers sur la periode de backtesting
    - `request_index_members.py`:  requete les composants de l'indice stoxx600 Europe à chaque date de rebalancement
    - `request_informations_members.py`: requete les informations sur les membres historiques de l'indice stoxx 600 Europe.
    - Pour aller plus loin, nous aurions pu développer un module update_data.py chargé de récupérer les composants du STOXX 600 à la prochaine date de rebalancement du portefeuille, de vérifier si les informations relatives aux tickers sont déjà disponibles et, le cas échéant, de les interroger via Bloomberg ; puis de contrôler leur présence dans la base de données des prix historiques afin, soit d’identifier la dernière date disponible pour mettre à jour les données jusqu’à aujourd’hui, soit, s’ils sont absents, de télécharger l’historique des prix avec une antériorité d’un an pour garantir la compatibilité avec le calcul des signaux.


## 2. Prérequis

- Python 3.11+ (3.12 recommandé)
- Paquets:
  - `pandas`
  - `numpy`
  - `scipy`
  - `plotly`
  - `pyarrow`
  - `openpyxl`
  - `blpapi` (uniquement pour les modules `request/` et la collecte Bloomberg)

## 3. Installation rapide - environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy plotly pyarrow openpyxl
```

Si vous utilisez la collecte Bloomberg:

```bash
pip install blpapi
```

## 9. Notebooks d'exemple : Exécution du Backtest

Notebook .ipynb de chaque stratégie d'allocation de portefeuille : 
- `examples/MomentumDual_EqualWeight.ipynb`
- `examples/MomentumDual_RiskParity.ipynb`
- `examples/MomentumDual_MinVar.ipynb`
- `examples/MomentumDual_SignalWeight.ipynb`

## 8. Sorties de reporting

Dans chaque dossier par stratégie dans `data/stockage/MomentumDual_<Method>`, par exemple:

`MomentumDual_EqualWeight_20161230_20260220/`

Fichiers générés:
- `metriques_MomentumDual_<Method>.parquet`
- `bbu_MomentumDual_<Method>.csv`
- `composition_detaillee_MomentumDual_<Method>.parquet`
- `nav_MomentumDual_<Method>.parquet`

