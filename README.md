# Prédiction d'attrition employés

Projet de machine learning classique : prédire si un employé va quitter l'entreprise
(attrition) à partir de ses caractéristiques (satisfaction, rémunération, ancienneté,
heures supplémentaires, etc.), avec exploration des données, entraînement, évaluation
et une API de prédiction.

## Pourquoi ce projet

Compléter mon portfolio Data avec un cas de machine learning "classique" (classification
supervisée sur données tabulaires), en complément de :
- [assistant documentaire RAG](https://github.com/Daoudbamba/docs-ai-assistant) — IA générative
- [observatoire du chômage](https://github.com/Daoudbamba/chomage-dashboard) — data engineering & dashboard

## Données

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) :
1470 employés, 35 variables. Jeu de données fictif créé par IBM pour l'apprentissage de la
data science — utilisé ici pour démontrer la démarche de modélisation, pas pour décrire des
employés réels.

## Démarche

1. **Exploration** (`notebooks/01_exploration.ipynb`) : distribution de la cible (classes
   déséquilibrées, ~16% d'attrition), variables les plus corrélées au départ (heures
   supplémentaires, satisfaction au travail, ancienneté, salaire).
2. **Prétraitement** (`src/data.py`) : suppression des colonnes constantes ou non
   informatives (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`), encodage
   de la cible.
3. **Entraînement** (`src/train.py`) : pipeline scikit-learn (encodage one-hot des
   variables catégorielles + standardisation), comparaison régression logistique vs
   forêt aléatoire, sélection du meilleur modèle par ROC-AUC (métrique adaptée au
   déséquilibre de classes, contrairement à l'accuracy).
4. **API** (`api/main.py`) : FastAPI expose le modèle entraîné via `/api/predict`.
5. **Démo** (`web/index.html`) : formulaire simple pour tester des prédictions sans
   passer par la documentation Swagger.

## Lancer le projet

```bash
docker compose up --build
```

- API + doc interactive : http://localhost:8020/docs
- Démo web : http://localhost:8020/web/

Le modèle est déjà entraîné et versionné (`models/attrition_model.joblib`). Pour le
ré-entraîner :

```bash
pip install -r requirements-dev.txt
python -m src.train
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

Tests sur la logique de prétraitement, sur l'entraînement (cohérence des métriques,
sélection du meilleur modèle) et sur l'API (prédiction, validation des entrées).

## Résultats

Voir `models/metrics.json` pour le détail des métriques par modèle (ROC-AUC, précision,
rappel, F1 sur la classe "attrition").
