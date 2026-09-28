from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "hr_attrition.csv"

TARGET = "Attrition"

# Constantes sur l'ensemble du jeu de données (une seule valeur possible) : aucune
# valeur prédictive. EmployeeNumber est un identifiant, pas une caractéristique.
DROP_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

CATEGORICAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

NUMERIC_FEATURES = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    # Le fichier source contient un BOM UTF-8 au début de la première colonne.
    return pd.read_csv(path, encoding="utf-8-sig")


def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    df = load_raw(path).drop(columns=DROP_COLUMNS)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET]
