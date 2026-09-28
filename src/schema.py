from typing import Literal

from pydantic import BaseModel, Field


class EmployeeFeatures(BaseModel):
    # Catégorielles
    BusinessTravel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    Department: Literal["Sales", "Research & Development", "Human Resources"]
    EducationField: Literal[
        "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"
    ]
    Gender: Literal["Male", "Female"]
    JobRole: str
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    OverTime: Literal["Yes", "No"]

    # Numériques
    Age: int = Field(..., ge=18, le=70)
    DailyRate: int
    DistanceFromHome: int = Field(..., ge=0)
    Education: int = Field(..., ge=1, le=5)
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    HourlyRate: int
    JobInvolvement: int = Field(..., ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5)
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int = Field(..., ge=0)
    PercentSalaryHike: int
    PerformanceRating: int = Field(..., ge=1, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TotalWorkingYears: int = Field(..., ge=0)
    TrainingTimesLastYear: int = Field(..., ge=0)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0)
    YearsInCurrentRole: int = Field(..., ge=0)
    YearsSinceLastPromotion: int = Field(..., ge=0)
    YearsWithCurrManager: int = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "BusinessTravel": "Travel_Rarely",
                "Department": "Research & Development",
                "EducationField": "Life Sciences",
                "Gender": "Female",
                "JobRole": "Laboratory Technician",
                "MaritalStatus": "Single",
                "OverTime": "Yes",
                "Age": 29,
                "DailyRate": 800,
                "DistanceFromHome": 8,
                "Education": 3,
                "EnvironmentSatisfaction": 2,
                "HourlyRate": 65,
                "JobInvolvement": 2,
                "JobLevel": 1,
                "JobSatisfaction": 2,
                "MonthlyIncome": 2800,
                "MonthlyRate": 15000,
                "NumCompaniesWorked": 2,
                "PercentSalaryHike": 13,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 3,
                "StockOptionLevel": 0,
                "TotalWorkingYears": 5,
                "TrainingTimesLastYear": 2,
                "WorkLifeBalance": 2,
                "YearsAtCompany": 3,
                "YearsInCurrentRole": 2,
                "YearsSinceLastPromotion": 0,
                "YearsWithCurrManager": 2,
            }
        }
    }


class PredictionResponse(BaseModel):
    attrition_probability: float
    attrition_predite: bool
