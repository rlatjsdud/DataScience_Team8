# DataScience_Team8
Forecasting CPI and National Credit Rating Using Macroeconomic Indicators

## Overview
This project aims to predict inflation rates and sovereign credit ratings based on various macroeconomic indicators using machine learning techniques. We employ both regression and classification models to analyze how economic conditions influence national financial outcomes.

## Objective
Predict *future inflation rates* using regression models.
Classify countries' *sovereign credit ratings* into predefined classes.
Evaluate model performance and analyze feature importance.

## Technologies & Libraries
- Python (pandas, numpy, matplotlib, seaborn)
- scikit-learn
- XGBoost
- Random Forest

## Models Used
| Task                     | Models                                      |
|--------------------------|---------------------------------------------|
| Inflation Prediction     | Linear Regression, Random Forest Regressor |
| Credit Rating Classification | Random Forest Classifier, XGBoost Classifier  |

## Results Summary
Random Forest Regressor achieved the best generalization performance on inflation prediction.
XGBoost Classifier provided better accuracy in multi-class classification of credit ratings.

## Key Findings
Certain macroeconomic features were highly influential in both tasks.
Predicting national credit ratings is more challenging due to class imbalance and complex interdependencies.

## Data Source
[World Bank Open Data](https://data.worldbank.org/) /
[ILO](https://www.ilo.org/topics/wages/minimum-wages) /
[tradingeconomics](https://tradingeconomics.com/country-list/rating)

## Team 8 Members
- 김선영
- 김현우
- 남경민
- 이규석
