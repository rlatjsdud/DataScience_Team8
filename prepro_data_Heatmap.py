import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler

# 설정: 연도/국가
years = ['2000', '2005', '2010', '2015', '2020']
advanced_countries = ['Australia', 'Canada', 'Iceland', 'Italy', 'United Kingdom', 'United States', 'Uruguay', 'Panama']
developing_countries = ['Brazil', 'Colombia', 'Costa Rica', 'Dominican Republic', 'Egypt, Arab Rep.', 'Honduras', 'India', 'Indonesia', 'Malaysia', 'Peru', 'Thailand', 'Bolivia']
selected_countries = advanced_countries + developing_countries

# 파일 경로
base_path = r"C:\Users\82102\OneDrive\바탕 화면\텀프\데과 데이터"
files = {
    "정제된_1인당_GDP_데이터.csv": "GDP_per_capita",
    "정제된_명목_GDP_데이터.csv": "GDP",
    "정제된_인구_데이터.csv": "Population",
    "정제된_빈곤율_데이터.csv": "Poverty_rate",
    "정제된_실업률_데이터.csv": "Unemployment",
    "정제된_실질금리_데이터.csv": "Real_interest_rate",
    "정제된_대출금리_데이터.csv": "Interest_rate",
    "정제된_경제성장률_데이터.csv": "Economic_growth",
    "정제된_지니계수_데이터_완성.csv": "Gini_index",
    "정제된_환율_데이터.csv": "Exchange_rate",
}

# 병합
merged_df = pd.DataFrame()
for file_name, col_name in files.items():
    df = pd.read_csv(os.path.join(base_path, file_name))
    df = df.rename(columns={"Country": "Country Name"}) if "Country" in df.columns else df
    df_filtered = df[df["Country Name"].isin(selected_countries)][["Country Name"] + years]
    df_melted = df_filtered.melt(id_vars="Country Name", var_name="Year", value_name=col_name)
    merged_df = df_melted if merged_df.empty else pd.merge(merged_df, df_melted, on=["Country Name", "Year"], how="outer")

# 최저임금 + 환율 → 달러 환산
min_wage_df = pd.read_csv(os.path.join(base_path, "최저임금.csv")).rename(columns={"Country": "Country Name"})
min_wage_df = min_wage_df[min_wage_df['Country Name'].isin(selected_countries)]
min_wage_long = min_wage_df.melt(id_vars="Country Name", var_name="Year", value_name="Minimum_Wage")
min_wage_long = min_wage_long[min_wage_long["Year"].isin(years)].reset_index(drop=True)

exchange_df = pd.read_csv(os.path.join(base_path, "정제된_환율_데이터.csv"))
exchange_df = exchange_df[exchange_df["Country Name"].isin(selected_countries)]
exchange_long = exchange_df.melt(id_vars="Country Name", value_vars=years, var_name="Year", value_name="Exchange_rate")
exchange_long["Year"] = exchange_long["Year"].astype(str)
min_wage_long["Year"] = min_wage_long["Year"].astype(str)

wage_merged = pd.merge(min_wage_long, exchange_long, on=["Country Name", "Year"], how="left")
wage_merged["Minimum_Wage_USD"] = wage_merged.apply(
    lambda row: row["Minimum_Wage"] / row["Exchange_rate"] if pd.notnull(row["Exchange_rate"]) and row["Exchange_rate"] != 0 else None,
    axis=1
)
merged_df = pd.merge(merged_df, wage_merged[["Country Name", "Year", "Minimum_Wage_USD"]], on=["Country Name", "Year"], how="left")

# 실업률 가중치 (정규화 전에 계산)
if "Unemployment" in merged_df.columns:
    merged_df["Unemployment_weighted"] = merged_df.groupby("Country Name")["Unemployment"].transform(lambda x: x / x.mean())
    merged_df.drop(columns=["Unemployment"], inplace=True)

# 결측치 처리: 국가별 평균으로 대체
features_to_fill = [col for col in merged_df.columns if col not in ["Country Name", "Year"]]
for col in features_to_fill:
    merged_df[col] = merged_df.groupby("Country Name")[col].transform(lambda x: x.fillna(x.mean()))

# 표준화
num_cols = ['GDP_per_capita', 'Economic_growth', 'Interest_rate', 'GDP',
            'Poverty_rate', 'Unemployment_weighted', 'Real_interest_rate',
            'Population', 'Gini_index', 'Exchange_rate', 'Minimum_Wage_USD']
scaler = StandardScaler()
merged_df[num_cols] = scaler.fit_transform(merged_df[num_cols])

# 최종 데이터프레임에서 20개국 × 5개년만 필터링
filtered_df = merged_df[
    (merged_df['Country Name'].isin(selected_countries)) &
    (merged_df['Year'].isin(years))
]

# Missing Value Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(filtered_df[num_cols].isnull(), cbar=False, cmap='Reds', yticklabels=False)
plt.title("Missing Value Heatmap (20 Countries × 5 Years)")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = filtered_df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()