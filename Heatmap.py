import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_paths = {
    "정제된_1인당_GDP_데이터.csv": "GDP_per_capita",
    "정제된_명목_GDP_데이터.csv": "GDP",
    "정제된_인구_데이터.csv": "Population",
    "정제된_빈곤율_데이터.csv": "Poverty_rate",
    "정제된_실업률_데이터.csv": "Unemployment",
    "정제된_실질금리_데이터.csv": "Real_interest_rate",
    "정제된_대출금리_데이터.csv": "Interest_rate",
    "정제된_경제성장률_데이터.csv": "Economic_growth",
    "정제된_CPI_데이터.csv": "CPI",
    "정제된_지니계수_데이터_완성.csv": "Gini_index",
    "정제된_환율_데이터.csv": "Exchange_rate",
    "최저임금.csv": "Minimum_Wage",
    "국가_신용등급.csv": "Credit_Rating"
}

# CSV 병합: long format으로 통일
base_dir = r"C:\Users\82102\OneDrive\바탕 화면\텀프\데과 데이터" # 파일 경로
merged_df = pd.DataFrame()

for file_name, col_name in file_paths.items():
    path = f"{base_dir}\\{file_name}"
    df = pd.read_csv(path)
    if 'Country Name' in df.columns:
        country_col = 'Country Name'
    elif 'Country' in df.columns:
        df = df.rename(columns={'Country': 'Country Name'})
        country_col = 'Country Name'
    else:
        continue

    # 연도 컬럼 추출 및 melt
    year_cols = [col for col in df.columns if col.isdigit()]
    df_melted = df.melt(id_vars=country_col, value_vars=year_cols,
                        var_name='Year', value_name=col_name)

    if merged_df.empty:
        merged_df = df_melted
    else:
        merged_df = pd.merge(merged_df, df_melted, on=['Country Name', 'Year'], how='outer')

# 시각화를 위한 숫자형 데이터프레임 구성
merged_df['Year'] = merged_df['Year'].astype(str)
merged_numeric = merged_df.drop(columns=['Country Name', 'Year'])

# Missing Value Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(merged_numeric.isnull(), cbar=False, cmap='Reds', yticklabels=False)
plt.title('Missing Value Heatmap')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = merged_numeric.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()
