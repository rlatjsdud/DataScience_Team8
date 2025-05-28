import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 연도 및 국가 리스트
years = ['2000', '2005', '2010', '2015', '2020']

# 선진국 8개
advanced_countries = [
    'Australia', 'Canada', 'Iceland', 'Italy', 
    'United Kingdom', 'United States', 'Uruguay', 'Panama'
]

# 중진국 12개
developing_countries = [
    'Brazil', 'Colombia', 'Costa Rica', 'Dominican Republic',
    'Egypt, Arab Rep.', 'Honduras', 'India', 'Indonesia',
    'Malaysia', 'Peru', 'Thailand', 'Bolivia'
]
# 국가별(20개) 연도(5기준) data = 전체 100개의 data

selected_countries = advanced_countries + developing_countries

# CSV 파일
base_path = r"C:\Users\82102\OneDrive\바탕 화면\텀프\데과 데이터" # 파일 경로
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

# 데이터 병합
merged_df = pd.DataFrame()

for file_path, indicator in files.items():
    df = pd.read_csv(os.path.join(base_path, file_path))
    df_filtered = df[df['Country Name'].isin(selected_countries)][['Country Name'] + years]
    df_melted = df_filtered.melt(id_vars='Country Name', var_name='Year', value_name=indicator)
    if merged_df.empty:
        merged_df = df_melted
    else:
        merged_df = pd.merge(merged_df, df_melted, on=['Country Name', 'Year'], how='outer')

# 국가 정렬 
income_group_df = df[['Country Name', 'IncomeGroup']].drop_duplicates()
merged_df = pd.merge(merged_df, income_group_df, on='Country Name', how='left')

country_order = advanced_countries + developing_countries
merged_df['Country_Order'] = merged_df['Country Name'].apply(lambda x: country_order.index(x))
merged_df = merged_df.sort_values(by=['Country_Order', 'Year']).reset_index(drop=True)

merged_df.drop(columns=['IncomeGroup', 'Country_Order'], inplace=True)


# -----------------------------------------------
# 최저임금 데이터 처리
# -----------------------------------------------
min_wage_df = pd.read_csv(os.path.join(base_path, "최저임금.csv"))

# wide → long 변환, Country 컬럼명 'Country Name'으로 변경
min_wage_df = min_wage_df.rename(columns={'Country': 'Country Name'})
min_wage_df = min_wage_df[min_wage_df['Country Name'].isin(selected_countries)]

min_wage_long = min_wage_df.melt(id_vars='Country Name', var_name='Year', value_name='Minimum_Wage')
min_wage_long = min_wage_long[min_wage_long['Year'].isin(years)].reset_index(drop=True)

# -----------------------------------------------
# 환율 데이터 전처리
# -----------------------------------------------
exchange_rate_df = pd.read_csv(os.path.join(base_path, "정제된_환율_데이터.csv"))
exchange_rate_df = exchange_rate_df[exchange_rate_df['Country Name'].isin(selected_countries)]
exchange_rate_df = exchange_rate_df[['Country Name'] + years]
exchange_rate_melted = exchange_rate_df.melt(id_vars='Country Name', var_name='Year', value_name='Exchange_rate')
exchange_rate_melted = exchange_rate_melted.drop_duplicates(subset=['Country Name', 'Year'])
exchange_rate_melted['Year'] = exchange_rate_melted['Year'].astype(str)
min_wage_long['Year'] = min_wage_long['Year'].astype(str)

# -----------------------------------------------
# 최저임금과 환율 병합 및 USD 환산
# -----------------------------------------------
wage_merged = pd.merge(min_wage_long, exchange_rate_melted, on=['Country Name', 'Year'], how='left')
wage_merged['Minimum_Wage_USD'] = wage_merged.apply(
    lambda row: row['Minimum_Wage'] / row['Exchange_rate'] if pd.notnull(row['Exchange_rate']) and row['Exchange_rate'] != 0 else None,
    axis=1
)

# -----------------------------------------------
# 데이터프레임과 최저임금 USD 병합
# -----------------------------------------------
merged_df = pd.merge(
    merged_df,
    wage_merged[['Country Name', 'Year', 'Minimum_Wage_USD']],
    on=['Country Name', 'Year'],
    how='left'
)

# -----------------------------------------------
# 결측치 처리 (국가별 연도 데이터 평균값으로 채우기)
# -----------------------------------------------
def fill_na_with_country_mean(df, features):
    for feature in features:
        if feature not in df.columns:
            continue
        country_means = df.groupby('Country Name')[feature].transform('mean')
        df[feature] = df[feature].fillna(country_means)
    return df

features_to_fill = [col for col in merged_df.columns 
                    if col not in ['Country Name', 'Year', 'IncomeGroup', 'Country_Order', 'Unemployment_weighted']]

if 'Unemployment' in merged_df.columns:
    features_to_fill.append('Unemployment')

merged_df = fill_na_with_country_mean(merged_df, features_to_fill)

# -----------------------------------------------
# 실업률 가중치 처리
# -----------------------------------------------
if 'Unemployment' in merged_df.columns:
    country_unemp_mean = merged_df.groupby('Country Name')['Unemployment'].transform('mean') # 국가별 실업률의 평균값으로 가중치 부여
    merged_df['Unemployment_weighted'] = merged_df['Unemployment'] / country_unemp_mean
    merged_df.drop(columns=['Unemployment'], inplace=True)

# -----------------------------------------------
# 표준화 처리 => 평균 0, 표준편차 1로 변환(모델 성능 향상 및 해석 용이)
# -----------------------------------------------
num_cols = [
    'GDP_per_capita',
    'Economic_growth',
    'Interest_rate',
    'GDP',
    'Poverty_rate',
    'Unemployment_weighted',
    'Real_interest_rate',
    'Population',
    'Gini_index',
    'Exchange_rate',
    'Minimum_Wage_USD'
]

scaler = StandardScaler()
merged_df[num_cols] = scaler.fit_transform(merged_df[num_cols])

# -----------------------------------------------
# 최종 결과 출력 (정렬 유지)
# -----------------------------------------------
print("\n 75개 샘플 선진국 -> 중진국 정렬:\n")
print(merged_df.head(100).to_string())

# --- CPI 데이터 불러오기 및 처리 ---
cpi_df = pd.read_csv(os.path.join(base_path, "정제된_CPI_데이터.csv"))
cpi_filtered = cpi_df[cpi_df['Country Name'].isin(selected_countries)][['Country Name'] + years]
cpi_melted = cpi_filtered.melt(id_vars='Country Name', var_name='Year', value_name='CPI')

# --- 신용등급 데이터 불러오기 및 처리 ---
credit_df = pd.read_csv(os.path.join(base_path,"국가_신용등급.csv"))
credit_filtered = credit_df[credit_df['Country Name'].isin(selected_countries)][['Country Name'] + years]
credit_melted = credit_filtered.melt(id_vars='Country Name', var_name='Year', value_name='Credit_Rating')

# --- Year 컬럼 타입 통일 (문자열) ---
merged_df['Year'] = merged_df['Year'].astype(str)
cpi_melted['Year'] = cpi_melted['Year'].astype(str)
credit_melted['Year'] = credit_melted['Year'].astype(str)

# --- CPI, 신용등급 데이터 병합 ---
merged_df = pd.merge(merged_df, cpi_melted, on=['Country Name', 'Year'], how='left')
merged_df = pd.merge(merged_df, credit_melted, on=['Country Name', 'Year'], how='left')

# --- 병합 후 컬럼명 확인 ---
print("Merged columns:", merged_df.columns.tolist())

# --- 결측치 확인 ---
print("Missing values in CPI and Credit_Rating:")
print(merged_df[['CPI', 'Credit_Rating']].isna().sum())

# --- 학습용 데이터 준비 ---
features = num_cols
df_reg = merged_df.dropna(subset=features + ['CPI'])
X_reg = df_reg[features] # CPI 회귀 예측을 위한 전체 dataset
y_reg = df_reg['CPI']

# --- 학습 / 테스트 데이터 분할 ---
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

df_clf = merged_df.dropna(subset=features + ['Credit_Rating'])
X_clf = df_clf[features] # 신용등급 분류 예측을 위한 전체 dataset
y_clf = df_clf['Credit_Rating']

# --- 분류 타겟 인코딩 ---
enc = OrdinalEncoder()
y_clf_enc = enc.fit_transform(y_clf.values.reshape(-1,1)).ravel()

# --- 1) 선형회귀 모델 학습 및 평가 ---
model_reg = LinearRegression()
model_reg.fit(X_train, y_train)

train_pred = model_reg.predict(X_train)
test_pred = model_reg.predict(X_test)

print("Train MSE (Linear Regression):", mean_squared_error(y_train, train_pred))
print("Test MSE (Linear Regression):", mean_squared_error(y_test, test_pred))

# --- 2) 랜덤포레스트 회귀 모델 학습 및 평가 ---
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

train_pred_rf = model_rf.predict(X_train)
test_pred_rf = model_rf.predict(X_test)

print("Train MSE (Random Forest):", mean_squared_error(y_train, train_pred_rf))
print("Test MSE (Random Forest):", mean_squared_error(y_test, test_pred_rf))

# --- 예측 결과를 별도 DataFrame에 저장 (인덱스 유지) ---
train_result_lr = pd.DataFrame({
    'index': X_train.index,
    'CPI_Predicted_LR': train_pred
})
test_result_lr = pd.DataFrame({
    'index': X_test.index,
    'CPI_Predicted_LR_Test': test_pred
})

train_result_rf = pd.DataFrame({
    'index': X_train.index,
    'CPI_Predicted_RF': train_pred_rf
})
test_result_rf = pd.DataFrame({
    'index': X_test.index,
    'CPI_Predicted_RF_Test': test_pred_rf
})

# --- df_reg reset_index 하여 'index' 컬럼 생성 ---
df_reg = df_reg.reset_index()

# --- 인덱스를 기준으로 merge 하여 NaN 없이 값 할당 ---
df_reg = df_reg.merge(train_result_lr, on='index', how='left')
df_reg = df_reg.merge(test_result_lr, on='index', how='left')
df_reg = df_reg.merge(train_result_rf, on='index', how='left')
df_reg = df_reg.merge(test_result_rf, on='index', how='left')

# --- CPI 예측 결과 정리 및 출력 ---
result_df = df_reg[['Country Name', 'Year', 'CPI', 'CPI_Predicted_LR', 'CPI_Predicted_LR_Test', 'CPI_Predicted_RF', 'CPI_Predicted_RF_Test']].copy()
result_df = result_df.sort_values(by=['Country Name', 'Year']).reset_index(drop=True)
result_df.insert(0, 'No', result_df.index + 1)

print("\n=== Country-wise CPI Actual vs Predicted ===")
print(result_df.to_string(index=False))

# --- 선형회귀 특성 중요도 출력 ---
importance_df_lr = pd.DataFrame({
    'Feature': features,
    'Coefficient': model_reg.coef_,
    'Abs_Coefficient': np.abs(model_reg.coef_)
}).sort_values(by='Abs_Coefficient', ascending=False)

print("\n=== Feature Importance (Linear Regression Coefficients) ===")
print(importance_df_lr)

# 랜덤포레스트 특성 중요도 추출
importance_rf = model_rf.feature_importances_
importance_df_rf = pd.DataFrame({
    'Feature': features,
    'Importance': importance_rf
}).sort_values(by='Importance', ascending=False)

print("\n=== Feature Importance (Random Forest Feature Importance) ===")
print(importance_df_rf)


# --- 3) 국가 신용등급 분류(RandomForestClassifier + k-fold 교차검증) ---

df_clf['Credit_Rating'] = df_clf['Credit_Rating'].str.replace('+', '', regex=False)
df_clf['Credit_Rating'] = df_clf['Credit_Rating'].str.replace('-', '', regex=False)

# 레이블 인코더로 신용등급 문자 → 숫자 변환
enc = LabelEncoder()
y_clf_enc = enc.fit_transform(df_clf['Credit_Rating'])

# 분류 모델 정의
model_clf = RandomForestClassifier(
    n_estimators=50,
    max_depth=4, # 깊이 제한으로 복잡도 줄이기
    min_samples_split=6, # 분기 조건 강화
    min_samples_leaf=3, # 잎 노드 최소 샘플 확보
    max_features='sqrt',
    random_state=42)

# KFold 설정 (5겹, 셔플, 시드 고정)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 교차검증 수행 (정확도 평가)
scores = cross_val_score(model_clf, X_clf, y_clf_enc, cv=kf, scoring='accuracy')

# 교차검증 결과 출력
print("\n=== 신용등급 분류 모델 평가 결과 ===")
print(f"K-Fold Accuracy 평균: {scores.mean():.4f}")
print(f"K-Fold Accuracy 각 Fold 점수: {scores}")

# 전체 데이터로 최종 모델 학습 및 예측
model_clf.fit(X_clf, y_clf_enc)
df_clf['Credit_Predicted'] = enc.inverse_transform(model_clf.predict(X_clf))

# 피처 중요도 추출
feature_importances = model_clf.feature_importances_
features = X_clf.columns

# 피처 중요도 DataFrame 생성 및 내림차순 정렬
feat_imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# 피처 중요도 터미널 출력
print("\n=== 국가 신용등급 예측에서 피처 중요도 ===")
for i, row in enumerate(feat_imp_df.itertuples(), 1):
    print(f"{i}. {row.Feature}: {row.Importance:.4f}")

# 예측 결과 일부 출력
print("\n=== 국가별 신용등급 예측 결과 (일부 출력) ===")
print(df_clf[['Country Name', 'Year', 'Credit_Rating', 'Credit_Predicted']].head(100).to_string())

### Confusion Matrix와 F1-score 분석 => overfitting 확인
# 예측 수행 (이미 학습한 model_clf 사용)
y_true = df_clf['Credit_Rating']
y_pred = df_clf['Credit_Predicted']

# 1. 혼동 행렬
cm = confusion_matrix(y_true, y_pred, labels=enc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=enc.classes_)

print("=== Confusion Matrix ===")
disp.plot(xticks_rotation=45, cmap='Blues')
plt.tight_layout()
plt.show()

# 2. 분류 성능 리포트
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, labels=enc.classes_))

###
# 교차검증 기반 예측 수행 (y_clf_enc는 숫자형)
y_cv_pred = cross_val_predict(model_clf, X_clf, y_clf_enc, cv=5)

# 예측 복원
y_cv_pred_label = enc.inverse_transform(y_cv_pred)
y_true = df_clf['Credit_Rating']

# 평가
print("=== Classification Report (Cross Validation) ===")
print(classification_report(y_true, y_cv_pred_label))

cm = confusion_matrix(y_true, y_cv_pred_label, labels=enc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=enc.classes_)
disp.plot(xticks_rotation=45, cmap='Blues')
plt.tight_layout()
plt.show()

# --- 4) 국가 신용등급 분류(XGBoost + k-fold 교차검증) ---
# XGBoost 모델 정의
model_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

### --- XGBoost로 국가 신용등급 분류 평가 ---
# 교차검증 기반 예측
y_pred_xgb_cv = cross_val_predict(model_xgb, X_clf, y_clf_enc, cv=5)
y_pred_xgb_label = enc.inverse_transform(y_pred_xgb_cv)

# 평가
print("\n=== Classification Report (XGBClassifier, Cross Validation) ===")
print(classification_report(y_true, y_pred_xgb_label))

cm_xgb = confusion_matrix(y_true, y_pred_xgb_label, labels=enc.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=enc.classes_).plot(
    xticks_rotation=45, cmap='Blues')
plt.tight_layout()
plt.show()
