import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load raw dataset
print("Reading raw Mendeley dataset...")
df = pd.read_excel("repositorio.xlsx")

# Match columns dynamically to handle character encodings safely
mapping = {}
for c in df.columns:
    c_lower = c.lower()
    if c == "edad":
        mapping[c] = "age"
    elif c == "sexonum":
        mapping[c] = "sex" # 0 = Female, 1 = Male
    elif c == "imc":
        mapping[c] = "bmi"
    elif "tposue" in c_lower and "fsem" not in c_lower and "trimmed" not in c_lower and "cat" not in c_lower:
        mapping[c] = "sleep_weekday_hrs"
    elif "tposue" in c_lower and "fsemtrimmed" in c_lower:
        mapping[c] = "sleep_weekend_hrs"
    elif "tpopantallatotalhs" == c_lower:
        mapping[c] = "screen_total_hrs"
    elif "tporedeshs" == c_lower:
        mapping[c] = "screen_social_media_hrs"
    elif "tpovideojuegoshs" == c_lower:
        mapping[c] = "screen_gaming_hrs"
    elif "tpotvonlinehs" == c_lower:
        mapping[c] = "screen_tv_streaming_hrs"
    elif "tpojuegotabletcelhs" == c_lower:
        mapping[c] = "screen_mobile_gaming_hrs"
    elif "cantdiascelulartrimmed" == c_lower:
        mapping[c] = "days_cellphone_per_week"
    elif "cantdiascompustrimmed" == c_lower:
        mapping[c] = "days_computer_per_week"
    elif "cantdiastablettrimmed" == c_lower:
        mapping[c] = "days_tablet_per_week"
    elif "cantdiasconsolajuegostrimmed" == c_lower:
        mapping[c] = "days_console_per_week"
    elif "cantdiastvtrimmed" == c_lower:
        mapping[c] = "days_tv_per_week"

df_sub = df[list(mapping.keys())].rename(columns=mapping).copy()

# Drop rows missing core predictors or targets
df_clean = df_sub.dropna(subset=['sleep_weekday_hrs', 'sleep_weekend_hrs', 'screen_total_hrs', 'age', 'sex', 'bmi']).copy()

# Filter clinically plausible bounds for adolescents
df_clean = df_clean[
    (df_clean['sleep_weekday_hrs'] >= 3.0) & (df_clean['sleep_weekday_hrs'] <= 14.0) &
    (df_clean['sleep_weekend_hrs'] >= 3.0) & (df_clean['sleep_weekend_hrs'] <= 16.0) &
    (df_clean['screen_total_hrs'] >= 0.0) & (df_clean['screen_total_hrs'] <= 24.0) &
    (df_clean['age'] >= 11) & (df_clean['age'] <= 20)
].copy()

# Fill sub-activity screen days/hours NaNs with 0
screen_sub_cols = [
    'screen_gaming_hrs', 'screen_mobile_gaming_hrs', 'screen_social_media_hrs',
    'screen_tv_streaming_hrs', 'days_cellphone_per_week', 'days_computer_per_week',
    'days_tablet_per_week', 'days_console_per_week', 'days_tv_per_week'
]
df_clean[screen_sub_cols] = df_clean[screen_sub_cols].fillna(0.0)

# Define Target Variables:
# Recommended adolescent sleep is 9.0 hours (American Academy of Sleep Medicine, Paruthi et al. 2016)
# Sleep Loss / Sleep Debt = 9.0 - actual sleep duration
df_clean['sleep_loss_weekday'] = 9.0 - df_clean['sleep_weekday_hrs']
df_clean['sleep_loss_weekend'] = 9.0 - df_clean['sleep_weekend_hrs']
df_clean['social_jetlag'] = df_clean['sleep_weekend_hrs'] - df_clean['sleep_weekday_hrs']

# Save clean English CSV
out_csv = "dataset_clean_english.csv"
df_clean.to_csv(out_csv, index=False)
print(f"Clean English dataset saved to '{out_csv}' with {len(df_clean)} adolescent records.\n", flush=True)

# 3. Model Training and Evaluation Pipeline
feature_cols = [
    'age', 'sex', 'bmi', 'screen_total_hrs', 'screen_social_media_hrs',
    'screen_gaming_hrs', 'screen_mobile_gaming_hrs', 'screen_tv_streaming_hrs',
    'days_cellphone_per_week', 'days_computer_per_week', 'days_tablet_per_week',
    'days_console_per_week', 'days_tv_per_week'
]

X = df_clean[feature_cols].values
# Multi-target output: [sleep_loss_weekday, sleep_loss_weekend]
y = df_clean[['sleep_loss_weekday', 'sleep_loss_weekend']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Split: {len(X_train)} training samples, {len(X_test)} testing samples.\n", flush=True)

# Baseline 1: Dummy Mean Regressor
dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)

# Baseline 2: Multiple Linear Regression (OLS)
ols = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', LinearRegression())
])
ols.fit(X_train, y_train)
y_pred_ols = ols.predict(X_test)

# Baseline 3: Polynomial Regression (Degree 2) with Ridge Regularization
poly = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=100.0))
])
poly.fit(X_train, y_train)
y_pred_poly = poly.predict(X_test)

models = {
    "Baseline: Mean Regressor": y_pred_dummy,
    "Linear Regression (OLS)": y_pred_ols,
    "Polynomial Regression (Deg 2 + Ridge)": y_pred_poly
}

print(f"{'Model':<38} | {'Wkday MAE':<10} | {'Wkday RMSE':<10} | {'Wkday R2':<10} | {'Wkend MAE':<10} | {'Wkend RMSE':<10} | {'Wkend R2':<10}")
print("-" * 108)

for name, preds in models.items():
    mae_wday = mean_absolute_error(y_test[:, 0], preds[:, 0])
    rmse_wday = np.sqrt(mean_squared_error(y_test[:, 0], preds[:, 0]))
    r2_wday = r2_score(y_test[:, 0], preds[:, 0])

    mae_wend = mean_absolute_error(y_test[:, 1], preds[:, 1])
    rmse_wend = np.sqrt(mean_squared_error(y_test[:, 1], preds[:, 1]))
    r2_wend = r2_score(y_test[:, 1], preds[:, 1])

    print(f"{name:<38} | {mae_wday:<10.3f} | {rmse_wday:<10.3f} | {r2_wday:<10.3f} | {mae_wend:<10.3f} | {rmse_wend:<10.3f} | {r2_wend:<10.3f}")

print("\n--- Summary Statistics of Target Variables ---")
print(f"Weekday Sleep Loss: Mean = {df_clean['sleep_loss_weekday'].mean():.2f} hrs, Std = {df_clean['sleep_loss_weekday'].std():.2f} hrs")
print(f"Weekend Sleep Loss: Mean = {df_clean['sleep_loss_weekend'].mean():.2f} hrs, Std = {df_clean['sleep_loss_weekend'].std():.2f} hrs")
print(f"Social Jetlag (Weekend - Weekday Sleep): Mean = {df_clean['social_jetlag'].mean():.2f} hrs")
