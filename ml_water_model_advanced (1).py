import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

# 1. Charger les données
file_path = 'donnees_completes.csv'
df = pd.read_csv(file_path)

def simulate_target(row):
    base = 6.2
    temp_factor = 1 + (row['Température (°C)'] - 20) * 0.02
    humidity_factor = 1 - (row['Humidité (%)'] - 50) * 0.005
    wind_factor = 1 + row['Vitesse du vent (m/s)'] * 0.01
    aqi_factor = 1 + (row['AQI'] - 50) * 0.002
    seasonal_factor = 1 + np.sin(2 * np.pi * pd.to_datetime(row['Timestamp']).dayofyear / 365) * 0.1
    hourly_factor = 1 + np.sin(2 * np.pi * pd.to_datetime(row['Timestamp']).hour / 24) * 0.2
    return max(base * temp_factor * humidity_factor * wind_factor * aqi_factor * seasonal_factor * hourly_factor, 3)

df['Consommation_Eau'] = df.apply(simulate_target, axis=1)

features = [
    'Température (°C)', 'Humidité (%)', 'Pression (hPa)',
    'Vitesse du vent (m/s)', 'Couverture nuageuse (%)',
    'AQI', 'PM2.5', 'PM10', 'O3', 'NO2'
]
df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
df['Month'] = pd.to_datetime(df['Timestamp']).dt.month
features += ['Hour', 'Month']
le = LabelEncoder()
df['Ville_encoded'] = le.fit_transform(df['Ville'])
features += ['Ville_encoded']

X = df[features]
y = df['Consommation_Eau']

# Imputation simple : remplacer les NaN par la moyenne de chaque colonne
X = X.fillna(X.mean())

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Entraînement GradientBoosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# 4. Validation croisée
cv_scores = cross_val_score(gb_model, X, y, cv=5, scoring='r2')
print(f"R² moyen (GradientBoosting, CV=5) : {cv_scores.mean():.3f}")

# 5. Évaluation sur le test set
y_pred = gb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\n--- GradientBoosting ---")
print(f"MAE : {mae:.3f} L/hab")
print(f"RMSE: {rmse:.3f} L/hab")
print(f"R²  : {r2:.3f}")


# 8. Sauvegarder le modèle
joblib.dump({'model': gb_model, 'label_encoder': le, 'features': features}, 'ml_gb_model.pkl')
print("\n✅ Modèle GradientBoosting sauvegardé dans ml_gb_model.pkl") 