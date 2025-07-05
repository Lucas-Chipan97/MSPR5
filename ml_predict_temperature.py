import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Chargement des données
df = pd.read_csv("donnees_completes.csv")

# 2. Prétraitement
df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
df['Month'] = pd.to_datetime(df['Timestamp']).dt.month

le = LabelEncoder()
df['Ville_encoded'] = le.fit_transform(df['Ville'])

features = [
    'Humidité (%)', 'Pression (hPa)', 'Vitesse du vent (m/s)',
    'Couverture nuageuse (%)', 'AQI', 'PM2.5', 'PM10', 'O3', 'NO2',
    'Hour', 'Month', 'Ville_encoded'
]
target = 'Température (°C)'

X = df[features].fillna(df[features].mean())
y = df[target]

# 3. Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Entraînement du modèle
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Sauvegarde du modèle
joblib.dump({
    'model': model,
    'label_encoder': le,
    'features': features
}, "ml_temperature_model.pkl")

# 6. Évaluation des performances
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 7. Affichage des résultats
print("\n✅ Modèle entraîné et sauvegardé dans 'ml_temperature_model.pkl'")
print("\n📊 Performances du modèle sur le jeu de test :")
print(f"• R² (score de prédiction)        : {r2:.3f}")
print(f"• MAE (erreur absolue moyenne)    : {mae:.2f} °C")
print(f"• RMSE (erreur quadratique moyenne): {rmse:.2f} °C")

if r2 > 0.9:
    print("🎯 Excellente performance du modèle !")
elif r2 > 0.8:
    print("👍 Bonne performance, modèle fiable.")
else:
    print("⚠️ Modèle perfectible. Pense à vérifier les données ou à tester un autre algorithme.")
