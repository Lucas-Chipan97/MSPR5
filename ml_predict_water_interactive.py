"""
Interface interactive pour le modèle GradientBoosting de prédiction de consommation d'eau
"""
import pandas as pd
import numpy as np
import joblib

# Charger le modèle GradientBoosting
model_data = joblib.load('ml_gb_model.pkl')
model = model_data['model']
le = model_data['label_encoder']
features = model_data['features']

# Liste des villes
villes = list(le.classes_)

def predict_ml():
    print("\n=== Prédiction interactive (modèle GradientBoosting) ===")
    print("Villes disponibles :")
    for i, v in enumerate(villes, 1):
        print(f"{i}. {v}")
    try:
        idx = int(input(f"Choisissez une ville (1-{len(villes)}): ")) - 1
        ville = villes[idx]
    except:
        print("Choix invalide.")
        return
    try:
        temperature = float(input("Température (°C): "))
        humidity = float(input("Humidité (%): "))
        pressure = float(input("Pression (hPa): "))
        wind = float(input("Vitesse du vent (m/s): "))
        cloud = float(input("Couverture nuageuse (%): "))
        aqi = float(input("AQI: "))
        pm25 = float(input("PM2.5: "))
        pm10 = float(input("PM10: "))
        o3 = float(input("O3: "))
        no2 = float(input("NO2: "))
        hour = int(input("Heure (0-23): "))
        month = int(input("Mois (1-12): "))
    except:
        print("Entrée invalide.")
        return
    ville_encoded = le.transform([ville])[0]
    X_pred = pd.DataFrame([{ 
        'Température (°C)': temperature,
        'Humidité (%)': humidity,
        'Pression (hPa)': pressure,
        'Vitesse du vent (m/s)': wind,
        'Couverture nuageuse (%)': cloud,
        'AQI': aqi,
        'PM2.5': pm25,
        'PM10': pm10,
        'O3': o3,
        'NO2': no2,
        'Hour': hour,
        'Month': month,
        'Ville_encoded': ville_encoded
    }])[features]
    # Imputation simple : remplacer les NaN par la moyenne de chaque colonne
    X_pred = X_pred.fillna(X_pred.mean())
    y_pred = model.predict(X_pred)[0]
    print(f"\n🌊 Consommation d'eau prédite (GradientBoosting) : {y_pred:.2f} L/habitant/heure")

def main():
    print("\n=== Interface de prédiction GradientBoosting ===")
    while True:
        print("\n1. Faire une prédiction")
        print("2. Quitter")
        choice = input("Votre choix : ").strip()
        if choice == '1':
            predict_ml()
        elif choice == '2':
            print("Au revoir !")
            break
        else:
            print("Choix invalide.")

if __name__ == "__main__":
    main() 