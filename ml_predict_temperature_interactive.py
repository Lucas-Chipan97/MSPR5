import pandas as pd
import numpy as np
import joblib

# Charger le modèle GradientBoosting pour la température
model_data = joblib.load('ml_temperature_model.pkl')
model = model_data['model']
le = model_data['label_encoder']
features = model_data['features']

# Liste des villes connues
villes = list(le.classes_)

def predict_temperature():
    print("\n=== Prédiction interactive de température (GradientBoosting) ===")
    print("Villes disponibles :")
    for i, v in enumerate(villes, 1):
        print(f"{i}. {v}")
    try:
        idx = int(input(f"Choisissez une ville (1-{len(villes)}): ")) - 1
        ville = villes[idx]
    except:
        print("❌ Choix invalide.")
        return
    try:
        humidite = float(input("Humidité (%): "))
        pression = float(input("Pression (hPa): "))
        vent = float(input("Vitesse du vent (m/s): "))
        nuages = float(input("Couverture nuageuse (%): "))
        aqi = float(input("AQI: "))
        pm25 = float(input("PM2.5: "))
        pm10 = float(input("PM10: "))
        o3 = float(input("O3: "))
        no2 = float(input("NO2: "))
        heure = int(input("Heure (0-23): "))
        mois = int(input("Mois (1-12): "))
    except:
        print("❌ Entrée invalide.")
        return

    ville_encoded = le.transform([ville])[0]

    X_pred = pd.DataFrame([{
        'Humidité (%)': humidite,
        'Pression (hPa)': pression,
        'Vitesse du vent (m/s)': vent,
        'Couverture nuageuse (%)': nuages,
        'AQI': aqi,
        'PM2.5': pm25,
        'PM10': pm10,
        'O3': o3,
        'NO2': no2,
        'Hour': heure,
        'Month': mois,
        'Ville_encoded': ville_encoded
    }])[features]

    X_pred = X_pred.fillna(X_pred.mean())
    prediction = model.predict(X_pred)[0]
    print(f"\n🌡️ Température prédite : {prediction:.2f} °C")

def main():
    print("\n=== Interface de prédiction : température ===")
    while True:
        print("\n1. Faire une prédiction")
        print("2. Quitter")
        choice = input("Votre choix : ").strip()
        if choice == '1':
            predict_temperature()
        elif choice == '2':
            print("Au revoir 👋")
            break
        else:
            print("Choix invalide.")

if __name__ == "__main__":
    main()
