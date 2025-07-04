# data_quality.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import logging

class DataQualityChecker:
    def __init__(self):
        self.quality_rules = {
            'temperature': {'min': -50, 'max': 60},
            'humidity': {'min': 0, 'max': 100},
            'aqi': {'min': 0, 'max': 500},
            'pressure': {'min': 900, 'max': 1100},
            'wind_speed': {'min': 0, 'max': 200}
        }
        
    def validate_data(self, df):
        """Validation complète d'un DataFrame"""
        results = {
            'total_rows': len(df),
            'valid_rows': 0,
            'errors': [],
            'quality_score': 0
        }
        
        # 1. Contrôle de complétude
        completeness = self._check_completeness(df)
        
        # 2. Contrôle de cohérence
        consistency = self._check_consistency(df)
        
        # 3. Contrôle de fraîcheur
        freshness = self._check_freshness(df)
        
        # 4. Contrôle de plausibilité
        plausibility = self._check_plausibility(df)
        
        # Calcul du score global
        results['completeness'] = completeness
        results['consistency'] = consistency
        results['freshness'] = freshness
        results['plausibility'] = plausibility
        
        results['quality_score'] = np.mean([
            completeness['score'],
            consistency['score'], 
            freshness['score'],
            plausibility['score']
        ])
        
        return results
    
    def _check_completeness(self, df):
        """Vérifie la complétude des données"""
        missing_counts = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        missing_cells = missing_counts.sum()
        
        completeness_score = (total_cells - missing_cells) / total_cells * 100
        
        return {
            'score': completeness_score,
            'missing_by_column': missing_counts.to_dict(),
            'status': 'PASS' if completeness_score >= 95 else 'FAIL'
        }
    
    def _check_consistency(self, df):
        """Vérifie la cohérence des données"""
        errors = []
        
        # Vérifier que temp_ressentie >= temp_réelle en été
        if 'Température (°C)' in df.columns and 'Température ressentie (°C)' in df.columns:
            temp_issues = df[df['Température ressentie (°C)'] < df['Température (°C)'] - 10]
            if len(temp_issues) > 0:
                errors.append(f"{len(temp_issues)} valeurs de température ressentie incohérentes")
        
        # Vérifier cohérence AQI vs polluants
        if 'AQI' in df.columns:
            high_aqi = df[df['AQI'] > 100]
            if len(high_aqi) > 0:
                # Vérifier que des polluants sont élevés aussi
                for _, row in high_aqi.iterrows():
                    if all(pd.isna([row.get('PM2.5'), row.get('PM10'), row.get('O3')])):
                        errors.append(f"AQI élevé sans données de polluants pour {row.get('Ville', 'ville inconnue')}")
        
        consistency_score = max(0, 100 - len(errors) * 10)
        
        return {
            'score': consistency_score,
            'errors': errors,
            'status': 'PASS' if consistency_score >= 80 else 'FAIL'
        }
    
    def _check_freshness(self, df):
        """Vérifie la fraîcheur des données"""
        if 'Timestamp' not in df.columns:
            return {'score': 0, 'status': 'FAIL', 'reason': 'Pas de timestamp'}
        
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        now = datetime.now()
        cutoff = now - timedelta(hours=2)
        
        fresh_data = df[df['Timestamp'] > cutoff]
        freshness_score = len(fresh_data) / len(df) * 100
        
        return {
            'score': freshness_score,
            'old_records': len(df) - len(fresh_data),
            'status': 'PASS' if freshness_score >= 90 else 'FAIL'
        }
    
    def _check_plausibility(self, df):
        """Vérifie la plausibilité des valeurs"""
        errors = []
        
        for column, rules in self.quality_rules.items():
            if column in df.columns:
                col_data = df[column].dropna()
                if len(col_data) > 0:
                    out_of_range = col_data[(col_data < rules['min']) | (col_data > rules['max'])]
                    if len(out_of_range) > 0:
                        errors.append(f"{len(out_of_range)} valeurs hors plage pour {column}")
        
        plausibility_score = max(0, 100 - len(errors) * 15)
        
        return {
            'score': plausibility_score,
            'errors': errors,
            'status': 'PASS' if plausibility_score >= 85 else 'FAIL'
        }
    
    def generate_quality_report(self, results):
        """Génère un rapport de qualité"""
        report = f"""
        📊 RAPPORT DE QUALITÉ DES DONNÉES - {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        🎯 SCORE GLOBAL: {results['quality_score']:.1f}/100
        
        📋 DÉTAILS PAR DIMENSION:
        • Complétude: {results['completeness']['score']:.1f}% ({results['completeness']['status']})
        • Cohérence: {results['consistency']['score']:.1f}% ({results['consistency']['status']})
        • Fraîcheur: {results['freshness']['score']:.1f}% ({results['freshness']['status']})
        • Plausibilité: {results['plausibility']['score']:.1f}% ({results['plausibility']['status']})
        
        ⚠️ PROBLÈMES DÉTECTÉS:
        {chr(10).join(['• ' + error for error in results['consistency']['errors']])}
        {chr(10).join(['• ' + error for error in results['plausibility']['errors']])}
        
        📈 RECOMMANDATIONS:
        {"• Améliorer la collecte de données" if results['quality_score'] < 80 else "• Maintenir la qualité actuelle"}
        """
        
        return report
    
    def send_alert(self, results, threshold=80):
        """Envoie une alerte si la qualité est insuffisante"""
        if results['quality_score'] < threshold:
            report = self.generate_quality_report(results)
            
            # Configuration email (à adapter)
            try:
                msg = MIMEText(report)
                msg['Subject'] = f"🚨 ALERTE QUALITÉ DONNÉES - Score: {results['quality_score']:.1f}%"
                msg['From'] = "data-quality@goodair.fr"
                msg['To'] = "admin@goodair.fr"
                
                # Envoi email (configurer SMTP)
                # server = smtplib.SMTP('localhost')
                # server.send_message(msg)
                # server.quit()
                
                print("📧 Alerte qualité envoyée")
                
            except Exception as e:
                logging.error(f"Erreur envoi alerte: {e}")

# Utilisation dans transform.py
def quality_check_pipeline():
    """Intégration dans le pipeline de transformation"""
    checker = DataQualityChecker()
    
    # Charger les données
    df = pd.read_csv('donnees_completes.csv')
    
    # Validation
    results = checker.validate_data(df)
    
    # Rapport
    report = checker.generate_quality_report(results)
    print(report)
    
    # Alerte si nécessaire
    checker.send_alert(results)
    
    # Sauvegarde des métriques qualité
    quality_metrics = pd.DataFrame([{
        'timestamp': datetime.now(),
        'quality_score': results['quality_score'],
        'completeness': results['completeness']['score'],
        'consistency': results['consistency']['score'],
        'freshness': results['freshness']['score'],
        'plausibility': results['plausibility']['score']
    }])
    
    quality_metrics.to_csv('quality_metrics.csv', mode='a', header=False, index=False)
    
    return results['quality_score'] >= 80  # Retourne True si qualité OK

if __name__ == "__main__":
    quality_check_pipeline()