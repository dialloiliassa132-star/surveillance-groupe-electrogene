"""
prediction.py — Moteur de prédiction à base de règles (pas de ML boîte noire).

Choix assumé : à ce stade du projet (pas encore de données réelles
collectées sur le terrain), un modèle basé sur des seuils physiques
+ détection de tendance est plus défendable devant un jury qu'un modèle
ML entraîné sur des données simulées. Il est aussi 100% explicable :
chaque alerte peut être justifiée point par point.

Quand des données réelles seront disponibles (après le partenaire pilote),
ces seuils pourront être calibrés, voire remplacés par un modèle appris,
sans changer l'interface de ce module (`evaluer_risque` reste le point
d'entrée unique du dashboard).
"""

from dataclasses import dataclass
from collections import deque


@dataclass
class SeuilsConfig:
    temp_alerte: float = 95.0
    temp_critique: float = 105.0
    vibration_alerte: float = 4.5
    vibration_critique: float = 6.5
    fenetre_tendance: int = 20          # nb de mesures pour calculer la pente affichée (indicatif)
    heures_vie_totale: float = 8000.0   # durée de vie moteur avant révision majeure
    seuil_derive_carburant_pct: float = 15.0  # % au-dessus de la conso attendue
    conso_nominale_l_h: float = 3.5     # L/h à pleine charge : À CALIBRER sur le vrai groupe
    conso_ratio_a_vide: float = 0.5     # conso à charge nulle / conso à pleine charge (hypothèse à mesurer)


class MoteurPrediction:
    """Analyse un historique de mesures et produit un diagnostic explicable."""

    def __init__(self, seuils: SeuilsConfig = None):
        self.seuils = seuils or SeuilsConfig()
        self.historique_temp = deque(maxlen=self.seuils.fenetre_tendance)
        self.historique_vib = deque(maxlen=self.seuils.fenetre_tendance)
        self.conso_attendue_l_h_reference = None  # calibrée sur les 1res mesures

    @staticmethod
    def _pente(valeurs: deque) -> float:
        """Pente approximative (régression linéaire simple) sur la fenêtre."""
        n = len(valeurs)
        if n < 2:
            return 0.0
        xs = list(range(n))
        moyenne_x = sum(xs) / n
        moyenne_y = sum(valeurs) / n
        num = sum((x - moyenne_x) * (y - moyenne_y) for x, y in zip(xs, valeurs))
        den = sum((x - moyenne_x) ** 2 for x in xs) or 1e-9
        return num / den

    def evaluer_risque(self, mesure: dict) -> dict:
        """
        Retourne un diagnostic : niveau de risque, raisons, estimation du
        temps restant avant intervention recommandée.
        """
        self.historique_temp.append(mesure["temperature_c"])
        self.historique_vib.append(mesure["vibration_mm_s"])

        raisons = []
        niveau = "faible"

        temp = mesure["temperature_c"]
        vib = mesure["vibration_mm_s"]
        pente_temp = self._pente(self.historique_temp)
        pente_vib = self._pente(self.historique_vib)

        # --- Seuils absolus ---
        if temp >= self.seuils.temp_critique:
            niveau = "eleve"
            raisons.append(f"Température critique atteinte ({temp:.1f}°C ≥ {self.seuils.temp_critique}°C)")
        elif temp >= self.seuils.temp_alerte:
            niveau = "modere"
            raisons.append(f"Température élevée ({temp:.1f}°C ≥ {self.seuils.temp_alerte}°C)")

        if vib >= self.seuils.vibration_critique:
            niveau = "eleve"
            raisons.append(f"Vibration critique ({vib:.2f} mm/s ≥ {self.seuils.vibration_critique} mm/s)")
        elif vib >= self.seuils.vibration_alerte and niveau != "eleve":
            niveau = "modere"
            raisons.append(f"Vibration élevée ({vib:.2f} mm/s ≥ {self.seuils.vibration_alerte} mm/s)")

        # --- Tendance (information seulement, n'escalade pas le niveau) ---
        # NOTE méthodologique : avec la charge électrique qui varie naturellement
        # (démarrage d'équipements, etc.), la pente instantanée de température/
        # vibration oscille normalement dans une plage comparable à celle d'une
        # dérive réelle en tout début de dégradation — elle n'est donc pas assez
        # fiable, seule, pour déclencher une alerte sans faux positifs. On
        # l'affiche à titre indicatif (utile pour le suivi et la calibration
        # future avec des données réelles), et on se base sur les seuils
        # absolus ci-dessus pour la détection, qui montent en deux paliers
        # (modéré puis élevé) et donnent donc déjà une alerte avant la panne
        # complète. Piste d'amélioration une fois des données réelles
        # disponibles : détrender la charge avant de calculer la pente.

        if not raisons:
            raisons.append("Paramètres dans les plages normales")

        # --- Estimation du temps restant avant intervention recommandée ---
        temps_restant_min = self._estimer_temps_restant(temp, vib, pente_temp, pente_vib)

        return {
            "niveau_risque": niveau,
            "raisons": raisons,
            "temps_restant_estime_min": temps_restant_min,
            "pente_temperature": round(pente_temp, 3),
            "pente_vibration": round(pente_vib, 4),
        }

    def _estimer_temps_restant(self, temp, vib, pente_temp, pente_vib) -> float | None:
        """
        Extrapole linéairement (sur la tendance actuelle) le temps avant
        d'atteindre le seuil critique. Retourne None si aucune dérive
        significative n'est en cours (pas d'urgence identifiable).
        """
        estimations = []
        if pente_temp > 1e-4:
            marge = self.seuils.temp_critique - temp
            if marge > 0:
                estimations.append(marge / pente_temp)
        if pente_vib > 1e-5:
            marge = self.seuils.vibration_critique - vib
            if marge > 0:
                estimations.append(marge / pente_vib)
        return round(min(estimations), 1) if estimations else None

    def detecter_derive_carburant(self, conso_instantanee_l_h: float, charge_pct: float) -> dict:
        """
        Compare la consommation observée à la consommation théorique attendue
        pour la charge actuelle. Une dérive persistante suggère un moteur mal
        réglé ou en fin de vie plutôt qu'une simple variation de charge.
        """
        # Modèle linéaire simple : conso = nominale * (ratio_a_vide + (1 - ratio_a_vide) * charge).
        # ATTENTION : c'est une hypothèse. Sur un vrai groupe, mesurer la conso à
        # plusieurs charges (0 %, 50 %, 100 %) puis ajuster conso_nominale_l_h
        # et conso_ratio_a_vide dans SeuilsConfig.
        ratio = self.seuils.conso_ratio_a_vide
        conso_attendue = self.seuils.conso_nominale_l_h * (ratio + (1 - ratio) * charge_pct / 100)
        if conso_attendue <= 0:
            return {"derive_detectee": False, "ecart_pct": 0.0, "conso_attendue_l_h": 0.0}

        ecart_pct = 100 * (conso_instantanee_l_h - conso_attendue) / conso_attendue
        derive = ecart_pct >= self.seuils.seuil_derive_carburant_pct

        return {
            "derive_detectee": derive,
            "ecart_pct": round(ecart_pct, 1),
            "conso_attendue_l_h": round(conso_attendue, 2),
        }

    def estimer_duree_vie_restante(self, heures_cumulees: float) -> dict:
        """
        Estimation simple basée sur les heures cumulées vs durée de vie
        totale attendue avant révision majeure.
        """
        heures_restantes = max(0.0, self.seuils.heures_vie_totale - heures_cumulees)
        pct_use = min(100.0, 100 * heures_cumulees / self.seuils.heures_vie_totale)
        return {
            "heures_restantes_estimees": round(heures_restantes, 1),
            "pourcentage_vie_consommee": round(pct_use, 1),
        }