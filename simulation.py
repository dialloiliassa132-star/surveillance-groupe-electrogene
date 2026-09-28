"""
simulation.py — Moteur de simulation de données capteurs pour groupe électrogène.

Génère des données réalistes (température, vibration, niveau/consommation
de carburant) pas à pas, avec la possibilité d'injecter un scénario de
panne progressive pour démontrer la détection.

Quand de vrais capteurs seront installés, il suffira de remplacer les
appels à `GeneratorSimulator.step()` par une lecture série / MQTT qui
retourne le même format de dictionnaire — le reste de l'app ne change pas.
"""

import random
import math
from dataclasses import dataclass, field


@dataclass
class SimulatorConfig:
    """Paramètres physiques par défaut (moteur diesel PME, quelques kVA)."""
    temp_ambiante: float = 30.0        # °C, climat sénégalais
    temp_nominale: float = 85.0        # °C en fonctionnement normal en charge
    temp_max_securite: float = 110.0   # °C seuil critique constructeur
    vibration_nominale: float = 2.5    # mm/s (RMS), fonctionnement sain
    vibration_max_securite: float = 7.0  # mm/s seuil critique
    conso_nominale_l_h: float = 3.5    # L/h à charge moyenne
    reservoir_capacite_l: float = 200.0
    bruit_temp: float = 0.8
    bruit_vibration: float = 0.15
    bruit_conso: float = 0.05
    pas_de_temps_min: float = 1.0      # 1 pas de simulation = 1 minute simulée


class GeneratorSimulator:
    """
    Simule l'état d'un groupe électrogène au fil du temps.

    scenario:
        "normal"                 -> fonctionnement sain
        "surchauffe_progressive" -> dérive lente de la température (ex: radiateur encrassé)
        "vibration_anormale"     -> montée de vibration (ex: désalignement, usure paliers)
        "derive_carburant"       -> surconsommation progressive (ex: mauvais réglage injection)
    """

    def __init__(self, config: SimulatorConfig = None, scenario: str = "normal", seed: int = None):
        self.cfg = config or SimulatorConfig()
        self.scenario = scenario
        self.rng = random.Random(seed)

        self.t_minutes = 0.0
        self.heures_cumulees = 0.0
        self.carburant_restant_l = self.cfg.reservoir_capacite_l
        self.carburant_consomme_total_l = 0.0

        self._temp_actuelle = self.cfg.temp_ambiante
        self._vib_actuelle = 0.3
        self._defaut_progression = 0.0  # avance du scénario de panne, 0 -> 1

    def _charge(self) -> float:
        """Simule une charge électrique variable (0 à 1) au cours du temps."""
        base = 0.55 + 0.25 * math.sin(self.t_minutes / 45.0)
        return max(0.15, min(1.0, base + self.rng.uniform(-0.05, 0.05)))

    def step(self) -> dict:
        """Avance la simulation d'un pas et retourne les mesures courantes."""
        dt = self.cfg.pas_de_temps_min
        self.t_minutes += dt
        self.heures_cumulees += dt / 60.0
        charge = self._charge()

        # --- Progression du défaut selon le scénario choisi ---
        if self.scenario != "normal":
            self._defaut_progression = min(1.0, self._defaut_progression + 0.0018)

        # --- Température : tend vers une cible dépendant de la charge, avec inertie ---
        temp_cible = self.cfg.temp_ambiante + (self.cfg.temp_nominale - self.cfg.temp_ambiante) * charge
        if self.scenario == "surchauffe_progressive":
            temp_cible += self._defaut_progression * 35.0
        self._temp_actuelle += (temp_cible - self._temp_actuelle) * 0.08
        temperature = self._temp_actuelle + self.rng.gauss(0, self.cfg.bruit_temp)

        # --- Vibration : corrélée à la charge et légèrement à la température ---
        vib_cible = self.cfg.vibration_nominale * (0.6 + 0.4 * charge)
        if self.scenario == "vibration_anormale":
            vib_cible += self._defaut_progression * 6.0
        self._vib_actuelle += (vib_cible - self._vib_actuelle) * 0.15
        vibration = max(0.0, self._vib_actuelle + self.rng.gauss(0, self.cfg.bruit_vibration))

        # --- Consommation carburant ---
        conso_horaire = self.cfg.conso_nominale_l_h * (0.5 + 0.5 * charge)
        if self.scenario == "derive_carburant":
            conso_horaire *= (1.0 + 0.5 * self._defaut_progression)
        conso_horaire = max(0.1, conso_horaire + self.rng.gauss(0, self.cfg.bruit_conso))
        conso_ce_pas_l = conso_horaire * (dt / 60.0)
        self.carburant_consomme_total_l += conso_ce_pas_l
        self.carburant_restant_l = max(0.0, self.carburant_restant_l - conso_ce_pas_l)

        return {
            "t_minutes": round(self.t_minutes, 1),
            "heures_cumulees": round(self.heures_cumulees, 2),
            "charge_pct": round(charge * 100, 1),
            "temperature_c": round(temperature, 1),
            "vibration_mm_s": round(vibration, 2),
            "conso_instantanee_l_h": round(conso_horaire, 2),
            "carburant_restant_l": round(self.carburant_restant_l, 1),
            "carburant_consomme_total_l": round(self.carburant_consomme_total_l, 2),
            "scenario": self.scenario,
        }