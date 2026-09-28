"""
source_donnees.py — Sources de mesures interchangeables pour le dashboard.

Principe : toute source expose une méthode `step()` qui renvoie UN dictionnaire
de mesures avec les clés de CLES_MESURE (le même format que
`GeneratorSimulator.step()` dans simulation.py).

Le dashboard et prediction.py ne savent donc pas d'où viennent les chiffres :
- GeneratorSimulator  -> données inventées (démo)
- SourceCSV           -> relevé réel enregistré dans un fichier CSV
- SourceCapteurs      -> lecture en direct (ESP32), à écrire plus tard
"""

import json
import os
import threading
import time
from collections import deque

import pandas as pd

CLES_MESURE = [
    "t_minutes",
    "heures_cumulees",
    "charge_pct",
    "temperature_c",
    "vibration_mm_s",
    "conso_instantanee_l_h",
    "carburant_restant_l",
    "carburant_consomme_total_l",
    "scenario",
]

# Colonnes que le CSV doit contenir. Le reste est calculé si absent.
COLONNES_OBLIGATOIRES = [
    "temperature_c",
    "vibration_mm_s",
    "charge_pct",
    "carburant_restant_l",
]


class FinDesDonnees(Exception):
    """Levée quand une source à durée limitée (CSV) n'a plus de mesures."""


class AttenteMesure(Exception):
    """Levée par une source en direct quand aucune nouvelle mesure n'est arrivée à temps."""


# ----------------------------------------------------------------------
# Fonctions utiles pour transformer les mesures brutes des capteurs
# ----------------------------------------------------------------------
def charge_pct_depuis_puissance(puissance_w: float, puissance_nominale_w: float) -> float:
    """Charge du groupe en % = puissance mesurée (ex. PZEM-004T) / puissance nominale."""
    if puissance_nominale_w <= 0:
        raise ValueError("La puissance nominale doit être > 0")
    return max(0.0, min(100.0, 100.0 * puissance_w / puissance_nominale_w))


def vibration_rms_mm_s(accel_g, freq_echantillonnage_hz: float,
                       f_min: float = 10.0, f_max: float = 1000.0) -> float:
    """
    Convertit des échantillons d'accélération (en g, ex. MPU6050/ADXL345)
    en vitesse de vibration RMS (mm/s), l'unité utilisée par les seuils
    de prediction.py (norme type ISO 10816 : bande 10 Hz - 1000 Hz).

    accel_g : liste d'au moins ~1 seconde d'échantillons, sur UN axe
    freq_echantillonnage_hz : fréquence d'échantillonnage réelle du capteur
    """
    import numpy as np

    a = np.asarray(accel_g, dtype=float)
    if a.size < 16:
        raise ValueError("Pas assez d'échantillons pour calculer la vibration")

    a = (a - a.mean()) * 9.80665            # g -> m/s², on retire la composante continue
    n = a.size
    spectre = np.fft.rfft(a)
    freqs = np.fft.rfftfreq(n, d=1.0 / freq_echantillonnage_hz)
    borne_haute = min(f_max, freq_echantillonnage_hz / 2.0)
    masque = (freqs >= f_min) & (freqs <= borne_haute)

    spectre_vitesse = np.zeros_like(spectre)
    spectre_vitesse[masque] = spectre[masque] / (2j * np.pi * freqs[masque])  # intégration
    vitesse = np.fft.irfft(spectre_vitesse, n)
    return float(np.sqrt(np.mean(vitesse ** 2)) * 1000.0)  # m/s -> mm/s


# ----------------------------------------------------------------------
# Source : fichier CSV de mesures réelles
# ----------------------------------------------------------------------
class SourceCSV:
    """
    Rejoue un relevé de mesures, une ligne par appel à step().

    Colonnes obligatoires : temperature_c, vibration_mm_s, charge_pct,
    carburant_restant_l
    Colonnes facultatives : t_minutes, heures_cumulees,
    conso_instantanee_l_h (sinon déduite de la baisse du niveau du réservoir)
    """

    def __init__(self, fichier, pas_de_temps_min: float = 1.0,
                 heures_initiales: float = 0.0, conso_nominale_l_h: float = 3.5):
        df = pd.read_csv(fichier)
        df.columns = [c.strip() for c in df.columns]

        manquantes = [c for c in COLONNES_OBLIGATOIRES if c not in df.columns]
        if manquantes:
            raise ValueError(f"Colonnes manquantes dans le CSV : {', '.join(manquantes)}")

        df = df.dropna(subset=COLONNES_OBLIGATOIRES).reset_index(drop=True)
        if df.empty:
            raise ValueError("Le CSV ne contient aucune ligne exploitable")

        # Temps
        if "t_minutes" not in df.columns:
            df["t_minutes"] = (df.index + 1) * pas_de_temps_min
        dt = df["t_minutes"].diff().fillna(pas_de_temps_min).clip(lower=0)

        # Consommation instantanée (L/h)
        if "conso_instantanee_l_h" not in df.columns:
            baisse_l = (-df["carburant_restant_l"].diff()).clip(lower=0)
            conso = baisse_l * 60.0 / dt.replace(0, pas_de_temps_min)
            conso = conso.rolling(10, min_periods=1).mean()   # lisse le bruit du capteur de niveau
            df["conso_instantanee_l_h"] = conso.bfill().fillna(conso_nominale_l_h)

        # Heures de fonctionnement cumulées
        if "heures_cumulees" not in df.columns:
            df["heures_cumulees"] = heures_initiales + df["t_minutes"] / 60.0

        # Carburant consommé cumulé
        df["carburant_consomme_total_l"] = (df["conso_instantanee_l_h"] * dt / 60.0).cumsum()

        self._mesures = []
        for _, r in df.iterrows():
            self._mesures.append({
                "t_minutes": round(float(r["t_minutes"]), 1),
                "heures_cumulees": round(float(r["heures_cumulees"]), 2),
                "charge_pct": round(float(r["charge_pct"]), 1),
                "temperature_c": round(float(r["temperature_c"]), 1),
                "vibration_mm_s": round(float(r["vibration_mm_s"]), 2),
                "conso_instantanee_l_h": round(float(r["conso_instantanee_l_h"]), 2),
                "carburant_restant_l": round(float(r["carburant_restant_l"]), 1),
                "carburant_consomme_total_l": round(float(r["carburant_consomme_total_l"]), 2),
                "scenario": "donnees_reelles",
            })
        self._i = 0

    def __len__(self) -> int:
        return len(self._mesures)

    def step(self) -> dict:
        if self._i >= len(self._mesures):
            raise FinDesDonnees("Fin du fichier de mesures")
        mesure = self._mesures[self._i]
        self._i += 1
        return dict(mesure)


# ----------------------------------------------------------------------
# Source : capteurs en direct (ESP32 -> broker MQTT -> ce programme)
# ----------------------------------------------------------------------
class SourceCapteurs:
    """
    Reçoit en direct les mesures d'UN groupe électrogène via MQTT.

    Topic écouté : groupes/<groupe_id>/mesures   (un topic par groupe)
    Pour surveiller plusieurs groupes : une SourceCapteurs par groupe_id.

    Message JSON publié par l'ESP32 (voir esp32_groupe/groupe_esp32.ino) :
      {
        "temperature_c": 84.5,
        "puissance_w": 2100,          # PZEM-004T
        "niveau_l": 143.2,            # niveau du réservoir
        "fe_hz": 500,                 # fréquence d'échantillonnage du MPU6050
        "accel_mg": [12, -30, ...],   # accélération en milli-g, 1 axe
        "conso_l_h": 3.1              # FACULTATIF (si débitmètre)
      }

    step() attend la prochaine mesure (max `timeout_s`), puis renvoie le
    même dictionnaire que GeneratorSimulator.step(). Sans mesure à temps,
    il lève AttenteMesure (le dashboard réessaie).
    """

    def __init__(self, groupe_id: str = "groupe1", broker: str = "localhost",
                 port: int = 1883, puissance_nominale_w: float = 5000.0,
                 conso_nominale_l_h: float = 3.5, heures_initiales: float = 0.0,
                 fichier_compteur: str = None, timeout_s: float = 5.0,
                 utilisateur: str = None, mot_de_passe: str = None,
                 seuil_marche_vib_mm_s: float = 0.5):
        self.groupe_id = groupe_id
        self.topic = f"groupes/{groupe_id}/mesures"
        self.puissance_nominale_w = puissance_nominale_w
        self.conso_nominale_l_h = conso_nominale_l_h
        self.timeout_s = timeout_s
        self.seuil_marche_vib = seuil_marche_vib_mm_s
        self.derniere_erreur = None
        self.connecte = False

        # Compteur d'heures sauvegardé sur disque (sinon remis à 0 au redémarrage)
        self._fichier_compteur = fichier_compteur or f"compteur_{groupe_id}.json"
        self._heures = self._charger_compteur(heures_initiales)
        self._derniere_sauvegarde = time.monotonic()

        self._t0 = time.monotonic()
        self._t_precedent = None
        self._conso_totale_l = 0.0
        self._historique_niveau = deque(maxlen=600)   # (temps_s, niveau_l)

        self._verrou = threading.Lock()
        self._nouvelle = threading.Event()
        self._derniere_mesure = None

        try:
            import paho.mqtt.client as mqtt
        except ImportError as e:
            raise ImportError("Installe paho-mqtt : pip install paho-mqtt") from e

        try:  # paho-mqtt >= 2.0
            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:  # paho-mqtt 1.x
            self._client = mqtt.Client()
        if utilisateur:
            self._client.username_pw_set(utilisateur, mot_de_passe)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client.connect_async(broker, port, keepalive=30)
        self._client.loop_start()          # thread réseau en arrière-plan

    # --- callbacks MQTT (signatures compatibles paho 1.x et 2.x) ---
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        self.connecte = True
        client.subscribe(self.topic, qos=1)

    def _on_disconnect(self, client, userdata, *args, **kwargs):
        self.connecte = False

    def _on_message(self, client, userdata, msg):
        self._traiter_message(msg.payload)

    # --- traitement d'un message (séparé pour pouvoir le tester sans broker) ---
    def _traiter_message(self, payload: bytes, maintenant: float = None):
        try:
            d = json.loads(payload)
            temperature = float(d["temperature_c"])
            puissance = float(d["puissance_w"])
            niveau = float(d["niveau_l"])
            vibration = vibration_rms_mm_s(
                [mg / 1000.0 for mg in d["accel_mg"]], float(d["fe_hz"]))
            charge = charge_pct_depuis_puissance(puissance, self.puissance_nominale_w)
        except (ValueError, KeyError, TypeError) as e:
            self.derniere_erreur = f"Message ignoré : {e}"
            return

        t = time.monotonic() if maintenant is None else maintenant
        dt_s = 0.0 if self._t_precedent is None else min(t - self._t_precedent, 60.0)
        self._t_precedent = t
        en_marche = vibration >= self.seuil_marche_vib

        # Consommation instantanée
        self._historique_niveau.append((t, niveau))
        conso = d.get("conso_l_h")
        if conso is None:
            conso = self._conso_depuis_niveau(charge)
        conso = max(0.0, float(conso))

        if en_marche:
            self._heures += dt_s / 3600.0
            self._conso_totale_l += conso * dt_s / 3600.0
        self._sauvegarder_compteur_si_besoin()

        mesure = {
            "t_minutes": round((t - self._t0) / 60.0, 2),
            "heures_cumulees": round(self._heures, 2),
            "charge_pct": round(charge, 1),
            "temperature_c": round(temperature, 1),
            "vibration_mm_s": round(vibration, 2),
            "conso_instantanee_l_h": round(conso, 2),
            "carburant_restant_l": round(niveau, 1),
            "carburant_consomme_total_l": round(self._conso_totale_l, 2),
            "scenario": f"direct:{self.groupe_id}",
        }
        with self._verrou:
            self._derniere_mesure = mesure
        self._nouvelle.set()

    def _conso_depuis_niveau(self, charge_pct: float) -> float:
        """Pente du niveau du réservoir sur ~10 min ; sinon estimation selon la charge."""
        import numpy as np
        pts = self._historique_niveau
        if len(pts) >= 5 and pts[-1][0] - pts[0][0] >= 120:
            t = np.array([p[0] for p in pts])
            n = np.array([p[1] for p in pts])
            pente = np.polyfit(t, n, 1)[0]               # L par seconde (négatif si ça baisse)
            return max(0.0, -pente * 3600.0)
        return self.conso_nominale_l_h * (0.5 + 0.5 * charge_pct / 100.0)

    # --- compteur d'heures persistant ---
    def _charger_compteur(self, defaut: float) -> float:
        try:
            with open(self._fichier_compteur, encoding="utf-8") as f:
                return float(json.load(f)["heures_cumulees"])
        except (OSError, ValueError, KeyError):
            return defaut

    def _sauvegarder_compteur_si_besoin(self, force: bool = False):
        if not force and time.monotonic() - self._derniere_sauvegarde < 60:
            return
        self._derniere_sauvegarde = time.monotonic()
        tmp = self._fichier_compteur + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"heures_cumulees": self._heures}, f)
            os.replace(tmp, self._fichier_compteur)      # écriture atomique
        except OSError as e:
            self.derniere_erreur = f"Sauvegarde du compteur impossible : {e}"

    # --- interface commune avec les autres sources ---
    def step(self) -> dict:
        if not self._nouvelle.wait(self.timeout_s):
            raise AttenteMesure(
                "Aucune mesure reçue"
                + ("" if self.connecte else " (broker MQTT injoignable)")
                + (f" — {self.derniere_erreur}" if self.derniere_erreur else ""))
        with self._verrou:
            mesure = dict(self._derniere_mesure)
            self._nouvelle.clear()
        return mesure

    def fermer(self):
        self._sauvegarder_compteur_si_besoin(force=True)
        self._client.loop_stop()
        self._client.disconnect()