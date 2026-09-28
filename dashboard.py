"""
dashboard.py — Application Streamlit : Surveillance & Prédiction pour
Groupes Électrogènes (PME sénégalaises).

Lancer avec :  streamlit run dashboard.py
"""

import base64
import os
import time
import pandas as pd
import streamlit as st

from simulation import GeneratorSimulator, SimulatorConfig
from prediction import MoteurPrediction, SeuilsConfig
from source_donnees import SourceCSV, SourceCapteurs, FinDesDonnees, AttenteMesure

st.set_page_config(page_title="Surveillance Groupe Électrogène", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------
# Entête (fixe, pleine largeur, avec photo) et pied de page (fixe)
# Pour ta photo : mets un fichier photo.jpg (ou .png / .jpeg) à côté de dashboard.py
# ----------------------------------------------------------------------
def _photo_html() -> str:
    dossier = os.path.dirname(os.path.abspath(__file__))
    for nom, mime in (("photo.jpg", "jpeg"), ("photo.jpeg", "jpeg"), ("photo.png", "png")):
        chemin = os.path.join(dossier, nom)
        if os.path.exists(chemin):
            with open(chemin, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f'<img class="photo" src="data:image/{mime};base64,{data}">'
    return '<div class="photo vide">👤</div>'


st.markdown("""
<style>
/* Entête fixe tout en haut, sur toute la largeur de l'écran */
.entete {position:fixed; top:0; left:0; right:0; height:76px; z-index:999;
         background:#0f3d5e; color:white; box-sizing:border-box;
         display:flex; align-items:center; justify-content:space-between;
         padding:0 16px 0 56px; box-shadow:0 2px 6px rgba(0,0,0,0.25);}
.entete .titres {min-width:0;}
.entete h2 {margin:0; padding:0; color:white; font-size:1.3rem; line-height:1.2;}
.entete p {margin:2px 0 0 0; font-size:0.8rem; opacity:0.85;}
.entete .photo {width:52px; height:52px; border-radius:50%; object-fit:cover;
                border:2px solid white; flex-shrink:0; margin-left:12px;}
.entete .photo.vide {display:flex; align-items:center; justify-content:center;
                     background:rgba(255,255,255,0.2); font-size:1.6rem;}
/* La barre Streamlit devient transparente : la flèche pour ouvrir la configuration reste visible */
header[data-testid="stHeader"] {background:transparent;}
header[data-testid="stHeader"] button, [data-testid="collapsedControl"] button,
[data-testid="stExpandSidebarButton"] {color:white;}
/* Pied de page fixe, plus haut */
.pied {position:fixed; left:0; right:0; bottom:0; height:80px; z-index:999;
       background:#0f3d5e; color:white; box-sizing:border-box;
       display:flex; align-items:center; justify-content:center; text-align:center;
       padding:0 16px; font-size:0.9rem;}
/* Le contenu ne passe pas sous l'entête ni sous le pied de page */
.block-container {padding-top:96px !important; padding-bottom:100px !important;}
@media (max-width: 640px) {
  .entete h2 {font-size:1rem;}
  .entete p {display:none;}
  .entete .photo {width:44px; height:44px;}
  .pied {font-size:0.75rem;}
}

/* Grande flèche d'aide : visible seulement tant que la configuration est fermée */
.fleche {position:fixed; top:84px; left:8px; z-index:999; background:#ffd400; color:#0f3d5e;
         font-weight:700; font-size:1rem; padding:8px 14px; border-radius:10px;
         box-shadow:0 2px 8px rgba(0,0,0,0.3); animation:pulse 1.2s infinite;}
.fleche .grande {display:block; font-size:2rem; line-height:1; text-align:left;}
@keyframes pulse {0%,100% {transform:translateY(0);} 50% {transform:translateY(-6px);}}
/* Le bouton d'ouverture de la configuration devient gros et jaune */
[data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"] {
    transform:scale(1.7); transform-origin:left center; background:#ffd400 !important;
    border-radius:8px; margin-left:6px;}
[data-testid="stExpandSidebarButton"], [data-testid="stExpandSidebarButton"] button,
[data-testid="collapsedControl"], [data-testid="collapsedControl"] button {color:#0f3d5e !important;}
/* Configuration ouverte : la flèche disparaît et la photo (et l'entête) grandissent */
.stApp:has(section[data-testid="stSidebar"][aria-expanded="true"]) .fleche {display:none;}
.stApp:has(section[data-testid="stSidebar"][aria-expanded="true"]) .entete {height:120px;}
.stApp:has(section[data-testid="stSidebar"][aria-expanded="true"]) .entete .photo {
    width:96px; height:96px; font-size:3rem;}
.stApp:has(section[data-testid="stSidebar"][aria-expanded="true"]) .block-container {
    padding-top:140px !important;}
@media (min-width: 993px) {
  .stApp:has(section[data-testid="stSidebar"][aria-expanded="true"]) .entete {padding-left:calc(21rem + 16px);}
}
</style>
<div class="entete">
  <div class="titres">
    <h2>⚡ Surveillance et prédiction — Groupe électrogène</h2>
    <p>Suivi en temps réel, risque de panne et durée de vie du moteur</p>
  </div>
  """ + _photo_html() + """
</div>
<div class="fleche"><span class="grande">⬆</span>Cliquez sur la flèche jaune<br>pour ouvrir la configuration</div>
<div class="pied">Surveillance & prédiction — Groupes électrogènes · PME sénégalaises</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Etat de session (persiste entre les reruns Streamlit)
# ----------------------------------------------------------------------
if "historique" not in st.session_state:
    st.session_state.historique = []
if "source" not in st.session_state:
    st.session_state.source = None
if "moteur_prediction" not in st.session_state:
    st.session_state.moteur_prediction = None
if "en_marche" not in st.session_state:
    st.session_state.en_marche = False

COULEUR_RISQUE = {"faible": "🟢", "modere": "🟠", "eleve": "🔴"}
LABEL_RISQUE = {"faible": "Faible", "modere": "Modéré", "eleve": "Élevé"}

# ----------------------------------------------------------------------
# Barre latérale — configuration
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ Configuration")

    type_source = st.radio("Source des données", ["Simulation", "Fichier CSV (mesures réelles)", "Capteurs en direct (MQTT)"])
    conso_nominale = st.number_input(
        "Consommation à pleine charge (L/h)", min_value=0.5, max_value=100.0, value=3.5, step=0.5,
        help="Valeur du groupe à étudier. Une seule valeur, utilisée par la source et par la prédiction.",
    )

    scenario = "normal"
    fichier_csv = None
    heures_initiales = 0.0
    if type_source == "Simulation":
        scenario = st.selectbox(
            "Scénario de démonstration",
            ["normal", "surchauffe_progressive", "vibration_anormale", "derive_carburant"],
            format_func=lambda s: {
                "normal": "Fonctionnement normal",
                "surchauffe_progressive": "Panne : surchauffe progressive",
                "vibration_anormale": "Panne : vibration anormale",
                "derive_carburant": "Panne : dérive de consommation",
            }[s],
        )
    elif type_source == "Capteurs en direct (MQTT)":
        groupe_id = st.text_input("Identifiant du groupe", value="groupe1")
        broker = st.text_input("Broker MQTT (adresse)", value="localhost")
        port_mqtt = st.number_input("Port", min_value=1, max_value=65535, value=1883)
        puissance_nominale_kw = st.number_input(
            "Puissance nominale du groupe (kW)", min_value=0.5, value=5.0, step=0.5)
        heures_initiales = st.number_input(
            "Heures de marche déjà au compteur (utilisé seulement au 1er lancement)",
            min_value=0.0, value=0.0, step=10.0)
    else:
        fichier_csv = st.file_uploader("Fichier de mesures (.csv)", type="csv")
        heures_initiales = st.number_input(
            "Heures de marche du moteur avant ce relevé", min_value=0.0, value=0.0, step=10.0
        )
        st.caption(
            "Colonnes obligatoires : temperature_c, vibration_mm_s, charge_pct, "
            "carburant_restant_l. Facultatives : t_minutes, heures_cumulees, conso_instantanee_l_h."
        )

    vitesse = st.slider("Vitesse de lecture (pas / seconde)", 1, 10, 4)

    col_a, col_b = st.columns(2)
    if col_a.button("▶️ Démarrer", use_container_width=True):
        try:
            if hasattr(st.session_state.source, "fermer"):
                st.session_state.source.fermer()
            if type_source == "Simulation":
                cfg = SimulatorConfig(conso_nominale_l_h=conso_nominale)
                st.session_state.source = GeneratorSimulator(cfg, scenario=scenario)
            elif type_source == "Capteurs en direct (MQTT)":
                st.session_state.source = SourceCapteurs(
                    groupe_id=groupe_id, broker=broker, port=int(port_mqtt),
                    puissance_nominale_w=puissance_nominale_kw * 1000.0,
                    conso_nominale_l_h=conso_nominale, heures_initiales=heures_initiales)
            elif fichier_csv is None:
                raise ValueError("Charge d'abord un fichier CSV.")
            else:
                st.session_state.source = SourceCSV(
                    fichier_csv, heures_initiales=heures_initiales, conso_nominale_l_h=conso_nominale
                )
            st.session_state.moteur_prediction = MoteurPrediction(
                SeuilsConfig(conso_nominale_l_h=conso_nominale)
            )
            st.session_state.historique = []
            st.session_state.en_marche = True
        except (ValueError, ImportError) as erreur:
            st.error(str(erreur))
    if col_b.button("⏸️ Arrêter", use_container_width=True):
        st.session_state.en_marche = False

    if st.button("🔄 Réinitialiser", use_container_width=True):
        if hasattr(st.session_state.source, "fermer"):
            st.session_state.source.fermer()
        st.session_state.historique = []
        st.session_state.source = None
        st.session_state.moteur_prediction = None
        st.session_state.en_marche = False

    if type_source == "Simulation":
        st.caption("⚠️ Données simulées, pas des mesures réelles.")
    elif type_source == "Capteurs en direct (MQTT)":
        st.caption("Mesures reçues en direct de l'ESP32 via MQTT.")
    else:
        st.caption("Les mesures viennent de ton fichier CSV.")


if st.session_state.source is None:
    st.info("Choisis une source de données dans la barre latérale puis clique sur **Démarrer**.")
    st.stop()

# ----------------------------------------------------------------------
# Boucle de simulation : un pas par rerun quand "en marche"
# ----------------------------------------------------------------------
if st.session_state.en_marche:
    try:
        mesure = st.session_state.source.step()
    except FinDesDonnees:
        st.session_state.en_marche = False
        st.info("Fin du fichier de mesures.")
    except AttenteMesure as e:
        st.warning(f"⏳ {e}")
    else:
        moteur = st.session_state.moteur_prediction
        diagnostic = moteur.evaluer_risque(mesure)
        derive_carburant = moteur.detecter_derive_carburant(
            mesure["conso_instantanee_l_h"], mesure["charge_pct"]
        )
        duree_vie = moteur.estimer_duree_vie_restante(mesure["heures_cumulees"])
        st.session_state.historique.append({**mesure, **diagnostic, **derive_carburant, **duree_vie})

if not st.session_state.historique:
    st.warning("En attente de la première mesure…")
    st.stop()

df = pd.DataFrame(st.session_state.historique)
dernier = df.iloc[-1]

# ----------------------------------------------------------------------
# 1. Alerte de risque (fonction "prédiction de panne")
# ----------------------------------------------------------------------
niveau = dernier["niveau_risque"]
emoji = COULEUR_RISQUE[niveau]
label = LABEL_RISQUE[niveau]

if niveau == "eleve":
    st.error(f"{emoji} **Risque {label}** — intervention recommandée rapidement")
elif niveau == "modere":
    st.warning(f"{emoji} **Risque {label}** — à surveiller")
else:
    st.success(f"{emoji} **Risque {label}** — fonctionnement normal")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("**Raisons du diagnostic :**")
    for raison in dernier["raisons"]:
        st.markdown(f"- {raison}")
with col2:
    if pd.notna(dernier["temps_restant_estime_min"]):
        heures = dernier["temps_restant_estime_min"] / 60
        st.metric("Temps avant seuil critique estimé", f"≈ {heures:.1f} h")
    else:
        st.metric("Temps avant seuil critique estimé", "Pas de dérive détectée")

st.divider()

# ----------------------------------------------------------------------
# 2. Surveillance temps réel
# ----------------------------------------------------------------------
st.subheader("📡 Surveillance en temps réel")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Température", f"{dernier['temperature_c']:.1f} °C")
m2.metric("Vibration", f"{dernier['vibration_mm_s']:.2f} mm/s")
m3.metric("Charge", f"{dernier['charge_pct']:.0f} %")
m4.metric("Carburant restant", f"{dernier['carburant_restant_l']:.1f} L")

g1, g2 = st.columns(2)
with g1:
    st.line_chart(df.set_index("t_minutes")[["temperature_c"]], height=220)
with g2:
    st.line_chart(df.set_index("t_minutes")[["vibration_mm_s"]], height=220)

st.divider()

# ----------------------------------------------------------------------
# 3. Suivi de consommation de carburant
# ----------------------------------------------------------------------
st.subheader("⛽ Suivi de consommation de carburant")
f1, f2, f3 = st.columns(3)
f1.metric("Consommation instantanée", f"{dernier['conso_instantanee_l_h']:.2f} L/h")
f2.metric("Consommation attendue", f"{dernier['conso_attendue_l_h']:.2f} L/h")
ecart = dernier["ecart_pct"]
f3.metric("Écart", f"{ecart:+.1f} %", delta_color="inverse" if ecart > 0 else "normal")

if dernier["derive_detectee"]:
    st.warning(
        f"⚠️ Dérive de consommation détectée : {ecart:+.1f}% au-dessus de l'attendu — "
        "signe possible d'un mauvais réglage ou d'usure."
    )

st.line_chart(df.set_index("t_minutes")[["conso_instantanee_l_h"]], height=200)

st.divider()

# ----------------------------------------------------------------------
# 4. Estimation de durée de vie restante
# ----------------------------------------------------------------------
st.subheader("🔧 Durée de vie restante estimée")
v1, v2 = st.columns(2)
v1.metric("Heures cumulées de fonctionnement", f"{dernier['heures_cumulees']:.1f} h")
v2.metric("Heures restantes avant révision majeure", f"{dernier['heures_restantes_estimees']:.0f} h")
st.progress(min(1.0, dernier["pourcentage_vie_consommee"] / 100))
st.caption(f"{dernier['pourcentage_vie_consommee']:.1f}% de la durée de vie estimée du moteur consommée")

# ----------------------------------------------------------------------
# Boucle d'actualisation automatique
# ----------------------------------------------------------------------
if st.session_state.en_marche:
    if type_source != "Capteurs en direct (MQTT)":
        time.sleep(1.0 / vitesse)   # en direct, step() attend déjà la mesure suivante
    st.rerun()