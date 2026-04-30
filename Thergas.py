
#
# Code d'estimation des Propriétés Thermodynamiques par la méthode de Benson
#
# THERGAS - En version gaz
#
# Roda - Version du 28 avril 2026
#

# Correction sur les polycyles, les métagroupes
# Ajout des conversions SMILES -> NLF et NLF-> SMILES
#

#!/usr/bin/env python3
"""
Code THERGAS
=================================================================
Roda - LRGP - Avril 2026
"""

#
# Bibliothèques python standard
#
import streamlit as st
from streamlit_ketcher import st_ketcher

import pandas as pd
import numpy as np

#
# Graphes
#
import matplotlib
import matplotlib.pyplot as plt #importer la lib graphique 
from matplotlib.pyplot import plot # tracage de courbe / evite d'ecrire plt.plot()
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#
# Bibliothèques python pour conversion
#
import re
from rdkit import Chem
from rdkit.Chem import RWMol, Atom, BondType
from rdkit.Chem import Draw, AllChem
import sys
from io import BytesIO
from collections import deque

import subprocess
import os

from Fonctions_SMILES import * # Importation des fonctions pour la conversion smiles <->nlf
#from Fonctions_DIPPR import * # Importation des fonctions pour fitting DIPPR
#from Fonctions_Polynome_Nasa import * # Importation des fonctions pour le polynome Nasa

# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Configuration de la page
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="THERGAS",
    page_icon="🧪",
    #layout="wide",
)

# ── Style CSS personnalisé ──────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-top: 0;
    }
    .result-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1E88E5;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        word-break: break-all;
        margin: 10px 0;
    }
    .nlf-box {
        border-left-color: #7B1FA2;
    }
    .smiles-box {
        border-left-color: #2E7D32;
    }
    .error-box {
        background-color: #fce4ec;
        border-left: 4px solid #c62828;
        padding: 15px;
        border-radius: 0 8px 8px 0;
    }
    .legend-title {
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Introduction
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

st.image('B2.png',width=850)

st.markdown('<h1 class="main-title">THERGAS</h1>', unsafe_allow_html=True)
st.subheader('A computer program for the evaluation of thermochemical data of molecules and free radicals in the gas phase')
st.subheader('The calculations are based on the methods developed by S.W. Benson: bond and group additivity')

st.markdown("***LRGP - Université de Lorraine, CNRS, LRGP, F-54000 Nancy, France***")
st.write("----------------------------------------------------------")
st.write("")
st.markdown('<a href="mailto:roda.bounaceur@univ-lorraine.fr"> If you have any problems please Contact us !</a>', unsafe_allow_html=True)

st.markdown("<hr style='height: 2px; background-color: #333;'>", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Début du code Streamlit de conversion
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

#
# Partie Ketcher
#
SMILES_Molecules = 'C1C=CC=C(OC)C=1' # Molécule par défaut

DEFAULT_MOL = SMILES_Molecules
molecule = st.text_input("Insert the SMILE notation of the molecule or draw it", DEFAULT_MOL)
SMILES_Molecules = st_ketcher(molecule)
st.markdown(f"Smile notation: ``{SMILES_Molecules}``")
st.write('')
#
# Partie Conversion
#

#molecule = 'CCCCC'
#molecule = r"C1C(CC([O-])=O)C(C/C=C\CCOS([O-])(=O)=O)C(=O)C1"

nlf = smiles_to_nlf(SMILES_Molecules)

nlf = re.sub(r"o/s\(//o\)\(o\)//o", "'so4'", nlf)
nlf = re.sub(r"s\(//o\)\(oh\)//o", "'so3h'", nlf)
nlf = re.sub(r"s\(//o\)\(o\)//o",  "'so3'", nlf)
nlf = re.sub(r"s\(//o\)\(//o\)",   "'so2'", nlf)

st.markdown(f"Thergas notation: ``{nlf}``")
st.write('')

# pour l'instant
molecule_thergas = nlf 


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Lancement du code THERGAS - Version 2
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

st.markdown("<hr style='height: 2px; background-color: #333;'>", unsafe_allow_html=True)

# Configuration
MOLECULE_FILE = "molecule.txt"
RESULT_FILE = "Results_Thergas.txt"

st.subheader("-- THERGAS --")

resultat_ok = 0

with open(MOLECULE_FILE, "w") as f:
    f.write(molecule_thergas)

#st.success(f"✅ {MOLECULE_FILE} créé dans l'environnement temporaire")

# Afficher le chemin
#st.info(f"Chemin : {os.path.abspath(MOLECULE_FILE)}")

if st.button("➡️ Launch the calculation"):
    if os.path.exists(MOLECULE_FILE):
        with st.spinner("Calculating..."):
            try:
                # Vérifier que l'exécutable existe
                if not os.path.exists("./thergaslinux"):
                    st.error("The thergaslinux executable cannot be found")
                else:
                    # Rendre exécutable
                    os.chmod("./thergaslinux", 0o755)
                    
                    # Exécuter
                    result = subprocess.run(
                        ["./thergaslinux"],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        st.success("Calculation complete !")
                        resultat_ok = 1
                        # Lire les résultats
                        #if os.path.exists(RESULT_FILE):
                        #    with open(RESULT_FILE, "r") as f:
                        #        st.download_button(
                        #            "📥 Télécharger résultats",
                        #            f.read(),
                        #            RESULT_FILE
                        #        )
                    else:
                        st.error(f"Error in the calculation : {result.stderr}")
                        
            except Exception as e:
                st.error(f"Exception : {e}")
    else:
        st.error(f"{MOLECULE_FILE} does not exist")

# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Traitement du fichier résultat avec affichage + courbes
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

# 1. Affichage des résultats
# Après l'exécution, dans la même section

if resultat_ok == 1:
    #st.success("✅ Calcul terminé !")
    
    if os.path.exists(RESULT_FILE):
        # Lire le fichier
        with open(RESULT_FILE, "r") as f:
            contenu = f.read()
        
        # Section des résultats
        st.subheader("📊 Résultats")
        
        # Option 1: Affichage dans un expander
        with st.expander("View detailed results", expanded=True):
            st.code(contenu, language="text")
        
        # Option 2: Téléchargement
        st.download_button(
            "📥 Download Results",
            contenu,
            RESULT_FILE,
            key="download"
        )
        
        # Option 3: Aperçu des premières lignes
        #lines = contenu.split('\n')
        #st.write(f"**Aperçu (premières lignes) :**")
        #for i, line in enumerate(lines[:10]):
        #    st.text(line)
        #if len(lines) > 10:
        #    st.info(f"... et {len(lines) - 10} lignes supplémentaires")


# 2. Graphs des résultats
R = 1.987  # cal/(mol.K)

# Lecture des valeurs
st.subheader("📈 Dimensionless thermodynamic properties")

# Lecture directe du fichier
with open(RESULT_FILE, "r") as f:
    lignes = f.readlines()

# Extraction des données
temperatures = []
cp_values = []
h_values = []
s_values = []

lecture_tableau = False
for ligne in lignes:
    if 'BENSON method' in ligne:
        lecture_tableau = True
        continue
    if lecture_tableau and ligne.strip() and not ligne.startswith('-'):
        parts = ligne.split()
        if len(parts) >= 4 and parts[0].replace('.', '').isdigit():
            try:
                T = float(parts[0])
                Cp = float(parts[1])
                H = float(parts[2])
                S = float(parts[3])
                
                temperatures.append(T)
                cp_values.append(Cp / R)
                h_values.append((H * 1000) / (R * T))
                s_values.append(S / R)
                
            except:
                pass

if not temperatures:
    st.error("Error: It is impossible to perform the calculation using Benson's method")
else:
    # Tracer
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Cp/R
    axes[0].plot(temperatures, cp_values, 'b-', linewidth=2, label='Benson Cp/R')
    #axes[0].axvline(Tmed, color='r', linestyle=':', label='Tmed', linewidth=5)
    axes[0].set_ylabel('Cp/R')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title(f'Thermodynamic properties versus Temperature')
    
    # H/(R·T)
    axes[1].plot(temperatures, h_values, 'g-', linewidth=2, label='Benson H/(R·T)')
    #axes[1].axvline(Tmed, color='r', linestyle=':', linewidth=5)
    axes[1].set_ylabel('H/(R·T)')
    axes[1].legend()
    axes[1].grid(True)
    
    # S/R
    axes[2].plot(temperatures, s_values, 'm-', linewidth=2, label='Benson S/R')
    #axes[2].axvline(Tmed, color='r', linestyle=':', linewidth=5)
    axes[2].set_ylabel('S/R')
    axes[2].set_xlabel('Temperature (K)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.xlim(200, 1500)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Polynomes NASA
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
