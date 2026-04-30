import streamlit as st

import matplotlib
import matplotlib as plt
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from scipy.optimize import curve_fit
from rdkit.Chem.rdchem import Mol
from typing import Union, Dict
import numpy.typing as npt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def cp_fit(temperatures: Union[npt.NDArray[np.float64], float],
           a1: float,
           a2: float,
           a3: float,
           a4: float,
           a5: float) -> Union[npt.NDArray[np.float64], float]:
    return a1 + a2 * temperatures + a3 * temperatures ** 2 + a4 * temperatures ** 3 + a5 * temperatures ** 4


def enthalpy_fit(temperatures: Union[npt.NDArray[np.float64], float],
                 a1: float,
                 a2: float,
                 a3: float,
                 a4: float,
                 a5: float,
                 a6: float) -> Union[npt.NDArray[np.float64], float]:
    #return a1 * temperatures + (a2 / 2) * temperatures ** 2 + (a3 / 3) * temperatures ** 3 + \
    #    (a4 / 4) * temperatures ** 4 + (a5 / 5) * temperatures ** 5 + a6

    return a1 + (a2 / 2) * temperatures ** 1 + (a3 / 3) * temperatures ** 2 + \
        (a4 / 4) * temperatures ** 3 + (a5 / 5) * temperatures ** 4 + a6 / temperatures 


def entropy_fit(temperatures: Union[npt.NDArray[np.float64], float],
                a1: float,
                a2: float,
                a3: float,
                a4: float,
                a5: float,
                a7: float) -> Union[npt.NDArray[np.float64], float]:
    #return a1 * np.log(temperatures) + a2 * temperatures + (a3 / 2) * temperatures ** 2 + \
    #    (a4 / 3) * temperatures ** 4 + (a5 / 5) * temperatures ** 5 + a7

    return a1 * np.log(temperatures) + a2 * temperatures + (a3 / 2) * temperatures ** 2 + \
        (a4 / 3) * temperatures ** 3 + (a5 / 4) * temperatures ** 4 + a7


def get_cp_coefficients(temperatures: Union[float, npt.NDArray[np.float64]],
                        cp_values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if len(cp_values.shape) > 1:
        coefficients = []
        for i in range(cp_values.shape[0]):
            popt, _ = curve_fit(cp_fit, temperatures, cp_values[i])
            coefficients.append(popt)
        coefficients_array = np.array(coefficients).T
    else:
        popt, _ = curve_fit(cp_fit, temperatures, cp_values)
        coefficients_array = np.array(popt)

    return coefficients_array


def get_nasa_coefficients(temp_debut, temperatures: Union[npt.NDArray[np.float64], float],
                          h298: Union[npt.NDArray[np.float64]],
                          s298: Union[npt.NDArray[np.float64]],
                          cp_values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    
    # Version originale avec pris en ecompte des unités de VanGeel
    #a1, a2, a3, a4, a5 = get_cp_coefficients(temperatures, cp_values * (4.184 / 8.314))
    #a6 = h298 * (4.184 / 8.314) * 1000 - enthalpy_fit(298.15, a1, a2, a3, a4, a5, 0)
    #a7 = s298 * (4.184 / 8.314) - entropy_fit(298.15, a1, a2, a3, a4, a5, 0)   
    
    # Roda - moi je tiens compte directement de Cp/R, H/RT et S/R
    a1, a2, a3, a4, a5 = get_cp_coefficients(temperatures, cp_values )
    a6 = temp_debut * ( h298 - enthalpy_fit(temp_debut, a1, a2, a3, a4, a5, 0) ) 
    a7 = s298 - entropy_fit(temp_debut, a1, a2, a3, a4, a5, 0)
    
    return np.array([a1, a2, a3, a4, a5, a6, a7]).T



def get_chemkin_file(name: str,
                     smiles: str,
                     method: str,
                     mol: Mol,
                     nasa_coefficients: npt.NDArray[np.float64]) -> str:
    atom_dict = {}  # type: Dict[str, int]
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in atom_dict:
            atom_dict[sym] += 1
        else:
            atom_dict[sym] = 1
    if len(atom_dict) > 4:
        raise ValueError("Chemkin format is only valid for molecules with up to 4 atom types.")
    else:
        num_types = len(atom_dict)

    chemkin = f"!\n! Filename: {name}\n! Smiles: {smiles}\n! method: {method}\n"
    name = 'Compound'                            
    chemkin += f"{name.upper(): <18}"
    chemkin += "      "
    for atom_type in atom_dict:
        chemkin += f"{atom_type: <2}{atom_dict[atom_type]: <3}"
    
    for i in range(4 - num_types):
        chemkin += "     "
    chemkin += "G   200.00    5000.00   1500.00    1\n"
    
    for i in range(5):
        formatted_coefficient = format_array(np.array(nasa_coefficients[i]))
        chemkin += formatted_coefficient
    chemkin += "    2\n"
    formatted_coefficient = format_array(np.array(nasa_coefficients[5]))
    chemkin += formatted_coefficient
    formatted_coefficient = format_array(np.array(nasa_coefficients[6]))
    chemkin += formatted_coefficient
    
    formatted_coefficient = format_array(np.array(nasa_coefficients[7]))
    chemkin += formatted_coefficient
    formatted_coefficient = format_array(np.array(nasa_coefficients[8]))
    chemkin += formatted_coefficient
    formatted_coefficient = format_array(np.array(nasa_coefficients[9]))
    chemkin += formatted_coefficient
    
    #for i in range(3):
    #    formatted_coefficient = format_array(np.array(nasa_coefficients[i]))
    #    chemkin += formatted_coefficient
    chemkin += "    3\n"
    
    for i in range(10, len(nasa_coefficients)):
        formatted_coefficient = format_array(np.array(nasa_coefficients[i]))
        chemkin += formatted_coefficient
    chemkin += "                   4\n"
    
    return chemkin

def format_array(arr: npt.NDArray[np.float64], decimals: int = 8) -> str:
    formatted_arr = np.array2string(arr, formatter={'float_kind': lambda x: ("{: ." + str(decimals) + "e}").format(x)})
    return formatted_arr.replace('\n', ', ')

#
# Nouveau Graph des Polynomes Nasa
#
def plot_nasa_validation_New(smile, coeffs, Tmin, Tmed, Tmax):
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14 = coeffs
    
    # Générer une grille fine
    T = np.linspace(Tmin, Tmax, 10)
    
    # 50 points de 10 à 1500 (1500 non inclus par défaut, donc on ajoute endpoint=True)
    part1 = np.linspace(Tmin, Tmed, 100, endpoint=True)
    #
    # 50 points de 1500 à 5000 (5000 inclus)
    part2 = np.linspace(Tmed, Tmax, 100, endpoint=True)
    #
    # Concatenation des trois parties
    T = np.concatenate((part1, part2))
    #

    # Calculer Cp/R, H/(RT), S/R avec les polynômes NASA
    Cp_R_fit = np.zeros_like(T)
    H_RT_fit = np.zeros_like(T)
    S_R_fit = np.zeros_like(T)
    
    for i, t in enumerate(T):
        if t >= Tmed:
            Cp_R_fit[i] = a1 + a2*t + a3*t**2 + a4*t**3 + a5*t**4
            #print(t,Cp_R_fit[i])
            H_RT_fit[i] = a1 + a2/2*t + a3/3*t**2 + a4/4*t**3 + a5/5*t**4 + a6/t
            S_R_fit[i] = a1*np.log(t) + a2*t + a3/2*t**2 + a4/3*t**3 + a5/4*t**4 + a7
        else:
            Cp_R_fit[i] = a8 + a9*t + a10*t**2 + a11*t**3 + a12*t**4
            #print(t,Cp_R_fit[i])
            H_RT_fit[i] = a8 + a9/2*t + a10/3*t**2 + a11/4*t**3 + a12/5*t**4 + a13/t
            S_R_fit[i] = a8*np.log(t) + a9*t + a10/2*t**2 + a11/3*t**3 + a12/4*t**4 + a14
    
    # Tracer
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Cp/R
    axes[0].plot(T, Cp_R_fit, 'b-', linewidth=2, label='NASA Cp/R')
    axes[0].axvline(Tmed, color='r', linestyle=':', label='Tmed', linewidth=5)
    axes[0].set_ylabel('Cp/R')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title(f'Validation NASA vs IA - {smile}')
    
    # H/(R·T)
    axes[1].plot(T, H_RT_fit, 'g-', linewidth=2, label='NASA H/(R·T)')
    axes[1].axvline(Tmed, color='r', linestyle=':', linewidth=5)
    axes[1].set_ylabel('H/(R·T)')
    axes[1].legend()
    axes[1].grid(True)
    
    # S/R
    axes[2].plot(T, S_R_fit, 'm-', linewidth=2, label='NASA S/R')
    axes[2].axvline(Tmed, color='r', linestyle=':', linewidth=5)
    axes[2].set_ylabel('S/R')
    axes[2].set_xlabel('Temperature (K)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)




###############################
###############################
def format_nasa_chemkin_line(smile, coeffs, Tmin=290, Tmax=5000, Tmed=1500):
    """
    Formate les 14 coefficients NASA dans le style Chemkin Therm.dat,
    avec alignement strict et numéros de ligne en colonne 79-80.
    """
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14 = coeffs

    # Inversion !!

    # Ligne 1 : SMILES (16) + 'G' (1) + Tmin (8) + Tmax (8) + Tmed (8) + '1' (position 79-80)
    #smile = smile + smile # pour le formatage je rajout des blancs
    smile_part = smile[:16].ljust(16)
    smile_part = 'Name_of_the_molecule___________________________'
    line1 = f"{smile_part} G {Tmin:8.2f} {Tmax:8.2f} {Tmed:8.2f}".ljust(78) + " 1"
    
    # Fonction pour formater UN coefficient : 15 caractères, notation scientifique, aligné à gauche dans le champ
    def fmt(x):
        s = f"{x:15.8E}"  # Ex: ' 1.23456789E+01' → 15 caractères
        return s  # Déjà 15 caractères, aligné à gauche dans le champ de 15

    # Ligne 2 : a1 à a5 (5 × 15 = 75 caractères) + 3 espaces + "2" en colonne 79-80
    #part2 = fmt(a1) + fmt(a2) + fmt(a3) + fmt(a4) + fmt(a5)
    part2 = fmt(a8) + fmt(a9) + fmt(a10) + fmt(a11) + fmt(a12)
    line2 = part2.ljust(78) + " 2"

    # Ligne 3 : a6 à a10
    #part3 = fmt(a6) + fmt(a7) + fmt(a8) + fmt(a9) + fmt(a10)
    part3 = fmt(a13) + fmt(a14) + fmt(a1) + fmt(a1) + fmt(a3)
    line3 = part3.ljust(78) + " 3"

    # Ligne 4 : a11 à a14 (4 × 15 = 60) + 18 espaces + "4"
    #part4 = fmt(a11) + fmt(a12) + fmt(a13) + fmt(a14)
    part4 = fmt(a4) + fmt(a5) + fmt(a6) + fmt(a7)
    #line4 = part4.ljust(78)  + " 4"
    line4 = part4.ljust(78) + "_____________" + " 4"
    return "\n".join([line1, line2, line3, line4]),line1, line2, line3, line4



def plot_nasa_validation(smile, coeffs, Tmin=290, Tmed=1500, Tmax=5000):
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14 = coeffs
    
    T = np.linspace(Tmin, Tmax, 500)
    Cp_R = np.zeros_like(T)
    H_RT = np.zeros_like(T)
    S_R = np.zeros_like(T)
    
    for i, t in enumerate(T):
        if t <= Tmed:
            Cp_R[i] = a1 + a2*t + a3*t**2 + a4*t**3 + a5*t**4
            H_RT[i] = a1 + a2/2*t + a3/3*t**2 + a4/4*t**3 + a5/5*t**4 + a6/t
            S_R[i] = a1*np.log(t) + a2*t + a3/2*t**2 + a4/3*t**3 + a5/4*t**4 + a7
        else:
            Cp_R[i] = a8 + a9*t + a10*t**2 + a11*t**3 + a12*t**4
            H_RT[i] = a8 + a9/2*t + a10/3*t**2 + a11/4*t**3 + a12/5*t**4 + a13/t
            S_R[i] = a8*np.log(t) + a9*t + a10/2*t**2 + a11/3*t**3 + a12/4*t**4 + a14
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    axes[0].plot(T, Cp_R, 'b-', linewidth=2)
    axes[0].set_ylabel('Cp/R')
    axes[0].grid(True)
    axes[0].set_title(f'Propriétés thermodynamiques ajustées - {smile}')
    
    axes[1].plot(T, H_RT, 'r-', linewidth=2)
    axes[1].set_ylabel('H/(R·T)')
    axes[1].grid(True)
    
    axes[2].plot(T, S_R, 'g-', linewidth=2)
    axes[2].set_ylabel('S/R')
    axes[2].set_xlabel('Température (K)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    st.pyplot(plt)


def fit_nasa_14coeff_complete_from_model(smile, modele, D3, noms_colonnes_retenues,
                                         Tmin=290, Tmed=1500, Tmax=5000, n_points=100):
    """
    Version CORRIGÉE — calcule H(T) et S(T) par intégration numérique
    et ajuste a6, a7, a13, a14 pour que les formules NASA coïncident
    avec les intégrales à T = Tmed (pour assurer la continuité).
    """
    # 1. Générer une grille de températures fines
    T_grid = np.linspace(Tmin, Tmax, n_points)
    
    # 2. Créer un DataFrame temporaire pour prédire Cp(T)
    df_sample = D3[D3['SMILES'] == smile].iloc[0:1].copy()
    df_pred = pd.DataFrame({'T(K)': T_grid})
    
    for col in noms_colonnes_retenues:
        if col != 'T(K)' and col in df_sample.columns:
            df_pred[col] = df_sample[col].values[0]
    
    # 3. CORRECTION : Récupérer CpMax et calculer Cp(T) réel
    CpMax = df_sample['CMax'].values[0]  # ← ADAPTE LE NOM DE COLONNE SI BESOIN
    Cp_ratio = modele.predict(df_pred[noms_colonnes_retenues])  # Cp/CpMax
    Cp_pred = Cp_ratio * CpMax  # Cp en J/mol·K
    Cp_R = Cp_pred / 8.31446261815324  # Cp/R → MAINTENANT CORRECT ✅
    
    # 4. Séparer les données
    mask_low = (T_grid <= Tmed) & (T_grid >= Tmin)
    mask_high = T_grid > Tmed
    
    T_low = T_grid[mask_low]
    Cp_R_low = Cp_R[mask_low]
    
    T_high = T_grid[mask_high]
    Cp_R_high = Cp_R[mask_high]
    
    # 5. Ajuster les polynômes pour Cp (a1-a5, a8-a12)
    poly = PolynomialFeatures(degree=4, include_bias=False)
    
    X_low = poly.fit_transform(T_low.reshape(-1, 1))
    model_low = LinearRegression(fit_intercept=True)
    model_low.fit(X_low, Cp_R_low)
    a1 = model_low.intercept_
    a2, a3, a4, a5 = model_low.coef_
    
    X_high = poly.fit_transform(T_high.reshape(-1, 1))
    model_high = LinearRegression(fit_intercept=True)
    model_high.fit(X_high, Cp_R_high)
    a8 = model_high.intercept_
    a9, a10, a11, a12 = model_high.coef_
    
    # =============================
    # 6. CORRECTION : Calculer H(Tmed)/R et S(Tmed)/R par intégration NUMÉRIQUE depuis Tmin
    # =============================
    
    T_ref = Tmed  # Température de référence = température de jonction
    
    # --- Intégrer H(T_ref)/R = ∫_{Tmin}^{T_ref} (Cp/R) dT ---
    T_int_H = np.linspace(Tmin, T_ref, 1000)
    # Interpoler Cp/R sur cette grille (on utilise les valeurs ajustées, pas les prédictions brutes)
    Cp_R_int_H = np.piecewise(
        T_int_H,
        [T_int_H <= Tmed],
        [
            lambda T: a1 + a2*T + a3*T**2 + a4*T**3 + a5*T**4,  # BT
            lambda T: a8 + a9*T + a10*T**2 + a11*T**3 + a12*T**4 # HT (normalement pas utilisé ici)
        ]
    )
    H_R_at_Tref = np.trapz(Cp_R_int_H, T_int_H)  # H(Tref)/R (en partant de H(Tmin)=0)
    
    # --- Intégrer S(T_ref)/R = ∫_{Tmin}^{T_ref} (Cp/R)/T dT ---
    S_R_at_Tref = np.trapz(Cp_R_int_H / T_int_H, T_int_H)
    
    # =============================
    # 7. Ajuster a6 et a7 pour que la formule BT donne EXACTEMENT H_R_at_Tref et S_R_at_Tref à T_ref
    # =============================
    
    # Formule NASA BT pour H/(R*T) à T_ref :
    # H/(R*T_ref) = a1 + a2/2*T_ref + a3/3*T_ref^2 + a4/4*T_ref^3 + a5/5*T_ref^4 + a6/T_ref
    # → Donc :
    poly_H_low = a1 + a2/2*T_ref + a3/3*T_ref**2 + a4/4*T_ref**3 + a5/5*T_ref**4
    a6 = T_ref * (H_R_at_Tref / T_ref - poly_H_low)  # H_R_at_Tref / T_ref = H/(R*T_ref)
    
    # Formule NASA BT pour S/R à T_ref :
    # S/R = a1*ln(T_ref) + a2*T_ref + a3/2*T_ref^2 + a4/3*T_ref^3 + a5/4*T_ref^4 + a7
    poly_S_low = a1*np.log(T_ref) + a2*T_ref + a3/2*T_ref**2 + a4/3*T_ref**3 + a5/4*T_ref**4
    a7 = S_R_at_Tref - poly_S_low
    
    # =============================
    # 8. Ajuster a13 et a14 pour que la formule HT donne EXACTEMENT les mêmes H et S à T_ref
    # =============================
    
    # Formule NASA HT pour H/(R*T_ref) :
    poly_H_high = a8 + a9/2*T_ref + a10/3*T_ref**2 + a11/4*T_ref**3 + a12/5*T_ref**4
    a13 = T_ref * (H_R_at_Tref / T_ref - poly_H_high)
    
    # Formule NASA HT pour S/R à T_ref :
    poly_S_high = a8*np.log(T_ref) + a9*T_ref + a10/2*T_ref**2 + a11/3*T_ref**3 + a12/4*T_ref**4
    a14 = S_R_at_Tref - poly_S_high
    
    # 9. Assembler les 14 coefficients
    coeffs = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14]
    # Inversion des coefficients car on calcul d'abord les BT !!
    coeffsBis = [ a8, a9, a10, a11, a12, a13, a14 , a1, a2, a3, a4, a5, a6, a7 ]
                                             
    st.write(f"✅ NASA Coefficients for {smile}:")
    labels = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14']
    for lab, val in zip(labels, coeffsBis):
        st.write(f"   {lab:3s} = {val:12.6e}")
    
    return coeffs


def plot_nasa_validation_corrected(smile, coeffs, Tmin=290, Tmed=1500, Tmax=5000, modele=None, D3=None, noms_colonnes_retenues=None):
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14 = coeffs
    
    # Générer une grille fine
    T = np.linspace(Tmin, Tmax, 1000)
    
    # Calculer Cp/R, H/(RT), S/R avec les polynômes NASA
    Cp_R_fit = np.zeros_like(T)
    H_RT_fit = np.zeros_like(T)
    S_R_fit = np.zeros_like(T)
    
    for i, t in enumerate(T):
        if t <= Tmed:
            Cp_R_fit[i] = a1 + a2*t + a3*t**2 + a4*t**3 + a5*t**4
            H_RT_fit[i] = a1 + a2/2*t + a3/3*t**2 + a4/4*t**3 + a5/5*t**4 + a6/t
            S_R_fit[i] = a1*np.log(t) + a2*t + a3/2*t**2 + a4/3*t**3 + a5/4*t**4 + a7
        else:
            Cp_R_fit[i] = a8 + a9*t + a10*t**2 + a11*t**3 + a12*t**4
            H_RT_fit[i] = a8 + a9/2*t + a10/3*t**2 + a11/4*t**3 + a12/5*t**4 + a13/t
            S_R_fit[i] = a8*np.log(t) + a9*t + a10/2*t**2 + a11/3*t**3 + a12/4*t**4 + a14
    
    # Comparer avec le modèle IA (optionnel mais recommandé)
    if modele is not None and D3 is not None:
        df_sample = D3[D3['SMILES'] == smile].iloc[0:1].copy()
        df_pred = pd.DataFrame({'T(K)': T})
        for col in noms_colonnes_retenues:
            if col != 'T(K)' and col in df_sample.columns:
                df_pred[col] = df_sample[col].values[0]
        Cp_pred = modele.predict(df_pred[noms_colonnes_retenues])
        Cp_R_ia = Cp_pred / 8.31446261815324
    else:
        Cp_R_ia = None
    
    # Tracer
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Cp/R
    axes[0].plot(T, Cp_R_fit, 'b-', linewidth=2, label='NASA Cp/R')
    #if Cp_R_ia is not None:
    #    axes[0].plot(T, Cp_R_ia, 'r--', linewidth=2, label='Modèle IA', alpha=0.7)
    axes[0].axvline(Tmed, color='r', linestyle=':', label='Tmed')
    axes[0].set_ylabel('Cp/R')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title(f'Validation NASA vs IA - {smile}')
    
    # H/(R·T)
    axes[1].plot(T, H_RT_fit, 'g-', linewidth=2, label='NASA H/(R·T)')
    axes[1].axvline(Tmed, color='k', linestyle=':')
    axes[1].set_ylabel('H/(R·T)')
    axes[1].legend()
    axes[1].grid(True)
    
    # S/R
    axes[2].plot(T, S_R_fit, 'm-', linewidth=2, label='NASA S/R')
    axes[2].axvline(Tmed, color='k', linestyle=':')
    axes[2].set_ylabel('S/R')
    axes[2].set_xlabel('Température (K)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)



