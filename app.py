import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from scipy.stats import poisson
import re
import os

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Dashboard Analisi Calcio", layout="wide", page_icon="⚽")

st.title("💎 DASHBOARD V22: Analisi Completa")
st.markdown("""
**Istruzioni:**
1. Carica un file CSV/Excel dal menu a sinistra.
2. Seleziona Paese, Lega e Squadre.
3. Premi 'Avvia Analisi'.
""")

# ==========================================
# 1. FUNZIONI DI UTILITÀ
# ==========================================

@st.cache_data
def load_dataset(file_obj):
    try:
        # 1. Leggiamo la prima riga per capire il separatore
        first_line = file_obj.readline()
        if isinstance(first_line, bytes):
            first_line = first_line.decode('latin1')
        
        # Conta se ci sono più punti e virgola o virgole
        if first_line.count(';') > first_line.count(','):
            sep = ';'
        else:
            sep = ','
            
        # IMPORTANTE: Riportiamo il "cursore" all'inizio del file
        file_obj.seek(0)
        
        # 2. Carichiamo il file con il separatore giusto
        df_raw = pd.read_csv(file_obj, sep=sep, encoding='latin1', on_bad_lines='skip')
        
        # 3. Pulizia Intestazioni (Mette tutto maiuscolo e toglie spazi)
        header = df_raw.columns.str.strip().str.upper().tolist()
        df_raw.columns = header
        
        # 4. Mappatura Colonne (Cerca i sinonimi)
        col_map = {
            'GOALMINH': ['GOALMINH', 'GOALMINCASA', 'MINUTI_CASA', 'MINUTI CASA', 'GOL CASA'],
            'GOALMINA': ['GOALMINA', 'GOALMINOSPITE', 'MINUTI_OSPITE', 'MINUTI OSPITE', 'GOL OSPITE'],
            'LEGA': ['LEGA', 'LEAGUE', 'DIVISION'],
            'PAESE': ['PAESE', 'COUNTRY', 'NAZIONE'],
            'CASA': ['CASA', 'HOME', 'TEAM1', 'SQUADRA CASA'],
            'OSPITE': ['OSPITE', 'AWAY', 'TEAM2', 'SQUADRA OSPITE']
        }
        
        for target, candidates in col_map.items():
            if target not in df_raw.columns:
                for candidate in candidates:
                    if candidate in df_raw.columns:
                        df_raw.rename(columns={candidate: target}, inplace=True)
                        break
        
        # Convertiamo le colonne chiave in testo
        for c in ['PAESE', 'LEGA', 'CASA', 'OSPITE']:
            if c in df_raw.columns:
                df_raw[c] = df_raw[c].astype(str).str.strip()
        
        return df_raw
    except Exception as e:
        st.error(f"Errore caricamento: {e}")
        return pd.DataFrame()
def get_minutes(val):
    # Estrae i numeri da celle sporche tipo "15; 45+2"
    if pd.isna(val): return []
    s = str(val).replace(',', '.').replace(';', ' ')
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    res = []
    for x in nums:
        try:
            n = int(float(x))
            if 0 <= n <= 130: res.append(n)
        except: pass
    return res

def calc_poisson_probs(lam_h, lam_a):
    # Calcola probabilità 1X2 usando Poisson
    probs = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            probs[i][j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
    p1 = np.sum(np.tril(probs, -1))
    px = np.sum(np.diag(probs))
    p2 = np.sum(np.triu(probs, 1))
    return p1, px, p2

# ==========================================
# 2. INTERFACCIA LATERALE
# ==========================================

st.sidebar.header("📂 Carica Dati")
uploaded_file = st.sidebar.file_uploader("Trascina qui il tuo file CSV o Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    global_df = load_dataset(uploaded_file)
    
    if not global_df.empty and 'PAESE' in global_df.columns:
        st.sidebar.success(f"File caricato: {len(global_df)} righe.")
        
        # --- FILTRI ---
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Filtri")
        
        paesi = sorted(global_df['PAESE'].unique())
        sel_paese = st.sidebar.selectbox("1. Paese", paesi)
        
        leghe = sorted(global_df[global_df['PAESE'] == sel_paese]['LEGA'].unique())
        sel_lega = st.sidebar.selectbox("2. Lega", leghe)
        
        mask = (global_df['PAESE'] == sel_paese) & (global_df['LEGA'] == sel_lega)
        teams = sorted(pd.concat([global_df[mask]['CASA'], global_df[mask]['OSPITE']]).unique())
        
        col1, col2 = st.sidebar.columns(2)
        sel_home = col1.selectbox("Casa", teams, index=0 if len(teams)>0 else None)
        sel_away = col2.selectbox("Ospite", teams, index=1 if len(teams)>1 else 0)
        
        run_analysis = st.sidebar.button("📊 AVVIA ANALISI", type="primary")
        
        # ==========================================
        # 3. ANALISI E OUTPUT
        # ==========================================
        if run_analysis:
            st.divider()
            st.subheader(f"Analisi: {sel_home} vs {sel_away}")
            
            # Filtra i dati della lega specifica
            df_league = global_df[(global_df['PAESE'] == sel_paese) & (global_df['LEGA'] == sel_lega)].copy()
            intervals = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
            
            # Recupera colonne gol minuti
            c_h = 'GOALMINH' if 'GOALMINH' in df_league.columns else 'GOALMINCASA'
            c_a = 'GOALMINA' if 'GOALMINA' in df_league.columns else 'GOALMINOSPITE'

            # Variabili accumulo dati
            goals_h = {'FT': 0, 'HT': 0, '2T': 0, 'S_FT': 0, 'S_HT': 0, 'S_2T': 0}
            goals_a = {'FT': 0, 'HT': 0, '2T': 0, 'S_FT': 0, 'S_HT': 0, 'S_2T': 0}
            match_h, match_a = 0, 0
            times_h, times_a, times_league = [], [], []
            
            # Inizializza statistiche intervalli
            stats_match = {
                sel_home: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
                sel_away: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
            }

            # --- CICLO ANALISI ---
            for _, row in df_league.iterrows():
                h, a_team = row['CASA'], row['OSPITE']
                min_h = get_minutes(row.get(c_h))
                min_a = get_minutes(row.get(c_a))
                
                # Raccolta tempi lega per confronto
                if min_h: times_league.append(min(min_h))
                if min_a: times_league.append(min(min_a))

                # --- POPOLA HEATMAP ---
                # Casa Gol Fatti / Subiti
                if h in stats_match:
                    for m in min_h: stats_match[h]['F'][intervals[min(5, (m-1)//15)]] += 1
                    for m in min_a: stats_match[h]['S'][intervals[min(5, (m-1)//15)]] += 1
                # Ospite Gol Fatti / Subiti
                if a_team in stats_match:
                    for m in min_a: stats_match[a_team]['F'][intervals[min(5, (m-1)//15)]] += 1
                    for m in min_h: stats_match[a_team]['S'][intervals[min(5, (m-1)//15)]] += 1

                # --- STATISTICHE CASA ---
                if h == sel_home:
                    match_h += 1
                    goals_h['FT'] += len(min_h)
                    goals_h['HT'] += len([x for x in min_h if x <= 45])
                    goals_h['2T'] += len([x for x in min_h if x > 45])
                    goals_h['S_FT'] += len(min_a)
                    goals_h['S_HT'] += len([x for x in min_a if x <= 45])
                    goals_h['S_2T'] += len([x for x in min_a if x > 45])
                    if min_h: times_h.append(min(min_h))
                
                # --- STATISTICHE OSPITE ---
                if a_team == sel_away:
                    match_a += 1
                    goals_a['FT'] += len(min_a)
                    goals_a['HT'] += len([x for x in min_a if x <= 45])
                    goals_a['2T'] += len([x for x in min_a if x > 45])
                    goals_a['S_FT'] += len(min_h)
                    goals_a['S_HT'] += len([x for x in min_h if x <= 45])
                    goals_a['S_2T'] += len([x for x in min_h if x > 45])
                    if min_a: times_a.append(min(min_a))

            # --- VISUALIZZAZIONE RISULTATI ---
            
            # 1. INFO PARTITE
            c1, c2 = st.columns(2)
            c1.info(f"Partite Casa analizzate: {match_h}")
            c2.info(f"Partite Ospite analizzate: {match_a}")

            # 2. TABELLA MEDIE
            st.subheader("📊 Medie Gol")
            def safe_div(n, d): return n / d if d else 0
            
            avg_h = {k: safe_div(goals_h[k], match_h) for k in goals_h}
            avg_a = {k: safe_div(goals_a[k], match_a) for k in goals_a}
            
            df_medie = pd.DataFrame({
                "Squadra": [sel_home, sel_away],
                "1° Tempo (F/S)": [f"{avg_h['HT']:.2f} / {avg_h['S_HT']:.2f}", f"{avg_a['HT']:.2f} / {avg_a['S_HT']:.2f}"],
                "2° Tempo (F/S)": [f"{avg_h['2T']:.2f} / {avg_h['S_2T']:.2f}", f"{avg_a['2T']:.2f} / {avg_a['S_2T']:.2f}"],
                "Finale (F/S)": [f"{avg_h['FT']:.2f} / {avg_h['S_FT']:.2f}", f"{avg_a['FT']:.2f} / {avg_a['S_FT']:.2f}"]
            })
            st.table(df_medie)

            # 3. PREVISIONI POISSON
            st.subheader("🎲 Previsioni Statistiche (Poisson)")
            exp_h = (avg_h['FT'] + avg_a['S_FT']) / 2
            exp_a = (avg_a['FT'] + avg_h['S_FT']) / 2
            p1, px, p2 = calc_poisson_probs(exp_h, exp_a)
            
            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.metric("Probabilità 1", f"{p1*100:.1f}%")
            col_p2.metric("Probabilità X", f"{px*100:.1f}%")
            col_p3.metric("Probabilità 2", f"{p2*100:.1f}%")

            # 4. GRAFICI HEATMAP
            st.subheader("🔥 Zone Gol (Minuti)")
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            
            # Attacco
            df_f = pd.DataFrame([stats_match[sel_home]['F'], stats_match[sel_away]['F']], index=[sel_home, sel_away])
            sns.heatmap(df_f, annot=True, cmap="Greens", fmt="d", ax=ax[0])
            ax[0].set_title("GOL FATTI")
            
            # Difesa
            df_s = pd.DataFrame([stats_match[sel_home]['S'], stats_match[sel_away]['S']], index=[sel_home, sel_away])
            sns.heatmap(df_s, annot=True, cmap="Reds", fmt="d", ax=ax[1])
            ax[1].set_title("GOL SUBITI")
            
            st.pyplot(fig)
            
            # 5. KAPLAN-MEIER
            st.subheader("⏱️ Tempo al 1° Gol")
            if times_h and times_a:
                fig_km, ax_km = plt.subplots(figsize=(10, 5))
                kmf = KaplanMeierFitter()
                
                kmf.fit(times_h, label=sel_home)
                kmf.plot_survival_function(ax=ax_km, ci_show=False, linewidth=3)
                
                kmf.fit(times_a, label=sel_away)
                kmf.plot_survival_function(ax=ax_km, ci_show=False, linewidth=3)
                
                st.pyplot(fig_km)
                st.caption("Il grafico mostra quanto a lungo le squadre resistono sullo 0-0.")
    else:
        st.error("Il file caricato non contiene la colonna 'PAESE'. Controlla il file.")
else:
    st.info("👈 Carica un file CSV per iniziare.")