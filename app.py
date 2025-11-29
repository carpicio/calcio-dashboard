import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from scipy.stats import poisson
import warnings
import re
import os

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="⚽ Dashboard Pro V38", layout="wide", page_icon="⚽")
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- TITOLO ---
st.title("💎 Dashboard Analisi Calcio V38")
st.markdown("**Statistiche Temporali, Poisson HT/FT & Analisi Ritmo**")
st.divider()

# ==========================================
# 1. CARICAMENTO DATI
# ==========================================
with st.sidebar:
    st.header("📂 Dati")
    uploaded_file = st.file_uploader("Carica file (CSV/Excel)", type=['csv', 'xlsx'])
    
    # File default (se presente nel repo)
    default_file = 'eng_tot_1.csv'
    file_to_use = uploaded_file if uploaded_file else (default_file if os.path.exists(default_file) else None)

    if file_to_use is None:
        st.info("👈 Carica un file per iniziare.")
        st.stop()

@st.cache_data
def load_data(file_input, is_path=False):
    try:
        # Lettura
        if is_path:
             with open(file_input, 'r', encoding='latin1', errors='replace') as f:
                line = f.readline()
                sep = ';' if line.count(';') > line.count(',') else ','
             df = pd.read_csv(file_input, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False)
        else:
            try:
                line = file_input.readline().decode('latin1')
                file_input.seek(0)
                sep = ';' if line.count(';') > line.count(',') else ','
                df = pd.read_csv(file_input, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False)
            except:
                file_input.seek(0)
                df = pd.read_excel(file_input)

        # Pulizia Colonne
        df.columns = df.columns.astype(str).str.strip().str.upper()
        df = df.loc[:, ~df.columns.duplicated()]

        # Mappatura Universale
        col_map = {
            'GOALMINH': ['GOALMINH', 'GOALMINCASA', 'MINUTI_CASA', 'GOALSH'],
            'GOALMINA': ['GOALMINA', 'GOALMINOSPITE', 'MINUTI_OSPITE', 'GOALSA'],
            'LEGA': ['LEGA', 'LEAGUE', 'DIVISION'],
            'PAESE': ['PAESE', 'COUNTRY'],
            'CASA': ['CASA', 'HOME', 'TXTECHIPA1'],
            'OSPITE': ['OSPITE', 'AWAY', 'TXTECHIPA2']
        }
        
        for target, candidates in col_map.items():
            if target not in df.columns:
                for candidate in candidates:
                    if candidate in df.columns:
                        df.rename(columns={candidate: target}, inplace=True)
                        break
        
        # Pulizia Dati
        for c in ['PAESE', 'LEGA', 'CASA', 'OSPITE']:
            if c in df.columns: df[c] = df[c].astype(str).str.strip()

        # ID Univoco
        if 'PAESE' in df.columns:
            df['ID_LEGA'] = df['PAESE'] + " - " + df['LEGA']
        else:
            df['ID_LEGA'] = df['LEGA']
            
        return df
    except Exception as e:
        st.error(f"Errore file: {e}")
        return pd.DataFrame()

df = load_data(file_to_use, isinstance(file_to_use, str))
if df.empty: st.stop()
st.sidebar.success(f"✅ {len(df)} righe caricate")

# ==========================================
# 2. SELEZIONE MATCH
# ==========================================
col1, col2, col3 = st.columns(3)
leghe = sorted(df['ID_LEGA'].unique())
with col1: sel_lega = st.selectbox("🏆 Campionato", leghe)

df_league = df[df['ID_LEGA'] == sel_lega].copy()
teams = sorted(pd.concat([df_league['CASA'], df_league['OSPITE']]).unique())

with col2: sel_home = st.selectbox("🏠 Casa", teams, index=0)
with col3: sel_away = st.selectbox("✈️ Ospite", teams, index=1 if len(teams)>1 else 0)

# ==========================================
# 3. ENGINE ANALISI
# ==========================================
if st.button("🚀 AVVIA ANALISI", type="primary"):
    st.divider()
    
    intervals = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
    
    def get_minutes(val):
        if pd.isna(val): return []
        s = str(val).replace(',', ' ').replace(';', ' ').replace('.', ' ').replace('"', '')
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        res = []
        for x in nums:
            try:
                n = int(float(x))
                if 0 <= n <= 130: res.append(n)
            except: pass
        return res

    c_h = 'GOALMINH' if 'GOALMINH' in df_league.columns else df_league.columns[0]
    c_a = 'GOALMINA' if 'GOALMINA' in df_league.columns else df_league.columns[0]

    # Accumulatori
    goals_h = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    goals_a = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    match_h, match_a = 0, 0
    
    # Liste per Kaplan-Meier
    times_h, times_a, times_league = [], [], []
    first_goals_h, first_goals_a = [], []
    
    # Heatmap Data
    hm_f = {sel_home: {i:0 for i in intervals}, sel_away: {i:0 for i in intervals}}
    hm_s = {sel_home: {i:0 for i in intervals}, sel_away: {i:0 for i in intervals}}

    for _, row in df_league.iterrows():
        h, a = row['CASA'], row['OSPITE']
        min_h = get_minutes(row.get(c_h))
        min_a = get_minutes(row.get(c_a))
        
        # Dati Lega (1° gol)
        if min_h: times_league.append(min(min_h))
        if min_a: times_league.append(min(min_a))
        
        # --- ANALISI SQUADRA CASA ---
        if h == sel_home:
            match_h += 1
            goals_h['FT'] += len(min_h)
            goals_h['HT'] += len([x for x in min_h if x <= 45])
            goals_h['S_FT'] += len(min_a)
            goals_h['S_HT'] += len([x for x in min_a if x <= 45])
            
            if min_h: 
                times_h.append(min(min_h))
                first_goals_h.append(min(min_h))
            
            # Popola Heatmap H
            for m in min_h:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                hm_f[sel_home][intervals[idx]] += 1
            for m in min_a:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                hm_s[sel_home][intervals[idx]] += 1

        # --- ANALISI SQUADRA OSPITE ---
        if a == sel_away:
            match_a += 1
            goals_a['FT'] += len(min_a)
            goals_a['HT'] += len([x for x in min_a if x <= 45])
            goals_a['S_FT'] += len(min_h)
            goals_a['S_HT'] += len([x for x in min_h if x <= 45])
            
            if min_a: 
                times_a.append(min(min_a))
                first_goals_a.append(min(min_a))

            # Popola Heatmap A
            for m in min_a:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                hm_f[sel_away][intervals[idx]] += 1
            for m in min_h:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                hm_s[sel_away][intervals[idx]] += 1

    # --- STATISTICHE CHIAVE ---
    def safe_avg(lst): return int(np.mean(lst)) if lst else 0
    def safe_div(n, d): return n / d if d > 0 else 0

    avg_min_h = safe_avg(first_goals_h)
    avg_min_a = safe_avg(first_goals_a)

    # Poisson Lambdas
    avg_h_ft = safe_div(goals_h['FT'], match_h)
    avg_h_ht = safe_div(goals_h['HT'], match_h)
    avg_h_conc_ft = safe_div(goals_h['S_FT'], match_h)
    avg_h_conc_ht = safe_div(goals_h['S_HT'], match_h)

    avg_a_ft = safe_div(goals_a['FT'], match_a)
    avg_a_ht = safe_div(goals_a['HT'], match_a)
    avg_a_conc_ft = safe_div(goals_a['S_FT'], match_a)
    avg_a_conc_ht = safe_div(goals_a['S_HT'], match_a)

    # --- CALCOLO POISSON ---
    exp_h_ft = (avg_h_ft + avg_a_conc_ft) / 2
    exp_a_ft = (avg_a_ft + avg_h_conc_ft) / 2
    exp_h_ht = (avg_h_ht + avg_a_conc_ht) / 2
    exp_a_ht = (avg_a_ht + avg_h_conc_ht) / 2

    def get_probs(lam_h, lam_a):
        probs = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                probs[i][j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
        p1 = np.sum(np.tril(probs, -1))
        px = np.sum(np.diag(probs))
        p2 = np.sum(np.triu(probs, 1))
        
        # O/U 2.5
        pu25 = 0
        for i in range(6):
            for j in range(6):
                if i+j <= 2: pu25 += probs[i][j]
        
        # 0-0 e U1.5
        p00 = probs[0][0]
        pu15 = probs[0][0] + probs[1][0] + probs[0][1]
        
        return p1, px, p2, pu25, p00, pu15

    p1_ft, px_ft, p2_ft, pu25_ft, _, _ = get_probs(exp_h_ft, exp_a_ft)
    _, _, _, _, p00_ht, pu15_ht = get_probs(exp_h_ht, exp_a_ht)

    def to_odd(p): return round(1/p, 2) if p > 0 else 99.00

    # --- VISUALIZZAZIONE DATI ---
    st.subheader("📊 Statistiche & Ritmo")
    m1, m2, m3 = st.columns(3)
    
    m1.info(f"**🏠 {sel_home}**")
    m1.write(f"Minuto Medio 1° Gol: **{avg_min_h}'**")
    m1.write(f"Media Gol Fatti (FT): **{avg_h_ft:.2f}**")
    
    m2.warning(f"**✈️ {sel_away}**")
    m2.write(f"Minuto Medio 1° Gol: **{avg_min_a}'**")
    m2.write(f"Media Gol Fatti (FT): **{avg_a_ft:.2f}**")

    m3.success("**🎲 Quote Reali (Fair Odds)**")
    m3.write(f"1: **@{to_odd(p1_ft)}** | X: **@{to_odd(px_ft)}** | 2: **@{to_odd(p2_ft)}**")
    m3.write(f"O 2.5: **@{to_odd(1-pu25_ft)}** | U 2.5: **@{to_odd(pu25_ft)}**")
    
    st.write("---")
    st.subheader("⏱️ Analisi Primo Tempo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prob. 0-0 HT", f"{p00_ht*100:.1f}%", f"@{to_odd(p00_ht)}")
    c2.metric("Prob. Under 1.5 HT", f"{pu15_ht*100:.1f}%", f"@{to_odd(pu15_ht)}")
    c3.metric("Media Gol HT Casa", f"{avg_h_ht:.2f}")
    c4.metric("Media Gol HT Ospite", f"{avg_a_ht:.2f}")

    st.divider()

    # --- GRAFICI ---
    tab1, tab2, tab3 = st.tabs(["📉 Ritmo Gol (Kaplan-Meier)", "⚽ Heatmap Fatti", "🛡️ Heatmap Subiti"])

    with tab1:
        plt.figure(figsize=(10, 5))
        kmf = KaplanMeierFitter()
        
        if times_h and times_a:
            # Casa
            kmf.fit(times_h, label=f'{sel_home}')
            ax = kmf.plot_survival_function(ci_show=False, linewidth=3, color='blue')
            # Ospite
            kmf.fit(times_a, label=f'{sel_away}')
            kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
            # Lega
            if len(times_league) > 10:
                kmf.fit(times_league, label='Media Campionato')
                kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color='gray', linestyle='--')
            
            plt.title("📉 Probabilità di 0-0 nel tempo (Ritmo Gol)")
            plt.xlabel("Minuti")
            plt.ylabel("Prob. che NON abbiano ancora segnato")
            plt.axhline(0.5, color='green', linestyle=':', label='Mediana 50%')
            plt.axvline(45, color='black', linestyle='--', alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.legend()
            st.pyplot(plt)
        else:
            st.warning("Dati insufficienti per il grafico KM.")

    # Heatmaps Sottili
    df_f = pd.DataFrame([hm_f[sel_home], hm_f[sel_away]], index=[sel_home, sel_away])
    df_s = pd.DataFrame([hm_s[sel_home], hm_s[sel_away]], index=[sel_home, sel_away])

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        sns.heatmap(df_f[intervals], annot=True, cmap="Greens", fmt="d", cbar=False, ax=ax)
        plt.title("Distribuzione Gol Fatti")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        sns.heatmap(df_s[intervals], annot=True, cmap="Reds", fmt="d", cbar=False, ax=ax)
        plt.title("Distribuzione Gol Subiti")
        st.pyplot(fig)
