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

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="⚽ Dashboard Universale V34", layout="wide", page_icon="⚽")
warnings.simplefilter(action='ignore', category=FutureWarning)

st.title("💎 Dashboard Analisi Calcio Universale")
st.markdown("**Analisi Tattica, Ritmo Gol & Previsioni (Compatibile con CGM Bet & EngTot)**")
st.divider()

# ==========================================
# 1. CARICAMENTO DATI
# ==========================================
with st.sidebar:
    st.header("📂 Gestione Dati")
    uploaded_file = st.file_uploader("Carica il file (CSV/Excel)", type=['csv', 'xlsx'])

if uploaded_file is None:
    st.info("👈 Carica un file per iniziare.")
    st.stop()

@st.cache_data
def load_data(file):
    try:
        # Rileva separatore leggendo la prima riga
        try:
            line = file.readline().decode('latin1')
            file.seek(0)
            sep = ';' if line.count(';') > line.count(',') else ','
            df = pd.read_csv(file, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False)
        except:
            file.seek(0)
            df = pd.read_excel(file)

        # PULIZIA NOMI COLONNE
        df.columns = df.columns.astype(str).str.strip().str.upper()
        df = df.loc[:, ~df.columns.duplicated()]

        # MAPPATURA INTELLIGENTE (Gestisce entrambi i formati)
        col_map = {
            'LEGA': ['LEGA', 'LEAGUE', 'DIVISION'],
            'PAESE': ['PAESE', 'COUNTRY', 'NATION'],
            'CASA': ['CASA', 'HOME', 'TXTECHIPA1', 'TEAM1'],
            'OSPITE': ['OSPITE', 'AWAY', 'TXTECHIPA2', 'TEAM2'],
            'GOALMINH': ['GOALMINH', 'GOALMINCASA', 'MINUTI_CASA', 'GOALSH'],
            'GOALMINA': ['GOALMINA', 'GOALMINOSPITE', 'MINUTI_OSPITE', 'GOALSA']
        }
        
        # Rinomina colonne
        for target, candidates in col_map.items():
            if target not in df.columns:
                for candidate in candidates:
                    # Cerca la colonna esatta (già in upper)
                    if candidate in df.columns:
                        df.rename(columns={candidate: target}, inplace=True)
                        break
        
        # Verifica colonne essenziali
        required = ['LEGA', 'CASA', 'OSPITE']
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            st.error(f"❌ Colonne mancanti: {missing}. Verifica il file.")
            return pd.DataFrame()

        # Pulizia Celle
        for c in ['PAESE', 'LEGA', 'CASA', 'OSPITE']:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        # Crea ID Lega
        if 'PAESE' in df.columns:
            df['ID_LEGA'] = df['PAESE'] + " - " + df['LEGA']
        else:
            df['ID_LEGA'] = df['LEGA']
            
        return df

    except Exception as e:
        st.error(f"Errore caricamento: {e}")
        return pd.DataFrame()

df = load_data(uploaded_file)

if df.empty:
    st.stop()

st.sidebar.success(f"✅ File caricato: {len(df)} righe")

# ==========================================
# 2. SELEZIONE MATCH
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    leghe = sorted(df['ID_LEGA'].unique())
    sel_lega = st.selectbox("🏆 Seleziona Campionato", leghe)

df_league = df[df['ID_LEGA'] == sel_lega].copy()
teams = sorted(pd.concat([df_league['CASA'], df_league['OSPITE']]).unique())

with col2:
    sel_home = st.selectbox("🏠 Squadra Casa", teams, index=0)

with col3:
    idx_away = 1 if len(teams) > 1 else 0
    sel_away = st.selectbox("✈️ Squadra Ospite", teams, index=idx_away)

# ==========================================
# 3. ENGINE DI ANALISI
# ==========================================
if st.button("🚀 AVVIA ANALISI MATCH", type="primary"):
    st.divider()
    st.subheader(f"⚔️ Analisi: {sel_home} vs {sel_away}")
    
    intervals = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
    
    def get_minutes(val):
        if pd.isna(val): return []
        # Pulisce tutto tranne i numeri
        s = str(val).replace('"', '').replace("'", "").replace('.', ' ').replace(',', ' ').replace(';', ' ')
        res = []
        for x in s.split():
            if x.isdigit():
                n = int(x)
                if 0 <= n <= 130: res.append(n)
        return res

    # Colonne minuti (fallback se mancano)
    c_h = 'GOALMINH' if 'GOALMINH' in df_league.columns else df_league.columns[0] 
    c_a = 'GOALMINA' if 'GOALMINA' in df_league.columns else df_league.columns[0]

    # Controlla se colonne minuti esistono davvero
    if 'GOALMINH' not in df_league.columns:
        st.warning("⚠️ Colonne minuti gol non trovate. I grafici temporali saranno vuoti.")

    # Accumulatori
    goals_h = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    goals_a = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    match_h, match_a = 0, 0
    times_h, times_a, times_league = [], [], []
    
    stats_match = {
        sel_home: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
        sel_away: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
    }

    for _, row in df_league.iterrows():
        h, a = row['CASA'], row['OSPITE']
        
        # Gestione sicura minuti
        min_h = get_minutes(row.get('GOALMINH'))
        min_a = get_minutes(row.get('GOALMINA'))
        
        if min_h: times_league.append(min(min_h))
        if min_a: times_league.append(min(min_a))

        # Heatmap
        if h in stats_match:
            for m in min_h:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[h]['F'][intervals[idx]] += 1
            for m in min_a:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[h]['S'][intervals[idx]] += 1
        
        if a in stats_match:
            for m in min_a:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[a]['F'][intervals[idx]] += 1
            for m in min_h:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[a]['S'][intervals[idx]] += 1

        # Stats
        if h == sel_home:
            match_h += 1
            goals_h['FT'] += len(min_h)
            goals_h['HT'] += len([x for x in min_h if x <= 45])
            goals_h['S_FT'] += len(min_a)
            goals_h['S_HT'] += len([x for x in min_a if x <= 45])
            if min_h: times_h.append(min(min_h))
        
        if a == sel_away:
            match_a += 1
            goals_a['FT'] += len(min_a)
            goals_a['HT'] += len([x for x in min_a if x <= 45])
            goals_a['S_FT'] += len(min_h)
            goals_a['S_HT'] += len([x for x in min_h if x <= 45])
            if min_a: times_a.append(min(min_a))

    # Medie
    def safe_div(n, d): return n / d if d > 0 else 0

    avg_h_ft = safe_div(goals_h['FT'], match_h)
    avg_h_ht = safe_div(goals_h['HT'], match_h)
    avg_h_conc_ft = safe_div(goals_h['S_FT'], match_h)
    avg_h_conc_ht = safe_div(goals_h['S_HT'], match_h)

    avg_a_ft = safe_div(goals_a['FT'], match_a)
    avg_a_ht = safe_div(goals_a['HT'], match_a)
    avg_a_conc_ft = safe_div(goals_a['S_FT'], match_a)
    avg_a_conc_ht = safe_div(goals_a['S_HT'], match_a)

    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**🏠 {sel_home}** ({match_h} match)\n\n1°T: {avg_h_ht:.2f} F / {avg_h_conc_ht:.2f} S\n\nFIN: {avg_h_ft:.2f} F / {avg_h_conc_ft:.2f} S")
    with c2:
        st.warning(f"**✈️ {sel_away}** ({match_a} match)\n\n1°T: {avg_a_ht:.2f} F / {avg_a_conc_ht:.2f} S\n\nFIN: {avg_a_ft:.2f} F / {avg_a_conc_ft:.2f} S")

    # Poisson
    exp_h_ft = (avg_h_ft + avg_a_conc_ft) / 2
    exp_a_ft = (avg_a_ft + avg_h_conc_ft) / 2
    exp_h_ht = (avg_h_ht + avg_a_conc_ht) / 2
    exp_a_ht = (avg_a_ht + avg_h_conc_ht) / 2

    def calc_poisson_probs(lam_h, lam_a):
        probs = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                probs[i][j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
        p1 = np.sum(np.tril(probs, -1))
        px = np.sum(np.diag(probs))
        p2 = np.sum(np.triu(probs, 1))
        pu25 = 0
        for i in range(6):
            for j in range(6):
                if i+j <= 2: pu25 += probs[i][j]
        return p1, px, p2, pu25

    p1_ft, px_ft, p2_ft, pu25_ft = calc_poisson_probs(exp_h_ft, exp_a_ft)
    prob_00_ht = poisson.pmf(0, exp_h_ht) * poisson.pmf(0, exp_a_ht)
    prob_u15_ht = prob_00_ht + (poisson.pmf(1, exp_h_ht) * poisson.pmf(0, exp_a_ht)) + (poisson.pmf(0, exp_h_ht) * poisson.pmf(1, exp_a_ht))
    
    def to_odd(p): return round(1/p, 2) if p > 0 else 99.00

    st.subheader("🎲 Previsioni (Poisson)")
    c1, c2, c3 = st.columns(3)
    c1.metric("1 (Casa)", f"{p1_ft*100:.0f}%", f"@{to_odd(p1_ft)}")
    c2.metric("X (Pareggio)", f"{px_ft*100:.0f}%", f"@{to_odd(px_ft)}")
    c3.metric("2 (Ospite)", f"{p2_ft*100:.0f}%", f"@{to_odd(p2_ft)}")
    st.caption(f"HT: 0-0 @{to_odd(prob_00_ht)} | Under 1.5 @{to_odd(prob_u15_ht)}")

    st.divider()

    # Grafici
    tab1, tab2, tab3 = st.tabs(["📉 Ritmo Gol", "⚽ Heatmap Fatti", "🛡️ Heatmap Subiti"])

    with tab1:
        if times_h and times_a:
            fig, ax = plt.subplots(figsize=(10, 5))
            kmf_h = KaplanMeierFitter()
            kmf_a = KaplanMeierFitter()
            kmf_l = KaplanMeierFitter()
            
            kmf_h.fit(times_h, label=f'{sel_home}')
            kmf_a.fit(times_a, label=f'{sel_away}')
            if len(times_league) > 10:
                kmf_l.fit(times_league, label='Media Lega')
                kmf_l.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color='gray', linestyle='--')
            
            kmf_h.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue')
            kmf_a.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
            
            try:
                med_h = kmf_h.median_survival_time_
                med_a = kmf_a.median_survival_time_
                plt.title(f"Tempo al 1° Gol: {sel_home} (~{med_h:.0f}') vs {sel_away} (~{med_a:.0f}')")
            except:
                plt.title("Tempo al 1° Gol")

            plt.axhline(y=0.5, color='green', linestyle=':', label='Mediana (50%)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            st.pyplot(fig)
        else:
            st.warning("Dati insufficienti per il grafico KM.")

    rows_f = []
    rows_s = []
    for t in [sel_home, sel_away]:
        d = stats_match[t]
        rows_f.append({**{'SQUADRA': t}, **d['F']})
        rows_s.append({**{'SQUADRA': t}, **d['S']})
    
    df_f = pd.DataFrame(rows_f).set_index('SQUADRA')
    df_s = pd.DataFrame(rows_s).set_index('SQUADRA')

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(df_f[intervals], annot=True, cmap="Greens", fmt="d", cbar=False, ax=ax)
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(df_s[intervals], annot=True, cmap="Reds", fmt="d", cbar=False, ax=ax)
        st.pyplot(fig)
