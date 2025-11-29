import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from scipy.stats import poisson
import warnings
import re

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="⚽ Dashboard Universale V35", layout="wide", page_icon="⚽")
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- TITOLO ---
st.title("💎 Dashboard Analisi Calcio V35 (Universale)")
st.markdown("**Analisi Tattica, Ritmo Gol & Previsioni (Compatibile con tutti i formati)**")
st.divider()

# ==========================================
# 1. CARICAMENTO DATI
# ==========================================
with st.sidebar:
    st.header("📂 Caricamento")
    uploaded_file = st.file_uploader("Carica file (CSV/Excel)", type=['csv', 'xlsx'])

    if st.button("🔄 Resetta Dati"):
        st.session_state.clear()
        st.rerun()

# Funzione di Caricamento Robusto
def load_data(file):
    try:
        # Tenta lettura CSV con separatore automatico
        try:
            # Legge prima riga per capire separatore
            line = file.readline().decode('latin1')
            file.seek(0) 
            sep = ';' if line.count(';') > line.count(',') else ','
            
            df = pd.read_csv(file, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False)
        except:
            file.seek(0)
            df = pd.read_excel(file)

        # --- PULIZIA E MAPPATURA COLONNE ---
        # 1. Standardizza nomi colonne (tutto maiuscolo, niente spazi)
        df.columns = df.columns.astype(str).str.strip().str.upper()
        
        # 2. Rimuove colonne duplicate
        df = df.loc[:, ~df.columns.duplicated()]

        # 3. Dizionario dei Sinonimi (Copre tutti i tuoi file)
        col_map = {
            'LEGA': ['LEGA', 'LEAGUE', 'DIVISION', 'DIV'],
            'PAESE': ['PAESE', 'COUNTRY', 'NATION', 'NAT'],
            'CASA': ['CASA', 'HOME', 'TXTECHIPA1', 'TEAM1', 'SQUADRA1', 'HT'],
            'OSPITE': ['OSPITE', 'AWAY', 'TXTECHIPA2', 'TEAM2', 'SQUADRA2', 'AT'],
            'GOALMINH': ['GOALMINH', 'GOALMINCASA', 'MINUTI_CASA', 'GOALSH', 'MINH'],
            'GOALMINA': ['GOALMINA', 'GOALMINOSPITE', 'MINUTI_OSPITE', 'GOALSA', 'MINA']
        }
        
        # 4. Rinomina le colonne trovate
        cols_found = []
        for target, candidates in col_map.items():
            for candidate in candidates:
                if candidate in df.columns:
                    df.rename(columns={candidate: target}, inplace=True)
                    cols_found.append(target)
                    break
        
        # 5. Verifica Colonne Essenziali
        required = ['LEGA', 'CASA', 'OSPITE']
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            st.error(f"❌ Errore: Colonne mancanti nel file: {missing}")
            st.write("Colonne trovate:", list(df.columns))
            return None

        # 6. Pulizia Celle
        for c in ['PAESE', 'LEGA', 'CASA', 'OSPITE']:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        # 7. Crea ID Lega Univoco
        if 'PAESE' in df.columns:
            df['ID_LEGA'] = df['PAESE'] + " - " + df['LEGA']
        else:
            df['ID_LEGA'] = df['LEGA']
            
        return df

    except Exception as e:
        st.error(f"Errore critico nel caricamento: {e}")
        return None

# Gestione Stato Sessione (Per non perdere i dati al reload)
if 'df' not in st.session_state:
    st.session_state.df = None

if uploaded_file is not None:
    df_loaded = load_data(uploaded_file)
    if df_loaded is not None:
        st.session_state.df = df_loaded

# Verifica se i dati sono pronti
if st.session_state.df is None:
    st.info("👈 Carica un file CSV o Excel dalla barra laterale.")
    st.stop()

df = st.session_state.df
st.sidebar.success(f"✅ Dati pronti: {len(df)} righe")

# ==========================================
# 2. SELEZIONE MATCH (CASCATA SICURA)
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    leghe = sorted(df['ID_LEGA'].unique())
    sel_lega = st.selectbox("🏆 Seleziona Campionato", leghe)

# Filtra dataframe per lega
df_league = df[df['ID_LEGA'] == sel_lega].copy()
teams = sorted(pd.concat([df_league['CASA'], df_league['OSPITE']]).unique())

with col2:
    sel_home = st.selectbox("🏠 Squadra Casa", teams, index=0)

with col3:
    # Cerca di selezionare una squadra diversa di default
    idx_away = 1 if len(teams) > 1 else 0
    sel_away = st.selectbox("✈️ Squadra Ospite", teams, index=idx_away)

# ==========================================
# 3. ENGINE DI ANALISI
# ==========================================
if st.button("🚀 AVVIA ANALISI MATCH", type="primary"):
    
    if sel_home == sel_away:
        st.warning("⚠️ Seleziona due squadre diverse.")
        st.stop()

    st.divider()
    st.subheader(f"⚔️ Analisi: {sel_home} vs {sel_away}")
    
    intervals = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
    
    def get_minutes(val):
        if pd.isna(val): return []
        # Pulisce tutto tranne numeri (gestisce 45+2, virgole, punti e virgola)
        s = str(val).replace('"', '').replace("'", "").replace('.', ' ').replace(',', ' ').replace(';', ' ')
        res = []
        for x in s.split():
            if x.isdigit():
                n = int(x)
                if 0 <= n <= 130: res.append(n)
        return res

    # Determina quali colonne minuti usare (se mancano usa default vuoti)
    c_h = 'GOALMINH' if 'GOALMINH' in df_league.columns else None
    c_a = 'GOALMINA' if 'GOALMINA' in df_league.columns else None

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
        
        min_h = get_minutes(row.get(c_h)) if c_h else []
        min_a = get_minutes(row.get(c_a)) if c_a else []
        
        # Dati Lega (per Media KM)
        if min_h: times_league.append(min(min_h))
        if min_a: times_league.append(min(min_a))

        # --- POPOLA HEATMAP ---
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

        # --- DATI STATISTICI ---
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

    # --- CALCOLI MEDIE ---
    def safe_div(n, d): return n / d if d > 0 else 0

    avg_h_ft = safe_div(goals_h['FT'], match_h)
    avg_h_ht = safe_div(goals_h['HT'], match_h)
    avg_h_conc_ft = safe_div(goals_h['S_FT'], match_h)
    avg_h_conc_ht = safe_div(goals_h['S_HT'], match_h)

    avg_a_ft = safe_div(goals_a['FT'], match_a)
    avg_a_ht = safe_div(goals_a['HT'], match_a)
    avg_a_conc_ft = safe_div(goals_a['S_FT'], match_a)
    avg_a_conc_ht = safe_div(goals_a['S_HT'], match_a)

    # --- VISUALIZZAZIONE MEDIE ---
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**🏠 {sel_home}** ({match_h} match)\n\n"
                f"**1°T:** F {avg_h_ht:.2f} | S {avg_h_conc_ht:.2f}\n\n"
                f"**FIN:** F {avg_h_ft:.2f} | S {avg_h_conc_ft:.2f}")
    with c2:
        st.warning(f"**✈️ {sel_away}** ({match_a} match)\n\n"
                 f"**1°T:** F {avg_a_ht:.2f} | S {avg_a_conc_ht:.2f}\n\n"
                 f"**FIN:** F {avg_a_ft:.2f} | S {avg_a_conc_ft:.2f}")

    st.divider()

    # --- POISSON ---
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
    p1_ht, px_ht, p2_ht, _ = calc_poisson_probs(exp_h_ht, exp_a_ht)
    
    prob_00_ht = poisson.pmf(0, exp_h_ht) * poisson.pmf(0, exp_a_ht)
    prob_u15_ht = prob_00_ht + (poisson.pmf(1, exp_h_ht) * poisson.pmf(0, exp_a_ht)) + (poisson.pmf(0, exp_h_ht) * poisson.pmf(1, exp_a_ht))
    
    def to_odd(p): return round(1/p, 2) if p > 0 else 99.00

    st.subheader("🎲 Previsioni & Quote Implicite")
    k1, k2, k3 = st.columns(3)
    k1.metric("1X2 Finale", f"1: {p1_ft*100:.0f}%", f"Quota: {to_odd(p1_ft)}")
    k1.caption(f"X: {px_ft*100:.0f}% (@{to_odd(px_ft)}) | 2: {p2_ft*100:.0f}% (@{to_odd(p2_ft)})")
    
    k2.metric("O/U 2.5 FT", f"Over: {(1-pu25_ft)*100:.0f}%", f"@{to_odd(1-pu25_ft)}")
    k2.caption(f"Under: {pu25_ft*100:.0f}% (@{to_odd(pu25_ft)})")
    
    k3.metric("1° Tempo", f"0-0: {prob_00_ht*100:.0f}%", f"@{to_odd(prob_00_ht)}")
    k3.caption(f"Under 1.5: {prob_u15_ht*100:.0f}% (@{to_odd(prob_u15_ht)})")

    st.divider()

    # --- GRAFICI ---
    tab1, tab2, tab3 = st.tabs(["📉 Ritmo Gol (Kaplan-Meier)", "⚽ Heatmap Fatti", "🛡️ Heatmap Subiti"])

    with tab1:
        if times_h and times_a:
            fig, ax = plt.subplots(figsize=(10, 5))
            kmf_h = KaplanMeierFitter()
            kmf_a = KaplanMeierFitter()
            kmf_l = KaplanMeierFitter()
            
            kmf_h.fit(times_h, label=f'{sel_home} (1° Gol)')
            kmf_a.fit(times_a, label=f'{sel_away} (1° Gol)')
            
            if len(times_league) > 10:
                kmf_l.fit(times_league, label='Media Lega')
                kmf_l.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color='gray', linestyle='--')

            kmf_h.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue')
            kmf_a.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
            
            try:
                med_h = kmf_h.median_survival_time_
                med_a = kmf_a.median_survival_time_
                plt.axhline(y=0.5, color='green', linestyle=':', label='Mediana (50%)')
                plt.title(f"Tempo al 1° Gol: {sel_home} (~{med_h:.0f}') vs {sel_away} (~{med_a:.0f}')")
            except:
                plt.title("Tempo al 1° Gol")
                
            plt.grid(True, alpha=0.3)
            plt.axvline(45, color='green', linestyle='--')
            plt.legend()
            st.pyplot(fig)
        else:
            st.warning("Dati insufficienti per il grafico KM (0 gol segnati).")

    # Dataframe Heatmaps
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
        plt.title("Densità Gol Fatti")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(df_s[intervals], annot=True, cmap="Reds", fmt="d", cbar=False, ax=ax)
        plt.title("Densità Gol Subiti")
        st.pyplot(fig)
