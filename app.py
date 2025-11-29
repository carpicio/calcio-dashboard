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
st.set_page_config(page_title="⚽ Dashboard V43 (Poisson Pro)", layout="wide", page_icon="⚽")
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- TITOLO ---
st.title("💎 Dashboard Analisi V43")
st.markdown("**Poisson 1X2 • Money Management • Analisi Ritmo Gol**")
st.divider()

# ==========================================
# 1. SIDEBAR: DATI & INPUT
# ==========================================
with st.sidebar:
    st.header("📂 1. Dati")
    uploaded_file = st.file_uploader("Carica file (CSV/Excel)", type=['csv', 'xlsx'])
    
    default_file = 'eng_tot.xlsx - eng_tot.csv'
    # Cerca file di default nella cartella se non caricato
    if uploaded_file is None:
        possible_files = [f for f in os.listdir() if 'eng_tot' in f or 'CGM' in f]
        if possible_files:
            default_file = possible_files[0]

    st.divider()
    
    # SEZIONE QUOTE (MONEY MANAGEMENT)
    st.header("💰 2. Quote Bookmaker")
    st.caption("Inserisci le quote per calcolare il valore:")
    
    c_q1, c_qx, c_q2 = st.columns(3)
    q_1 = c_q1.number_input("1", value=1.00, step=0.01, format="%.2f")
    q_x = c_qx.number_input("X", value=1.00, step=0.01, format="%.2f")
    q_2 = c_q2.number_input("2", value=1.00, step=0.01, format="%.2f")
    
    c_ou1, c_ou2 = st.columns(2)
    q_over25 = c_ou1.number_input("Over 2.5", value=1.00, step=0.01, format="%.2f")
    q_under25 = c_ou2.number_input("Under 2.5", value=1.00, step=0.01, format="%.2f")
    
    st.divider()
    w_cassa = st.number_input("Cassa Totale (€)", value=1000.0, step=10.0)

@st.cache_data
def load_data(file_input):
    try:
        # Logica di caricamento file (Path o Buffer)
        if isinstance(file_input, str): # Se è un percorso file
            with open(file_input, 'r', encoding='latin1', errors='replace') as f:
                line = f.readline()
                sep = ';' if line.count(';') > line.count(',') else ','
            df = pd.read_csv(file_input, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False)
        else: # Se è un file caricato
            try:
                line = file_input.readline().decode('latin1')
                file_input.seek(0)
                sep = ';' if line.count(';') > line.count(',') else ','
                df = pd.read_csv(file_input, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False)
            except:
                file_input.seek(0)
                df = pd.read_excel(file_input)

        # Pulizia Nomi Colonne
        df.columns = df.columns.astype(str).str.strip().str.upper()
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Mappatura Universale
        col_map = {
            'GOALMINH': ['GOALMINH', 'GOALMINCASA', 'MINUTI_CASA', 'GOALSH'],
            'GOALMINA': ['GOALMINA', 'GOALMINOSPITE', 'MINUTI_OSPITE', 'GOALSA'],
            'LEGA': ['LEGA', 'LEAGUE', 'DIVISION'],
            'PAESE': ['PAESE', 'COUNTRY'],
            'CASA': ['CASA', 'HOME', 'TXTECHIPA1'],
            'OSPITE': ['OSPITE', 'AWAY', 'TXTECHIPA2'],
            'ELO_H': ['ELOHOMEO', 'ELO_HOME'],
            'ELO_A': ['ELOAWAYO', 'ELO_AWAY']
        }
        
        for target, candidates in col_map.items():
            if target not in df.columns:
                for candidate in candidates:
                    if candidate in df.columns:
                        df.rename(columns={candidate: target}, inplace=True)
                        break

        # Pulizia Celle
        for c in ['PAESE', 'LEGA', 'CASA', 'OSPITE']:
            if c in df.columns: df[c] = df[c].astype(str).str.strip()

        if 'PAESE' in df.columns:
            df['ID_LEGA'] = df['PAESE'] + " - " + df['LEGA']
        else:
            df['ID_LEGA'] = df['LEGA']
            
        return df
    except Exception as e:
        return pd.DataFrame()

# Caricamento effettivo
if uploaded_file:
    df = load_data(uploaded_file)
elif os.path.exists(default_file):
    df = load_data(default_file)
else:
    st.stop() # Aspetta caricamento

if df.empty:
    st.error("File non valido.")
    st.stop()

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
# 3. ENGINE DI ANALISI
# ==========================================
if st.button("🚀 AVVIA ANALISI", type="primary"):
    st.divider()
    
    intervals = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
    
    def get_minutes(val):
        if pd.isna(val): return []
        s = str(val).replace(',', ' ').replace(';', ' ').replace('.', ' ').replace('"', '').replace("'", "")
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

    # Accumulatori Stats
    goals_h = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    goals_a = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    match_h, match_a = 0, 0
    
    # Liste per Grafici
    times_h, times_a, times_league = [], [], []
    first_goal_h, first_goal_a = [], []
    
    # Heatmap Data
    stats_match = {
        sel_home: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
        sel_away: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
    }

    for _, row in df_league.iterrows():
        h, a = row['CASA'], row['OSPITE']
        min_h = get_minutes(row.get(c_h))
        min_a = get_minutes(row.get(c_a))
        
        if min_h: times_league.append(min(min_h))
        if min_a: times_league.append(min(min_a))

        # Stats Casa
        if h == sel_home:
            match_h += 1
            goals_h['FT'] += len(min_h)
            goals_h['HT'] += len([x for x in min_h if x <= 45])
            goals_h['S_FT'] += len(min_a)
            goals_h['S_HT'] += len([x for x in min_a if x <= 45])
            if min_h: 
                times_h.extend(min_h)
                first_goal_h.append(min(min_h))
            
            # Heatmap
            for m in min_h:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                stats_match[h]['F'][intervals[idx]] += 1
            for m in min_a:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                stats_match[h]['S'][intervals[idx]] += 1

        # Stats Ospite
        if a == sel_away:
            match_a += 1
            goals_a['FT'] += len(min_a)
            goals_a['HT'] += len([x for x in min_a if x <= 45])
            goals_a['S_FT'] += len(min_h)
            goals_a['S_HT'] += len([x for x in min_h if x <= 45])
            if min_a: 
                times_a.extend(min_a)
                first_goal_a.append(min(min_a))

            # Heatmap
            for m in min_a:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                stats_match[a]['F'][intervals[idx]] += 1
            for m in min_h:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                stats_match[a]['S'][intervals[idx]] += 1

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

    # --- POISSON ---
    # Lambda FT
    lam_h_ft = (avg_h_ft + avg_a_conc_ft) / 2
    lam_a_ft = (avg_a_ft + avg_h_conc_ft) / 2
    # Lambda HT
    lam_h_ht = (avg_h_ht + avg_a_conc_ht) / 2
    lam_a_ht = (avg_a_ht + avg_h_conc_ht) / 2

    def calc_poisson_probs(lh, la):
        probs = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                probs[i][j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
        p1 = np.sum(np.tril(probs, -1))
        px = np.sum(np.diag(probs))
        p2 = np.sum(np.triu(probs, 1))
        pu25 = 0
        for i in range(6):
            for j in range(6):
                if i+j <= 2: pu25 += probs[i][j]
        return p1, px, p2, pu25

    p1_ft, px_ft, p2_ft, pu25_ft = calc_poisson_probs(lam_h_ft, lam_a_ft)
    p1_ht, px_ht, p2_ht, _ = calc_poisson_probs(lam_h_ht, lam_a_ht)
    
    # HT Specifics
    p_00_ht = poisson.pmf(0, lam_h_ht) * poisson.pmf(0, lam_a_ht)
    p_u15_ht = p_00_ht + (poisson.pmf(1, lam_h_ht) * poisson.pmf(0, lam_a_ht)) + (poisson.pmf(0, lam_h_ht) * poisson.pmf(1, lam_a_ht))

    def to_odd(p): return round(1/p, 2) if p > 0 else 99.00

    # --- VISUALIZZAZIONE PREVISIONI ---
    st.subheader("🎲 PREVISIONI & VALORE")
    
    # Kelly Criterion
    def calc_kelly(prob, quota, bankroll):
        if prob <= 0 or quota <= 1: return 0, 0
        b = quota - 1
        f = (b * prob - (1 - prob)) / b
        stake_pct = max(0, f * 0.3)
        return stake_pct * 100, bankroll * stake_pct

    # Card Valore
    def show_value_card(label, prob, quota_book):
        odd_real = to_odd(prob)
        valore = (prob * quota_book) - 1
        pct, eur = calc_kelly(prob, quota_book, w_cassa)
        
        color = "green" if valore > 0 else "red"
        icon = "✅ VALUE" if valore > 0 else "❌ NO VALUE"
        
        st.markdown(f"""
        <div style="border:1px solid #444; padding:10px; border-radius:8px; margin-bottom:10px;">
            <strong>{label}</strong><br>
            Prob: <b>{prob*100:.1f}%</b> (Fair: {odd_real})<br>
            Book: <b>{quota_book:.2f}</b><br>
            <span style="color:{color}; font-weight:bold;">{icon} ({valore*100:.1f}%)</span>
            {f"<br><small style='color:#00FF00'>Punta: € {eur:.2f}</small>" if valore > 0 else ""}
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: show_value_card(f"Vittoria {sel_home}", p1_ft, q_1)
    with c2: show_value_card("Pareggio", px_ft, q_x)
    with c3: show_value_card(f"Vittoria {sel_away}", p2_ft, q_2)
    
    c4, c5, c6 = st.columns(3)
    with c4: show_value_card("Over 2.5 FT", 1-pu25_ft, q_over25)
    with c5: show_value_card("Under 2.5 FT", pu25_ft, q_under25)
    with c6: 
        st.markdown(f"""
        <div style="border:1px solid #666; padding:10px; border-radius:8px;">
            <strong>Speciale 1° Tempo</strong><br>
            1 ({p1_ht*100:.0f}%) - X ({px_ht*100:.0f}%) - 2 ({p2_ht*100:.0f}%)<br>
            0-0 HT: <b>{p_00_ht*100:.1f}%</b> (@{to_odd(p_00_ht)})<br>
            U1.5 HT: <b>{p_u15_ht*100:.1f}%</b> (@{to_odd(p_u15_ht)})
        </div>
        """, unsafe_allow_html=True)

    # --- GRAFICI ---
    st.divider()
    tab1, tab2 = st.tabs(["📉 Ritmo Gol (1° Gol)", "🔥 Densità Fatti/Subiti"])

    with tab1:
        if first_goal_h or first_goal_a:
            fig, ax = plt.subplots(figsize=(10, 4))
            kmf = KaplanMeierFitter()
            
            if first_goal_h:
                kmf.fit(first_goal_h, label=f'{sel_home}')
                kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue')
            if first_goal_a:
                kmf.fit(first_goal_a, label=f'{sel_away}')
                kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
            
            if len(times_league) > 5:
                kmf.fit(times_league, label='Media Lega')
                kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color='gray', linestyle='--')
            
            plt.title(f"Tempo al 1° Gol: {sel_home} (~{int(np.mean(first_goal_h)) if first_goal_h else 0}') vs {sel_away} (~{int(np.mean(first_goal_a)) if first_goal_a else 0}')")
            plt.axhline(0.5, color='green', linestyle=':', label='50% Prob.')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("Dati insufficienti per Kaplan-Meier.")

    with tab2:
        rows_f = []
        rows_s = []
        for t in [sel_home, sel_away]:
            d = stats_match[t]
            rows_f.append({**{'SQUADRA': t}, **d['F']})
            rows_s.append({**{'SQUADRA': t}, **d['S']})
        
        df_f = pd.DataFrame(rows_f).set_index('SQUADRA')
        df_s = pd.DataFrame(rows_s).set_index('SQUADRA')

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        sns.heatmap(df_f[intervals], annot=True, cmap="Greens", fmt="d", cbar=False, ax=axes[0])
        axes[0].set_title('GOL FATTI')
        sns.heatmap(df_s[intervals], annot=True, cmap="Reds", fmt="d", cbar=False, ax=axes[1])
        axes[1].set_title('GOL SUBITI')
        plt.tight_layout()
        st.pyplot(fig)
