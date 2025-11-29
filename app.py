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
st.set_page_config(page_title="⚽ Dashboard Tattica V31", layout="wide", page_icon="⚽")
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- TITOLO ---
st.title("💎 Dashboard Analisi Calcio V31")
st.markdown("**Analisi Tattica, Quote Reali & Caccia al Valore**")
st.divider()

# ==========================================
# 1. CARICAMENTO DATI (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("📂 1. Dati")
    uploaded_file = st.file_uploader("Carica file (CSV/Excel)", type=['csv', 'xlsx'])
    
    # File default se presente su GitHub
    default_file = 'eng_tot_1.csv'
    file_to_use = uploaded_file if uploaded_file else (default_file if os.path.exists(default_file) else None)

    if file_to_use is None:
        st.warning("Carica un file per iniziare.")
        st.stop()

    st.divider()
    st.header("💰 2. Quote Bookmaker")
    st.caption("Inserisci le quote reali per calcolare il valore:")
    
    col_q1, col_q2, col_q3 = st.columns(3)
    b_1 = col_q1.number_input("1", value=1.00, step=0.01, format="%.2f")
    b_x = col_q2.number_input("X", value=1.00, step=0.01, format="%.2f")
    b_2 = col_q3.number_input("2", value=1.00, step=0.01, format="%.2f")
    
    col_qu1, col_qu2 = st.columns(2)
    b_u05ht = col_qu1.number_input("U 0.5 HT", value=1.00, step=0.01, format="%.2f")
    b_u15ht = col_qu2.number_input("U 1.5 HT", value=1.00, step=0.01, format="%.2f")

@st.cache_data
def load_data(file_input, is_path=False):
    try:
        if is_path:
             with open(file_input, 'r', encoding='latin1', errors='replace') as f:
                line = f.readline()
                sep = ';' if line.count(';') > line.count(',') else ','
             df = pd.read_csv(file_input, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False, header=None)
        else:
            try:
                line = file_input.readline().decode('latin1')
                file_input.seek(0)
                sep = ';' if line.count(';') > line.count(',') else ','
                df = pd.read_csv(file_input, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False, header=None)
            except:
                file_input.seek(0)
                df = pd.read_excel(file_input, header=None)

        header = df.iloc[0].astype(str).str.strip().str.upper().tolist()
        seen = {}
        unique_header = []
        for col in header:
            if col in seen:
                seen[col] += 1
                unique_header.append(f"{col}.{seen[col]}")
            else:
                seen[col] = 0
                unique_header.append(col)
        df = df.iloc[1:].copy()
        df.columns = unique_header
        
        col_map = {
            'GOALMINH': ['GOALMINH', 'GOALMINCASA', 'MINUTI_CASA'],
            'GOALMINA': ['GOALMINA', 'GOALMINOSPITE', 'MINUTI_OSPITE'],
            'LEGA': ['LEGA', 'LEAGUE', 'DIVISION'],
            'PAESE': ['PAESE', 'COUNTRY'],
            'CASA': ['CASA', 'HOME', 'TEAM1'],
            'OSPITE': ['OSPITE', 'AWAY', 'TEAM2']
        }
        
        for target, candidates in col_map.items():
            if target not in df.columns:
                for candidate in candidates:
                    found = next((c for c in df.columns if c == candidate), None)
                    if found:
                        df.rename(columns={found: target}, inplace=True)
                        break
        
        for c in ['PAESE', 'LEGA', 'CASA', 'OSPITE']:
            if c in df.columns: df[c] = df[c].astype(str).str.strip()

        if 'PAESE' in df.columns:
            df['ID_LEGA'] = df['PAESE'] + " - " + df['LEGA']
        else:
            df['ID_LEGA'] = df['LEGA']
            
        return df
    except Exception as e:
        return pd.DataFrame()

is_path = isinstance(file_to_use, str)
df = load_data(file_to_use, is_path)

if df.empty:
    st.error("File non valido.")
    st.stop()

# ==========================================
# 2. SELEZIONE MATCH
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    leghe = sorted(df['ID_LEGA'].unique())
    sel_lega = st.selectbox("🏆 Campionato", leghe)

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
if st.button("🚀 AVVIA ANALISI", type="primary"):
    st.divider()
    st.subheader(f"⚔️ {sel_home} vs {sel_away}")
    
    intervals = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
    
    def get_minutes(val):
        if pd.isna(val): return []
        s = str(val).replace(',', '.').replace(';', ' ').replace('"', '').replace("'", "")
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        res = []
        for x in nums:
            try:
                n = int(float(x))
                if 0 <= n <= 130: res.append(n)
            except: pass
        return res

    c_h = 'GOALMINH' if 'GOALMINH' in df_league.columns else 'GOALMINCASA'
    c_a = 'GOALMINA' if 'GOALMINA' in df_league.columns else 'GOALMINOSPITE'

    goals_h = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    goals_a = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    match_h, match_a = 0, 0
    times_h, times_a = [], []
    
    stats_match = {
        sel_home: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
        sel_away: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
    }

    for _, row in df_league.iterrows():
        h, a = row['CASA'], row['OSPITE']
        min_h = get_minutes(row.get(c_h))
        min_a = get_minutes(row.get(c_a))
        
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

    def safe_div(n, d): return n / d if d > 0 else 0

    # Medie
    avg_h_ft = safe_div(goals_h['FT'], match_h)
    avg_h_ht = safe_div(goals_h['HT'], match_h)
    avg_h_conc_ft = safe_div(goals_h['S_FT'], match_h)
    avg_h_conc_ht = safe_div(goals_h['S_HT'], match_h)

    avg_a_ft = safe_div(goals_a['FT'], match_a)
    avg_a_ht = safe_div(goals_a['HT'], match_a)
    avg_a_conc_ft = safe_div(goals_a['S_FT'], match_a)
    avg_a_conc_ht = safe_div(goals_a['S_HT'], match_a)

    # Poisson
    exp_h_ft = (avg_h_ft + avg_a_conc_ft) / 2
    exp_a_ft = (avg_a_ft + avg_h_conc_ft) / 2
    exp_h_ht = (avg_h_ht + avg_a_conc_ht) / 2
    exp_a_ht = (avg_a_ht + avg_h_conc_ht) / 2

    def get_poisson_probs(lam_h, lam_a):
        probs = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                probs[i][j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
        p1 = np.sum(np.tril(probs, -1))
        px = np.sum(np.diag(probs))
        p2 = np.sum(np.triu(probs, 1))
        return p1, px, p2

    p1_ft, px_ft, p2_ft = get_poisson_probs(exp_h_ft, exp_a_ft)
    
    # Probabilità HT Specifiche
    # 0-0 HT: P(0) casa * P(0) ospite
    prob_00_ht = poisson.pmf(0, exp_h_ht) * poisson.pmf(0, exp_a_ht)
    
    # Under 1.5 HT: 0-0 + 1-0 + 0-1
    prob_10_ht = poisson.pmf(1, exp_h_ht) * poisson.pmf(0, exp_a_ht)
    prob_01_ht = poisson.pmf(0, exp_h_ht) * poisson.pmf(1, exp_a_ht)
    prob_u15_ht = prob_00_ht + prob_10_ht + prob_01_ht

    def to_odd(p): return 1/p if p > 0 else 99.00

    # --- VISUALIZZAZIONE VALORE ---
    def show_value_card(label, prob, book_odd):
        real_odd = to_odd(prob)
        delta = book_odd - real_odd
        valore = (prob * book_odd) - 1
        
        color = "green" if valore > 0 else "red"
        icon = "✅ VALUE BET" if valore > 0 else "❌ NO VALUE"
        
        st.markdown(f"""
        <div style="padding:10px; border-radius:5px; background-color:rgba(200,200,200,0.1); margin-bottom:10px;">
            <strong>{label}</strong><br>
            Prob. Reale: {prob*100:.1f}% (Quota: {real_odd:.2f})<br>
            Bookmaker: {book_odd:.2f}<br>
            <span style="color:{color}; font-weight:bold;">{icon} (ROI: {valore*100:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 1X2 Finale")
        show_value_card("Vittoria Casa (1)", p1_ft, b_1)
        show_value_card("Pareggio (X)", px_ft, b_x)
        show_value_card("Vittoria Ospite (2)", p2_ft, b_2)
        
    with c2:
        st.subheader("⏱️ Primo Tempo")
        show_value_card("Under 0.5 HT (0-0)", prob_00_ht, b_u05ht)
        show_value_card("Under 1.5 HT", prob_u15_ht, b_u15ht)

    st.divider()
    
    # --- GRAFICI ---
    tab1, tab2 = st.tabs(["📉 Ritmo Gol (Kaplan-Meier)", "🔥 Heatmaps"])

    with tab1:
        if times_h and times_a:
            fig, ax = plt.subplots(figsize=(10, 4))
            kmf_h = KaplanMeierFitter()
            kmf_a = KaplanMeierFitter()
            
            kmf_h.fit(times_h, label=f'{sel_home}')
            kmf_a.fit(times_a, label=f'{sel_away}')
            
            kmf_h.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue')
            kmf_a.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
            
            plt.title('Probabilità di restare a 0 gol (Ritmo)')
            plt.grid(True, alpha=0.3)
            plt.axvline(45, color='green', linestyle='--')
            st.pyplot(fig)
        else:
            st.warning("Dati insufficienti per KM.")

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
