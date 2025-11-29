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
st.set_page_config(page_title="⚽ Dashboard V46 (Full + Matrix)", layout="wide", page_icon="⚽")
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- TITOLO ---
st.title("💎 Dashboard V46: Analisi Totale")
st.markdown("**Statistiche • ELO • Money Management • Poisson Matrix • Ritmi**")
st.divider()

# ==========================================
# 1. SIDEBAR: TUTTI GLI INPUT
# ==========================================
with st.sidebar:
    st.header("📂 1. Dati")
    uploaded_file = st.file_uploader("Carica file (CSV/Excel)", type=['csv', 'xlsx'])
    
    default_file = 'eng_tot.xlsx - eng_tot.csv' 
    file_to_use = uploaded_file if uploaded_file else (default_file if os.path.exists(default_file) else None)

    if file_to_use is None:
        st.warning("Carica un file per iniziare.")
        st.stop()

    st.divider()
    
    # ELO
    st.header("👑 2. ELO Rating")
    ce1, ce2 = st.columns(2)
    user_elo_h = ce1.number_input("ELO Casa", value=1500, step=10)
    user_elo_a = ce2.number_input("ELO Ospite", value=1500, step=10)
    
    st.divider()
    
    # QUOTE E CASSA
    st.header("💰 3. Quote & Cassa")
    c_q1, c_qx, c_q2 = st.columns(3)
    q_1 = c_q1.number_input("1", value=1.00, step=0.01, format="%.2f")
    q_x = c_qx.number_input("X", value=1.00, step=0.01, format="%.2f")
    q_2 = c_q2.number_input("2", value=1.00, step=0.01, format="%.2f")
    
    c_ou1, c_ou2 = st.columns(2)
    q_over25 = c_ou1.number_input("Quota Over 2.5", value=1.00, step=0.01, format="%.2f")
    q_under25 = c_ou2.number_input("Quota Under 2.5", value=1.00, step=0.01, format="%.2f")
    
    st.divider()
    w_cassa = st.number_input("Cassa Totale (€)", value=1000.0, step=10.0)
    w_soglia_trappola = st.slider("Soglia Allarme Trappola (%)", 10, 100, 30, 5)

# ==========================================
# 2. CARICAMENTO DATI
# ==========================================
@st.cache_data
def load_data(file_input, is_path=False):
    try:
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

        df.columns = df.columns.astype(str).str.strip().str.upper()
        df = df.loc[:, ~df.columns.duplicated()]
        
        col_map = {
            'GOALMINH': ['GOALMINH', 'GOALMINCASA', 'MINUTI_CASA', 'GOALSH'],
            'GOALMINA': ['GOALMINA', 'GOALMINOSPITE', 'MINUTI_OSPITE', 'GOALSA'],
            'LEGA': ['LEGA', 'LEAGUE', 'DIVISION'],
            'PAESE': ['PAESE', 'COUNTRY'],
            'CASA': ['CASA', 'HOME', 'TXTECHIPA1'],
            'OSPITE': ['OSPITE', 'AWAY', 'TXTECHIPA2'],
            'ELO_H': ['ELOHOMEO', 'ELO_HOME', 'ELO1'],
            'ELO_A': ['ELOAWAYO', 'ELO_AWAY', 'ELO2']
        }
        
        for target, candidates in col_map.items():
            if target not in df.columns:
                for candidate in candidates:
                    if candidate in df.columns:
                        df.rename(columns={candidate: target}, inplace=True)
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

st.sidebar.success(f"✅ {len(df)} righe caricate")
has_elo = 'ELO_H' in df.columns and 'ELO_A' in df.columns

# ==========================================
# 3. SELEZIONE MATCH
# ==========================================
col1, col2, col3 = st.columns(3)
leghe = sorted(df['ID_LEGA'].unique())
with col1: sel_lega = st.selectbox("🏆 Campionato", leghe)

df_league = df[df['ID_LEGA'] == sel_lega].copy()
teams = sorted(pd.concat([df_league['CASA'], df_league['OSPITE']]).unique())

with col2: sel_home = st.selectbox("🏠 Casa", teams, index=0)
with col3: sel_away = st.selectbox("✈️ Ospite", teams, index=1 if len(teams)>1 else 0)

# ==========================================
# 4. ENGINE ANALISI
# ==========================================
if st.button("🚀 AVVIA ANALISI COMPLETA", type="primary"):
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

    # Accumulatori
    goals_h = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    goals_a = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    match_h, match_a = 0, 0
    times_h, times_a, times_league = [], [], []
    first_goal_h, first_goal_a = [], []
    
    # Dati per Heatmap
    stats_match = {
        sel_home: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
        sel_away: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
    }
    
    # Elo Storico
    hist_elo_h, hist_elo_a = [], []

    for _, row in df_league.iterrows():
        h, a = row['CASA'], row['OSPITE']
        min_h = get_minutes(row.get(c_h))
        min_a = get_minutes(row.get(c_a))
        
        if has_elo:
            if h == sel_home: hist_elo_h.append(float(row['ELO_H']))
            if a == sel_away: hist_elo_a.append(float(row['ELO_A']))

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
                times_h.append(min(min_h))
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
                times_a.append(min(min_a))
                first_goal_a.append(min(min_a))

            # Heatmap
            for m in min_a:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                stats_match[a]['F'][intervals[idx]] += 1
            for m in min_h:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                stats_match[a]['S'][intervals[idx]] += 1

    def safe_div(n, d): return n / d if d > 0 else 0
    def safe_mean(lst): return np.mean(lst) if lst else 0

    # Medie
    avg_h_ft = safe_div(goals_h['FT'], match_h)
    avg_h_ht = safe_div(goals_h['HT'], match_h)
    avg_h_conc_ft = safe_div(goals_h['S_FT'], match_h)
    avg_h_conc_ht = safe_div(goals_h['S_HT'], match_h)
    avg_min_h = safe_mean(first_goal_h)

    avg_a_ft = safe_div(goals_a['FT'], match_a)
    avg_a_ht = safe_div(goals_a['HT'], match_a)
    avg_a_conc_ft = safe_div(goals_a['S_FT'], match_a)
    avg_a_conc_ht = safe_div(goals_a['S_HT'], match_a)
    avg_min_a = safe_mean(first_goal_a)

    # --- DISPLAY STATS ---
    st.subheader("📊 Statistiche & Ritmi")
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**🏠 {sel_home}** ({match_h} match)")
        st.write(f"⏱️ Minuto medio 1° Gol: **{int(avg_min_h)}'**")
        st.write(f"**1°T:** {avg_h_ht:.2f} F / {avg_h_conc_ht:.2f} S")
        st.write(f"**FIN:** {avg_h_ft:.2f} F / {avg_h_conc_ft:.2f} S")
    with c2:
        st.warning(f"**✈️ {sel_away}** ({match_a} match)")
        st.write(f"⏱️ Minuto medio 1° Gol: **{int(avg_min_a)}'**")
        st.write(f"**1°T:** {avg_a_ht:.2f} F / {avg_a_conc_ht:.2f} S")
        st.write(f"**FIN:** {avg_a_ft:.2f} F / {avg_a_conc_ft:.2f} S")

    st.divider()

    # --- ELO RATING ---
    if has_elo:
        st.subheader("👑 Analisi ELO")
        diff_elo = (user_elo_h + 100) - user_elo_a
        prob_elo_h = 1 / (1 + 10 ** (-diff_elo / 400))
        prob_elo_a = 1 - prob_elo_h
        
        ec1, ec2 = st.columns(2)
        ec1.metric("Prob. ELO Casa", f"{prob_elo_h*100:.1f}%")
        ec2.metric("Prob. ELO Ospite", f"{prob_elo_a*100:.1f}%")

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
        return p1, px, p2, pu25, probs

    p1_ft, px_ft, p2_ft, pu25_ft, matrix_ft = calc_poisson_probs(exp_h_ft, exp_a_ft)
    
    # HT Specifics
    prob_00_ht = poisson.pmf(0, exp_h_ht) * poisson.pmf(0, exp_a_ht)
    prob_10_ht = poisson.pmf(1, exp_h_ht) * poisson.pmf(0, exp_a_ht)
    prob_01_ht = poisson.pmf(0, exp_h_ht) * poisson.pmf(1, exp_a_ht)
    prob_u15_ht = prob_00_ht + prob_10_ht + prob_01_ht

    def to_odd(p): return round(1/p, 2) if p > 0 else 99.00

    # --- MONEY MANAGEMENT ---
    st.subheader("💰 Money Management & Valore")
    
    def calc_kelly(prob, quota, bankroll):
        if prob <= 0 or quota <= 1: return 0, 0
        b = quota - 1
        f = (b * prob - (1 - prob)) / b
        stake_pct = max(0, f * 0.3)
        return stake_pct * 100, bankroll * stake_pct

    quotes = {'1': q_1, 'X': q_x, '2': q_2, 'O 2.5': q_over25, 'U 2.5': q_under25}
    probs_real = {'1': p1_ft, 'X': px_ft, '2': p2_ft, 'O 2.5': 1-pu25_ft, 'U 2.5': pu25_ft}
    
    cols = st.columns(3)
    for i, (segno, quota) in enumerate(quotes.items()):
        if quota > 1.0:
            prob = probs_real.get(segno, 0)
            fair = to_odd(prob)
            valore = (prob * quota) - 1
            diff_pct = (quota - fair) / fair if fair > 0 else 0
            trappola = diff_pct > (w_soglia_trappola / 100.0)
            
            with cols[i % 3]:
                color = "green" if (valore > 0 and not trappola) else "red"
                icon = "✅" if (valore > 0 and not trappola) else "⚠️" if trappola else "❌"
                msg = "VALUE" if (valore > 0 and not trappola) else "TRAPPOLA" if trappola else "NO VALUE"
                
                st.markdown(f"""
                <div style="border:1px solid #444; padding:10px; border-radius:5px; margin-bottom:5px;">
                    <strong>{segno}</strong> (Q: {quota:.2f})<br>
                    Fair: <b>{fair:.2f}</b> | Prob: {prob*100:.1f}%<br>
                    <span style="color:{color}; font-weight:bold;">{icon} {msg}</span>
                </div>
                """, unsafe_allow_html=True)

    st.write("---")
    
    # Output HT
    c_ht1, c_ht2 = st.columns(2)
    c_ht1.metric("Under 0.5 HT (0-0)", f"{prob_00_ht*100:.1f}%", f"Fair: @{to_odd(prob_00_ht)}")
    c_ht2.metric("Under 1.5 HT", f"{prob_u15_ht*100:.1f}%", f"Fair: @{to_odd(prob_u15_ht)}")

    st.divider()

    # --- GRAFICI ---
    t1, t2, t3, t4 = st.tabs(["📉 Ritmo Gol (KM)", "⚽ Heatmap Fatti", "🛡️ Heatmap Subiti", "🎯 Poisson Matrix"])

    with t1:
        fig, ax = plt.subplots(figsize=(10, 5))
        kmf_h = KaplanMeierFitter()
        kmf_a = KaplanMeierFitter()
        kmf_l = KaplanMeierFitter()
        
        if times_h and times_a:
            kmf_h.fit(times_h, label=f'{sel_home} (1° Gol)')
            kmf_a.fit(times_a, label=f'{sel_away} (1° Gol)')
            if len(times_league) > 10:
                kmf_l.fit(times_league, label='Media Lega')
                kmf_l.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color='gray', linestyle='--')
            
            kmf_h.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue')
            kmf_a.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
            
            plt.axhline(y=0.5, color='green', linestyle=':', label='Mediana (50%)')
            plt.title(f'Tempo al 1° Gol (Probabilità 0-0)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            st.pyplot(fig)
        else:
            st.warning("Dati insufficienti per Kaplan-Meier.")

    rows_f = []
    rows_s = []
    for t in [sel_home, sel_away]:
        d = stats_match[t]
        rows_f.append({**{'SQUADRA': t}, **d['F']})
        rows_s.append({**{'SQUADRA': t}, **d['S']})
    
    df_f = pd.DataFrame(rows_f).set_index('SQUADRA')
    df_s = pd.DataFrame(rows_s).set_index('SQUADRA')

    with t2:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        sns.heatmap(df_f[intervals], annot=True, cmap="Greens", fmt="d", cbar=False)
        st.pyplot(fig)

    with t3:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        sns.heatmap(df_s[intervals], annot=True, cmap="Reds", fmt="d", cbar=False)
        st.pyplot(fig)
        
    with t4:
        # Matrice Poisson
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix_ft, annot=True, fmt=".1%", cmap="Blues", cbar=False)
        ax.set_xlabel(f"Gol {sel_away}")
        ax.set_ylabel(f"Gol {sel_home}")
        ax.set_title("Probabilità Risultato Esatto (FT)")
        st.pyplot(fig)
