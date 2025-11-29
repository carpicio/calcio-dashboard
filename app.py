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
st.set_page_config(page_title="⚽ Dashboard Pro V39", layout="wide", page_icon="⚽")
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- TITOLO ---
st.title("💎 Dashboard Analisi Calcio V39 (Full)")
st.markdown("**Analisi Tattica, Previsioni, Money Management & Ritmi di Gioco**")
st.divider()

# ==========================================
# 1. CARICAMENTO DATI (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("📂 1. Dati")
    uploaded_file = st.file_uploader("Carica file (CSV/Excel)", type=['csv', 'xlsx'])
    
    default_file = 'eng_tot_1.csv'
    file_to_use = uploaded_file if uploaded_file else (default_file if os.path.exists(default_file) else None)

    if file_to_use is None:
        st.warning("Carica un file per iniziare.")
        st.stop()

    st.divider()
    
    # SEZIONE QUOTE (MONEY MANAGEMENT)
    st.header("💰 2. Quote Bookmaker")
    st.caption("Inserisci le quote per calcolare il valore:")
    
    col_b1, col_b2, col_b3 = st.columns(3)
    q_1 = col_b1.number_input("1", value=1.00, step=0.01, format="%.2f")
    q_x = col_b2.number_input("X", value=1.00, step=0.01, format="%.2f")
    q_2 = col_b3.number_input("2", value=1.00, step=0.01, format="%.2f")
    
    col_bu1, col_bu2 = st.columns(2)
    q_over25 = col_bu1.number_input("Over 2.5", value=1.00, step=0.01, format="%.2f")
    q_under25 = col_bu2.number_input("Under 2.5", value=1.00, step=0.01, format="%.2f")
    
    st.divider()
    w_cassa = st.number_input("Cassa Totale (€)", value=1000.0, step=10.0)

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
# 3. ANALISI COMPLETA
# ==========================================
if st.button("🚀 AVVIA ANALISI COMPLETA", type="primary"):
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

    c_h = 'GOALMINH' if 'GOALMINH' in df_league.columns else df_league.columns[0]
    c_a = 'GOALMINA' if 'GOALMINA' in df_league.columns else df_league.columns[0]

    # Accumulatori Stats
    goals_h = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    goals_a = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    match_h, match_a = 0, 0
    
    # Dati per KM (minuti 1° gol)
    times_h, times_a, times_league = [], [], []
    
    # Dati per Heatmap (tutti i gol)
    stats_match = {
        sel_home: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
        sel_away: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
    }
    
    for _, row in df_league.iterrows():
        h, a = row['CASA'], row['OSPITE']
        min_h = get_minutes(row.get(c_h))
        min_a = get_minutes(row.get(c_a))
        
        # Media Lega (1° gol)
        if min_h: times_league.append(min(min_h))
        if min_a: times_league.append(min(min_a))

        # Stats per Squadra
        if h == sel_home:
            match_h += 1
            goals_h['FT'] += len(min_h)
            goals_h['HT'] += len([x for x in min_h if x <= 45])
            goals_h['S_FT'] += len(min_a)
            goals_h['S_HT'] += len([x for x in min_a if x <= 45])
            if min_h: times_h.append(min(min_h))
            
            # Heatmap
            for m in min_h:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                stats_match[h]['F'][intervals[idx]] += 1
            for m in min_a:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                stats_match[h]['S'][intervals[idx]] += 1

        if a == sel_away:
            match_a += 1
            goals_a['FT'] += len(min_a)
            goals_a['HT'] += len([x for x in min_a if x <= 45])
            goals_a['S_FT'] += len(min_h)
            goals_a['S_HT'] += len([x for x in min_h if x <= 45])
            if min_a: times_a.append(min(min_a))

            # Heatmap
            for m in min_a:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                stats_match[a]['F'][intervals[idx]] += 1
            for m in min_h:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                stats_match[a]['S'][intervals[idx]] += 1

    def safe_div(n, d): return n / d if d > 0 else 0

    # Medie
    avg_h_ft = safe_div(goals_h['FT'], match_h)
    avg_h_ht = safe_div(goals_h['HT'], match_h)
    avg_h_conc = safe_div(goals_h['S_FT'], match_h)
    avg_h_conc_ht = safe_div(goals_h['S_HT'], match_h)

    avg_a_ft = safe_div(goals_a['FT'], match_a)
    avg_a_ht = safe_div(goals_a['HT'], match_a)
    avg_a_conc = safe_div(goals_a['S_FT'], match_a)
    avg_a_conc_ht = safe_div(goals_a['S_HT'], match_a)

    # Lambda Poisson
    exp_h = (avg_h_ft + avg_a_conc) / 2
    exp_a = (avg_a_ft + avg_h_conc) / 2
    exp_h_ht = (avg_h_ht + avg_a_conc_ht) / 2
    exp_a_ht = (avg_a_ht + avg_h_conc_ht) / 2

    # Calcolo Probabilità
    def calc_probs(lh, la):
        probs = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                probs[i][j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
        p1 = np.sum(np.tril(probs, -1))
        px = np.sum(np.diag(probs))
        p2 = np.sum(np.triu(probs, 1))
        return p1, px, p2, probs

    p1, px, p2, matrix_ft = calc_probs(exp_h, exp_a)
    _, _, _, matrix_ht = calc_probs(exp_h_ht, exp_a_ht)
    
    # O/U 2.5 FT
    pu25 = 0
    for i in range(6):
        for j in range(6):
            if i+j <= 2: pu25 += matrix_ft[i][j]
    po25 = 1 - pu25

    # HT Specifics
    prob_00_ht = matrix_ht[0][0]
    prob_u15_ht = matrix_ht[0][0] + matrix_ht[1][0] + matrix_ht[0][1]

    def to_odd(p): return 1/p if p > 0 else 99.00
    
    # Kelly Criterion
    def calc_kelly(prob, quota, bankroll):
        if prob <= 0 or quota <= 1: return 0, 0
        b = quota - 1
        f = (b * prob - (1 - prob)) / b
        stake_pct = max(0, f * 0.3) # Kelly 30%
        return stake_pct * 100, bankroll * stake_pct

    # --- DISPLAY ANALISI VALORE ---
    def show_card(title, prob, quote_book):
        odd_real = to_odd(prob)
        valore = (prob * quote_book) - 1
        pct, eur = calc_kelly(prob, quote_book, w_cassa)
        
        color = "green" if valore > 0 else "red"
        icon = "✅ VALUE" if valore > 0 else "❌ NO"
        
        st.markdown(f"""
        <div style="border:1px solid #444; padding:15px; border-radius:8px; margin-bottom:10px;">
            <h4 style="margin:0;">{title}</h4>
            <hr style="margin:5px 0;">
            <div>Prob. Reale: <b>{prob*100:.1f}%</b> (Q. {odd_real:.2f})</div>
            <div>Quota Book: <b>{quote_book:.2f}</b></div>
            <div style="color:{color}; font-weight:bold; margin-top:5px;">
                {icon} (ROI {valore*100:.1f}%)
            </div>
            {f"<div style='color:#00FF00'>💰 Puntata: € {eur:.2f}</div>" if valore > 0 else ""}
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📊 Analisi Valore (Money Management)")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a: show_card(f"Vittoria {sel_home} (1)", p1, q_1)
    with col_b: show_card("Pareggio (X)", px, q_x)
    with col_c: show_card(f"Vittoria {sel_away} (2)", p2, q_2)
    
    col_d, col_e = st.columns(2)
    with col_d: show_card("Over 2.5 FT", po25, q_over25)
    with col_e: show_card("Under 2.5 FT", pu25, q_under25)

    st.divider()

    # --- DISPLAY STATS ---
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**🏠 {sel_home}** ({match_h} match)\n\n"
                f"**1°T:** {avg_h_ht:.2f} F | {avg_h_conc_ht:.2f} S\n\n"
                f"**FIN:** {avg_h_ft:.2f} F | {avg_h_conc:.2f} S")
        # Minuto 1° Gol
        if times_h:
             st.write(f"⏱️ Minuto medio 1° gol: **{int(np.mean(times_h))}'**")

    with c2:
        st.warning(f"**✈️ {sel_away}** ({match_a} match)\n\n"
                 f"**1°T:** {avg_a_ht:.2f} F | {avg_a_conc_ht:.2f} S\n\n"
                 f"**FIN:** {avg_a_ft:.2f} F | {avg_a_conc:.2f} S")
        if times_a:
             st.write(f"⏱️ Minuto medio 1° gol: **{int(np.mean(times_a))}'**")

    st.write(f"**Speciale 1° Tempo:** 0-0 @{to_odd(prob_00_ht):.2f} ({prob_00_ht*100:.1f}%) | U1.5 @{to_odd(prob_u15_ht):.2f} ({prob_u15_ht*100:.1f}%)")

    # --- GRAFICI ---
    tab1, tab2, tab3 = st.tabs(["📉 Ritmo Gol (KM)", "⚽ Heatmap Fatti", "🛡️ Heatmap Subiti"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        kmf = KaplanMeierFitter()
        
        has_data = False
        if times_h:
            kmf.fit(times_h, label=f'{sel_home} (1° Gol)')
            kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue')
            has_data = True
        if times_a:
            kmf.fit(times_a, label=f'{sel_away} (1° Gol)')
            kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
            has_data = True
        if times_league:
            kmf.fit(times_league, label='Media Campionato')
            kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color='gray', linestyle='--')

        if has_data:
            plt.title('📉 Probabilità 0-0 nel tempo (Kaplan-Meier)')
            plt.axhline(0.5, color='green', linestyle=':', label='Mediana (50%)')
            plt.axvline(45, color='black', linestyle='--')
            plt.grid(True, alpha=0.3)
            plt.legend()
            st.pyplot(fig)
        else:
            st.warning("Dati insufficienti per il grafico KM.")

    # Heatmaps
    rows_f = []
    rows_s = []
    for t in [sel_home, sel_away]:
        d = stats_match[t]
        rows_f.append({**{'SQUADRA': t}, **d['F']})
        rows_s.append({**{'SQUADRA': t}, **d['S']})
    
    df_f = pd.DataFrame(rows_f).set_index('SQUADRA')
    df_s = pd.DataFrame(rows_s).set_index('SQUADRA')

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        sns.heatmap(df_f[intervals], annot=True, cmap="Greens", fmt="d", cbar=False, ax=ax)
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        sns.heatmap(df_s[intervals], annot=True, cmap="Reds", fmt="d", cbar=False, ax=ax)
        st.pyplot(fig)
