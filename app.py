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
st.set_page_config(page_title="⚽ Dashboard Pro V41 (Full)", layout="wide", page_icon="⚽")
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- TITOLO ---
st.title("💎 Dashboard Analisi Calcio V41 (Full)")
st.markdown("**Statistiche, Poisson, ELO, Ritmi di Gioco & Money Management**")
st.divider()

# ==========================================
# 1. CARICAMENTO DATI (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("📂 1. Dati")
    uploaded_file = st.file_uploader("Carica file (CSV/Excel)", type=['csv', 'xlsx'])
    
    default_file = 'eng_tot_1.csv' # O il nome che hai su GitHub
    file_to_use = uploaded_file if uploaded_file else (default_file if os.path.exists(default_file) else None)

    if file_to_use is None:
        st.warning("Carica un file per iniziare.")
        st.stop()

    st.divider()
    
    # SEZIONE QUOTE BOOKMAKER
    st.header("💰 2. Quote & Filtri")
    st.caption("Inserisci le quote reali:")
    
    col_b1, col_b2, col_b3 = st.columns(3)
    q_1 = col_b1.number_input("1", value=1.00, step=0.01, format="%.2f")
    q_x = col_b2.number_input("X", value=1.00, step=0.01, format="%.2f")
    q_2 = col_b3.number_input("2", value=1.00, step=0.01, format="%.2f")
    
    col_bu1, col_bu2 = st.columns(2)
    q_over25 = col_bu1.number_input("Over 2.5", value=1.00, step=0.01, format="%.2f")
    q_under25 = col_bu2.number_input("Under 2.5", value=1.00, step=0.01, format="%.2f")

    st.divider()
    w_cassa = st.number_input("Cassa Totale (€)", value=1000.0, step=10.0)
    w_soglia_trappola = st.slider("Soglia Allarme Trappola (%)", 10, 100, 30, 5)

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

        # Header e Pulizia
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
        
        # Mappatura Completa (incluso ELO)
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

st.sidebar.success(f"✅ {len(df)} righe caricate")

# Check ELO
has_elo = 'ELO_H' in df.columns and 'ELO_A' in df.columns

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
    
    # Dati per KM
    times_h, times_a, times_league = [], [], []
    
    # Dati per Heatmap
    stats_match = {
        sel_home: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
        sel_away: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
    }
    
    # Dati Elo
    elo_h_val = 1500; elo_a_val = 1500

    for _, row in df_league.iterrows():
        h, a = row['CASA'], row['OSPITE']
        min_h = get_minutes(row.get(c_h))
        min_a = get_minutes(row.get(c_a))
        
        # Recupero Elo
        if has_elo:
            if h == sel_home: elo_h_val = float(row['ELO_H'])
            if a == sel_away: elo_a_val = float(row['ELO_A'])
        
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
        pu25 = 0
        for i in range(6):
            for j in range(6):
                if i+j <= 2: pu25 += probs[i][j]
        return p1, px, p2, pu25, probs

    p1, px, p2, pu25, matrix_probs = calc_probs(exp_h, exp_a)
    
    # Probabilità HT
    prob_00_ht = poisson.pmf(0, exp_h_ht) * poisson.pmf(0, exp_a_ht)
    prob_u15_ht = prob_00_ht + (poisson.pmf(1, exp_h_ht) * poisson.pmf(0, exp_a_ht)) + (poisson.pmf(0, exp_h_ht) * poisson.pmf(1, exp_a_ht))

    def to_odd(p): return 1/p if p > 0 else 99.00

    # Kelly Criterion
    def calc_kelly(prob, quota, bankroll):
        if prob <= 0 or quota <= 1: return 0, 0
        b = quota - 1
        f = (b * prob - (1 - prob)) / b
        stake_pct = max(0, f * 0.3) # Kelly 30%
        return stake_pct * 100, bankroll * stake_pct

    # --- VISUALIZZAZIONE ---

    # 1. Statistiche Medie
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**🏠 {sel_home}** ({match_h} match)\n\n1°T: {avg_h_ht:.2f} F / {avg_h_conc_ht:.2f} S\n\nFIN: {avg_h_ft:.2f} F / {avg_h_conc:.2f} S")
    with c2:
        st.warning(f"**✈️ {sel_away}** ({match_a} match)\n\n1°T: {avg_a_ht:.2f} F / {avg_a_conc_ht:.2f} S\n\nFIN: {avg_a_ft:.2f} F / {avg_a_conc:.2f} S")

    # 2. Analisi Valore & Money Management
    st.subheader("💰 Caccia al Valore")
    quotes = {'1': q_1, 'X': q_x, '2': q_2, 'O 2.5': q_over25, 'U 2.5': q_under25}
    probs_real = {'1': p1, 'X': px, '2': p2, 'O 2.5': 1-pu25, 'U 2.5': pu25}
    
    cols = st.columns(3)
    for i, (segno, quota) in enumerate(quotes.items()):
        if quota > 1.0:
            prob = probs_real.get(segno, 0)
            valore = (prob * quota) - 1
            
            # Filtro Trappola
            fair = to_odd(prob)
            diff_pct = (quota - fair) / fair if fair > 0 else 0
            trappola = diff_pct > (w_soglia_trappola / 100.0)
            
            with cols[i % 3]:
                color = "green" if (valore > 0 and not trappola) else "red"
                icon = "✅" if (valore > 0 and not trappola) else "⚠️" if trappola else "❌"
                msg = "VALUE" if (valore > 0 and not trappola) else "TRAPPOLA" if trappola else "NO VALUE"
                
                st.markdown(f"""
                <div style="border:1px solid #444; padding:10px; border-radius:5px; margin-bottom:5px;">
                    <strong>{segno}</strong> (Q: {quota})<br>
                    Fair Odd: <b>{fair:.2f}</b><br>
                    <span style="color:{color}; font-weight:bold;">{icon} {msg}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if valore > 0 and not trappola:
                    pct, eur = calc_kelly(prob, quota, w_cassa)
                    st.caption(f"Puntata: € {eur:.2f} ({pct:.1f}%)")

    # 3. Probabilità HT
    st.write("---")
    c_ht1, c_ht2 = st.columns(2)
    c_ht1.metric("Prob. 0-0 HT", f"{prob_00_ht*100:.1f}%", f"Fair: @{to_odd(prob_00_ht):.2f}")
    c_ht2.metric("Prob. U 1.5 HT", f"{prob_u15_ht*100:.1f}%", f"Fair: @{to_odd(prob_u15_ht):.2f}")

    # 4. ELO Analysis (se disponibile)
    if has_elo:
        st.subheader("👑 Analisi ELO")
        e1, e2, e3 = st.columns(3)
        e1.metric(f"Elo {sel_home}", int(elo_h_val))
        e2.metric(f"Elo {sel_away}", int(elo_a_val))
        diff_elo = (elo_h_val + 100) - elo_a_val # +100 Home Adv
        prob_elo = 1 / (1 + 10 ** (-diff_elo / 400))
        e3.metric("Prob. ELO Casa", f"{prob_elo*100:.1f}%", f"Fair: @{to_odd(prob_elo):.2f}")

    st.divider()

    # --- GRAFICI ---
    tab1, tab2, tab3 = st.tabs(["📉 Ritmo Gol (KM)", "⚽ Heatmap Densità", "🎯 Poisson Matrix"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        kmf = KaplanMeierFitter()
        
        if times_h and times_a:
            kmf.fit(times_h, label=f'{sel_home}')
            kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue')
            kmf.fit(times_a, label=f'{sel_away}')
            kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
            
            if len(times_league) > 10:
                kmf.fit(times_league, label='Media Lega')
                kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color='gray', linestyle='--')

            plt.title('Ritmo Gol (Probabilità 0-0)')
            plt.axhline(0.5, color='green', linestyle=':', label='Mediana')
            plt.axvline(45, color='black', linestyle='--')
            plt.grid(True, alpha=0.3)
            plt.legend()
            st.pyplot(fig)
        else:
            st.warning("Dati insufficienti per KM.")

    with tab2:
        # Heatmap H2H
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
        axes[0].set_title('Densità Gol FATTI')
        sns.heatmap(df_s[intervals], annot=True, cmap="Reds", fmt="d", cbar=False, ax=axes[1])
        axes[1].set_title('Densità Gol SUBITI')
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix_probs, annot=True, fmt=".1%", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel(f"Gol {sel_away}")
        ax.set_ylabel(f"Gol {sel_home}")
        ax.set_title("Probabilità Risultato Esatto")
        st.pyplot(fig)
