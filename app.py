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
st.set_page_config(page_title="⚽ Dashboard Pro V42", layout="wide", page_icon="⚽")
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- TITOLO ---
st.title("💎 Dashboard Analisi Calcio V42 (Full + ELO Manuale)")
st.markdown("**Statistiche, Poisson, ELO User vs Storico, Ritmi & Money Management**")
st.divider()

# ==========================================
# 1. SIDEBAR: DATI & INPUT UTENTE
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
    
    # SEZIONE ELO MANUALE (NUOVO)
    st.header("👑 2. ELO Rating (Manuale)")
    st.caption("Inserisci l'ELO attuale delle squadre per il confronto:")
    col_elo1, col_elo2 = st.columns(2)
    user_elo_h = col_elo1.number_input("ELO Casa", value=1500, step=10)
    user_elo_a = col_elo2.number_input("ELO Ospite", value=1500, step=10)

    st.divider()

    # SEZIONE QUOTE (MONEY MANAGEMENT)
    st.header("💰 3. Quote Bookmaker")
    
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
        
        # Mappatura Completa (ELO incluso)
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

# Check ELO nel file
has_elo_file = 'ELO_H' in df.columns and 'ELO_A' in df.columns

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
    
    # Liste Minuti per KM e Media 1° Gol
    times_h_all, times_a_all = [], [] # Tutti i gol
    first_goal_h, first_goal_a = [], [] # Solo il primo gol per partita
    times_league = []
    
    # Dati per Heatmap
    stats_match = {
        sel_home: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
        sel_away: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
    }
    
    # Dati Elo Storico (Media)
    hist_elo_h = []
    hist_elo_a = []

    for _, row in df_league.iterrows():
        h, a = row['CASA'], row['OSPITE']
        min_h = get_minutes(row.get(c_h))
        min_a = get_minutes(row.get(c_a))
        
        # ELO Storico
        if has_elo_file:
            if h == sel_home: hist_elo_h.append(float(row['ELO_H']))
            if a == sel_away: hist_elo_a.append(float(row['ELO_A']))

        # Media Lega (1° gol)
        if min_h: times_league.append(min(min_h))
        if min_a: times_league.append(min(min_a))

        # --- Stats Squadra CASA ---
        if h == sel_home:
            match_h += 1
            goals_h['FT'] += len(min_h)
            goals_h['HT'] += len([x for x in min_h if x <= 45])
            goals_h['S_FT'] += len(min_a)
            goals_h['S_HT'] += len([x for x in min_a if x <= 45])
            
            times_h_all.extend(min_h)
            if min_h: first_goal_h.append(min(min_h))
            
            # Heatmap
            for m in min_h:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                stats_match[h]['F'][intervals[idx]] += 1
            for m in min_a:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                stats_match[h]['S'][intervals[idx]] += 1

        # --- Stats Squadra OSPITE ---
        if a == sel_away:
            match_a += 1
            goals_a['FT'] += len(min_a)
            goals_a['HT'] += len([x for x in min_a if x <= 45])
            goals_a['S_FT'] += len(min_h)
            goals_a['S_HT'] += len([x for x in min_h if x <= 45])
            
            times_a_all.extend(min_a)
            if min_a: first_goal_a.append(min(min_a))

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
    avg_min_h = safe_mean(first_goal_h) # Minuto medio 1° gol

    avg_a_ft = safe_div(goals_a['FT'], match_a)
    avg_a_ht = safe_div(goals_a['HT'], match_a)
    avg_a_conc_ft = safe_div(goals_a['S_FT'], match_a)
    avg_a_conc_ht = safe_div(goals_a['S_HT'], match_a)
    avg_min_a = safe_mean(first_goal_a) # Minuto medio 1° gol

    # --- 4. VISUALIZZAZIONE DATI ---
    
    # A. Statistiche Medie (Con Minuto 1° Gol ripristinato!)
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

    # --- 5. ANALISI ELO & POISSON ---
    st.subheader("⚖️ Confronto Modelli: Poisson vs ELO")

    # Poisson
    exp_h = (avg_h_ft + avg_a_conc_ft) / 2
    exp_a = (avg_a_ft + avg_h_conc_ft) / 2
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

    p1_ft, px_ft, p2_ft, pu25_ft = calc_poisson_probs(exp_h, exp_a)
    
    # HT Specifics
    prob_00_ht = poisson.pmf(0, exp_h_ht) * poisson.pmf(0, exp_a_ht)
    prob_u15_ht = prob_00_ht + (poisson.pmf(1, exp_h_ht) * poisson.pmf(0, exp_a_ht)) + (poisson.pmf(0, exp_h_ht) * poisson.pmf(1, exp_a_ht))

    # ELO Calculation
    HOME_ADVANTAGE = 100
    dr = (user_elo_h + HOME_ADVANTAGE) - user_elo_a
    prob_elo_h = 1 / (1 + 10 ** (-dr / 400))
    prob_elo_a = 1 - prob_elo_h
    
    # Tabella Confronto
    col_mod1, col_mod2, col_mod3 = st.columns(3)
    
    def to_odd(p): return round(1/p, 2) if p > 0 else 99.00

    with col_mod1:
        st.markdown("#### 🎲 Poisson (Forma)")
        st.write(f"1: **{p1_ft*100:.1f}%** (@{to_odd(p1_ft)})")
        st.write(f"2: **{p2_ft*100:.1f}%** (@{to_odd(p2_ft)})")
        st.caption(f"0-0 HT: {prob_00_ht*100:.1f}%")
    
    with col_mod2:
        st.markdown("#### 👑 ELO User (Forza)")
        st.write(f"1: **{prob_elo_h*100:.1f}%** (@{to_odd(prob_elo_h)})")
        st.write(f"2: **{prob_elo_a*100:.1f}%** (@{to_odd(prob_elo_a)})")
        st.caption(f"Delta ELO: {user_elo_h - user_elo_a}")

    with col_mod3:
        st.markdown("#### 📜 ELO Storico (File)")
        if has_elo_file:
            avg_hist_h = safe_mean(hist_elo_h)
            avg_hist_a = safe_mean(hist_elo_a)
            st.write(f"Media Casa: {int(avg_hist_h)}")
            st.write(f"Media Ospite: {int(avg_hist_a)}")
            
            # Check coerenza
            diff_user = user_elo_h - user_elo_a
            diff_hist = avg_hist_h - avg_hist_a
            if abs(diff_user - diff_hist) > 100:
                st.error("⚠️ L'ELO inserito diverge molto dallo storico!")
            else:
                st.success("✅ ELO inserito coerente con storico.")
        else:
            st.write("Dati storici non disponibili.")

    # --- MONEY MANAGEMENT ---
    st.divider()
    st.subheader("💰 Caccia al Valore (Money Management)")
    
    def calc_kelly(prob, quota, bankroll):
        if prob <= 0 or quota <= 1: return 0, 0
        b = quota - 1
        f = (b * prob - (1 - prob)) / b
        stake_pct = max(0, f * 0.3)
        return stake_pct * 100, bankroll * stake_pct

    # Usiamo una media ponderata tra Poisson ed ELO per il calcolo valore?
    # Per ora usiamo Poisson che è più reattivo alla forma recente per il betting
    
    quotes = {'1': q_1, 'X': q_x, '2': q_2, 'O 2.5': q_over25, 'U 2.5': q_under25}
    probs_real = {'1': p1_ft, 'X': px_ft, '2': p2_ft, 'O 2.5': 1-pu25_ft, 'U 2.5': pu25_ft}
    
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
                icon = "✅ VALUE" if (valore > 0 and not trappola) else "⚠️ TRAP" if trappola else "❌ NO"
                
                st.markdown(f"""
                <div style="border:1px solid #555; padding:10px; border-radius:5px; margin-bottom:10px;">
                    <strong>{segno}</strong> (Q: {quota:.2f})<br>
                    Fair: <b>{fair:.2f}</b> | Prob: {prob*100:.1f}%<br>
                    <span style="color:{color}; font-weight:bold;">{icon}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if valore > 0 and not trappola:
                    pct, eur = calc_kelly(prob, quota, w_cassa)
                    st.caption(f"Punta: € {eur:.2f} ({pct:.1f}%)")

    st.divider()

    # --- GRAFICI ---
    tab1, tab2, tab3 = st.tabs(["📉 Ritmo Gol (KM)", "⚽ Heatmap Fatti", "🛡️ Heatmap Subiti"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        kmf_h = KaplanMeierFitter()
        kmf_a = KaplanMeierFitter()
        
        if first_goal_h and first_goal_a: # Uso i dati del primo gol salvati prima
            kmf_h.fit(first_goal_h, label=f'{sel_home} (1° Gol)')
            kmf_a.fit(first_goal_a, label=f'{sel_away} (1° Gol)')
            
            kmf_h.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue')
            kmf_a.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
            
            # Media Lega
            if times_league:
                kmf_l = KaplanMeierFitter()
                kmf_l.fit(times_league, label='Media Lega')
                kmf_l.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color='gray', linestyle='--')

            plt.axhline(y=0.5, color='green', linestyle=':', label='Mediana (50%)')
            plt.title('Tempo al 1° Gol (Probabilità 0-0)')
            plt.grid(True, alpha=0.3)
            plt.axvline(45, color='green', linestyle='--')
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
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(df_f[intervals], annot=True, cmap="Greens", fmt="d", cbar=False, ax=ax)
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(df_s[intervals], annot=True, cmap="Reds", fmt="d", cbar=False, ax=ax)
        st.pyplot(fig)
