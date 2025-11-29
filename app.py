import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from scipy.stats import poisson
import warnings

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="⚽ Dashboard Tattica V33", layout="wide", page_icon="⚽")
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- TITOLO ---
st.title("💎 Dashboard Analisi Calcio V33")
st.markdown("**Versione Stabile: Memoria di Sessione Attiva**")
st.divider()

# ==========================================
# 1. FUNZIONI DI CARICAMENTO (Con Cache)
# ==========================================
@st.cache_data
def load_data(file):
    try:
        # Tenta lettura robusta del separatore
        try:
            # Legge come testo per inferire il separatore
            content = file.getvalue().decode('latin1')
            first_line = content.split('\n')[0]
            sep = ';' if first_line.count(';') > first_line.count(',') else ','
            file.seek(0) # Reset pointer
            df = pd.read_csv(file, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False)
        except:
            file.seek(0)
            df = pd.read_excel(file)

        # --- PULIZIA COLONNE ---
        df.columns = df.columns.astype(str).str.strip().str.upper()
        df = df.loc[:, ~df.columns.duplicated()]

        # Mappatura Nomi Standard
        col_map = {
            'GOALMINH': ['GOALMINH', 'GOALMINCASA', 'MINUTI_CASA', 'GOALMIN_H'],
            'GOALMINA': ['GOALMINA', 'GOALMINOSPITE', 'MINUTI_OSPITE', 'GOALMIN_A'],
            'LEGA': ['LEGA', 'LEAGUE', 'DIVISION'],
            'PAESE': ['PAESE', 'COUNTRY', 'NATION'],
            'CASA': ['CASA', 'HOME', 'TEAM1'],
            'OSPITE': ['OSPITE', 'AWAY', 'TEAM2']
        }
        
        for target, candidates in col_map.items():
            if target not in df.columns:
                for candidate in candidates:
                    # Cerca anche versioni upper/lower
                    found = next((c for c in df.columns if c == candidate.upper()), None)
                    if found:
                        df.rename(columns={found: target}, inplace=True)
                        break
        
        # Verifica colonne minime
        required = ['LEGA', 'CASA', 'OSPITE']
        if not all(c in df.columns for c in required):
            return None # Segnala errore

        # Pulizia celle stringa
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
        return None

# ==========================================
# 2. SIDEBAR & GESTIONE STATO
# ==========================================
with st.sidebar:
    st.header("📂 Dati")
    uploaded_file = st.file_uploader("Carica file (CSV/Excel)", type=['csv', 'xlsx'])

# Inizializza session state per i dati se non esiste
if 'df_main' not in st.session_state:
    st.session_state.df_main = None

# Se l'utente carica un file, aggiorna lo stato
if uploaded_file is not None:
    df_loaded = load_data(uploaded_file)
    if df_loaded is not None:
        st.session_state.df_main = df_loaded
        st.success(f"✅ Dati caricati: {len(df_loaded)} righe")
    else:
        st.error("❌ Errore nel file: Colonne mancanti o formato errato.")

# Se non ci sono dati, ferma tutto
if st.session_state.df_main is None:
    st.info("👈 Carica un file dalla barra laterale per iniziare.")
    st.stop()

df = st.session_state.df_main

# ==========================================
# 3. FILTRI A CASCATA (Senza ricaricamento)
# ==========================================
col1, col2, col3 = st.columns(3)

# 1. Campionato
with col1:
    leghe_disponibili = sorted(df['ID_LEGA'].unique())
    sel_lega = st.selectbox("🏆 Seleziona Campionato", leghe_disponibili)

# Filtra dataset (questo è veloce perché è in memoria)
df_league = df[df['ID_LEGA'] == sel_lega].copy()

# 2. Squadre
teams = sorted(pd.concat([df_league['CASA'], df_league['OSPITE']]).unique())

with col2:
    sel_home = st.selectbox("🏠 Squadra Casa", teams, index=0 if len(teams)>0 else None)

with col3:
    # Cerca di selezionare una diversa dalla home
    default_idx = 1 if len(teams) > 1 else 0
    sel_away = st.selectbox("✈️ Squadra Ospite", teams, index=default_idx)

# ==========================================
# 4. ENGINE DI ANALISI
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
        s = str(val).replace(',', '.').replace(';', ' ').replace('"', '').replace("'", "")
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        res = []
        for x in nums:
            try:
                n = int(float(x))
                if 0 <= n <= 130: res.append(n)
            except: pass
        return res

    # Colonne minuti
    c_h = 'GOALMINH' if 'GOALMINH' in df.columns else 'GOALMINCASA'
    c_a = 'GOALMINA' if 'GOALMINA' in df.columns else 'GOALMINOSPITE'
    
    if c_h not in df.columns: # Fallback estremo
        st.error("Colonne minuti gol non trovate nel file.")
        st.stop()

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
        min_h = get_minutes(row.get(c_h))
        min_a = get_minutes(row.get(c_a))
        
        # Media Lega
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

        # Stats Medie (Casa vs Fuori)
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

    # Calcoli
    def safe_div(n, d): return n / d if d > 0 else 0

    avg_h_ft = safe_div(goals_h['FT'], match_h)
    avg_h_ht = safe_div(goals_h['HT'], match_h)
    avg_h_conc_ft = safe_div(goals_h['S_FT'], match_h)
    avg_h_conc_ht = safe_div(goals_h['S_HT'], match_h)

    avg_a_ft = safe_div(goals_a['FT'], match_a)
    avg_a_ht = safe_div(goals_a['HT'], match_a)
    avg_a_conc_ft = safe_div(goals_a['S_FT'], match_a)
    avg_a_conc_ht = safe_div(goals_a['S_HT'], match_a)

    # Visualizzazione Medie
    st.write("### 📊 Statistiche Medie")
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**🏠 {sel_home}** ({match_h} match)\n\n"
                f"**1°T:** F {avg_h_ht:.2f} | S {avg_h_conc_ht:.2f}\n\n"
                f"**FIN:** F {avg_h_ft:.2f} | S {avg_h_conc_ft:.2f}")
    with c2:
        st.warning(f"**✈️ {sel_away}** ({match_a} match)\n\n"
                 f"**1°T:** F {avg_a_ht:.2f} | S {avg_a_conc_ht:.2f}\n\n"
                 f"**FIN:** F {avg_a_ft:.2f} | S {avg_a_conc_ft:.2f}")

    # Poisson
    exp_h_ft = (avg_h_ft + avg_a_conc_ft) / 2
    exp_a_ft = (avg_a_ft + avg_h_conc_ft) / 2
    exp_h_ht = (avg_h_ht + avg_a_conc_ht) / 2
    exp_a_ht = (avg_a_ht + avg_h_conc_ht) / 2

    def calc_poisson(lam_h, lam_a):
        probs = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                probs[i][j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
        return np.sum(np.tril(probs, -1)), np.sum(np.diag(probs)), np.sum(np.triu(probs, 1))

    p1_ft, px_ft, p2_ft = calc_poisson(exp_h_ft, exp_a_ft)
    
    prob_00_ht = poisson.pmf(0, exp_h_ht) * poisson.pmf(0, exp_a_ht)
    prob_u15_ht = prob_00_ht + (poisson.pmf(1, exp_h_ht) * poisson.pmf(0, exp_a_ht)) + (poisson.pmf(0, exp_h_ht) * poisson.pmf(1, exp_a_ht))

    def to_odd(p): return round(1/p, 2) if p > 0 else 99.00

    st.divider()
    st.subheader("🎲 Previsioni & Quote Implicite")
    k1, k2, k3 = st.columns(3)
    k1.metric("1X2 Finale", f"1: {p1_ft*100:.0f}%", f"Quota: {to_odd(p1_ft)}")
    k2.metric("1X2 Finale", f"X: {px_ft*100:.0f}%", f"Quota: {to_odd(px_ft)}")
    k3.metric("1X2 Finale", f"2: {p2_ft*100:.0f}%", f"Quota: {to_odd(p2_ft)}")
    
    st.write(f"**Speciale 1° Tempo:** 0-0 ({prob_00_ht*100:.1f}% | @{to_odd(prob_00_ht)}) — Under 1.5 ({prob_u15_ht*100:.1f}% | @{to_odd(prob_u15_ht)})")

    st.divider()

    # Grafici
    tab1, tab2, tab3 = st.tabs(["📉 Ritmo Gol (KM)", "⚽ Heatmap Fatti", "🛡️ Heatmap Subiti"])

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
            
            plt.axhline(y=0.5, color='green', linestyle=':', label='Mediana (50%)')
            plt.title('Tempo al 1° Gol')
            plt.grid(True, alpha=0.3)
            plt.legend()
            st.pyplot(fig)
        else:
            st.warning("Dati insufficienti per Kaplan-Meier (0 gol segnati in casa/trasferta).")

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
