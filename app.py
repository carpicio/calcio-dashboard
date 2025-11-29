import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from scipy.stats import poisson
import warnings
import os
import re

# Configurazione
st.set_page_config(page_title="⚽ Dashboard V-Final", layout="wide")
warnings.simplefilter(action='ignore', category=FutureWarning)

st.title("🛡️ Dashboard Calcio: Versione Blindata")
st.markdown("Analisi Statistica e Predittiva - Rilevamento Automatico Dati")

# ==========================================
# 1. CARICAMENTO UNIVERSALE
# ==========================================
with st.sidebar:
    st.header("1. Dati")
    uploaded_file = st.file_uploader("Trascina qui il tuo file (CSV o Excel)", type=['csv', 'xlsx'])
    
    # Cerca file locale se non caricato (per debug locale o GitHub)
    if uploaded_file is None:
        files_locali = [f for f in os.listdir() if f.endswith(('.csv', '.xlsx')) and 'requirements' not in f]
        if files_locali:
            st.info(f"Trovato file locale: {files_locali[0]}")
            # Opzionale: caricamento automatico file locale per test
            # uploaded_file = files_locali[0] 

if uploaded_file is None:
    st.warning("👈 Per favore, carica il file CSV o Excel dalla barra a sinistra.")
    st.stop()

@st.cache_data
def load_data_super_safe(file):
    # 1. Tenta lettura CSV con vari separatori
    separators = [';', ',', '\t']
    df = None
    
    # Se è un file caricato da Streamlit
    if hasattr(file, 'read'):
        file.seek(0)
        try:
            # Prova Excel
            df = pd.read_excel(file)
        except:
            # Prova CSV
            for sep in separators:
                file.seek(0)
                try:
                    df = pd.read_csv(file, sep=sep, encoding='latin1', on_bad_lines='skip')
                    # Se ha creato almeno 5 colonne, è probabilmente giusto
                    if df.shape[1] > 4:
                        break
                except:
                    continue
    
    # 2. Pulizia Nomi Colonne
    if df is not None:
        df.columns = [str(c).strip().upper() for c in df.columns]
        # Rimuovi colonne duplicate
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Pulizia Dati (Stringhe)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            
    return df

df = load_data_super_safe(uploaded_file)

if df is None or df.empty:
    st.error("❌ Non sono riuscito a leggere il file. Assicurati che sia un CSV o Excel valido.")
    st.stop()

st.success(f"✅ File caricato! Righe: {len(df)} | Colonne: {len(df.columns)}")

# ==========================================
# 2. MAPPATURA INTELLIGENTE (AUTO + MANUALE)
# ==========================================
st.subheader("🔧 Verifica Colonne")
col1, col2, col3 = st.columns(3)

all_cols = list(df.columns)

# Funzione per trovare la colonna migliore
def find_best_match(keywords, columns):
    for col in columns:
        for k in keywords:
            if k in col: return col
    return columns[0] if columns else None

# Auto-rilevamento
default_paese = find_best_match(['PAESE', 'COUNTRY', 'NAT'], all_cols)
default_lega = find_best_match(['LEGA', 'LEAGUE', 'DIV'], all_cols)
default_casa = find_best_match(['CASA', 'HOME', 'TEAM1', 'TXTECHIPA1'], all_cols)
default_ospite = find_best_match(['OSPITE', 'AWAY', 'TEAM2', 'TXTECHIPA2'], all_cols)
default_minH = find_best_match(['GOALMINH', 'GOALMINCASA', 'MINH', 'GOALSH'], all_cols)
default_minA = find_best_match(['GOALMINA', 'GOALMINOSPITE', 'MINA', 'GOALSA'], all_cols)

# Widget di conferma (L'utente può correggere se l'auto-rilevamento sbaglia)
with col1:
    c_paese = st.selectbox("Colonna PAESE", all_cols, index=all_cols.index(default_paese) if default_paese in all_cols else 0)
    c_lega = st.selectbox("Colonna LEGA", all_cols, index=all_cols.index(default_lega) if default_lega in all_cols else 0)
with col2:
    c_casa = st.selectbox("Colonna CASA", all_cols, index=all_cols.index(default_casa) if default_casa in all_cols else 0)
    c_ospite = st.selectbox("Colonna OSPITE", all_cols, index=all_cols.index(default_ospite) if default_ospite in all_cols else 0)
with col3:
    c_minH = st.selectbox("Minuti Gol CASA", all_cols, index=all_cols.index(default_minH) if default_minH in all_cols else 0)
    c_minA = st.selectbox("Minuti Gol OSPITE", all_cols, index=all_cols.index(default_minA) if default_minA in all_cols else 0)

# ==========================================
# 3. FILTRI & ANALISI
# ==========================================
st.divider()

# Filtro Campionato
paesi_disp = sorted(df[c_paese].unique())
sel_paese = st.selectbox("Seleziona Paese", paesi_disp)

leghe_disp = sorted(df[df[c_paese] == sel_paese][c_lega].unique())
sel_lega = st.selectbox("Seleziona Campionato", leghe_disp)

# Filtro Squadre
df_league = df[(df[c_paese] == sel_paese) & (df[c_lega] == sel_lega)].copy()
teams = sorted(pd.concat([df_league[c_casa], df_league[c_ospite]]).unique())

c_sel1, c_sel2 = st.columns(2)
sel_home = c_sel1.selectbox("🏠 Squadra Casa", teams, index=0)
sel_away = c_sel2.selectbox("✈️ Squadra Ospite", teams, index=1 if len(teams)>1 else 0)

# Quote (Opzionali)
with st.expander("💰 Inserisci Quote (Opzionale)"):
    qc1, qc2, qc3 = st.columns(3)
    q1 = qc1.number_input("Quota 1", 1.0)
    qx = qc2.number_input("Quota X", 1.0)
    q2 = qc3.number_input("Quota 2", 1.0)

if st.button("🚀 AVVIA ANALISI", type="primary"):
    
    # --- PREPARAZIONE DATI ---
    intervals = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
    
    def parse_mins(val):
        if pd.isna(val): return []
        # Tritatutto universale per numeri
        s = str(val).replace(',', ' ').replace(';', ' ').replace('.', ' ').replace('"', '').replace("'", "")
        nums = re.findall(r'\d+', s)
        res = []
        for x in nums:
            try:
                n = int(x)
                if 0 <= n <= 130: res.append(n)
            except: pass
        return res

    # Accumulatori
    goals_h = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    goals_a = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    match_h, match_a = 0, 0
    times_h, times_a, times_league = [], [], []
    
    # Heatmap
    hm_f = {sel_home: {i:0 for i in intervals}, sel_away: {i:0 for i in intervals}}
    hm_s = {sel_home: {i:0 for i in intervals}, sel_away: {i:0 for i in intervals}}

    for _, row in df_league.iterrows():
        h, a = row[c_casa], row[c_ospite]
        mins_h = parse_mins(row.get(c_minH))
        mins_a = parse_mins(row.get(c_minA))
        
        # Media Lega
        if mins_h: times_league.append(min(mins_h))
        if mins_a: times_league.append(min(mins_a))

        # Heatmaps
        if h in [sel_home, sel_away]:
            target = h
            for m in mins_h:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                hm_f[target][intervals[idx]] += 1
            for m in mins_a:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                hm_s[target][intervals[idx]] += 1 # Subiti

        if a in [sel_home, sel_away]:
            target = a
            for m in mins_a:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                hm_f[target][intervals[idx]] += 1
            for m in mins_h:
                idx = min(5, (m-1)//15); idx = 3 if (m>45 and m<=60 and idx<3) else idx
                hm_s[target][intervals[idx]] += 1 # Subiti

        # Stats & KM
        if h == sel_home:
            match_h += 1
            goals_h['FT'] += len(mins_h)
            goals_h['HT'] += len([x for x in mins_h if x <= 45])
            goals_h['S_FT'] += len(mins_a)
            goals_h['S_HT'] += len([x for x in mins_a if x <= 45])
            if mins_h: times_h.append(min(mins_h))
            
        if a == sel_away:
            match_a += 1
            goals_a['FT'] += len(mins_a)
            goals_a['HT'] += len([x for x in mins_a if x <= 45])
            goals_a['S_FT'] += len(mins_h)
            goals_a['S_HT'] += len([x for x in mins_h if x <= 45])
            if mins_a: times_a.append(min(mins_a))

    # --- OUTPUT ---
    
    # 1. Medie
    def safe_div(n, d): return n/d if d > 0 else 0
    
    avg_h_ft = safe_div(goals_h['FT'], match_h)
    avg_h_ht = safe_div(goals_h['HT'], match_h)
    avg_h_s_ft = safe_div(goals_h['S_FT'], match_h)
    avg_h_s_ht = safe_div(goals_h['S_HT'], match_h)
    
    avg_a_ft = safe_div(goals_a['FT'], match_a)
    avg_a_ht = safe_div(goals_a['HT'], match_a)
    avg_a_s_ft = safe_div(goals_a['S_FT'], match_a)
    avg_a_s_ht = safe_div(goals_a['S_HT'], match_a)
    
    st.subheader("📊 Analisi Statistica")
    c1, c2 = st.columns(2)
    c1.info(f"**🏠 {sel_home}** ({match_h} match)\n\n**1°T:** F {avg_h_ht:.2f} | S {avg_h_s_ht:.2f}\n\n**FIN:** F {avg_h_ft:.2f} | S {avg_h_s_ft:.2f}")
    c2.warning(f"**✈️ {sel_away}** ({match_a} match)\n\n**1°T:** F {avg_a_ht:.2f} | S {avg_a_s_ht:.2f}\n\n**FIN:** F {avg_a_ft:.2f} | S {avg_a_s_ft:.2f}")

    # 2. Poisson
    lam_h = (avg_h_ft + avg_a_s_ft) / 2
    lam_a = (avg_a_ft + avg_h_s_ft) / 2
    
    probs = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            probs[i][j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            
    p1 = np.sum(np.tril(probs, -1))
    px = np.sum(np.diag(probs))
    p2 = np.sum(np.triu(probs, 1))
    
    st.subheader("🎲 Previsioni (Poisson)")
    m1, m2, m3 = st.columns(3)
    m1.metric("1 (Casa)", f"{p1*100:.1f}%", f"Q. Reale: {1/p1:.2f}" if p1>0 else "")
    m2.metric("X (Pareggio)", f"{px*100:.1f}%", f"Q. Reale: {1/px:.2f}" if px>0 else "")
    m3.metric("2 (Ospite)", f"{p2*100:.1f}%", f"Q. Reale: {1/p2:.2f}" if p2>0 else "")

    # 3. Grafici
    st.divider()
    t1, t2, t3 = st.tabs(["📉 Ritmo Gol (KM)", "⚽ Heatmap Fatti", "🛡️ Heatmap Subiti"])
    
    with t1:
        fig, ax = plt.subplots(figsize=(10, 5))
        kmf = KaplanMeierFitter()
        if times_h and times_a:
            kmf.fit(times_h, label=f'{sel_home}')
            kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue')
            kmf.fit(times_a, label=f'{sel_away}')
            kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
            
            if len(times_league) > 10:
                kmf.fit(times_league, label="Media Lega")
                kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color='gray', linestyle='--')
                
            plt.title("Tempo al 1° Gol (Probabilità 0-0)")
            plt.axhline(0.5, color='green', linestyle=':')
            plt.axvline(45, color='black', linestyle='--')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("Dati insufficienti per il grafico del Ritmo.")
            
    # Prepare Heatmaps
    df_f = pd.DataFrame([hm_f[sel_home], hm_f[sel_away]], index=[sel_home, sel_away], columns=intervals)
    df_s = pd.DataFrame([hm_s[sel_home], hm_s[sel_away]], index=[sel_home, sel_away], columns=intervals)

    with t2:
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(df_f, annot=True, cmap="Greens", fmt="d", cbar=False)
        plt.title("Densità Gol FATTI")
        st.pyplot(fig)

    with t3:
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(df_s, annot=True, cmap="Reds", fmt="d", cbar=False)
        plt.title("Densità Gol SUBITI")
        st.pyplot(fig)
