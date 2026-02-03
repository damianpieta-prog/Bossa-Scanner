import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="BOSSA 2.0 PRO (Global Anchor)", layout="wide")

# --- CSS (Stylizacja) ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNKCJE POMOCNICZE ---

@st.cache_data(ttl=300)
def get_data(ticker):
    """
    Pobiera dane dzienne (10 lat - aby znaleźć stare dołki) 
    i godzinowe (60 dni - do momentum).
    """
    try:
        # ZMIANA: period="10y" aby złapać dołki sprzed lat
        df_d1 = yf.download(ticker, period="10y", interval="1d", progress=False)
        df_h1 = yf.download(ticker, period="60d", interval="1h", progress=False)
        return df_d1, df_h1
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def lin_reg_channel(series, window):
    # Zabezpieczenie
    if len(series) < window or window < 5:
        return None, None, None, None
    
    # Bierzemy dane z okna
    y = series[-window:].values
    x = np.arange(len(y))
    
    # Regresja Liniowa
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Linie kanału
    basis = intercept + slope * x
    residuals = y - basis
    sigma = np.std(residuals)
    
    # Wartości na dzisiaj
    last_basis = intercept + slope * (window - 1)
    
    # Slope Score
    slope_score = (slope * (window - 1)) / sigma if sigma != 0 else 0
    
    return last_basis, sigma, slope_score, slope

# --- GŁÓWNA LOGIKA APLIKACJI ---

st.title("📊 BOSSA 2.0 PRO")
st.caption("Tryb: **Global Anchor** (Szuka ekstremum w całej historii 10 lat)")

col1, col2 = st.columns([1, 3])

with col1:
    symbol = st.text_input("Symbol (np. BTC-USD, NVDA, SPY):", value="BTC-USD").upper()
    if symbol == "GOLD": symbol = "GLD"
    
    st.markdown("---")
    st.markdown("**Zasada działania:**")
    st.info("Algorytm sprawdza trend (EMA 200). Jeśli rośnie -> szuka najniższego dołka w historii (nawet sprzed lat). Jeśli spada -> szuka najwyższego szczytu.")

if symbol:
    try:
        with st.spinner(f"Analizuję pełną historię {symbol}..."):
            df_d1, df_h1 = get_data(symbol)
            
            if df_d1.empty or df_h1.empty:
                st.error("Nie znaleziono danych.")
            else:
                if isinstance(df_d1.columns, pd.MultiIndex):
                    try:
                        df_d1 = df_d1.xs(symbol, axis=1, level=1)
                        df_h1 = df_h1.xs(symbol, axis=1, level=1)
                    except: pass

                # === 1. ANALIZA D1 (GLOBAL ANCHOR) ===
                close_d1 = df_d1['Close']
                current_price = float(close_d1.iloc[-1])
                
                ema200 = calculate_ema(close_d1, 200).iloc[-1]
                
                # --- NOWA LOGIKA KOTWICZENIA ---
                # Nie tniemy danych (brak subset). Szukamy na całym df_d1 (10 lat)
                
                if current_price > ema200:
                    # HOSSA: Znajdź ABSOLUTNE MINIMUM z pobranych danych
                    anchor_date = close_d1.idxmin()
                    anchor_type = "LOW (Dołek wieloletni)"
                    trend_direction = 1
                else:
                    # BESSA: Znajdź ABSOLUTNE MAKSIMUM z pobranych danych
                    anchor_date = close_d1.idxmax()
                    anchor_type = "HIGH (Szczyt wieloletni)"
                    trend_direction = -1
                
                # Obliczamy długość kanału od znalezionej daty
                anchor_idx = close_d1.index.get_loc(anchor_date)
                current_idx = len(close_d1)
                dynamic_window = current_idx - anchor_idx
                
                # Zabezpieczenie przed błędem (gdyby ekstremum było dzisiaj)
                if dynamic_window < 5:
                    dynamic_window = 100
                    anchor_info = "Ekstremum jest teraz. Używam domyślnie 100 dni."
                else:
                    anchor_info = f"Start kanału: {anchor_date.strftime('%Y-%m-%d')} ({dynamic_window} sesji temu)"

                # === 2. OBLICZENIA KANAŁU ===
                basis, sigma, slope_score, raw_slope = lin_reg_channel(close_d1, window=dynamic_window)
                z_score = (current_price - basis) / sigma if sigma else 0
                
                # === 3. MOMENTUM H4 ===
                df_h4 = df_h1.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
                close_h4 = df_h4['Close']
                ema7_h4 = calculate_ema(close_h4, 7).iloc[-1]
                ema19_h4 = calculate_ema(close_h4, 19).iloc[-1]
                mom_up = ema7_h4 > ema19_h4
                mom_down = ema7_h4 < ema19_h4

                # === 4. DECYZJE ===
                c_green, c_red, c_yellow, c_grey = "#21c354", "#ff4b4b", "#ffa421", "#333333"
                signal = "WAIT"
                bg_color = c_grey
                desc = "..."

                if trend_direction == -1: 
                    signal = "WAIT (BESSA)"
                    bg_color = c_red
                    desc = "Cena poniżej EMA200. Szukam szczytu do rysowania kanału spadkowego."
                elif slope_score < 0.1:
                    signal = "WAIT (PŁASKO)"
                    bg_color = c_yellow
                    desc = "Bardzo długi trend boczny."
                else:
                    if z_score >= 2.0:
                        signal = "TAKE PROFIT"
                        bg_color = c_yellow
                        desc = "Cena przy górnej bandzie długoterminowego kanału."
                    elif mom_up and z_score < 1.0:
                        signal = "BUY (LONG TERM)"
                        bg_color = c_green
                        desc = "Trend wieloletni wzrostowy + Momentum H4."
                    elif mom_down:
                        signal = "WAIT / KOREKTA"
                        bg_color = c_red
                        desc = "Długi trend OK, ale lokalnie H4 spada."

                # === 5. GUI ===
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 25px;">
                    <h1 style="color: white; margin:0;">{signal}</h1>
                    <p style="color: #eee;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Cena", f"${current_price:,.2f}")
                m2.metric("Z-Score", f"{z_score:.2f}")
                m3.metric("Slope", f"{slope_score:.2f}")
                m4.metric("Długość", f"{dynamic_window}", help=anchor_info)

                st.subheader(f"Wykres ({anchor_type} z {anchor_date.strftime('%Y-%m-%d')})")
                
                # Wykres - pokazujemy cały zakres kanału + mały margines
                plot_start_idx = max(0, anchor_idx - 50)
                plot_data = df_d1.iloc[plot_start_idx:].copy()
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_data.index,
                                open=plot_data['Open'], high=plot_data['High'],
                                low=plot_data['Low'], close=plot_data['Close'], name='Cena'))

                if sigma is not None:
                    # Rysujemy kanał od punktu zakotwiczenia do teraz
                    channel_dates = df_d1.index[-dynamic_window:]
                    x_vals = np.arange(dynamic_window)
                    
                    start_val = basis - (raw_slope * (dynamic_window - 1))
                    vals_basis = start_val + raw_slope * x_vals
                    vals_upper = vals_basis + 2 * sigma
                    vals_lower = vals_basis - 2 * sigma
                    
                    fig.add_trace(go.Scatter(x=channel_dates, y=vals_basis, mode='lines', line=dict(color='gray', dash='dash'), name='Środek'))
                    fig.add_trace(go.Scatter(x=channel_dates, y=vals_upper, mode='lines', line=dict(color='red'), name='+2 Sigma'))
                    fig.add_trace(go.Scatter(x=channel_dates, y=vals_lower, mode='lines', line=dict(color='#00ff00'), name='-2 Sigma'))

                fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Błąd: {e}")
