import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="BOSSA 2.0 PRO", layout="wide")

# --- FUNKCJE (Pamięć podręczna na 5 min żeby nie blokowali) ---
@st.cache_data(ttl=300)
def get_data(ticker):
    # D1 (Dzienny) - 2 lata wstecz
    df_d1 = yf.download(ticker, period="2y", interval="1d", progress=False)
    # H1 (Godzinowy) - 60 dni wstecz (max dla yfinance)
    df_h1 = yf.download(ticker, period="60d", interval="1h", progress=False)
    return df_d1, df_h1

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def lin_reg_channel(series, window=120):
    if len(series) < window:
        return None, None, None, None
    
    y = series[-window:].values
    x = np.arange(len(y))
    
    # Matematyka regresji
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Linie kanału
    basis = intercept + slope * x
    residuals = y - basis
    sigma = np.std(residuals)
    
    # Wartości na dzisiaj
    last_basis = intercept + slope * (window - 1)
    slope_score = (slope * (window - 1)) / sigma if sigma != 0 else 0
    
    return last_basis, sigma, slope_score, slope

# --- INTERFEJS APLIKACJI ---
st.title("📊 BOSSA 2.0 PRO")
st.markdown("Strategia: **Kanał Regresji (D1)** + **Momentum (H4)**")

symbol = st.text_input("Wpisz symbol (np. BTC-USD, GOOG, GLD):", value="BTC-USD").upper()

# Poprawka dla złota
if symbol == "GOLD":
    st.warning("Dla złota używam symbolu GLD.")
    symbol = "GLD"

if symbol:
    try:
        with st.spinner("Analizuję wykresy..."):
            df_d1, df_h1 = get_data(symbol)
            
            if df_d1.empty or df_h1.empty:
                st.error("Nie znaleziono danych. Sprawdź symbol.")
            else:
                # Obsługa MultiIndex (naprawa błędu w nowych bibliotekach)
                if isinstance(df_d1.columns, pd.MultiIndex):
                    try:
                        df_d1 = df_d1.xs(symbol, axis=1, level=1)
                        df_h1 = df_h1.xs(symbol, axis=1, level=1)
                    except: pass

                # --- OBLICZENIA D1 ---
                close_d1 = df_d1['Close']
                current_price = float(close_d1.iloc[-1])
                
                ema100 = calculate_ema(close_d1, 100).iloc[-1]
                ema200 = calculate_ema(close_d1, 200).iloc[-1]
                
                basis, sigma, slope_score, raw_slope = lin_reg_channel(close_d1, window=120)
                z_score = (current_price - basis) / sigma if sigma else 0
                
                # --- OBLICZENIA H4 (z danych H1) ---
                df_h4 = df_h1.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
                close_h4 = df_h4['Close']
                
                ema7_h4 = calculate_ema(close_h4, 7).iloc[-1]
                ema19_h4 = calculate_ema(close_h4, 19).iloc[-1]
                
                # --- WERDYKT ---
                trend_ok = ema100 > ema200
                slope_ok = slope_score > 0.50
                cheap_ok = z_score < 2.0
                
                mom_up = ema7_h4 > ema19_h4
                mom_down = ema7_h4 < ema19_h4
                
                signal = "WAIT"
                bg_color = "#333333" # Szary
                desc = "Brak warunków do wejścia."

                if not trend_ok:
                    signal = "WAIT (BESSA)"
                    bg_color = "#ff4b4b" # Czerwony
                    desc = "Trend główny spadkowy (EMA 100 < 200)."
                elif not slope_ok:
                    signal = "WAIT (PŁASKO)"
                    bg_color = "#ffa421" # Żółty
                    desc = "Kanał regresji jest płaski (Slope < 0.5)."
                else:
                    # Jesteśmy w trendzie
                    if z_score >= 2.0:
                        signal = "TAKE PROFIT"
                        bg_color = "#ffa421"
                        desc = "Cena na szczycie kanału (+2 Sigma). Realizuj zyski."
                    elif mom_down:
                        signal = "WAIT / EXIT"
                        bg_color = "#ff4b4b"
                        desc = "H4 Momentum spadkowe (Korekta)."
                    elif mom_up and cheap_ok:
                        signal = "BUY (STRONG)"
                        bg_color = "#21c354" # Zielony
                        desc = "Trend OK + Cena w kanale + H4 Momentum Wzrostowe."

                # --- WYŚWIETLANIE ---
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                    <h1 style="color: white; margin:0;">{signal}</h1>
                    <p style="color: white; margin:0;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Cena", f"${current_price:,.2f}")
                c2.metric("Z-Score", f"{z_score:.2f}", help="Im mniej tym taniej. Powyżej 2.0 drogo.")
                c3.metric("Slope", f"{slope_score:.2f}", help="Musi być > 0.50")

                # --- WYKRES ---
                st.subheader("Wizualizacja Kanału (D1)")
                
                # Bierzemy ostatnie 150 dni do rysowania
                plot_data = df_d1.iloc[-150:].copy()
                fig = go.Figure()

                # Świece
                fig.add_trace(go.Candlestick(x=plot_data.index, 
                                open=plot_data['Open'], high=plot_data['High'],
                                low=plot_data['Low'], close=plot_data['Close'], name='Cena'))

                # Rysowanie kanału (projekcja ostatnich parametrów)
                if sigma is not None:
                    # Generujemy daty dla ostatnich 120 świeczek na wykresie
                    dates = plot_data.index[-120:]
                    
                    # Odtwarzamy linię środkową wstecz
                    # Wzór: Y = Basis_dzisiaj - (Slope * (odległość_wstecz))
                    vals_basis = [basis - (raw_slope * (119 - i)) for i in range(120)]
                    vals_upper = [v + 2*sigma for v in vals_basis]
                    vals_lower = [v - 2*sigma for v in vals_basis]
                    
                    fig.add_trace(go.Scatter(x=dates, y=vals_basis, line=dict(color='white', width=1, dash='dash'), name='Środek'))
                    fig.add_trace(go.Scatter(x=dates, y=vals_upper, line=dict(color='red', width=2), name='+2 Sigma'))
                    fig.add_trace(go.Scatter(x=dates, y=vals_lower, line=dict(color='green', width=2), name='-2 Sigma'))

                fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Legenda:** Kupuj, gdy cena jest blisko **Zielonej Linii**, a werdykt to **BUY**.")

    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")
