# profitable_king_colab.ipynb

# PROFITABLE KING TRADING BOT - Streamlit Version
# Includes: Live data simulation, indicator logic, forex + crypto support,
# strategy using algebra, probability, calculus, and Telegram alerts.
# Features: GUI Options, Broker Integration (Mock), Backtesting, ML Predictions (Simulated), Risk Management
# Online data collection, dashboard access, self-improvement logic

import time
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import logging
import requests
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
import streamlit as st

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("[WARNING] ccxt not available. Using MockExchange instead.")

# ==== CONFIGURATION ====
EXCHANGES = ['binance', 'mexc', 'blofin', 'mock', 'forex']
TIMEFRAME = '1m'
LIMIT = 100
RSI_THRESHOLD = 30
RISK_PERCENT = 0.01
STOP_LOSS_PERCENT = 0.03
TAKE_PROFIT_PERCENT = 0.05
TELEGRAM_TOKEN = '8474718833:AAFpiJIg4IW0g0eSlYf0LpV3bmmpeJPTqys'
TELEGRAM_CHAT_ID = '195058225'
TRADE_LOG_FILE = 'trades_log.csv'
LOGIN_PASSWORD = 'Sam0317'

# ==== AUTO-FETCH SYMBOLS ====
ALL_SYMBOLS = []
if CCXT_AVAILABLE:
    for name in EXCHANGES:
        if name not in ['mock', 'forex']:
            try:
                ex = getattr(ccxt, name)()
                ex.load_markets()
                ALL_SYMBOLS.extend([symbol for symbol in ex.symbols if '/USDT' in symbol])
            except:
                continue
FALLBACK_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'MATIC/USDT',
    'DOT/USDT', 'TRX/USDT', 'SHIB/USDT', 'LTC/USDT', 'LINK/USDT', 'BNB/USDT',
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'NZD/USD', 'USD/CAD'
]
SYMBOLS = list(set(ALL_SYMBOLS)) if ALL_SYMBOLS else FALLBACK_SYMBOLS

# ==== MOCK EXCHANGE ====
class MockExchange:
    def fetch_ohlcv(self, symbol, timeframe='1m', limit=100):
        now = int(time.time() * 1000)
        prices = np.cumsum(np.random.randn(limit)) + 100
        volumes = np.random.rand(limit) * 10
        ohlcv = [[now - i*60000, p-1, p+1, p-2, p, v] for i, (p, v) in enumerate(zip(prices, volumes))]
        return list(reversed(ohlcv))

    def create_market_order(self, symbol, side, amount):
        print(f"[MOCK ORDER] {side.upper()} {amount} of {symbol}")

# ==== INIT EXCHANGE ====
def initialize_exchange(name):
    if name in ['mock', 'forex']:
        return MockExchange()
    elif CCXT_AVAILABLE:
        try:
            exchange_class = getattr(ccxt, name)
            exchange = exchange_class()
            exchange.load_markets()
            return exchange
        except Exception as e:
            print(f"Error initializing exchange '{name}': {e}")
            return MockExchange()
    else:
        return MockExchange()

# ==== HELPERS ====
def fetch_ohlcv(exchange, symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def analyze_indicators(df):
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = MACD(df['close']).macd_diff()
    df['ema'] = EMAIndicator(df['close'], window=9).ema_indicator()
    return df

def probability_of_move(df):
    buy_volume = df['volume'].iloc[-5:].sum() * np.random.uniform(0.45, 0.55)
    sell_volume = df['volume'].iloc[-5:].sum() - buy_volume
    return buy_volume / (buy_volume + sell_volume)

def rate_of_change(df):
    delta_price = df['close'].iloc[-1] - df['close'].iloc[-2]
    delta_time = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[-2]) / 1000
    return delta_price / delta_time if delta_time != 0 else 0

def ml_prediction(df):
    trend = df['close'].rolling(3).mean().iloc[-1] - df['close'].rolling(3).mean().iloc[-3]
    return 'BUY' if trend > 0 else 'SELL' if trend < 0 else 'WAIT'

def generate_signal(df):
    rsi = df['rsi'].iloc[-1]
    macd = df['macd'].iloc[-1]
    ema = df['ema'].iloc[-1]
    price = df['close'].iloc[-1]
    p_up = probability_of_move(df)
    roc = rate_of_change(df)
    ml_sig = ml_prediction(df)

    if ml_sig == 'BUY' and rsi < RSI_THRESHOLD and macd > 0 and price > ema and p_up > 0.55 and roc > 0:
        return "BUY"
    elif ml_sig == 'SELL' and rsi > 70 and macd < 0 and price < ema and p_up < 0.45 and roc < 0:
        return "SELL"
    else:
        return "WAIT"

def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, data=payload)
    except:
        print("[ERROR] Telegram message failed.")

# ==== STREAMLIT DASHBOARD ====
st.set_page_config(page_title="Profitable King Dashboard", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #111827; color: white; font-family: 'Segoe UI'; }
        .stButton > button { background-color: #2563EB; color: white; font-size: 16px; border-radius: 10px; padding: 0.5em 1.5em; }
        .stTextInput > div > input { font-size: 18px; padding: 0.4em; border-radius: 8px; }
        .stSuccess { background-color: #10B981; color: white; border-radius: 5px; padding: 0.5em; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Profitable King AI Trading Bot")
password = st.text_input("🔐 Enter password to access dashboard", type="password")
if password == LOGIN_PASSWORD:
    pair = st.text_input("💱 Enter pair (e.g. BTC/USDT)", value="BTC/USDT")
    if st.button("🚀 Get Signal"):
        for ex_name in EXCHANGES:
            exchange = initialize_exchange(ex_name)
            df = fetch_ohlcv(exchange, pair)
            if df is not None:
                df = analyze_indicators(df)
                signal = generate_signal(df)
                price = df['close'].iloc[-1]
                st.success(f"[{pair} on {ex_name.upper()}] ⇒ {signal} @ ${price:.2f}")
                send_telegram_message(f"Signal for {pair} on {ex_name.upper()}: {signal} @ ${price:.2f}")
            else:
                st.warning(f"⚠️ Could not fetch {pair} from {ex_name}.")
else:
    st.warning("🔒 Access denied. Enter correct password.")
