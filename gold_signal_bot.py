import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import yfinance as yf

# ============================================================
# CONFIGURATION
# ============================================================
TICKER = "GC=F"
MTF_TICKER = TICKER
FAST_LEN, SLOW_LEN = 9, 21
TREND_LEN = 200
RSI_LEN, RSI_LONG_MAX, RSI_SHORT_MIN = 14, 70, 30
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_LEN, BB_MULT = 20, 2.0
ADX_LEN, ADX_THRESHOLD = 14, 20
MTF_EMA_LEN = 50

SL_PERC, TP1_PERC, TP2_PERC, TP3_PERC = 1.0, 1.0, 2.0, 3.0
TRAIL_PERC = 1.0
BE_BUFFER_PERC = 0.1
MAX_PENDING_BARS = 5

STATE_FILE = "state.json"
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"Erreur envoi Telegram: {e}")

def load_state():
    default = {
        "pending_long": False, "pending_short": False, "pending_start": None,
        "in_long": False, "in_short": False,
        "entry_price": None, "sl": None, "tp1": None, "tp2": None, "tp3": None,
        "tp1_hit": False, "trail_stop": None,
        "tp1_sent": False, "tp2_sent": False, "tp3_sent": False
    }
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            saved = json.load(f)
        default.update(saved)
    return default

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def fetch_15m():
    df = yf.download(TICKER, period="60d", interval="15m", progress=False, auto_adjust=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df.dropna()

def fetch_mtf_trend():
    df = yf.download(MTF_TICKER, period="180d", interval="1h", progress=False, auto_adjust=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.dropna()
    df_4h = df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    ema50 = EMAIndicator(df_4h["Close"], window=MTF_EMA_LEN).ema_indicator()
    return df_4h["Close"].iloc[-1], ema50.iloc[-1]

def compute_indicators(df):
    close = df["Close"]
    ind = {}
    ind["ema_fast"] = EMAIndicator(close, window=FAST_LEN).ema_indicator()
    ind["ema_slow"] = EMAIndicator(close, window=SLOW_LEN).ema_indicator()
    ind["ema_trend"] = EMAIndicator(close, window=TREND_LEN).ema_indicator()
    ind["rsi"] = RSIIndicator(close, window=RSI_LEN).rsi()
    macd = MACD(close, window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    ind["macd_line"] = macd.macd()
    ind["macd_signal"] = macd.macd_signal()
    bb = BollingerBands(close, window=BB_LEN, window_dev=BB_MULT)
    ind["bb_upper"] = bb.bollinger_hband()
    ind["bb_lower"] = bb.bollinger_lband()
    adx = ADXIndicator(df["High"], df["Low"], close, window=ADX_LEN)
    ind["adx"] = adx.adx()
    return ind

def main():
    state = load_state()
    df = fetch_15m()
    ind = compute_indicators(df)
    mtf_close, mtf_ema = fetch_mtf_trend()

    close = df["Close"].iloc[-1]
    price_time = df.index[-1]

    cross_up = ind["ema_fast"].iloc[-2] <= ind["ema_slow"].iloc[-2] and ind["ema_fast"].iloc[-1] > ind["ema_slow"].iloc[-1]
    cross_down = ind["ema_fast"].iloc[-2] >= ind["ema_slow"].iloc[-2] and ind["ema_fast"].iloc[-1] < ind["ema_slow"].iloc[-1]

    core_long_no_cross = close > ind["ema_trend"].iloc[-1] and ind["macd_line"].iloc[-1] > ind["macd_signal"].iloc[-1] and close < ind["bb_upper"].iloc[-1]
    core_short_no_cross = close < ind["ema_trend"].iloc[-1] and ind["macd_line"].iloc[-1] < ind["macd_signal"].iloc[-1] and close > ind["bb_lower"].iloc[-1]

    confirm_long = ind["rsi"].iloc[-1] < RSI_LONG_MAX and mtf_close > mtf_ema and ind["adx"].iloc[-1] > ADX_THRESHOLD
    confirm_short = ind["rsi"].iloc[-1] > RSI_SHORT_MIN and mtf_close < mtf_ema and ind["adx"].iloc[-1] > ADX_THRESHOLD

    core_long = cross_up and core_long_no_cross
    core_short = cross_down and core_short_no_cross

    if not state["in_long"] and not state["in_short"]:
        if state["pending_long"] and not core_long_no_cross:
            state["pending_long"] = False
        if state["pending_short"] and not core_short_no_cross:
            state["pending_short"] = False

        if state["pending_long"] and state["pending_start"]:
            elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(state["pending_start"])
            if elapsed > timedelta(minutes=MAX_PENDING_BARS * 15):
                state["pending_long"] = False
                send_telegram("⌛ Opportunité LONG expirée (or) sans confirmation.")
        if state["pending_short"] and state["pending_start"]:
            elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(state["pending_start"])
            if elapsed > timedelta(minutes=MAX_PENDING_BARS * 15):
                state["pending_short"] = False
                send_telegram("⌛ Opportunité SHORT expirée (or) sans confirmation.")

        if not state["pending_long"] and not state["pending_short"]:
            if core_long and not confirm_long:
                state["pending_long"] = True
                state["pending_start"] = datetime.now(timezone.utc).isoformat()
                missing = []
                if ind["rsi"].iloc[-1] >= RSI_LONG_MAX: missing.append("RSI")
                if mtf_close <= mtf_ema: missing.append("MTF")
                if ind["adx"].iloc[-1] <= ADX_THRESHOLD: missing.append("ADX")
                send_telegram(f"🟡 OPPORTUNITÉ LONG (Or)\nPrix: {close:.2f}\nManque: {', '.join(missing)}\n⏳ En attente de confirmation...")
            elif core_short and not confirm_short:
                state["pending_short"] = True
                state["pending_start"] = datetime.now(timezone.utc).isoformat()
                missing = []
                if ind["rsi"].iloc[-1] <= RSI_SHORT_MIN: missing.append("RSI")
                if mtf_close >= mtf_ema: missing.append("MTF")
                if ind["adx"].iloc[-1] <= ADX_THRESHOLD: missing.append("ADX")
                send_telegram(f"🟡 OPPORTUNITÉ SHORT (Or)\nPrix: {close:.2f}\nManque: {', '.join(missing)}\n⏳ En attente de confirmation...")

        entry_long = (core_long and confirm_long) or (state["pending_long"] and core_long_no_cross and confirm_long)
        entry_short = (core_short and confirm_short) or (state["pending_short"] and core_short_no_cross and confirm_short)

        if entry_long:
            state.update({
                "in_long": True, "pending_long": False,
                "entry_price": close, "sl": close * (1 - SL_PERC / 100),
                "tp1": close * (1 + TP1_PERC / 100), "tp2": close * (1 + TP2_PERC / 100), "tp3": close * (1 + TP3_PERC / 100),
                "tp1_hit": False, "trail_stop": None, "tp1_sent": False, "tp2_sent": False, "tp3_sent": False
            })
            send_telegram(f"🟢 ENTRÉE LONG CONFIRMÉE (Or)\nPrix: {close:.2f}\nSL: {state['sl']:.2f}\nTP1: {state['tp1']:.2f}\nTP2: {state['tp2']:.2f}\nTP3: {state['tp3']:.2f}")

        elif entry_short:
            state.update({
                "in_short": True, "pending_short": False,
                "entry_price": close, "sl": close * (1 + SL_PERC / 100),
                "tp1": close * (1 - TP1_PERC / 100), "tp2": close * (1 - TP2_PERC / 100), "tp3": close * (1 - TP3_PERC / 100),
                "tp1_hit": False, "trail_stop": None, "tp1_sent": False, "tp2_sent": False, "tp3_sent": False
            })
            send_telegram(f"🔴 ENTRÉE SHORT CONFIRMÉE (Or)\nPrix: {close:.2f}\nSL: {state['sl']:.2f}\nTP1: {state['tp1']:.2f}\nTP2: {state['tp2']:.2f}\nTP3: {state['tp3']:.2f}")

    elif state["in_long"]:
        if not state["tp1_hit"] and close >= state["tp1"]:
            state["tp1_hit"] = True
            state["trail_stop"] = close * (1 - TRAIL_PERC / 100)
        if state["tp1_hit"]:
            new_trail = close * (1 - TRAIL_PERC / 100)
            state["trail_stop"] = max(state["trail_stop"], new_trail)
            be_level = state["entry_price"] * (1 + BE_BUFFER_PERC / 100)
            active_stop = max(state["trail_stop"], be_level)
        else:
            active_stop = state["sl"]

        if not state["tp1_sent"] and close >= state["tp1"]:
            state["tp1_sent"] = True
            send_telegram(f"✅ TP1 atteint (Or LONG) à {close:.2f} — stop remonté au break-even")
        if not state["tp2_sent"] and close >= state["tp2"]:
            state["tp2_sent"] = True
            send_telegram(f"✅ TP2 atteint (Or LONG) à {close:.2f}")
        if close >= state["tp3"]:
            send_telegram(f"🏁 TP3 atteint (Or LONG) à {close:.2f} — position fermée")
            state.update({"in_long": False, "entry_price": None, "sl": None, "tp1": None, "tp2": None, "tp3": None, "tp1_hit": False, "trail_stop": None})
        elif close <= active_stop:
            reason = "trailing stop / break-even" if state["tp1_hit"] else "stop loss"
            send_telegram(f"⛔ Position LONG (Or) fermée à {close:.2f} ({reason})")
            state.update({"in_long": False, "entry_price": None, "sl": None, "tp1": None, "tp2": None, "tp3": None, "tp1_hit": False, "trail_stop": None})

    elif state["in_short"]:
        if not state["tp1_hit"] and close <= state["tp1"]:
            state["tp1_hit"] = True
            state["trail_stop"] = close * (1 + TRAIL_PERC / 100)
        if state["tp1_hit"]:
            new_trail = close * (1 + TRAIL_PERC / 100)
            state["trail_stop"] = min(state["trail_stop"], new_trail)
            be_level = state["entry_price"] * (1 - BE_BUFFER_PERC / 100)
            active_stop = min(state["trail_stop"], be_level)
        else:
            active_stop = state["sl"]

        if not state["tp1_sent"] and close <= state["tp1"]:
            state["tp1_sent"] = True
            send_telegram(f"✅ TP1 atteint (Or SHORT) à {close:.2f} — stop remonté au break-even")
        if not state["tp2_sent"] and close <= state["tp2"]:
            state["tp2_sent"] = True
            send_telegram(f"✅ TP2 atteint (Or SHORT) à {close:.2f}")
        if close <= state["tp3"]:
            send_telegram(f"🏁 TP3 atteint (Or SHORT) à {close:.2f} — position fermée")
            state.update({"in_short": False, "entry_price": None, "sl": None, "tp1": None, "tp2": None, "tp3": None, "tp1_hit": False, "trail_stop": None})
        elif close >= active_stop:
            reason = "trailing stop / break-even" if state["tp1_hit"] else "stop loss"
            send_telegram(f"⛔ Position SHORT (Or) fermée à {close:.2f} ({reason})")
            state.update({"in_short": False, "entry_price": None, "sl": None, "tp1": None, "tp2": None, "tp3": None, "tp1_hit": False, "trail_stop": None})

    save_state(state)
    print(f"[{price_time}] Vérification terminée. Prix: {close:.2f}")

if __name__ == "__main__":
    main()
