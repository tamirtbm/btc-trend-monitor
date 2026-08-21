"""
BTC Trend Monitor
------------------
מושך נתוני מחירים ש�  ביטקוין מ-Binance, מחשב אינדיקטורים טכניים לשני מסלולים
(קצר וארוך), ושולח איתות לטלגרם רק כשיש שינוי מגמה אמיתי (Long/Short/Neutral).

מסלול קצר (SHORT_TERM):  interval=1h, EMA 9/21 + MACD + OBV
מסלול ארוך (LONG_TERM):  interval=1d, EMA 50/200 + ADX + OBV

הרצה:
    python btc_trend_monitor.py short   # בודק ושולח איתות למסלול הקצר
    python btc_trend_monitor.py long    # בודק ושולח איתות למסלול הארוך
"""

import os
import sys
import json
import requests
import numpy as np
import pandas as pd

# ---------- הגדרות ----------

BINANCE_KLINE_URL = "https://data-api.binance.vision/api/v3/klines"
SYMBOL = "BTCUSDT"

TELEGRAM_BOT_TOKEN = os.environ.get("CRYPTO_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("CRYPTO_TELEGRAM_CHAT_ID")

STATE_FILE = "state.json"

TIMEFRAMES = {
    "short": {
        "interval": "1h",       # 1 שעה
        "limit": 300,
        "state_key": "short_term",
        "label": "טווח קצר (1H)",
    },
    "long": {
        "interval": "1d",       # יומי
        "limit": 300,
        "state_key": "long_term",
        "label": "טווח ארוך (יומי)",
    },
}


# ---------- משיכת נתונים ----------

def fetch_klines(interval: str, limit: int = 300) -> pd.DataFrame:
    """מושך נרות היסטוריים מ-Binance ומחזיר DataFrame ממוין מהישן לחדש."""
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "limit": limit,
    }
    resp = requests.get(BINANCE_KLINE_URL, params=params, timeout=15)
    resp.raise_for_status()
    rows = resp.json()

    if not isinstance(rows, list):
        raise RuntimeError(f"Binance API error: {rows}")

    # Binance מחזיר לכל נר: [openTime, open, high, low, close, volume, closeTime,
    # quoteAssetVolume, numTrades, takerBuyBaseVol, takerBuyQuoteVol, ignore]
    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype({
        "timestamp": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------- אינדיקטורים (מחושבים ידנית, בלי ספריית ta) ----------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


# ---------- לוגיקת מגמה לכל מסלול ----------

def evaluate_short_term(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    macd_line, signal_line, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["obv"] = obv(df)

    last = df.iloc[-1]

    ema_bullish = last["ema9"] > last["ema21"]
    macd_bullish = last["macd"] > last["macd_signal"]
    obv_rising = df["obv"].iloc[-1] > df["obv"].iloc[-5]  # מגמת OBV ב-5 נרות אחרונים

    if ema_bullish and macd_bullish:
        trend = "LONG"
    elif (not ema_bullish) and (not macd_bullish):
        trend = "SHORT"
    else:
        trend = "NEUTRAL"

    return {
        "trend": trend,
        "price": round(last["close"], 2),
        "ema9": round(last["ema9"], 2),
        "ema21": round(last["ema21"], 2),
        "macd": round(last["macd"], 2),
        "macd_signal": round(last["macd_signal"], 2),
        "obv_rising": bool(obv_rising),
    }


def evaluate_long_term(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["adx"] = adx(df)
    df["obv"] = obv(df)

    last = df.iloc[-1]

    ema_bullish = last["ema50"] > last["ema200"]
    trend_strong = last["adx"] > 20
    obv_rising = df["obv"].iloc[-1] > df["obv"].iloc[-10]

    if trend_strong and ema_bullish:
        trend = "LONG"
    elif trend_strong and not ema_bullish:
        trend = "SHORT"
    else:
        trend = "NEUTRAL"  # אין מגמה מספיק חזקה (ADX נמוך)

    return {
        "trend": trend,
        "price": round(last["close"], 2),
        "ema50": round(last["ema50"], 2),
        "ema200": round(last["ema200"], 2),
        "adx": round(last["adx"], 2),
        "obv_rising": bool(obv_rising),
    }


# ---------- ניהול state (למניעת איתותים כפולים) ----------

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------- טלגרם ----------

def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("חסרים CRYPTO_TELEGRAM_BOT_TOKEN / CRYPTO_TELEGRAM_CHAT_ID - לא נשלחה הודעה")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }, timeout=15)
    resp.raise_for_status()


def format_signal_message(label: str, prev_trend: str, result: dict) -> str:
    trend_emoji = {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪"}
    obv_text = "עולה (תומך במגמה)" if result["obv_rising"] else "לא תומך"

    lines = [
        f"<b>⚡ שינוי מגמה - BTC ⚡</b>",
        f"מסלול: {label}",
        f"{prev_trend} → <b>{trend_emoji.get(result['trend'], '')} {result['trend']}</b>",
        f"מחיר נוכחי: ${result['price']:,}",
        f"OBV: {obv_text}",
    ]
    return "\n".join(lines)


# ---------- ריצה ----------

def run(track: str):
    cfg = TIMEFRAMES[track]
    df = fetch_klines(cfg["interval"], cfg["limit"])

    if track == "short":
        result = evaluate_short_term(df)
    else:
        result = evaluate_long_term(df)

    state = load_state()
    prev_trend = state.get(cfg["state_key"], "NEUTRAL")
    new_trend = result["trend"]

    print(f"[{cfg['label']}] מחיר: {result['price']} | מגמה קודמת: {prev_trend} | מגמה נוכחית: {new_trend}")

    if new_trend != prev_trend:
        message = format_signal_message(cfg["label"], prev_trend, result)
        send_telegram_message(message)
        print("נשלח איתות בטלגרם")
    else:
        print("אין שינוי מגמה - לא נשלח איתות")

    state[cfg["state_key"]] = new_trend
    save_state(state)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("short", "long"):
        print("שימוש: python btc_trend_monitor.py [short|long]")
        sys.exit(1)
    run(sys.argv[1])
