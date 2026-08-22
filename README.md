# btc-trend-monitor

מנטר מגמות למספר מטבעות קריפטו מ-Binance, ושולח איתות לטלגרם כשיש שינוי מגמה (Long/Short/Neutral).

## מטבעות נתמכים

כרגע רץ אוטומטית (דרך GitHub Actions) על: `BTCUSDT`, `SOLUSDT`, `ETHUSDT`, `ENAUSDT`.
כל מטבע נשמר בנפרד ב-`state.json` (למעט BTCUSDT ששומר על שם המפתח ההיסטורי, לתאימות לאחור).

## הרצה ידנית

```
python btc_trend_monitor.py short              # BTCUSDT, מסלול קצר
python btc_trend_monitor.py long                # BTCUSDT, מסלול ארוך
python btc_trend_monitor.py short SOLUSDT       # מטבע אחר, מסלול קצר
python btc_trend_monitor.py long ETHUSDT        # מטבע אחר, מסלול ארוך
```

כדי להוסיף מטבע נוסף לבדיקה האוטומטית, יש להוסיף את הסמל (למשל `DOGEUSDT`) לרשימה בתוך
`.github/workflows/short-term.yml` ו-`.github/workflows/long-term.yml`.
