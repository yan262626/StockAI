import os, json, numpy as np, requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    import pandas as pd
    import yfinance as yf
    import xgboost as xgb
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

LSTM_AVAILABLE = False

app = Flask(__name__)
CORS(app)

FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "")
GROQ_KEY    = os.environ.get("GROQ_KEY", "")


def fetch_data(symbol):
    tk   = yf.Ticker(symbol)
    hist = tk.history(period="6mo", interval="1d")
    if hist.empty:
        raise ValueError(f"Aucune donnée pour {symbol}")
    rows = [
        {"date": str(d.date()),
         "open":  round(float(r["Open"]), 4),
         "high":  round(float(r["High"]), 4),
         "low":   round(float(r["Low"]),  4),
         "close": round(float(r["Close"]),4),
         "volume":int(r["Volume"])}
        for d, r in hist.iterrows()
    ]
    info = {}
    try:
        raw  = tk.info
        info = {
            "shortName":    raw.get("shortName", symbol),
            "sector":       raw.get("sector", "—"),
            "industry":     raw.get("industry", "—"),
            "currentPrice": raw.get("currentPrice") or rows[-1]["close"],
            "previousClose":raw.get("previousClose") or rows[-2]["close"],
            "dayChange":    raw.get("regularMarketChangePercent", 0),
            "currency":     raw.get("currency", "USD"),
            "peRatio":      raw.get("trailingPE"),
            "eps":          raw.get("trailingEps"),
            "beta":         raw.get("beta"),
            "52WeekHigh":   raw.get("fiftyTwoWeekHigh"),
            "52WeekLow":    raw.get("fiftyTwoWeekLow"),
            "marketCap":    raw.get("marketCap"),
            "divYield":     raw.get("dividendYield"),
            "roe":          raw.get("returnOnEquity"),
            "netMargin":    raw.get("profitMargins"),
        }
    except Exception:
        info = {
            "shortName": symbol, "currency": "USD",
            "currentPrice":  rows[-1]["close"],
            "previousClose": rows[-2]["close"] if len(rows) > 1 else rows[-1]["close"],
            "dayChange": 0,
        }
    return rows, info


def finnhub_recs(symbol):
    if not FINNHUB_KEY:
        return []
    try:
        r = requests.get(
            f"https://finnhub.io/api/v1/stock/recommendation"
            f"?symbol={symbol}&token={FINNHUB_KEY}", timeout=8)
        return r.json()[:3]
    except Exception:
        return []


def monte_carlo(prices, horizons, n=5000):
    log_r  = np.diff(np.log(prices))
    mu     = float(np.mean(log_r))
    sigma  = float(np.std(log_r))
    S0     = prices[-1]
    out    = {}
    for label, T in horizons.items():
        steps  = max(1, int(T * 252))
        dt     = T / steps
        Z      = np.random.standard_normal((n, steps))
        paths  = S0 * np.exp(
            np.cumsum((mu - .5*sigma**2)*dt + sigma*np.sqrt(dt)*Z, axis=1))
        final  = paths[:, -1]
        mean_p = float(np.mean(final))
        std_p  = float(np.std(final))
        pct    = (mean_p - S0) / S0 * 100
        prob_u = float(np.mean(final > S0) * 100)
        cov    = std_p / mean_p if mean_p else 1
        qual   = max(0, min(100, int(100*(1 - min(cov*2, 1)))))
        out[label] = {
            "model":      "Monte Carlo",
            "price":      round(mean_p, 4),
            "pct_change": round(pct, 2),
            "prob_up":    round(prob_u, 1),
            "ci_low":     round(float(np.percentile(final, 5)), 4),
            "ci_high":    round(float(np.percentile(final, 95)), 4),
            "quality":    qual,
        }
    return out


def build_features(rows):
    if not ML_AVAILABLE:
        return None
    try:
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        c = df["close"]
        df["ma5"]  = c.rolling(5).mean()
        df["ma20"] = c.rolling(20).mean()
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"]  = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        bb_mid, bb_std = c.rolling(20).mean(), c.rolling(20).std()
        df["bb_pct"]    = (c - bb_mid) / (2 * bb_std + 1e-9)
        df["log_ret"]   = np.log(c / c.shift(1))
        df["volatility"]= df["log_ret"].rolling(20).std() * np.sqrt(252)
        df["vol_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1)
        df.dropna(inplace=True)
        return df
    except Exception:
        return None


FEAT = ["ma5","ma20","rsi","macd","bb_pct","volatility","vol_ratio"]

def xgboost_pred(df, horizons_days):
    if not ML_AVAILABLE or df is None or len(df) < 20:
        return {}
    S0  = float(df["close"].iloc[-1])
    out = {}
    for label, days in horizons_days.items():
        try:
            shift = max(1, round(days))
            d2    = df.copy()
            d2["target"] = d2["close"].shift(-shift)
            d2.dropna(inplace=True)
            X, y  = d2[FEAT].values, d2["target"].values
            split = max(10, int(len(X)*0.8))
            if split >= len(X) - 1:
                continue
            mdl = xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0)
            mdl.fit(X[:split], y[:split])
            pred = float(mdl.predict(X[-1].reshape(1, -1))[0])
            pct  = (pred - S0) / S0 * 100
            Xt, yt = X[split:], y[split:]
            if len(Xt):
                rmse = float(np.sqrt(mean_squared_error(yt, mdl.predict(Xt))))
                qual = max(0, min(100, int(100*(1 - min(rmse/S0*10, 1)))))
            else:
                qual = 60
            out[label] = {
                "model":      "XGBoost",
                "price":      round(pred, 4),
                "pct_change": round(pct, 2),
                "prob_up":    round(min(95, max(5, 50+pct*3)), 1),
                "quality":    qual,
            }
        except Exception:
            continue
    return out


def make_ensemble(mc, xgb_r):
    W   = {"Monte Carlo": .40, "XGBoost": .60}
    out = {}
    for h in set(mc) | set(xgb_r):
        parts = [d[h] for d in [mc, xgb_r] if h in d]
        if not parts:
            continue
        tw = sum(W.get(p["model"], .5) for p in parts)
        def wa(k):
            return sum(p[k]*W.get(p["model"],.5) for p in parts) / tw
        out[h] = {
            "model":       "Ensemble",
            "price":       round(wa("price"), 4),
            "pct_change":  round(wa("pct_change"), 2),
            "prob_up":     round(wa("prob_up"), 1),
            "quality":     int(wa("quality")),
            "models_used": [p["model"] for p in parts],
        }
    return out


def groq_analysis(symbol, info, preds, recs):
    ens = preds.get("ensemble", {})
    h24 = ens.get("24h", {})
    h7d = ens.get("7d",  {})
    h1m = ens.get("1m",  {})

    if not GROQ_KEY:
        return _fallback(symbol, info, h24, h7d, h1m, recs)

    prompt = (
        f"Analyse en français l'action {symbol} ({info.get('shortName','')}).\n"
        f"Prix : {info.get('currentPrice')} {info.get('currency','USD')} "
        f"| Variation : {info.get('dayChange','N/A')}%\n"
        f"Secteur : {info.get('sector','N/A')} | P/E : {info.get('peRatio','N/A')}\n"
        f"Prédiction 24h : {h24.get('pct_change','?')}% "
        f"(prob↑ {h24.get('prob_up','?')}%, qualité {h24.get('quality','?')}/100)\n"
        f"7j : {h7d.get('pct_change','?')}% | 1m : {h1m.get('pct_change','?')}%\n\n"
        f"Rédige en markdown :\n"
        f"## Résumé exécutif\n## Facteurs haussiers\n"
        f"## Facteurs baissiers\n## Conclusion"
    )
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 700, "temperature": 0.4},
            timeout=30)
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return _fallback(symbol, info, h24, h7d, h1m, recs)


def _fallback(symbol, info, h24, h7d, h1m, recs):
    chg   = info.get("dayChange", 0) or 0
    trend = "haussière" if (h24.get("pct_change") or 0) > 0 else "baissière"
    qual  = h24.get("quality", 50)
    cons  = "Non disponible"
    if recs:
        r0  = recs[0]
        tot = sum(r0.get(k,0) for k in
                  ["strongBuy","buy","hold","sell","strongSell"])
        buy = r0.get("strongBuy",0) + r0.get("buy",0)
        if tot:
            cons = (f"{round(buy/tot*100)}% achat "
                    f"({r0.get('buy',0)} Buy, "
                    f"{r0.get('hold',0)} Hold, "
                    f"{r0.get('sell',0)} Sell)")
    return f"""## Analyse — {symbol}

**Résumé** : {symbol} à {info.get('currentPrice')} {info.get('currency','USD')} ({chg:+.2f}%).
Biais **{trend}** à 24h : **{h24.get('pct_change','N/A')}%**, prob ↑ **{h24.get('prob_up','N/A')}%**, qualité **{qual}/100**.

**Haussier** : 7j {h7d.get('pct_change','N/A')}% | 1m {h1m.get('pct_change','N/A')}%

**Consensus** : {cons}

*Ajoutez une clé Groq sur console.groq.com pour une analyse IA complète.*"""


@app.route("/api/predict", methods=["POST"])
def predict():
    body   = request.get_json(force=True)
    ticker = (body.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker requis"}), 400
    try:
        rows, info = fetch_data(ticker)
        prices     = [r["close"] for r in rows]
        if len(prices) < 15:
            return jsonify({"error": "Données insuffisantes"}), 400

        recs = finnhub_recs(ticker)
        consensus = {"strongBuy":0,"buy":0,"hold":0,"sell":0,"strongSell":0}
        if recs:
            r0 = recs[0]
            consensus = {k: r0.get(k,0) for k in consensus}

        mc_h  = {"1h":  1/(252*6.5), "6h": 6/(252*6.5),
                 "24h": 1/252,        "7d": 7/252,
                 "1m":  21/252,       "6m": 126/252}
        xgb_h = {"1h": 0.04, "6h": 0.25, "24h": 1,
                  "7d": 5,    "1m": 21,   "6m": 126}

        mc_r  = monte_carlo(prices, mc_h)
        df    = build_features(rows)
        xgb_r = xgboost_pred(df, xgb_h)
        ens_r = make_ensemble(mc_r, xgb_r)

        metrics = {
            "peRatio":    info.get("peRatio"),
            "eps":        info.get("eps"),
            "beta":       info.get("beta"),
            "52WeekHigh": info.get("52WeekHigh"),
            "52WeekLow":  info.get("52WeekLow"),
            "marketCap":  info.get("marketCap"),
            "divYield":   info.get("divYield"),
            "roe":        info.get("roe"),
            "netMargin":  info.get("netMargin"),
        }

        preds  = {"monte_carlo": mc_r, "xgboost": xgb_r,
                  "lstm": {}, "ensemble": ens_r}
        ai_txt = groq_analysis(ticker, info, preds, recs)

        return jsonify({
            "symbol":      ticker,
            "shortName":   info.get("shortName", ticker),
            "timestamp":   datetime.utcnow().isoformat(),
            "quote": {
                "c":  info["currentPrice"],
                "pc": info["previousClose"],
                "dp": info.get("dayChange") or round(
                    (info["currentPrice"] - info["previousClose"])
                    / info["previousClose"] * 100, 2),
            },
            "metrics":     metrics,
            "info":        info,
            "predictions": preds,
            "consensus":   consensus,
            "ai_analysis": ai_txt,
            "history":     rows[-90:],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "ml":      ML_AVAILABLE,
        "lstm":    False,
        "groq":    bool(GROQ_KEY),
        "finnhub": bool(FINNHUB_KEY),
    })


@app.route("/")
def index():
    return jsonify({"message": "StockAI API is running. Use /health or /api/predict"})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
