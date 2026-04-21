"""
StockOracle Pro — Backend 100% GRATUIT
IA : Groq (llama-3.3-70b, GRATUIT)
Données : yfinance (GRATUIT, sans clé) + Finnhub (plan gratuit)
ML : Monte Carlo + XGBoost + LSTM
"""

import os, json, numpy as np, requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    import pandas as pd, yfinance as yf
    import xgboost as xgb
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

app = Flask(__name__)
CORS(app)

FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "")
GROQ_KEY    = os.environ.get("GROQ_KEY",    "")


def resolve_isin(isin):
    try:
        r = requests.post(
            "https://api.openfigi.com/v3/mapping",
            headers={"Content-Type": "application/json"},
            json=[{"idType": "ID_ISIN", "idValue": isin}],
            timeout=10,
        )
        return r.json()[0]["data"][0]["ticker"]
    except:
        return isin


def fetch_yfinance(symbol):
    tk   = yf.Ticker(symbol)
    hist = tk.history(period="6mo", interval="1d")
    if hist.empty:
        raise ValueError(f"Aucune donnée pour '{symbol}'. Vérifiez le ticker.")
    rows = [
        {"date": str(d.date()), "open": round(float(r["Open"]),4),
         "high": round(float(r["High"]),4), "low": round(float(r["Low"]),4),
         "close": round(float(r["Close"]),4), "volume": int(r["Volume"])}
        for d, r in hist.iterrows()
    ]
    info = {}
    try:
        raw = tk.info
        info = {
            "peRatio":      raw.get("trailingPE"),
            "eps":          raw.get("trailingEps"),
            "beta":         raw.get("beta"),
            "52WeekHigh":   raw.get("fiftyTwoWeekHigh"),
            "52WeekLow":    raw.get("fiftyTwoWeekLow"),
            "marketCap":    raw.get("marketCap"),
            "divYield":     raw.get("dividendYield"),
            "roe":          raw.get("returnOnEquity"),
            "netMargin":    raw.get("profitMargins"),
            "shortName":    raw.get("shortName", symbol),
            "sector":       raw.get("sector", "—"),
            "industry":     raw.get("industry", "—"),
            "currentPrice": raw.get("currentPrice") or rows[-1]["close"],
            "previousClose":raw.get("previousClose") or rows[-2]["close"],
            "dayChange":    raw.get("regularMarketChangePercent"),
            "currency":     raw.get("currency", "USD"),
        }
    except:
        info = {"currentPrice": rows[-1]["close"],
                "previousClose": rows[-2]["close"] if len(rows)>1 else rows[-1]["close"],
                "shortName": symbol, "currency": "USD"}
    return rows, info


def finnhub_recs(symbol):
    if not FINNHUB_KEY: return []
    try:
        r = requests.get(
            f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={FINNHUB_KEY}",
            timeout=8)
        return r.json()[:3]
    except: return []

def finnhub_metrics(symbol):
    if not FINNHUB_KEY: return {}
    try:
        r = requests.get(
            f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={FINNHUB_KEY}",
            timeout=8)
        return r.json().get("metric", {})
    except: return {}


def build_features(rows):
    if not ML_AVAILABLE: return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    c = df["close"]
    df["ma5"]      = c.rolling(5).mean()
    df["ma20"]     = c.rolling(20).mean()
    df["ma50"]     = c.rolling(50).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]      = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12, ema26   = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    df["macd"]     = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    bb_mid, bb_std = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_pct"]   = (c - bb_mid) / (2 * bb_std + 1e-9)
    df["log_ret"]  = np.log(c / c.shift(1))
    df["volatility"]= df["log_ret"].rolling(20).std() * np.sqrt(252)
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1)
    df.dropna(inplace=True)
    return df


def monte_carlo(prices, horizons, n=6000):
    log_r = np.diff(np.log(prices))
    mu, sigma, S0 = float(np.mean(log_r)), float(np.std(log_r)), prices[-1]
    out = {}
    for label, T in horizons.items():
        steps = max(1, int(T * 252)); dt = T / steps
        Z     = np.random.standard_normal((n, steps))
        paths = S0 * np.exp(np.cumsum((mu-.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z, axis=1))
        final = paths[:, -1]
        mean_p = float(np.mean(final)); std_p = float(np.std(final))
        pct    = (mean_p - S0) / S0 * 100
        prob_u = float(np.mean(final > S0) * 100)
        cov    = std_p / mean_p if mean_p else 1
        qual   = max(0, min(100, int(100*(1-min(cov*2,1)))))
        out[label] = {
            "model": "Monte Carlo GBM", "price": round(mean_p,4),
            "pct_change": round(pct,2), "prob_up": round(prob_u,1),
            "ci_low": round(float(np.percentile(final,5)),4),
            "ci_high": round(float(np.percentile(final,95)),4),
            "quality": qual
        }
    return out


FEAT = ["ma5","ma20","rsi","macd","bb_pct","volatility","vol_ratio"]

def xgboost_pred(df, horizons_days):
    if not ML_AVAILABLE or df is None or len(df) < 20: return {}
    S0, out = float(df["close"].iloc[-1]), {}
    for label, days in horizons_days.items():
        shift = max(1, round(days))
        d2 = df.copy(); d2["target"] = d2["close"].shift(-shift); d2.dropna(inplace=True)
        X, y = d2[FEAT].values, d2["target"].values
        split = max(10, int(len(X)*0.8))
        if split >= len(X)-1: continue
        mdl = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.04,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, verbosity=0)
        mdl.fit(X[:split], y[:split])
        pred = float(mdl.predict(X[-1].reshape(1,-1))[0])
        pct  = (pred - S0) / S0 * 100
        Xt, yt = X[split:], y[split:]
        qual = max(0, min(100, int(100*(1-min(
            float(np.sqrt(mean_squared_error(yt, mdl.predict(Xt))))/S0*10, 1))))) if len(Xt) else 60
        out[label] = {
            "model": "XGBoost", "price": round(pred,4),
            "pct_change": round(pct,2),
            "prob_up": round(min(95, max(5, 50+pct*3)), 1),
            "quality": qual
        }
    return out


def lstm_pred(prices, labels):
    if not LSTM_AVAILABLE or len(prices) < 40: return {}
    SEQ = 20; sc = MinMaxScaler()
    scp = sc.fit_transform(np.array(prices).reshape(-1,1))
    X, y = [], []
    for i in range(SEQ, len(scp)):
        X.append(scp[i-SEQ:i,0]); y.append(scp[i,0])
    X, y = np.array(X).reshape(-1,SEQ,1), np.array(y)
    sp = max(10, int(len(X)*0.85))
    mdl = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ,1)),
        Dropout(0.2), LSTM(32), Dropout(0.2), Dense(1)
    ])
    mdl.compile(optimizer="adam", loss="mse")
    mdl.fit(X[:sp], y[:sp], epochs=25, batch_size=16,
            verbose=0, validation_split=0.1)
    S0   = prices[-1]
    pred = float(sc.inverse_transform(
        [[float(mdl.predict(scp[-SEQ:].reshape(1,SEQ,1), verbose=0)[0][0])]])[0][0])
    pct  = (pred - S0) / S0 * 100
    qual = max(0, min(100, int(100*(1-min(
        float(np.sqrt(mean_squared_error(
            sc.inverse_transform(y[sp:].reshape(-1,1)),
            sc.inverse_transform(mdl.predict(X[sp:], verbose=0))
        )))/S0*10, 1))))) if len(X) > sp else 58
    out = {}
    for lbl in labels:
        out[lbl] = {
            "model": "LSTM", "price": round(pred,4),
            "pct_change": round(pct,2),
            "prob_up": round(min(95, max(5, 50+pct*4)), 1),
            "quality": qual
        }
    return out


def make_ensemble(mc, xgb_r, lstm_r):
    W = {"Monte Carlo GBM": .30, "XGBoost": .40, "LSTM": .30}
    out = {}
    for h in set(mc)|set(xgb_r)|set(lstm_r):
        parts = [d[h] for d in [mc, xgb_r, lstm_r] if h in d]
        if not parts: continue
        tw = sum(W.get(p["model"], .33) for p in parts)
        def wa(k): return sum(p[k]*W.get(p["model"],.33) for p in parts)/tw
        out[h] = {
            "model": "Ensemble", "price": round(wa("price"),4),
            "pct_change": round(wa("pct_change"),2),
            "prob_up": round(wa("prob_up"),1),
            "quality": int(wa("quality")),
            "models_used": [p["model"] for p in parts]
        }
    return out


def ai_analysis(symbol, info, preds, recs):
    ens = preds.get("ensemble", {})
    h24 = ens.get("24h", {}); h7d = ens.get("7d", {}); h1m = ens.get("1m", {})

    if not GROQ_KEY:
        return _fallback(symbol, info, h24, h7d, h1m, recs)

    prompt = f"""Tu es un analyste quantitatif senior. Analyse en français l'action {symbol} ({info.get('shortName','')}).

Prix actuel : {info.get('currentPrice')} {info.get('currency','USD')} | Variation : {info.get('dayChange','N/A')}%
Secteur : {info.get('sector','N/A')} | P/E : {info.get('peRatio','N/A')} | Beta : {info.get('beta','N/A')}
Prédictions Ensemble :
- 24h : {h24.get('pct_change','?')}% (prob↑ {h24.get('prob_up','?')}%, qualité {h24.get('quality','?')}/100)
- 7j  : {h7d.get('pct_change','?')}%
- 1m  : {h1m.get('pct_change','?')}% (qualité {h1m.get('quality','?')}/100)
Consensus : {json.dumps(recs[:1]) if recs else 'N/A'}

Rédige en markdown :
## Résumé exécutif (3 phrases)
## Facteurs haussiers (3 points)
## Facteurs baissiers (3 points)
## Conclusion et biais directionnel"""

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 800, "temperature": 0.4},
            timeout=30,
        )
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return _fallback(symbol, info, h24, h7d, h1m, recs) + f"\n\n*(Groq error: {e})*"


def _fallback(symbol, info, h24, h7d, h1m, recs):
    chg   = info.get("dayChange", 0) or 0
    trend = "haussière" if (h24.get("pct_change") or 0) > 0 else "baissière"
    qual  = h24.get("quality", 50)
    qt    = "élevée" if qual>70 else "modérée" if qual>45 else "faible"
    cons  = "Non disponible (ajoutez une clé Finnhub)"
    if recs:
        r0  = recs[0]
        tot = sum(r0.get(k,0) for k in ["strongBuy","buy","hold","sell","strongSell"])
        buy = r0.get("strongBuy",0) + r0.get("buy",0)
        if tot: cons = f"{round(buy/tot*100)}% achat ({r0.get('buy',0)} Buy, {r0.get('hold',0)} Hold, {r0.get('sell',0)} Sell)"
    return f"""## Analyse automatique — {symbol}

**Résumé exécutif**

{symbol} s'échange à **{info.get('currentPrice')} {info.get('currency','USD')}** ({chg:+.2f}% aujourd'hui).
Biais **{trend}** à 24h : **{h24.get('pct_change','N/A')}%** prévu, probabilité de hausse **{h24.get('prob_up','N/A')}%**.
Fiabilité globale : **{qt}** ({qual}/100).

**Facteurs haussiers**
- Prédiction 7 jours : **{h7d.get('pct_change','N/A')}%** (prob ↑ {h7d.get('prob_up','N/A')}%)
- Prédiction 1 mois : **{h1m.get('pct_change','N/A')}%** (qualité {h1m.get('quality','N/A')}/100)
- Secteur : {info.get('sector','N/A')}

**Facteurs de risque**
- Qualité prédiction : {qual}/100
- Volatilité dans les intervalles Monte Carlo
- Facteurs macro/news non intégrés

**Consensus analystes**
{cons}

---
*💡 Ajoutez une clé Groq gratuite sur console.groq.com pour une analyse LLaMA rédigée.*"""


@app.route("/api/predict", methods=["POST"])
def predict():
    body   = request.get_json(force=True)
    ticker = (body.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker requis"}), 400

    if len(ticker) == 12 and ticker[:2].isalpha():
        ticker = resolve_isin(ticker)

    try:
        rows, info = fetch_yfinance(ticker)
        prices = [r["close"] for r in rows]
        if len(prices) < 15:
            return jsonify({"error": "Données insuffisantes"}), 400

        recs   = finnhub_recs(ticker)
        fh_met = finnhub_metrics(ticker)

        consensus = {"strongBuy":0,"buy":0,"hold":0,"sell":0,"strongSell":0}
        if recs:
            r0 = recs[0]; consensus = {k: r0.get(k,0) for k in consensus}

        mc_h  = {"1h":1/(252*6.5),"6h":6/(252*6.5),"24h":1/252,
                 "7d":7/252,"1m":21/252,"6m":126/252}
        xgb_h = {"1h":0.04,"6h":0.25,"24h":1,"7d":5,"1m":21,"6m":126}

        mc_r   = monte_carlo(prices, mc_h)
        df     = build_features(rows)
        xgb_r  = xgboost_pred(df, xgb_h)
        lstm_r = lstm_pred(prices, list(mc_h.keys()))
        ens_r  = make_ensemble(mc_r, xgb_r, lstm_r)

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
            **{k: fh_met.get(k) for k in
               ["roeTTM","netProfitMarginTTM","debtToEquity"] if fh_met.get(k)},
        }

        preds  = {"monte_carlo":mc_r,"xgboost":xgb_r,"lstm":lstm_r,"ensemble":ens_r}
        ai_t   = ai_analysis(ticker, info, preds, recs)

        return jsonify({
            "symbol":      ticker,
            "shortName":   info.get("shortName", ticker),
            "timestamp":   datetime.utcnow().isoformat(),
            "quote": {
                "c":  info["currentPrice"],
                "pc": info["previousClose"],
                "dp": info.get("dayChange") or round(
                    (info["currentPrice"]-info["previousClose"])/info["previousClose"]*100, 2)
            },
            "metrics":     metrics,
            "info":        info,
            "predictions": preds,
            "consensus":   consensus,
            "ai_analysis": ai_t,
            "history":     rows[-90:],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status":"ok","ml":ML_AVAILABLE,"lstm":LSTM_AVAILABLE,
                    "groq":bool(GROQ_KEY),"finnhub":bool(FINNHUB_KEY)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
