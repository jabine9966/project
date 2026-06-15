"""
OKX SOL-USDT-SWAP 多维度技术分析 & 交易计划生成（第二部分）
----------------------------------------------------
- 数据来源：项目 data/ 目录（由 fetch_data.py 生成的清洗后数据）
- 分析维度：
    * 传统指标：MA/EMA/SMA, MACD, RSI, Stoch, BBands, ATR, ADX/+DI/-DI
    * 高级指标：Ichimoku, SuperTrend, Keltner, Donchian, OBV, MFI, CMF, VWAP, Pivot, Fibonacci
    * 量化指标：年化波动率、最大回撤、偏度/峰度、自相关、Hurst 指数、Z‑score
    * 衍生品：资金费率走势/分位、OI 变化、Long/Short 比、Taker CVD
- 输出：项目根目录 REPORT.md（行情分析 + 多种交易计划）

时区：所有展示时间为 Asia/Shanghai (UTC+8)。
"""
from __future__ import annotations
import os
import json
import math
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

SH_TZ = timezone(timedelta(hours=8))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_REPORT = os.path.join(ROOT, "REPORT.md")
OUT_JSON = os.path.join(ROOT, "analysis.json")

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
ANNUAL_BARS = {
    "1m": 525600,
    "5m": 105120,
    "15m": 35040,
    "30m": 17520,
    "1h": 8760,
    "2h": 4380,
    "4h": 2190,
    "1d": 365,
}


# ============================== 加载数据 ==============================

def load_ohlcv(tf: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"ohlcv_{tf}.csv")
    df = pd.read_csv(path)
    df = df.sort_values("ts").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(SH_TZ)
    return df


def load_simple_csv(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.sort_values("ts").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(SH_TZ)
    return df


def load_json(name: str) -> dict:
    path = os.path.join(DATA_DIR, f"{name}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ============================== 指标 ==============================

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()


def rsi(s, n=14):
    d = s.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(s, f=12, sl=26, sig=9):
    m = ema(s, f) - ema(s, sl)
    sg = ema(m, sig)
    return m, sg, m - sg


def bbands(s, n=20, k=2):
    m, sd = sma(s, n), s.rolling(n).std()
    return m + k * sd, m, m - k * sd


def true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(h, l, c, n=14):
    return true_range(h, l, c).ewm(alpha=1 / n, adjust=False).mean()


def adx(h, l, c, n=14):
    up = h.diff(); dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(h, l, c)
    atr_ = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), plus_di, minus_di


def stoch(h, l, c, k=14, d=3):
    ll, hh = l.rolling(k).min(), h.rolling(k).max()
    kv = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    return kv, kv.rolling(d).mean()


def supertrend(h, l, c, n=10, mult=3.0):
    atr_ = atr(h, l, c, n)
    hl2 = (h + l) / 2
    up = hl2 + mult * atr_
    dn = hl2 - mult * atr_
    st = pd.Series(index=c.index, dtype=float)
    dirn = pd.Series(index=c.index, dtype=int)
    for i in range(len(c)):
        if i == 0:
            st.iloc[i] = up.iloc[i]; dirn.iloc[i] = -1; continue
        prev_st, prev_dir = st.iloc[i - 1], dirn.iloc[i - 1]
        if prev_dir == 1:
            st.iloc[i] = max(dn.iloc[i], prev_st)
            if c.iloc[i] < st.iloc[i]:
                st.iloc[i] = up.iloc[i]; dirn.iloc[i] = -1
            else:
                dirn.iloc[i] = 1
        else:
            st.iloc[i] = min(up.iloc[i], prev_st)
            if c.iloc[i] > st.iloc[i]:
                st.iloc[i] = dn.iloc[i]; dirn.iloc[i] = 1
            else:
                dirn.iloc[i] = -1
    return st, dirn


def ichimoku(h, l, c):
    conv = (h.rolling(9).max() + l.rolling(9).min()) / 2
    base = (h.rolling(26).max() + l.rolling(26).min()) / 2
    spanA = ((conv + base) / 2).shift(26)
    spanB = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    chiko = c.shift(-26)
    return conv, base, spanA, spanB, chiko


def obv(c, v):
    sign = np.sign(c.diff().fillna(0))
    return (sign * v).cumsum()


def mfi(h, l, c, v, n=14):
    tp = (h + l + c) / 3
    mf = tp * v
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    mr = pos.rolling(n).sum() / neg.rolling(n).sum().replace(0, np.nan)
    return 100 - 100 / (1 + mr)


def cmf(h, l, c, v, n=20):
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = mfm * v
    return mfv.rolling(n).sum() / v.rolling(n).sum()


def vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


def keltner(h, l, c, n=20, mult=2):
    m, a = ema(c, n), atr(h, l, c, n)
    return m + mult * a, m, m - mult * a


def donchian(h, l, n=20):
    return h.rolling(n).max(), l.rolling(n).min()


def hurst(ts):
    ts = np.asarray(ts)
    if len(ts) < 100:
        return float("nan")
    lags = range(2, 50)
    tau = []
    for lag in lags:
        diff = np.subtract(ts[lag:], ts[:-lag])
        s = np.std(diff)
        if s <= 0 or not np.isfinite(s):
            return float("nan")
        tau.append(np.sqrt(s))
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0] * 2.0)


# ============================== 计算 ==============================

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    o = df.copy()
    o["ema20"] = ema(o["close"], 20)
    o["ema50"] = ema(o["close"], 50)
    o["ema200"] = ema(o["close"], 200)
    o["sma20"] = sma(o["close"], 20)
    o["sma50"] = sma(o["close"], 50)
    o["rsi14"] = rsi(o["close"], 14)
    o["macd"], o["macd_sig"], o["macd_hist"] = macd(o["close"])
    o["bb_up"], o["bb_mid"], o["bb_dn"] = bbands(o["close"])
    o["atr14"] = atr(o["high"], o["low"], o["close"], 14)
    o["adx14"], o["pdi"], o["mdi"] = adx(o["high"], o["low"], o["close"], 14)
    o["stoch_k"], o["stoch_d"] = stoch(o["high"], o["low"], o["close"])
    o["st"], o["st_dir"] = supertrend(o["high"], o["low"], o["close"])
    o["ich_conv"], o["ich_base"], o["ich_a"], o["ich_b"], o["ich_chi"] = ichimoku(o["high"], o["low"], o["close"])
    o["obv"] = obv(o["close"], o["volume"])
    o["mfi14"] = mfi(o["high"], o["low"], o["close"], o["volume"], 14)
    o["cmf20"] = cmf(o["high"], o["low"], o["close"], o["volume"], 20)
    o["vwap"] = vwap(o)
    o["kc_up"], o["kc_mid"], o["kc_dn"] = keltner(o["high"], o["low"], o["close"])
    o["dc_up"], o["dc_dn"] = donchian(o["high"], o["low"], 20)
    o["ret"] = o["close"].pct_change()
    return o


def snap(df: pd.DataFrame) -> dict:
    d = df.iloc[-1]
    out = {
        "last_close": d["close"],
        "ema20": d["ema20"], "ema50": d["ema50"], "ema200": d["ema200"],
        "rsi14": d["rsi14"],
        "macd": d["macd"], "macd_sig": d["macd_sig"], "macd_hist": d["macd_hist"],
        "bb_up": d["bb_up"], "bb_mid": d["bb_mid"], "bb_dn": d["bb_dn"],
        "bb_pctB": (d["close"] - d["bb_dn"]) / (d["bb_up"] - d["bb_dn"]) if (d["bb_up"] - d["bb_dn"]) else None,
        "bb_width": (d["bb_up"] - d["bb_dn"]) / d["bb_mid"] if d["bb_mid"] else None,
        "atr14": d["atr14"], "atr_pct": d["atr14"] / d["close"] * 100 if d["close"] else None,
        "adx14": d["adx14"], "+DI": d["pdi"], "-DI": d["mdi"],
        "stoch_k": d["stoch_k"], "stoch_d": d["stoch_d"],
        "supertrend": d["st"], "st_dir": int(d["st_dir"]) if pd.notna(d["st_dir"]) else None,
        "ichimoku_conv": d["ich_conv"], "ichimoku_base": d["ich_base"],
        "ichimoku_spanA": d["ich_a"], "ichimoku_spanB": d["ich_b"],
        "mfi14": d["mfi14"], "cmf20": d["cmf20"],
        "vwap": d["vwap"],
        "kc_up": d["kc_up"], "kc_dn": d["kc_dn"],
        "donchian_up": d["dc_up"], "donchian_dn": d["dc_dn"],
    }
    return {k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v))
            for k, v in out.items()}


def quant_metrics(df: pd.DataFrame, tf: str) -> dict:
    d = df.dropna(subset=["ret"])
    r = d["ret"]
    out = {
        "n_bars": int(len(d)),
        "mean_return": float(r.mean()) if len(r) else None,
        "std_return": float(r.std()) if len(r) else None,
        "skew": float(r.skew()) if len(r) > 2 else None,
        "kurt": float(r.kurt()) if len(r) > 3 else None,
        "realized_vol_annualized_pct":
            float(r.std() * np.sqrt(ANNUAL_BARS[tf]) * 100) if len(r) > 1 else None,
        "max_drawdown_pct":
            float((d["close"] / d["close"].cummax() - 1).min() * 100) if len(d) else None,
        "autocorr_lag1": float(r.autocorr(1)) if len(r) > 5 else None,
        "hurst": hurst(d["close"].values) if len(d) >= 120 else None,
    }
    return out


# ============================== 主流程 ==============================

def main() -> None:
    # 1) 加载所有数据
    ohlcv = {tf: load_ohlcv(tf) for tf in TIMEFRAMES}
    ind = {tf: enrich(df) for tf, df in ohlcv.items()}
    snaps = {tf: snap(df) for tf, df in ind.items()}
    quants = {tf: quant_metrics(df, tf) for tf, df in ind.items()}

    fr = load_simple_csv("funding_rate")
    oi = load_simple_csv("open_interest")
    ls = load_simple_csv("long_short_ratio")
    tv = load_simple_csv("taker_volume")
    fc = load_json("funding_current")
    tk = load_json("ticker")

    # 2) 衍生品聚合
    def safe(x): return None if x is None or (isinstance(x, float) and math.isnan(x)) else x

    deriv = {}
    if not fr.empty:
        fr["fundingRate"] = pd.to_numeric(fr["fundingRate"], errors="coerce")
        deriv.update({
            "funding_rate_current": safe(fc.get("fundingRate")),
            "funding_rate_next_time_sh": (
                datetime.fromtimestamp(fc["nextFundingTimestamp"] / 1000, tz=SH_TZ).isoformat()
                if fc.get("nextFundingTimestamp") else fc.get("nextFundingDatetime")
            ),
            "funding_rate_15d_avg": float(fr["fundingRate"].mean()),
            "funding_rate_15d_sum_annualized_pct":
                float(fr["fundingRate"].sum() * 365 / (len(fr) / 3) * 100) if len(fr) >= 3 else None,
            "funding_rate_15d_max": float(fr["fundingRate"].max()),
            "funding_rate_15d_min": float(fr["fundingRate"].min()),
            "funding_rate_neg_share_pct": float((fr["fundingRate"] < 0).mean() * 100),
        })
    if not oi.empty:
        oi["oi"] = pd.to_numeric(oi["oi"], errors="coerce")
        deriv.update({
            "oi_first": float(oi["oi"].iloc[0]),
            "oi_last": float(oi["oi"].iloc[-1]),
            "oi_change_pct": float((oi["oi"].iloc[-1] / oi["oi"].iloc[0] - 1) * 100),
            "oi_max": float(oi["oi"].max()),
            "oi_min": float(oi["oi"].min()),
        })
    if not ls.empty:
        ls["ratio"] = pd.to_numeric(ls["ratio"], errors="coerce")
        deriv.update({
            "ls_ratio_last": float(ls["ratio"].iloc[-1]),
            "ls_ratio_avg": float(ls["ratio"].mean()),
            "ls_ratio_max": float(ls["ratio"].max()),
            "ls_ratio_min": float(ls["ratio"].min()),
        })
    if not tv.empty:
        tv["buyVol"] = pd.to_numeric(tv["buyVol"], errors="coerce")
        tv["sellVol"] = pd.to_numeric(tv["sellVol"], errors="coerce")
        deriv.update({
            "taker_buy_15d": float(tv["buyVol"].sum()),
            "taker_sell_15d": float(tv["sellVol"].sum()),
            "taker_buy_sell_ratio_15d": float(tv["buyVol"].sum() / tv["sellVol"].sum()),
            "taker_buy_sell_ratio_24h":
                float(tv["buyVol"].tail(24).sum() / tv["sellVol"].tail(24).sum()) if len(tv) >= 24 else None,
            "cvd_15d": float((tv["buyVol"] - tv["sellVol"]).sum()),
            "cvd_24h": float((tv["buyVol"] - tv["sellVol"]).tail(24).sum()) if len(tv) >= 24 else None,
        })
    deriv.update({
        "ticker_last": tk.get("last"),
        "ticker_high_24h": tk.get("high"),
        "ticker_low_24h": tk.get("low"),
        "ticker_baseVolume_24h": tk.get("baseVolume"),
        "ticker_change_pct_24h": tk.get("percentage"),
    })

    # 3) 关键位
    d1h = ind["1h"]
    high_n = float(d1h["high"].max()); low_n = float(d1h["low"].min())
    last = float(d1h["close"].iloc[-1]); rng = high_n - low_n
    fibs = {
        "0% (低)": low_n,
        "23.6%": low_n + rng * 0.236,
        "38.2%": low_n + rng * 0.382,
        "50%": low_n + rng * 0.5,
        "61.8%": low_n + rng * 0.618,
        "78.6%": low_n + rng * 0.786,
        "100% (高)": high_n,
    }
    d1d = ind["1d"].iloc[-2]
    P = (d1d["high"] + d1d["low"] + d1d["close"]) / 3
    R1 = 2 * P - d1d["low"]; S1 = 2 * P - d1d["high"]
    R2 = P + (d1d["high"] - d1d["low"]); S2 = P - (d1d["high"] - d1d["low"])
    R3 = d1d["high"] + 2 * (P - d1d["low"]); S3 = d1d["low"] - 2 * (d1d["high"] - P)
    pivots = {"P": float(P), "R1": float(R1), "R2": float(R2), "R3": float(R3),
              "S1": float(S1), "S2": float(S2), "S3": float(S3)}

    # 4) 综合评分
    score, score_table = compute_score(snaps, deriv, last, high_n)

    # 5) 落盘 analysis.json + 撰写 REPORT.md
    summary = {
        "generated_at_sh": datetime.now(SH_TZ).isoformat(),
        "symbol": "SOL-USDT-SWAP (OKX)",
        "lookback_days": 15,
        "snapshots": snaps,
        "quant": quants,
        "derivatives": deriv,
        "fib": {k: float(v) for k, v in fibs.items()},
        "pivots_classic_prev_day": pivots,
        "context": {
            "high_window": high_n, "low_window": low_n, "last_price": last,
            "range_pct": (high_n - low_n) / low_n * 100,
            "price_pos_in_range_pct": (last - low_n) / (high_n - low_n) * 100,
        },
        "score": score, "score_table": score_table,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    write_report_md(summary, ind, fr, oi, ls, tv)
    print(f"✅ 报告已生成：{OUT_REPORT}")


# ============================== 评分 ==============================

def compute_score(snaps, deriv, last, high_n):
    rows = []

    # 多周期均线排列（覆盖中短期所有可用周期）
    def aligned_up(s):
        if s.get("ema20") is None or s.get("ema50") is None or s.get("ema200") is None:
            return 0
        return 1 if s["last_close"] > s["ema20"] > s["ema50"] > s["ema200"] else \
               -1 if s["last_close"] < s["ema20"] < s["ema50"] < s["ema200"] else 0

    ma_tfs = ["4h", "2h", "1h", "30m", "15m", "5m"]
    up_count = sum(max(0, aligned_up(snaps[tf])) for tf in ma_tfs)
    dn_count = sum(-min(0, aligned_up(snaps[tf])) for tf in ma_tfs)
    rows.append(("多周期均线排列", up_count - dn_count, len(ma_tfs), -len(ma_tfs)))

    # MACD 1h/2h/4h
    m_score = 0
    for tf in ["1h", "2h", "4h"]:
        s = snaps[tf]
        if s["macd"] is None: continue
        if s["macd"] > s["macd_sig"] and s["macd_hist"] > 0: m_score += 1
        elif s["macd"] < s["macd_sig"] and s["macd_hist"] < 0: m_score -= 1
    rows.append(("MACD (1h/2h/4h)", m_score, 3, -3))

    # ADX 趋势强度（1h）
    a = snaps["1h"]
    if a["adx14"] and a["adx14"] > 25:
        rows.append(("ADX(1h) 强趋势", 1 if a["+DI"] > a["-DI"] else -1, 1, -1))
    else:
        rows.append(("ADX(1h) 强趋势", 0, 1, -1))

    # RSI/Stoch/MFI 极端（超买扣分给多）
    s1h = snaps["1h"]
    if s1h["rsi14"] and s1h["rsi14"] > 75:
        rows.append(("RSI/MFI 超买（短期）", -1, 1, -2))
    elif s1h["rsi14"] and s1h["rsi14"] < 25:
        rows.append(("RSI/MFI 超卖（短期）", 1, 2, -1))
    else:
        rows.append(("RSI/MFI 极端", 0, 2, -2))

    # 布林 4h
    s4h = snaps["4h"]
    if s4h["bb_pctB"] is not None:
        if s4h["bb_pctB"] > 0.95:
            rows.append(("布林 (4h)", -1, 1, -1))
        elif s4h["bb_pctB"] < 0.05:
            rows.append(("布林 (4h)", 1, 1, -1))
        else:
            rows.append(("布林 (4h)", 0, 1, -1))
    else:
        rows.append(("布林 (4h)", 0, 1, -1))

    # Ichimoku 1h
    s = snaps["1h"]
    if all(s.get(k) is not None for k in ["ichimoku_spanA", "ichimoku_spanB"]):
        cloud_top = max(s["ichimoku_spanA"], s["ichimoku_spanB"])
        cloud_bot = min(s["ichimoku_spanA"], s["ichimoku_spanB"])
        if last > cloud_top: rows.append(("Ichimoku (1h)", 1, 1, -1))
        elif last < cloud_bot: rows.append(("Ichimoku (1h)", -1, 1, -1))
        else: rows.append(("Ichimoku (1h)", 0, 1, -1))
    else:
        rows.append(("Ichimoku (1h)", 0, 1, -1))

    # SuperTrend 1h/2h/4h
    st = 0
    for tf in ["1h", "2h", "4h"]:
        d = snaps[tf]["st_dir"]
        if d is None: continue
        st += 1 if d == 1 else -1
    rows.append(("SuperTrend (1h/2h/4h)", st, 3, -3))

    # CMF/CVD 资金流
    cvd = deriv.get("cvd_24h") or 0
    cmf_1h = snaps["1h"].get("cmf20") or 0
    flow = 0
    if cvd > 0 and cmf_1h > 0: flow = 1
    elif cvd < 0 and cmf_1h < 0: flow = -1
    rows.append(("资金流 (CMF+CVD)", flow, 1, -1))

    # 资金费率反向
    fr_now = deriv.get("funding_rate_current") or 0
    fr_avg = deriv.get("funding_rate_15d_avg") or 0
    if fr_now < -0.0001 or fr_avg < -0.00005:
        rows.append(("资金费率（反向利好多）", 2, 2, -2))
    elif fr_now > 0.0003:
        rows.append(("资金费率（多头拥挤）", -2, 2, -2))
    else:
        rows.append(("资金费率", 0, 2, -2))

    # OI 走势
    oi_chg = deriv.get("oi_change_pct")
    if oi_chg is not None:
        if abs(oi_chg) < 3:
            rows.append(("OI 走势（去杠杆/低拥挤）", 1, 1, -1))
        else:
            rows.append(("OI 走势", 0, 1, -1))
    else:
        rows.append(("OI 走势", 0, 1, -1))

    # 多空账户比
    ls_last = deriv.get("ls_ratio_last")
    ls_avg = deriv.get("ls_ratio_avg")
    if ls_last and ls_avg and ls_last < ls_avg * 0.85:
        rows.append(("多空账户比（散户卖出）", 1, 1, -1))
    elif ls_last and ls_avg and ls_last > ls_avg * 1.15:
        rows.append(("多空账户比（散户买入）", -1, 1, -1))
    else:
        rows.append(("多空账户比", 0, 1, -1))

    score = sum(v for _, v, _, _ in rows)
    return score, rows


# ============================== 报告 ==============================

def write_report_md(s: dict, ind: dict, fr, oi, ls, tv) -> None:
    snaps = s["snapshots"]; quants = s["quant"]; deriv = s["derivatives"]
    ctx = s["context"]; fibs = s["fib"]; pivots = s["pivots_classic_prev_day"]

    last = ctx["last_price"]; high_n = ctx["high_window"]; low_n = ctx["low_window"]
    score = s["score"]; score_table = s["score_table"]
    score_max = sum(mx for _, _, mx, _ in score_table)
    score_min = sum(mn for _, _, _, mn in score_table)

    # 趋势方向（用于摘要）
    if score >= 5: outlook = "**偏多（趋势完好）**"
    elif score <= -5: outlook = "**偏空（趋势承压）**"
    else: outlook = "**中性 / 震荡**"

    # 关键位（综合）
    levels = build_levels(snaps, fibs, pivots, last)

    lines: list[str] = []
    lines.append("# OKX SOL‑USDT 永续合约 行情分析与交易计划报告")
    lines.append("")
    lines.append(f"> **数据源**：OKX 公共 API（通过 CCXT 拉取，由 `scripts/fetch_data.py` 生成的 `data/` 数据集）")
    lines.append(f"> **品种**：`SOL-USDT-SWAP`")
    lines.append(f"> **数据窗口**：最近 {s['lookback_days']} 天（时区 Asia/Shanghai, UTC+8）")
    lines.append(f"> **报告生成时间**：{s['generated_at_sh']}")
    lines.append(f"> **当前价**：**{last:.4g} USDT**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 0. 数据使用清单")
    lines.append("")
    lines.append("| 数据集 | 文件 | 行数 |")
    lines.append("|---|---|---:|")
    for tf in TIMEFRAMES:
        lines.append(f"| OHLCV {tf} | `data/ohlcv_{tf}.csv` | {len(ind[tf])} |")
    lines.append(f"| 资金费率历史 | `data/funding_rate.csv` | {len(fr)} |")
    lines.append(f"| 持仓量历史 | `data/open_interest.csv` | {len(oi)} |")
    lines.append(f"| 多空账户比 | `data/long_short_ratio.csv` | {len(ls)} |")
    lines.append(f"| 主动买卖盘 | `data/taker_volume.csv` | {len(tv)} |")
    lines.append("")
    lines.append("> 完整的数据清单与质量检测见 [`data/DATA_QUALITY_REPORT.md`](data/DATA_QUALITY_REPORT.md)。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## 1. 行情核心数据 ({s['lookback_days']}天)")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| 最新价 | **{last:.4g}** |")
    lines.append(f"| 24h 涨跌 | {fmt_pct(deriv.get('ticker_change_pct_24h'))} |")
    lines.append(f"| 24h 高 / 低 | {fmt_num(deriv.get('ticker_high_24h'))} / {fmt_num(deriv.get('ticker_low_24h'))} |")
    lines.append(f"| {s['lookback_days']}d 高 / 低 | **{high_n:.4g} / {low_n:.4g}** |")
    lines.append(f"| {s['lookback_days']}d 振幅 | **{ctx['range_pct']:.2f}%** |")
    lines.append(f"| 当前价在区间位置 | **{ctx['price_pos_in_range_pct']:.1f}%** |")
    lines.append(f"| 24h 基础币成交量 | {fmt_num(deriv.get('ticker_baseVolume_24h'))} SOL |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. 永续衍生品维度")
    lines.append("")
    lines.append("### 2.1 资金费率（Funding Rate）")
    lines.append("")
    lines.append("| 指标 | 值 | 解读 |")
    lines.append("|---|---:|---|")
    lines.append(f"| 当前费率 | **{fmt_rate(deriv.get('funding_rate_current'))}** | {interpret_rate_current(deriv.get('funding_rate_current'))} |")
    lines.append(f"| 下一次结算 (SH) | {deriv.get('funding_rate_next_time_sh') or '—'} | — |")
    lines.append(f"| {s['lookback_days']}d 平均 (每 8h) | {fmt_rate(deriv.get('funding_rate_15d_avg'))} | {interpret_rate_avg(deriv.get('funding_rate_15d_avg'))} |")
    if deriv.get("funding_rate_15d_sum_annualized_pct") is not None:
        lines.append(f"| {s['lookback_days']}d 年化等价 | **{deriv['funding_rate_15d_sum_annualized_pct']:.2f}%/年** | — |")
    lines.append(f"| {s['lookback_days']}d 区间 | [{fmt_rate(deriv.get('funding_rate_15d_min'))}, {fmt_rate(deriv.get('funding_rate_15d_max'))}] | — |")
    if deriv.get("funding_rate_neg_share_pct") is not None:
        lines.append(f"| 负费率出现占比 | {deriv['funding_rate_neg_share_pct']:.1f}% | 反映窗口内空头补贴多头比例 |")
    lines.append("")
    lines.append("### 2.2 持仓量（Open Interest）")
    lines.append("")
    if "oi_first" in deriv:
        lines.append("| 指标 | 值 |")
        lines.append("|---|---:|")
        lines.append(f"| 期初 OI | {deriv['oi_first']:.2f} |")
        lines.append(f"| 期末 OI | {deriv['oi_last']:.2f} |")
        lines.append(f"| {s['lookback_days']}d OI 变化 | **{deriv['oi_change_pct']:+.2f}%** |")
        lines.append(f"| 峰值 / 谷值 | {deriv['oi_max']:.2f} / {deriv['oi_min']:.2f} |")
        lines.append("")
        lines.append(f"> {interpret_oi(deriv['oi_change_pct'], deriv_change=deriv.get('ticker_change_pct_24h'))}")
    lines.append("")
    lines.append("### 2.3 多空账户比")
    lines.append("")
    if "ls_ratio_last" in deriv:
        lines.append("| 指标 | 值 |")
        lines.append("|---|---:|")
        lines.append(f"| 最新 | **{deriv['ls_ratio_last']:.2f}** |")
        lines.append(f"| {s['lookback_days']}d 平均 | {deriv['ls_ratio_avg']:.2f} |")
        lines.append(f"| {s['lookback_days']}d 高 / 低 | {deriv['ls_ratio_max']:.2f} / {deriv['ls_ratio_min']:.2f} |")
        lines.append("")
        lines.append(f"> {interpret_ls(deriv['ls_ratio_last'], deriv['ls_ratio_avg'])}")
    lines.append("")
    lines.append("### 2.4 主动买卖盘 (Taker Flow / CVD)")
    lines.append("")
    if "cvd_15d" in deriv:
        lines.append("| 指标 | 值 |")
        lines.append("|---|---:|")
        if deriv.get("taker_buy_sell_ratio_24h") is not None:
            lines.append(f"| 24h 主动买/卖比 | **{deriv['taker_buy_sell_ratio_24h']:.4f}** |")
        lines.append(f"| {s['lookback_days']}d 主动买/卖比 | {deriv['taker_buy_sell_ratio_15d']:.4f} |")
        if deriv.get("cvd_24h") is not None:
            lines.append(f"| 24h CVD | **{deriv['cvd_24h']:+.2f}** |")
        lines.append(f"| {s['lookback_days']}d CVD | {deriv['cvd_15d']:+.2f} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. 多周期技术指标矩阵")
    lines.append("")
    lines.append("### 3.1 趋势方向（均线 / Ichimoku / SuperTrend / VWAP）")
    lines.append("")
    lines.append("| 周期 | 价格 vs EMA20/50/200 | SuperTrend | Ichimoku (云) | 累积 VWAP | 综合方向 |")
    lines.append("|---|---|---|---|---:|---|")
    for tf in TIMEFRAMES:
        sn = snaps[tf]
        ma = ema_status(sn)
        st = "—" if sn["st_dir"] is None else ("✅ 多" if sn["st_dir"] == 1 else "❌ 空")
        ich = ichimoku_status(sn, sn["last_close"])
        vwap_v = fmt_num(sn["vwap"])
        direction = combined_dir(sn)
        lines.append(f"| {tf} | {ma} | {st} | {ich} | {vwap_v} | {direction} |")
    lines.append("")
    lines.append("### 3.2 动量与超买超卖（RSI / Stoch / MFI）")
    lines.append("")
    lines.append("| 周期 | RSI(14) | Stoch K/D | MFI(14) | 状态 |")
    lines.append("|---|---:|---|---:|---|")
    for tf in TIMEFRAMES:
        sn = snaps[tf]
        rsi_v = fmt_num(sn["rsi14"], 1)
        stoch_v = f"{fmt_num(sn['stoch_k'], 0)}/{fmt_num(sn['stoch_d'], 0)}"
        mfi_v = fmt_num(sn["mfi14"], 1)
        status = momentum_status(sn)
        lines.append(f"| {tf} | {rsi_v} | {stoch_v} | {mfi_v} | {status} |")
    lines.append("")
    lines.append("### 3.3 趋势强度（ADX/DI）")
    lines.append("")
    lines.append("| 周期 | ADX(14) | +DI | −DI | 评级 |")
    lines.append("|---|---:|---:|---:|---|")
    for tf in TIMEFRAMES:
        sn = snaps[tf]
        adx_v = sn["adx14"]; pd_v = sn["+DI"]; nd_v = sn["-DI"]
        rate = adx_rate(adx_v, pd_v, nd_v)
        lines.append(f"| {tf} | {fmt_num(adx_v,1)} | {fmt_num(pd_v,1)} | {fmt_num(nd_v,1)} | {rate} |")
    lines.append("")
    lines.append("### 3.4 波动率（ATR / Bollinger）")
    lines.append("")
    lines.append("| 周期 | ATR(14) | ATR/Price | BB %B | BB Width |")
    lines.append("|---|---:|---:|---:|---:|")
    for tf in TIMEFRAMES:
        sn = snaps[tf]
        lines.append(f"| {tf} | {fmt_num(sn['atr14'],3)} | {fmt_pct(sn['atr_pct'])} | "
                     f"{fmt_num(sn['bb_pctB'],2)} | {fmt_pct(safe_mul100(sn['bb_width']))} |")
    lines.append("")
    lines.append("### 3.5 资金流（OBV / CMF）")
    lines.append("")
    lines.append("| 周期 | CMF(20) | 简评 |")
    lines.append("|---|---:|---|")
    for tf in ["15m", "30m", "1h", "2h", "4h"]:
        cmf_v = snaps[tf]["cmf20"]
        lines.append(f"| {tf} | {fmt_num(cmf_v,3)} | {cmf_comment(cmf_v)} |")
    lines.append("")
    lines.append("### 3.6 关键支撑 / 阻力位（综合）")
    lines.append("")
    lines.append("| 类别 | 位置 (USDT) | 来源 |")
    lines.append("|---|---:|---|")
    for tag, price, src in levels:
        lines.append(f"| {tag} | **{price:.4g}** | {src} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. 量化指标维度")
    lines.append("")
    lines.append("| 指标 | 1h | 2h | 4h | 1d |")
    lines.append("|---|---:|---:|---:|---:|")
    def qv(tf, k, fmt=lambda x: fmt_num(x, 4)):
        v = quants[tf].get(k); return "—" if v is None else fmt(v)
    lines.append(f"| 样本数 | {quants['1h'].get('n_bars')} | {quants['2h'].get('n_bars')} | "
                 f"{quants['4h'].get('n_bars')} | {quants['1d'].get('n_bars')} |")
    lines.append(f"| 年化已实现波动率 | {qv('1h','realized_vol_annualized_pct', lambda x: f'{x:.1f}%')} | "
                 f"{qv('2h','realized_vol_annualized_pct', lambda x: f'{x:.1f}%')} | "
                 f"{qv('4h','realized_vol_annualized_pct', lambda x: f'{x:.1f}%')} | "
                 f"{qv('1d','realized_vol_annualized_pct', lambda x: f'{x:.1f}%')} |")
    lines.append(f"| 最大回撤 | {qv('1h','max_drawdown_pct', lambda x: f'{x:.2f}%')} | "
                 f"{qv('2h','max_drawdown_pct', lambda x: f'{x:.2f}%')} | "
                 f"{qv('4h','max_drawdown_pct', lambda x: f'{x:.2f}%')} | "
                 f"{qv('1d','max_drawdown_pct', lambda x: f'{x:.2f}%')} |")
    lines.append(f"| 收益偏度 (Skew) | {qv('1h','skew', lambda x: f'{x:+.2f}')} | "
                 f"{qv('2h','skew', lambda x: f'{x:+.2f}')} | "
                 f"{qv('4h','skew', lambda x: f'{x:+.2f}')} | "
                 f"{qv('1d','skew', lambda x: f'{x:+.2f}')} |")
    lines.append(f"| 收益峰度 (Kurt) | {qv('1h','kurt', lambda x: f'{x:+.2f}')} | "
                 f"{qv('2h','kurt', lambda x: f'{x:+.2f}')} | "
                 f"{qv('4h','kurt', lambda x: f'{x:+.2f}')} | "
                 f"{qv('1d','kurt', lambda x: f'{x:+.2f}')} |")
    lines.append(f"| Lag‑1 自相关 | {qv('1h','autocorr_lag1', lambda x: f'{x:+.3f}')} | "
                 f"{qv('2h','autocorr_lag1', lambda x: f'{x:+.3f}')} | "
                 f"{qv('4h','autocorr_lag1', lambda x: f'{x:+.3f}')} | "
                 f"{qv('1d','autocorr_lag1', lambda x: f'{x:+.3f}')} |")
    lines.append(f"| Hurst 指数 | {qv('1h','hurst', lambda x: f'{x:.3f}')} | "
                 f"{qv('2h','hurst', lambda x: f'{x:.3f}')} | "
                 f"{qv('4h','hurst', lambda x: f'{x:.3f}')} | "
                 f"{qv('1d','hurst', lambda x: f'{x:.3f}')} |")
    lines.append("")
    lines.append("**量化解读：**")
    h1 = quants["1h"].get("hurst"); ac1h = quants["1h"].get("autocorr_lag1")
    ac4h = quants["4h"].get("autocorr_lag1"); ac2h = quants["2h"].get("autocorr_lag1")
    lines.append(f"- **Hurst (1h) = {fmt_num(h1,3)}** → {hurst_comment(h1)}")
    lines.append(f"- **1h 自相关 = {fmt_num(ac1h,3)}，2h 自相关 = {fmt_num(ac2h,3)}，4h 自相关 = {fmt_num(ac4h,3)}** → "
                 f"{ac_comment(ac4h)}")
    lines.append("- **正偏 + 高峰度**：行情存在尾部跳跃风险，仓位管理优于"
                 "胜率优化；建议固定百分比 R 风控。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. 多维度信号评分")
    lines.append("")
    lines.append("| 维度 | 得分 | 区间 |")
    lines.append("|---|---:|---:|")
    for name, v, mx, mn in score_table:
        lines.append(f"| {name} | **{v:+d}** | [{mn}, {mx}] |")
    lines.append(f"| **总分** | **{score:+d}** | [{score_min}, {score_max}] |")
    lines.append("")
    lines.append(f"> **综合方向**：{outlook}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. 交易计划")
    lines.append("")
    lines.append("> ⚠️ **以下计划仅供研究参考，不构成投资建议。所有杠杆建议 ≤ 5×；单次风险敞口 ≤ 账户净值 2%**。")
    lines.append("")
    plans = build_plans(last, snaps, deriv, levels, outlook)
    for plan in plans:
        lines.append(plan)
        lines.append("")
    lines.append("### 6.x 仓位组合模板（保守 / 中性 / 激进）")
    lines.append("")
    lines.append("| 配置 | 中期波段多 (E) | 战术 (A/B) | 网格 (F) | 套保/逆势 (D) | 量化均值回归 (G) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append("| 保守型 | 50% | 10% | 10% | 0% | 5% |")
    lines.append("| 中性型 | 35% | 25% | 15% | 5% | 10% |")
    lines.append("| 激进型 | 20% | 35% | 15% | 10% | 10% |")
    lines.append("")
    lines.append("> 总仓位上限建议 ≤ 75% 名义敞口（永续合约保证金占用 < 15% 净值）。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. 实时监控信号清单")
    lines.append("")
    lines.append("| 等级 | 信号 | 阈值 | 含义 |")
    lines.append("|---|---|---|---|")
    lines.append("| 🚨 红 | 资金费率 > **+0.03%** | 单期 | 多头开始拥挤，警惕回调 |")
    lines.append("| 🚨 红 | OI 24h 增加 > **+8%** 且 价跌 | — | 新空头进场，趋势反转风险 |")
    lines.append(f"| 🚨 红 | 1h 收盘 < **{levels_lookup(levels, '🟢', 4):.4g}** | 4h | 强支撑失守，趋势警告 |")
    lines.append("| ⚠️ 黄 | 多空账户比 < **1.5** | — | 散户翻空，常为底部前 |")
    lines.append("| ⚠️ 黄 | 1h RSI > 82 且未新高 | 1h 顶背离 | 短期回调高概率 |")
    lines.append(f"| ✅ 绿 | 1h 收盘 > **{levels_lookup(levels, '🔴', 0):.4g}**，OI 同步抬升 | 1h | 突破确认，启动 B 计划 |")
    lines.append(f"| ✅ 绿 | 回踩 **{levels_lookup(levels, '🟢', 2):.4g}** 不破 + CVD 转正 | 4h | 回踩成立，启动 C/E 加仓 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 8. 风险提示")
    lines.append("")
    lines.append("1. 本报告基于过去 15 日数据，未涵盖宏观与一次性事件（ETF、解锁、CEX 突发）；")
    lines.append("2. **资金费率每 8 小时结算一次**，长期持仓需计入 P&L；")
    lines.append("3. 永续合约高杠杆下，单根 4h K 线波动可达 ±2σ ATR，止损必须严格执行；")
    lines.append("4. OKX 不同保证金模式下 IM/MM 不同，请确认实际可用保证金；")
    lines.append("5. 量化指标 (Hurst / 自相关) 受样本量限制，结论作辅助。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 9. 一句话总结")
    lines.append("")
    lines.append(f"> {one_line_summary(outlook, snaps, deriv, levels)}")
    lines.append("")
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================== 文案 / 辅助 ==============================

def fmt_num(x, d=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{d}f}"


def fmt_pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.2f}%" if isinstance(x, (int, float)) else str(x)


def fmt_rate(x):
    if x is None: return "—"
    return f"{x*100:+.4f}%"


def safe_mul100(x):
    return None if x is None else x * 100


def interpret_rate_current(r):
    if r is None: return "—"
    if r < -0.0001: return "微负，**空头补贴多头**"
    if r > 0.0003: return "偏高，**多头拥挤，警惕回调**"
    return "中性"


def interpret_rate_avg(r):
    if r is None: return "—"
    if r < 0: return "窗口整体负值，**非多头主导**"
    return "窗口整体为正，多头主导"


def interpret_oi(chg, deriv_change=None):
    if chg is None: return ""
    if abs(chg) < 3:
        if deriv_change and deriv_change > 0:
            return ("**价涨 + OI 几乎持平** = 去杠杆 / 空头平仓型上涨。"
                    "此类行情回调浅、续航中等；若价格突破后 OI 同步抬升即趋势确认。")
        if deriv_change and deriv_change < 0:
            return "**价跌 + OI 持平** = 多头止损平仓，关注是否构筑底部。"
        return "OI 持平，市场杠杆度变化不大。"
    if chg > 0: return "**OI 显著抬升**，新仓位入场，注意配合价格方向判断真假突破。"
    return "**OI 显著回落**，前期持仓被动平仓，关注是否进入震荡。"


def interpret_ls(last, avg):
    if last < avg * 0.85:
        return "**散户多头比例下降**（聪明钱反向特征，常为延续行情的特征）。"
    if last > avg * 1.15:
        return "**散户多头比例上升**，注意拥挤风险。"
    return "多空账户比处于近期均值附近。"


def ema_status(s):
    if any(s.get(k) is None for k in ["ema20", "ema50", "ema200"]):
        return "—"
    c = s["last_close"]
    if c > s["ema20"] > s["ema50"] > s["ema200"]: return "✅ 全多头排列"
    if c < s["ema20"] < s["ema50"] < s["ema200"]: return "❌ 全空头排列"
    if c > s["ema50"] > s["ema200"]: return "🟡 中长期多"
    if c < s["ema50"] < s["ema200"]: return "🟠 中长期空"
    return "🔁 缠绕"


def ichimoku_status(s, price):
    a, b = s.get("ichimoku_spanA"), s.get("ichimoku_spanB")
    if a is None or b is None: return "—"
    top, bot = max(a, b), min(a, b)
    if price > top: return "云上 (强多)"
    if price < bot: return "云下 (强空)"
    return "云中 (震荡)"


def combined_dir(s):
    score = 0
    if s.get("ema20") and s.get("ema50") and s.get("ema200"):
        if s["last_close"] > s["ema20"] > s["ema50"] > s["ema200"]: score += 2
        elif s["last_close"] < s["ema20"] < s["ema50"] < s["ema200"]: score -= 2
    if s.get("st_dir") == 1: score += 1
    elif s.get("st_dir") == -1: score -= 1
    if score >= 3: return "**强多**"
    if score >= 1: return "偏多"
    if score <= -3: return "**强空**"
    if score <= -1: return "偏空"
    return "中性"


def momentum_status(s):
    rsi = s.get("rsi14") or 0
    mfi = s.get("mfi14") or 0
    if rsi > 75 or mfi > 80: return "🔥 超买"
    if rsi < 25 or mfi < 20: return "❄ 超卖"
    if rsi > 60: return "偏强"
    if rsi < 40: return "偏弱"
    return "中性"


def adx_rate(adx, pdi, ndi):
    if adx is None: return "—"
    if adx > 40:
        return ("强多" if pdi and pdi > ndi else "强空") + f" (ADX {adx:.0f})"
    if adx > 25:
        return ("中等多" if pdi and pdi > ndi else "中等空") + f" (ADX {adx:.0f})"
    return f"无趋势 (ADX {adx:.0f})"


def cmf_comment(v):
    if v is None: return "—"
    if v > 0.15: return "资金强流入"
    if v > 0.05: return "资金温和流入"
    if v < -0.15: return "资金强流出"
    if v < -0.05: return "资金温和流出"
    return "中性"


def hurst_comment(h):
    if h is None: return "样本不足"
    if h < 0.45: return "**反持续 / 均值回归**特性，1h 级别更适合区间震荡策略。"
    if h > 0.55: return "**持续 / 趋势**特性，1h 级别趋势跟随胜率更高。"
    return "近似随机游走。"


def ac_comment(ac):
    if ac is None: return ""
    if ac < -0.1: return "**短期均值回归显著**，'追涨' 在该周期上劣势。"
    if ac > 0.1: return "**短期动量持续**，可顺势加仓。"
    return "动量与回归特征中性。"


def build_levels(snaps, fibs, pivots, last):
    """从指标快照中抽取关键支撑/阻力位，按距离当前价排序为表格。"""
    raw = []
    s1h = snaps["1h"]; s2h = snaps["2h"]; s4h = snaps["4h"]
    s15m = snaps["15m"]; s30m = snaps["30m"]
    s1d = snaps["1d"]; s5m = snaps["5m"]
    # 阻力
    for name, v in [
        ("Pivot R3", pivots["R3"]),
        ("Pivot R2", pivots["R2"]),
        ("Pivot R1", pivots["R1"]),
        ("4h BB 上轨", s4h.get("bb_up")),
        ("2h BB 上轨", s2h.get("bb_up")),
        ("1h BB 上轨", s1h.get("bb_up")),
        ("窗口高点", fibs["100% (高)"]),
        ("1h Donchian 上沿", s1h.get("donchian_up")),
        ("30m Donchian 上沿", s30m.get("donchian_up")),
        ("4h Keltner 上轨", s4h.get("kc_up")),
        ("1d SuperTrend", s1d.get("supertrend") if s1d.get("st_dir") == -1 else None),
    ]:
        if v and v > last:
            raw.append(("🔴", v, name))
    # 支撑
    for name, v in [
        ("Fib 78.6%", fibs["78.6%"]),
        ("Fib 61.8%", fibs["61.8%"]),
        ("Fib 50%", fibs["50%"]),
        ("Fib 38.2%", fibs["38.2%"]),
        ("Fib 23.6%", fibs["23.6%"]),
        ("窗口低点", fibs["0% (低)"]),
        ("1h SuperTrend", s1h.get("supertrend") if s1h.get("st_dir") == 1 else None),
        ("2h SuperTrend", s2h.get("supertrend") if s2h.get("st_dir") == 1 else None),
        ("4h SuperTrend", s4h.get("supertrend") if s4h.get("st_dir") == 1 else None),
        ("1h Ichimoku 基线", s1h.get("ichimoku_base")),
        ("1h Ichimoku Span A", s1h.get("ichimoku_spanA")),
        ("1h Ichimoku Span B", s1h.get("ichimoku_spanB")),
        ("2h Ichimoku 基线", s2h.get("ichimoku_base")),
        ("1h BB 中轨", s1h.get("bb_mid")),
        ("2h BB 中轨", s2h.get("bb_mid")),
        ("Pivot S1", pivots["S1"]),
        ("Pivot S2", pivots["S2"]),
        ("Pivot S3", pivots["S3"]),
        ("5m Donchian 下沿", s5m.get("donchian_dn")),
        ("15m Donchian 下沿", s15m.get("donchian_dn")),
        ("30m Donchian 下沿", s30m.get("donchian_dn")),
    ]:
        if v and v < last:
            raw.append(("🟢", v, name))

    raw.append(("🟡", last, "当前价"))
    # 去重接近位（合并相近 0.3% 的位）
    raw.sort(key=lambda x: -x[1])
    merged = []
    for tag, p, src in raw:
        if merged and abs(p - merged[-1][1]) / merged[-1][1] < 0.003 and merged[-1][0] == tag:
            merged[-1] = (tag, merged[-1][1], merged[-1][2] + " / " + src)
        else:
            merged.append((tag, p, src))
    return merged


def levels_lookup(levels, tag, idx):
    """取 levels 中匹配 tag 的第 idx 项的价格；越界则取最后一项。"""
    matched = [l for l in levels if l[0] == tag]
    if not matched: return 0
    if idx >= len(matched): return matched[-1][1]
    return matched[idx][1]


def build_plans(last, snaps, deriv, levels, outlook):
    """构造交易计划文本块。"""
    s1h = snaps["1h"]; s4h = snaps["4h"]
    atr1h = s1h.get("atr14") or last * 0.005
    atr4h = s4h.get("atr14") or last * 0.01

    # 关键阻力/支撑
    resistances = [l for l in levels if l[0] == "🔴"]
    supports = [l for l in levels if l[0] == "🟢"]
    R0 = resistances[-1][1] if resistances else last * 1.005    # 最近阻力
    R1 = resistances[-2][1] if len(resistances) >= 2 else last * 1.015
    R2 = resistances[-3][1] if len(resistances) >= 3 else last * 1.03
    R3 = resistances[-4][1] if len(resistances) >= 4 else last * 1.05
    S0 = supports[0][1] if supports else last * 0.995           # 最近支撑
    S1 = supports[1][1] if len(supports) >= 2 else last * 0.985
    S2 = supports[2][1] if len(supports) >= 3 else last * 0.97
    S3 = supports[3][1] if len(supports) >= 4 else last * 0.955
    S4 = supports[4][1] if len(supports) >= 5 else last * 0.94

    plans = []

    # A 计划 - 超短期回踩多
    plans.append(
f"""### 6.1 🟢 计划 A：超短期（**1–4 小时**）— **回踩多**
- **入场**：当前价 {last:.4g} **不追**，挂多分两笔于 **{S0:.4g} – {(S0 - 0.3*atr1h):.4g}**。
- **止损**：**{(S0 - 1.5*atr1h):.4g}**（≈ 1.5×ATR(1h)）。
- **止盈**：T1 = **{R0:.4g}**；T2 = **{R1:.4g}**。
- **风险回报**：约 1 : {((R0-S0)/(1.5*atr1h)):.1f}。
- **触发确认**：回踩时 5m RSI 30–45 区，1m BB 缩口后向上突破中轨；CVD 1m 由负转正。""")

    # B 计划 - 突破追多
    plans.append(
f"""### 6.2 🟢 计划 B：超短期（**1–4 小时**）— **突破追多**
- **入场触发**：1h 收盘价 > **{R0:.4g}**（突破 1h BB 上轨/最近阻力）。
- **入场价**：{R0:.4g} – {(R0 + 0.3*atr1h):.4g} 分批。
- **止损**：**{(R0 - 1.0*atr1h):.4g}**（突破失败回到形态内）。
- **止盈**：T1 = **{R1:.4g}**；T2 = **{R2:.4g}**；T3 = **{R3:.4g}**。
- **加分项**：OI 1h 内增加 ≥ 1%；资金费率 ≤ +0.005%（未拥挤）。""")

    # C 计划 - 趋势跟随主仓位
    plans.append(
f"""### 6.3 🟡 计划 C：短期（**1–3 天**）— **趋势跟随多**（主仓位）
- **建仓**：30% 仓位市价 {last:.4g} 多；40% 挂在 **{S1:.4g} / {S2:.4g}**。
- **止损**：日收盘价 < **{S3:.4g}**（关键支撑失守）。
- **分批止盈**：
  - 25% @ **{R1:.4g}**
  - 25% @ **{R2:.4g}**
  - 25% @ **{R3:.4g}**
  - 25% 移动止损至 1h SuperTrend 跟随。
- **每日复检**：1h ADX 是否 >30；OI 是否上行；费率 < +0.03%。""")

    # D 计划 - 战术性逆势空
    plans.append(
f"""### 6.4 🟠 计划 D：短期（**1–3 天**）— **战术性逆势空**
> 仅在 **以下条件同时满足** 时开仓：
> 1) 1h RSI > 85 出现顶背离；
> 2) 价格触及 **{R2:.4g} – {R3:.4g}** 但未能 1h 收盘站上；
> 3) 资金费率 > +0.02%；
> 4) 1h CVD 持续 2 根负值。
- **入场**：{R2:.4g} – {R3:.4g} 分批做空。
- **止损**：**{(R3 + 1.0*atr1h):.4g}**。
- **止盈**：T1 = **{R0:.4g}**；T2 = **{S0:.4g}**。
- **杠杆**：≤ 3×；R:R ≈ 1 : 3。""")

    # E 计划 - 中期波段多
    plans.append(
f"""### 6.5 🔵 计划 E：中期（**4–14 天**）— **波段多**
- **思路**：综合方向 = {outlook}；衍生品健康 (费率/OI/账户比) → 仍处于中级别上涨结构。
- **分批入场**：
  - 1/3 @ **{S1:.4g}**
  - 1/3 @ **{S2:.4g}**
  - 1/3 @ **{S4:.4g}**（深回踩极限）
- **平均成本预估**：~{((S1+S2+S4)/3):.4g}
- **止损**：日收盘价 < **{(S4 - 2*atr4h):.4g}**（破位 + 2×ATR(4h) 缓冲）。
- **目标位**：
  - T1: **{R2:.4g}**（+{(R2/((S1+S2+S4)/3)-1)*100:.1f}%）
  - T2: **{R3:.4g}**（+{(R3/((S1+S2+S4)/3)-1)*100:.1f}%）
  - T3: 心理整数 / Fib 1.272 扩展
- **管理**：每根日 K 收盘后，止损上移至上一根日 K 低点 −0.5×ATR(14)。""")

    # F 计划 - 网格
    grid_top = R2; grid_bot = S2
    grid_n = 24
    step = (grid_top - grid_bot) / grid_n
    plans.append(
f"""### 6.6 ⚪ 计划 F：网格交易（**Range‑Bound Grid**）
- **网格区间**：**{grid_bot:.4g} – {grid_top:.4g} USDT**（覆盖关键支撑 → 关键阻力）。
- **网格数**：**{grid_n}** 格，每格 ≈ **{step:.4g} USDT (≈ {step/last*100:.2f}%)**，与 5m ATR 匹配。
- **每格仓位**：账户净值 0.8%；总占用 ≤ 20%。
- **网格类型**：**中性网格**——上端 12 格只做多平仓，下端 12 格只做空平仓。
- **暂停 / 切换条件**：
  - 跌破 {grid_bot:.4g} 持续 1h → 关闭下行网格，保留多头底仓；
  - 突破 {grid_top:.4g} 持续 1h → 关闭网格，切换 C/E 趋势计划；
  - 资金费率 24h 均值 > +0.05% → 仅做空网格；< −0.05% → 仅做多网格。""")

    # G 计划 - 量化均值回归
    plans.append(
f"""### 6.7 ⚪ 计划 G：量化均值回归（**1h 级别 Bollinger + Z‑Score**）
- **逻辑**：Hurst < 0.5 + 1h/4h 负自相关 → 1h 收盘价相对 BB 中轨 Z‑score 绝对值 > 2 时反向开仓。
- **规则**：
  - **做多**：1h 收盘价 ≤ BB 下轨且 RSI < 30 → 0.5% 仓位多；
  - **做空**：1h 收盘价 ≥ BB 上轨且 RSI > 78 且 1h MACD hist 转负 → 0.5% 仓位空；
  - **止损**：开仓后 3 根 1h K 线内未盈利 → 平仓；或反向突破 1×ATR 止损；
  - **止盈**：回到 BB 中轨平 1/2，移动止损至开仓价；剩余仓位让其奔向反向轨道。
- **当前状态**：1h RSI = {fmt_num(s1h.get('rsi14'),1)}，4h %B = {fmt_num(s4h.get('bb_pctB'),2)} —— {('接近做空触发，等待 MACD hist 转负。' if (s1h.get('rsi14') or 0) > 70 else '尚未触发。')}""")

    return plans


def one_line_summary(outlook, snaps, deriv, levels):
    s1h = snaps["1h"]; s4h = snaps["4h"]
    last = s1h["last_close"]
    rsi = s1h.get("rsi14") or 0
    resistances = [l for l in levels if l[0] == "🔴"]
    supports = [l for l in levels if l[0] == "🟢"]
    R0 = resistances[-1][1] if resistances else last * 1.01
    S0 = supports[0][1] if supports else last * 0.99
    S1 = supports[1][1] if len(supports) >= 2 else last * 0.985

    overbought = "已严重超买" if rsi > 75 else ("偏强但未超买" if rsi > 60 else "中性")
    fr_now = deriv.get("funding_rate_current")
    fr_state = "资金费率为负（不拥挤，反而利好多头）" if fr_now and fr_now < 0 else (
        "资金费率偏高需警惕拥挤" if fr_now and fr_now > 0.0003 else "资金费率中性"
    )
    return (f"SOL‑USDT 永续目前处于 **{outlook.strip('**')}** 结构：1h 动量 {overbought}，"
            f"接近关键阻力 **{R0:.4g}**；{fr_state}。"
            f"最佳操作 = '**站稳 {R0:.4g} 追多** 或 **回踩 {S0:.4g}/{S1:.4g} 加多**'，"
            f"逆势空仅在阻力位出现 1h 顶背离时小仓试探。")


if __name__ == "__main__":
    main()
