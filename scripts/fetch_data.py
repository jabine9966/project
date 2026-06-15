"""
OKX SOL-USDT-SWAP 数据下载脚本（第一部分）
----------------------------------------
- 拉取最近 15 天的：1m/5m/15m/30m/1h/2h/4h/1d OHLCV、资金费率、持仓量、多空账户比、主动买卖盘、Ticker
- 时区：所有写入的 `dt` 列为 Asia/Shanghai (UTC+8)
- 内置 OKX 防封机制：自适应限流、随机抖动、指数退避重试、小批量分页、UA 伪装
- 落盘前进行清洗与质量检测，最终生成 data/DATA_QUALITY_REPORT.md

用法:
    python scripts/fetch_data.py
"""
from __future__ import annotations
import os
import json
import time
import math
import random
import logging
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

import ccxt
import pandas as pd
import numpy as np

# ============================== 配置 ==============================

SYMBOL = "SOL/USDT:USDT"        # CCXT 统一符号 → OKX SOL-USDT-SWAP
INST_CCY = "SOL"
LOOKBACK_DAYS = 15
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]

SH_TZ = timezone(timedelta(hours=8))   # Asia/Shanghai

# 防封参数
OHLCV_BATCH = 100               # OKX 公共行情接口 limit ≤ 300，这里保守取 100
SLEEP_BASE = 0.25               # 基础休眠 (秒)
SLEEP_JITTER = 0.35             # 随机抖动上限
RETRY_MAX = 6                   # 最大重试次数
RETRY_BACKOFF = [1, 2, 4, 8, 16, 32]   # 指数退避秒数

# 输出目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================== 日志 ==============================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fetch")

# ============================== 通用工具 ==============================

def sleep_jitter(base: float = SLEEP_BASE, jitter: float = SLEEP_JITTER) -> None:
    """请求间随机抖动，避免被风控识别为机器人节奏。"""
    time.sleep(base + random.uniform(0, jitter))


def with_retry(fn: Callable, *args, desc: str = "", **kwargs):
    """指数退避重试。捕获 ccxt 限流/网络/可用性等可恢复错误。"""
    last_err: Optional[Exception] = None
    for attempt in range(RETRY_MAX):
        try:
            return fn(*args, **kwargs)
        except (
            ccxt.RateLimitExceeded,
            ccxt.DDoSProtection,
            ccxt.NetworkError,
            ccxt.ExchangeNotAvailable,
            ccxt.RequestTimeout,
        ) as e:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            wait += random.uniform(0, 1.0)
            last_err = e
            log.warning(
                f"{desc} 触发限流/网络异常 ({type(e).__name__})，"
                f"第 {attempt+1}/{RETRY_MAX} 次重试，等待 {wait:.2f}s ..."
            )
            time.sleep(wait)
        except ccxt.BaseError as e:
            # 业务类错误不重试
            log.error(f"{desc} 业务异常：{e}")
            raise
    raise RuntimeError(f"{desc} 重试 {RETRY_MAX} 次后仍失败：{last_err}")


def to_sh(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=SH_TZ)


def add_sh_dt(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    df["dt"] = pd.to_datetime(df[ts_col], unit="ms", utc=True).dt.tz_convert(SH_TZ)
    return df


# ============================== 交易所实例 ==============================

def make_exchange() -> ccxt.okx:
    ex = ccxt.okx({
        "enableRateLimit": True,      # 启用 ccxt 内置节流
        "timeout": 30000,
        "options": {"defaultType": "swap"},
        "headers": {
            # UA 伪装，减少被风控识别
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
    })
    with_retry(ex.load_markets, desc="load_markets")
    return ex


# ============================== OHLCV ==============================

TF_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def fetch_ohlcv_window(ex: ccxt.okx, symbol: str, tf: str, since_ms: int, end_ms: int) -> pd.DataFrame:
    """分页拉取一个时间窗内的 OHLCV，带防封与去重。"""
    all_rows: list[list] = []
    cursor = since_ms
    page = 0
    while cursor < end_ms:
        rows = with_retry(
            ex.fetch_ohlcv,
            symbol,
            timeframe=tf,
            since=cursor,
            limit=OHLCV_BATCH,
            desc=f"ohlcv {tf} page={page}",
        )
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        # 已抵达数据末端
        if last_ts >= end_ms or len(rows) < OHLCV_BATCH:
            break
        cursor = last_ts + TF_MS[tf]
        page += 1
        sleep_jitter()
    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    return df


# ============================== 衍生品端点 ==============================

def fetch_funding_history(ex: ccxt.okx, symbol: str, since_ms: int, end_ms: int) -> pd.DataFrame:
    rows_all = []
    cursor = since_ms
    page = 0
    while cursor < end_ms:
        rows = with_retry(
            ex.fetch_funding_rate_history,
            symbol,
            since=cursor,
            limit=100,
            desc=f"funding page={page}",
        )
        if not rows:
            break
        rows_all.extend(rows)
        last_ts = rows[-1]["timestamp"]
        if last_ts >= end_ms or len(rows) < 2:
            break
        cursor = last_ts + 1
        page += 1
        sleep_jitter()
    df = pd.DataFrame([{"ts": r["timestamp"], "fundingRate": r["fundingRate"]} for r in rows_all])
    return df


def fetch_oi_history(ex: ccxt.okx, symbol: str, since_ms: int) -> pd.DataFrame:
    rows = with_retry(
        ex.fetch_open_interest_history,
        symbol,
        timeframe="1H",
        since=since_ms,
        limit=500,
        desc="open_interest",
    )
    df = pd.DataFrame([{
        "ts": r["timestamp"],
        "oi": r.get("openInterestAmount") or r.get("openInterestValue"),
        "oi_value": r.get("openInterestValue"),
    } for r in rows])
    return df


def fetch_long_short_ratio(ex: ccxt.okx, ccy: str, since_ms: int, end_ms: int) -> pd.DataFrame:
    """OKX 公共统计：合约多空账户比 (1H)。"""
    resp = with_retry(
        ex.publicGetRubikStatContractsLongShortAccountRatio,
        {"ccy": ccy, "period": "1H", "begin": str(since_ms), "end": str(end_ms)},
        desc="long_short_ratio",
    )
    rows = resp.get("data", []) or []
    df = pd.DataFrame(rows, columns=["ts", "ratio"])
    if not df.empty:
        df["ts"] = df["ts"].astype("int64")
        df["ratio"] = df["ratio"].astype(float)
    return df


def fetch_taker_volume(ex: ccxt.okx, ccy: str, since_ms: int, end_ms: int) -> pd.DataFrame:
    """OKX 公共统计：合约主动买/卖量 (1H)。instType 必须为 CONTRACTS。"""
    resp = with_retry(
        ex.publicGetRubikStatTakerVolume,
        {"ccy": ccy, "instType": "CONTRACTS", "period": "1H",
         "begin": str(since_ms), "end": str(end_ms)},
        desc="taker_volume",
    )
    rows = resp.get("data", []) or []
    df = pd.DataFrame(rows, columns=["ts", "sellVol", "buyVol"])
    if not df.empty:
        df["ts"] = df["ts"].astype("int64")
        df["sellVol"] = df["sellVol"].astype(float)
        df["buyVol"] = df["buyVol"].astype(float)
    return df


# ============================== 清洗 & 质量检测 ==============================

def clean_ohlcv(df: pd.DataFrame, tf: str) -> tuple[pd.DataFrame, dict]:
    """清洗 OHLCV：去重、排序、丢异常、统计完整度与离群点。"""
    raw_n = len(df)
    if df.empty:
        return df, {"raw_rows": 0, "clean_rows": 0, "duplicates": 0,
                    "bad_ohlc_dropped": 0, "ohlc_relation_anomalies": 0,
                    "expected_bars": 0, "completeness_pct": 0.0,
                    "missing_bars": 0, "return_outliers_8sigma": 0,
                    "first_ts": None, "last_ts": None}

    # 1) 去重 + 排序
    df = df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    dedup_n = len(df)
    dup = raw_n - dedup_n

    # 2) 丢弃 OHLC 异常 (NaN / ≤0)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    mask_bad = (
        df[["open", "high", "low", "close"]].isna().any(axis=1)
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    dropped_bad = int(mask_bad.sum())
    df = df.loc[~mask_bad].reset_index(drop=True)

    # 3) OHLC 关系异常（仅标记，不丢弃）
    rel_anom = int(((df["high"] < df[["open", "close"]].max(axis=1))
                    | (df["low"] > df[["open", "close"]].min(axis=1))).sum())

    # 4) 完整度
    first_ts, last_ts = int(df["ts"].iloc[0]), int(df["ts"].iloc[-1])
    expected = (last_ts - first_ts) // TF_MS[tf] + 1
    completeness = len(df) / expected * 100 if expected else 0
    missing = max(expected - len(df), 0)

    # 5) 收益率离群点 (8σ)
    rets = df["close"].pct_change().dropna()
    if len(rets) > 30:
        sigma = rets.std()
        outliers = int(((rets - rets.mean()).abs() > 8 * sigma).sum())
    else:
        outliers = 0

    # 6) 加 Asia/Shanghai 时间列
    df = add_sh_dt(df, "ts")

    return df, {
        "raw_rows": raw_n,
        "clean_rows": len(df),
        "duplicates": dup,
        "bad_ohlc_dropped": dropped_bad,
        "ohlc_relation_anomalies": rel_anom,
        "expected_bars": int(expected),
        "completeness_pct": round(completeness, 3),
        "missing_bars": int(missing),
        "return_outliers_8sigma": outliers,
        "first_ts": to_sh(first_ts).isoformat(),
        "last_ts": to_sh(last_ts).isoformat(),
    }


def clean_funding(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw_n = len(df)
    if df.empty:
        return df, {"raw_rows": 0, "clean_rows": 0, "duplicates": 0,
                    "rate_range_anomalies": 0,
                    "first_ts": None, "last_ts": None}
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    dup = raw_n - len(df)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.dropna(subset=["fundingRate"]).reset_index(drop=True)
    rate_anom = int((df["fundingRate"].abs() > 0.0075).sum())  # OKX 单期上限 ±0.75%
    df = add_sh_dt(df, "ts")
    return df, {
        "raw_rows": raw_n,
        "clean_rows": len(df),
        "duplicates": dup,
        "rate_range_anomalies": rate_anom,
        "first_ts": df["dt"].iloc[0].isoformat() if len(df) else None,
        "last_ts": df["dt"].iloc[-1].isoformat() if len(df) else None,
    }


def clean_generic_ts(df: pd.DataFrame, value_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    raw_n = len(df)
    if df.empty:
        return df, {"raw_rows": 0, "clean_rows": 0, "duplicates": 0,
                    "null_values_dropped": 0,
                    "first_ts": None, "last_ts": None}
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    dup = raw_n - len(df)
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    null_n = int(df[value_cols].isna().any(axis=1).sum())
    df = df.dropna(subset=value_cols).reset_index(drop=True)
    df = add_sh_dt(df, "ts")
    return df, {
        "raw_rows": raw_n,
        "clean_rows": len(df),
        "duplicates": dup,
        "null_values_dropped": null_n,
        "first_ts": df["dt"].iloc[0].isoformat() if len(df) else None,
        "last_ts": df["dt"].iloc[-1].isoformat() if len(df) else None,
    }


# ============================== 主流程 ==============================

def main() -> None:
    ex = make_exchange()

    now_ms = ex.milliseconds()
    since_ms = now_ms - LOOKBACK_DAYS * 24 * 3600 * 1000
    log.info(f"窗口 (Shanghai): {to_sh(since_ms)} → {to_sh(now_ms)} （{LOOKBACK_DAYS} 天）")

    quality: dict = {
        "generated_at_sh": datetime.now(SH_TZ).isoformat(),
        "exchange": "OKX",
        "symbol": SYMBOL,
        "inst_id": "SOL-USDT-SWAP",
        "lookback_days": LOOKBACK_DAYS,
        "window_start_sh": to_sh(since_ms).isoformat(),
        "window_end_sh": to_sh(now_ms).isoformat(),
        "datasets": {},
    }

    # ---------- OHLCV ----------
    for tf in TIMEFRAMES:
        log.info(f"下载 OHLCV {tf} ...")
        raw = fetch_ohlcv_window(ex, SYMBOL, tf, since_ms, now_ms)
        clean, stats = clean_ohlcv(raw, tf)
        path = os.path.join(DATA_DIR, f"ohlcv_{tf}.csv")
        clean.to_csv(path, index=False)
        stats["file"] = os.path.relpath(path, ROOT)
        quality["datasets"][f"ohlcv_{tf}"] = stats
        log.info(f"  保存 {path}: clean={stats['clean_rows']}, 完整度={stats['completeness_pct']}%")
        sleep_jitter()

    # ---------- Ticker ----------
    log.info("下载 Ticker ...")
    ticker = with_retry(ex.fetch_ticker, SYMBOL, desc="ticker")
    ticker_safe = {k: v for k, v in ticker.items() if k != "info"}
    with open(os.path.join(DATA_DIR, "ticker.json"), "w") as f:
        json.dump(ticker_safe, f, indent=2, default=str)
    quality["datasets"]["ticker"] = {
        "file": "data/ticker.json",
        "last": ticker.get("last"),
        "high_24h": ticker.get("high"),
        "low_24h": ticker.get("low"),
        "baseVolume_24h": ticker.get("baseVolume"),
        "percentage_24h": ticker.get("percentage"),
    }
    sleep_jitter()

    # ---------- 资金费率（当前 + 历史）----------
    log.info("下载 当前资金费率 ...")
    try:
        cur = with_retry(ex.fetch_funding_rate, SYMBOL, desc="funding_current")
        cur_safe = {k: v for k, v in cur.items() if k != "info"}
        with open(os.path.join(DATA_DIR, "funding_current.json"), "w") as f:
            json.dump(cur_safe, f, indent=2, default=str)
        quality["datasets"]["funding_current"] = {
            "file": "data/funding_current.json",
            "rate": cur.get("fundingRate"),
            "next_funding_time_sh": (
                to_sh(cur["nextFundingTimestamp"]).isoformat()
                if cur.get("nextFundingTimestamp") else None
            ),
        }
    except Exception as e:
        log.warning(f"funding_current 失败：{e}")
        quality["datasets"]["funding_current"] = {"error": str(e)}
    sleep_jitter()

    log.info("下载 资金费率历史 ...")
    raw_fr = fetch_funding_history(ex, SYMBOL, since_ms, now_ms)
    clean_fr, stats_fr = clean_funding(raw_fr)
    clean_fr.to_csv(os.path.join(DATA_DIR, "funding_rate.csv"), index=False)
    stats_fr["file"] = "data/funding_rate.csv"
    quality["datasets"]["funding_rate_history"] = stats_fr
    log.info(f"  保存 funding_rate.csv: clean={stats_fr['clean_rows']}")
    sleep_jitter()

    # ---------- OI 历史 ----------
    log.info("下载 持仓量历史 ...")
    try:
        raw_oi = fetch_oi_history(ex, SYMBOL, since_ms)
        clean_oi, stats_oi = clean_generic_ts(raw_oi, ["oi"])
        clean_oi.to_csv(os.path.join(DATA_DIR, "open_interest.csv"), index=False)
        stats_oi["file"] = "data/open_interest.csv"
        quality["datasets"]["open_interest"] = stats_oi
        log.info(f"  保存 open_interest.csv: clean={stats_oi['clean_rows']}")
    except Exception as e:
        log.warning(f"open_interest 失败：{e}")
        quality["datasets"]["open_interest"] = {"error": str(e)}
    sleep_jitter()

    # ---------- 多空账户比 ----------
    log.info("下载 多空账户比 ...")
    try:
        raw_ls = fetch_long_short_ratio(ex, INST_CCY, since_ms, now_ms)
        clean_ls, stats_ls = clean_generic_ts(raw_ls, ["ratio"])
        clean_ls.to_csv(os.path.join(DATA_DIR, "long_short_ratio.csv"), index=False)
        stats_ls["file"] = "data/long_short_ratio.csv"
        quality["datasets"]["long_short_ratio"] = stats_ls
        log.info(f"  保存 long_short_ratio.csv: clean={stats_ls['clean_rows']}")
    except Exception as e:
        log.warning(f"long_short_ratio 失败：{e}")
        quality["datasets"]["long_short_ratio"] = {"error": str(e)}
    sleep_jitter()

    # ---------- 主动买卖盘 ----------
    log.info("下载 主动买卖盘 ...")
    try:
        raw_tv = fetch_taker_volume(ex, INST_CCY, since_ms, now_ms)
        clean_tv, stats_tv = clean_generic_ts(raw_tv, ["buyVol", "sellVol"])
        clean_tv.to_csv(os.path.join(DATA_DIR, "taker_volume.csv"), index=False)
        stats_tv["file"] = "data/taker_volume.csv"
        quality["datasets"]["taker_volume"] = stats_tv
        log.info(f"  保存 taker_volume.csv: clean={stats_tv['clean_rows']}")
    except Exception as e:
        log.warning(f"taker_volume 失败：{e}")
        quality["datasets"]["taker_volume"] = {"error": str(e)}
    sleep_jitter()

    # ---------- 写质量报告 (Markdown + JSON) ----------
    with open(os.path.join(DATA_DIR, "quality.json"), "w") as f:
        json.dump(quality, f, indent=2, default=str, ensure_ascii=False)
    write_quality_report_md(quality)
    log.info("数据下载与清洗完成 ✅")


# ============================== 质量报告 Markdown ==============================

def write_quality_report_md(q: dict) -> None:
    path = os.path.join(DATA_DIR, "DATA_QUALITY_REPORT.md")
    lines: list[str] = []
    lines.append("# 📋 数据质量报告 (Data Quality Report)")
    lines.append("")
    lines.append(f"- **生成时间 (Asia/Shanghai)**：{q['generated_at_sh']}")
    lines.append(f"- **交易所**：{q['exchange']}")
    lines.append(f"- **品种**：`{q['symbol']}`（OKX `{q['inst_id']}`）")
    lines.append(f"- **回看窗口**：{q['lookback_days']} 天")
    lines.append(f"- **窗口起 (SH)**：{q['window_start_sh']}")
    lines.append(f"- **窗口止 (SH)**：{q['window_end_sh']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. 数据清单 (Inventory)")
    lines.append("")
    lines.append("| # | 数据集 | 文件 | 行数 | 起始 (SH) | 截止 (SH) | 说明 |")
    lines.append("|---|---|---|---:|---|---|---|")

    pretty_names = {
        "ohlcv_1m": "OHLCV 1 分钟",
        "ohlcv_5m": "OHLCV 5 分钟",
        "ohlcv_15m": "OHLCV 15 分钟",
        "ohlcv_30m": "OHLCV 30 分钟",
        "ohlcv_1h": "OHLCV 1 小时",
        "ohlcv_2h": "OHLCV 2 小时",
        "ohlcv_4h": "OHLCV 4 小时",
        "ohlcv_1d": "OHLCV 1 日",
        "ticker": "Ticker（实时）",
        "funding_current": "当前资金费率",
        "funding_rate_history": "资金费率历史 (8h)",
        "open_interest": "持仓量历史 (1h)",
        "long_short_ratio": "多空账户比 (1h)",
        "taker_volume": "主动买/卖盘量 (1h)",
    }
    for i, (name, info) in enumerate(q["datasets"].items(), 1):
        if "error" in info:
            lines.append(f"| {i} | {pretty_names.get(name, name)} | — | 0 | — | — | ❌ {info['error']} |")
            continue
        rows = info.get("clean_rows", "—")
        first = info.get("first_ts") or "—"
        last = info.get("last_ts") or "—"
        file = info.get("file", "—")
        note = ""
        if "completeness_pct" in info:
            note = f"完整度 {info['completeness_pct']}%（缺 {info['missing_bars']} 根）"
        elif name == "ticker":
            note = f"最新价 {info.get('last')}"
        elif name == "funding_current":
            note = f"当前费率 {info.get('rate')}"
        lines.append(f"| {i} | {pretty_names.get(name, name)} | `{file}` | {rows} | {first} | {last} | {note} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. OHLCV 数据质量明细")
    lines.append("")
    lines.append("| 周期 | 原始行数 | 清洗后行数 | 重复 | 丢弃异常 | OHLC 关系异常 | 期望 bar | 实际 bar | 缺口 | 完整度 % | 收益率离群 (8σ) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tf in TIMEFRAMES:
        s = q["datasets"].get(f"ohlcv_{tf}")
        if not s or "error" in s:
            lines.append(f"| {tf} | — | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {tf} | {s['raw_rows']} | {s['clean_rows']} | {s['duplicates']} | {s['bad_ohlc_dropped']} | "
            f"{s['ohlc_relation_anomalies']} | {s['expected_bars']} | {s['clean_rows']} | "
            f"{s['missing_bars']} | {s['completeness_pct']} | {s['return_outliers_8sigma']} |"
        )
    lines.append("")
    lines.append("> **完整度**：`实际 bar 数 / 期望 bar 数 × 100%`，期望基于第一根 → 最后一根 bar 的时间跨度。")
    lines.append("> **8σ 离群**：以 close 的 1 期收益率为口径，超过 8 倍标准差的 bar（仅记录，不删除）。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. 衍生品数据质量")
    lines.append("")
    fr = q["datasets"].get("funding_rate_history", {})
    oi = q["datasets"].get("open_interest", {})
    ls = q["datasets"].get("long_short_ratio", {})
    tv = q["datasets"].get("taker_volume", {})

    lines.append("| 数据集 | 原始 | 清洗后 | 重复 | 空值/异常 | 备注 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    if "error" not in fr:
        lines.append(f"| 资金费率历史 | {fr.get('raw_rows','—')} | {fr.get('clean_rows','—')} | {fr.get('duplicates','—')} | {fr.get('rate_range_anomalies','—')} | 阈值 \\|rate\\| > 0.75% |")
    else:
        lines.append(f"| 资金费率历史 | — | — | — | — | ❌ {fr['error']} |")
    if "error" not in oi:
        lines.append(f"| 持仓量历史 | {oi.get('raw_rows','—')} | {oi.get('clean_rows','—')} | {oi.get('duplicates','—')} | {oi.get('null_values_dropped','—')} | 周期 1H |")
    else:
        lines.append(f"| 持仓量历史 | — | — | — | — | ❌ {oi['error']} |")
    if "error" not in ls:
        lines.append(f"| 多空账户比 | {ls.get('raw_rows','—')} | {ls.get('clean_rows','—')} | {ls.get('duplicates','—')} | {ls.get('null_values_dropped','—')} | OKX rubik 1H |")
    else:
        lines.append(f"| 多空账户比 | — | — | — | — | ❌ {ls['error']} |")
    if "error" not in tv:
        lines.append(f"| 主动买卖盘 | {tv.get('raw_rows','—')} | {tv.get('clean_rows','—')} | {tv.get('duplicates','—')} | {tv.get('null_values_dropped','—')} | OKX rubik CONTRACTS 1H |")
    else:
        lines.append(f"| 主动买卖盘 | — | — | — | — | ❌ {tv['error']} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. 防封 / 限流策略说明")
    lines.append("")
    lines.append("| 机制 | 当前配置 |")
    lines.append("|---|---|")
    lines.append("| ccxt enableRateLimit | ✅ 已启用 |")
    lines.append(f"| 请求基础休眠 | {SLEEP_BASE}s |")
    lines.append(f"| 随机抖动上限 | {SLEEP_JITTER}s |")
    lines.append(f"| 最大重试次数 | {RETRY_MAX} |")
    lines.append(f"| 退避序列 (秒) | {RETRY_BACKOFF} |")
    lines.append(f"| OHLCV 单次 limit | {OHLCV_BATCH} |")
    lines.append("| User‑Agent 伪装 | ✅ 已启用 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. 数据使用说明")
    lines.append("")
    lines.append("- 全部 CSV 含两列时间：`ts`(UTC 毫秒) 与 `dt`(Asia/Shanghai 时区字符串)。")
    lines.append("- OHLCV 列：`open, high, low, close, volume`，`volume` 为合约张数。")
    lines.append("- 资金费率每 8 小时结算，对应 OKX SOL‑USDT‑SWAP。")
    lines.append("- 数据可直接被 `scripts/analyze_report.py` 加载进行多维度技术分析与交易计划生成。")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"质量报告已写入 {path}")


if __name__ == "__main__":
    main()
