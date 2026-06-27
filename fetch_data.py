# -*- coding: utf-8 -*-
"""
Binance SOL 永续合约数据下载脚本
=================================
使用 CCXT 库接入 Binance 交易所，下载 SOL/USDT 永续合约最近 15 天的：
- K 线数据（OHLCV）
- 实时 Ticker 与订单簿数据
- 量价分析衍生数据：持仓量、资金费率、多空比、主动买卖量

内置防封机制：速率限制、指数退避重试、备用域名自动切换、请求抖动。
数据下载后自动检查与清洗，结果保存到 DATA/ 目录。

依赖：
    pip install ccxt pandas numpy
"""

import ccxt
import pandas as pd
import numpy as np
import time
import json
import os
import sys
from datetime import datetime, timedelta
import logging
import random
from typing import Dict, Any, Optional

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fetch_data.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 全局配置
# ============================================================
CONFIG = {
    'symbol': 'SOL/USDT:USDT',   # CCXT 永续合约格式
    'raw_symbol': 'SOLUSDT',     # Binance API 原始符号
    'days': 15,                   # 回溯天数
    'timeframe': '1h',           # K 线周期
    'data_dir': 'DATA',          # 输出目录
    'max_retries': 5,            # 单次接口最大重试次数
    'base_delay': 1.0,           # 基础退避秒数
    'request_jitter': 2.0,       # 随机抖动上限
}

# Binance USD-M 期货 API 备用域名（用于防封自动切换）
BINANCE_DOMAINS = [
    'fapi.binance.com',
    'fapi1.binance.com',
    'fapi2.binance.com',
    'fapi3.binance.com',
]


# ============================================================
# 防封数据获取器
# ============================================================
class SafeBinanceFetcher:
    """
    封装 CCXT Binance 连接，提供：
    - 多域名备用自动切换
    - 指数退避 + 随机抖动重试
    - 429/IP 限流特殊等待
    """

    def __init__(self):
        self.domain_idx = 0
        self.exchange = self._init_exchange()
        self.raw_symbol = CONFIG['raw_symbol']
        self.symbol = CONFIG['symbol']

    def _init_exchange(self) -> ccxt.binance:
        domain = BINANCE_DOMAINS[self.domain_idx]
        logger.info(f'初始化 Binance 连接 -> {domain}')
        config = {
            'enableRateLimit': True,
            'rateLimit': 100,          # 基础速率限制(ms)
            'options': {
                'defaultType': 'swap',  # 永续合约
                'adjustForTimeDifference': True,
            },
            'urls': {
                'api': {
                    'public': f'https://{domain}',
                    'private': f'https://{domain}',
                }
            },
            'timeout': 30000,
        }
        return ccxt.binance(config)

    def _switch_domain(self) -> bool:
        """切换到下一个备用域名"""
        self.domain_idx += 1
        if self.domain_idx < len(BINANCE_DOMAINS):
            logger.warning(f'切换备用域名: {BINANCE_DOMAINS[self.domain_idx]}')
            self.exchange = self._init_exchange()
            return True
        logger.error('所有备用域名均已耗尽！')
        return False

    def _safe_request(self, func, *args, **kwargs):
        """
        通用请求包装器：
        1. 每次请求前增加动态延迟（指数退避 + 随机抖动）
        2. 捕获 NetworkError / ExchangeError 并重试
        3. 检测到 429/418 时执行额外长等待
        4. 重试耗尽后自动切换域名
        """
        for attempt in range(CONFIG['max_retries']):
            try:
                # 动态延迟：指数退避 + 随机抖动（防封核心）
                delay = (
                    CONFIG['base_delay'] * (2 ** attempt)
                    + random.uniform(0, CONFIG['request_jitter'])
                )
                time.sleep(delay)

                result = func(*args, **kwargs)
                return result

            except ccxt.NetworkError as e:
                logger.warning(
                    f'NetworkError [{attempt + 1}/{CONFIG["max_retries"]}]: {e}'
                )
                if attempt == CONFIG['max_retries'] - 1:
                    if self._switch_domain():
                        return self._safe_request(func, *args, **kwargs)
                    raise

            except ccxt.ExchangeError as e:
                err_str = str(e)
                logger.warning(
                    f'ExchangeError [{attempt + 1}/{CONFIG["max_retries"]}]: {err_str}'
                )
                # 针对限流/IP 封禁的特殊处理
                if any(code in err_str for code in ['429', '418', 'IP']):
                    extra_wait = 30 + random.uniform(0, 30)
                    logger.warning(
                        f'触发限流/IP 检测，额外等待 {extra_wait:.1f} 秒'
                    )
                    time.sleep(extra_wait)
                if attempt == CONFIG['max_retries'] - 1:
                    if self._switch_domain():
                        return self._safe_request(func, *args, **kwargs)
                    raise

            except Exception as e:
                logger.error(
                    f'未知错误 [{attempt + 1}/{CONFIG["max_retries"]}]: {e}'
                )
                if attempt == CONFIG['max_retries'] - 1:
                    raise

        return None

    # --------------------------------------------------------
    # 1. K 线数据
    # --------------------------------------------------------
    def fetch_ohlcv(self) -> pd.DataFrame:
        """获取最近 N 天 K 线数据（自动分页）"""
        logger.info(
            f'开始获取 {self.symbol} {CONFIG["timeframe"]} K 线数据，'
            f'回溯 {CONFIG["days"]} 天'
        )
        since = int(
            (datetime.now() - timedelta(days=CONFIG['days'])).timestamp() * 1000
        )
        all_data = []
        current_since = since

        while True:
            batch = self._safe_request(
                self.exchange.fetch_ohlcv,
                self.symbol,
                CONFIG['timeframe'],
                current_since,
                1500,  # Binance 单次最大 1500
            )
            if not batch or len(batch) == 0:
                break

            all_data.extend(batch)
            current_since = batch[-1][0] + 1

            if len(batch) < 1500:
                break
            logger.info(f'  已累积 {len(all_data)} 条 K 线...')

        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    # --------------------------------------------------------
    # 2. Ticker 数据
    # --------------------------------------------------------
    def fetch_ticker(self) -> dict:
        """获取 24h Ticker 统计"""
        logger.info(f'获取 {self.symbol} Ticker 数据...')
        return self._safe_request(self.exchange.fetch_ticker, self.symbol)

    # --------------------------------------------------------
    # 3. 订单簿数据
    # --------------------------------------------------------
    def fetch_order_book(self, limit: int = 100) -> dict:
        """获取深度订单簿"""
        logger.info(f'获取 {self.symbol} 订单簿 (depth={limit})...')
        return self._safe_request(
            self.exchange.fetch_order_book, self.symbol, limit
        )

    # --------------------------------------------------------
    # 4. 持仓量历史 (Open Interest)
    # --------------------------------------------------------
    def fetch_open_interest_history(self) -> pd.DataFrame:
        """获取历史持仓量 (Open Interest)"""
        logger.info(f'获取 {self.symbol} 历史持仓量...')
        period = CONFIG['timeframe']
        since = int(
            (datetime.now() - timedelta(days=CONFIG['days'])).timestamp() * 1000
        )
        all_data = []
        current_since = since

        while True:
            params = {
                'symbol': self.raw_symbol,
                'period': period,
                'limit': 500,
                'startTime': current_since,
            }
            try:
                batch = self._safe_request(
                    self.exchange.fapiPublic_get_openInterestHist, params
                )
                if not batch or len(batch) == 0:
                    break
                all_data.extend(batch)
                current_since = int(batch[-1]['timestamp']) + 1
                if len(batch) < 500:
                    break
            except Exception as e:
                logger.error(f'获取持仓量失败: {e}')
                break

        df = pd.DataFrame(all_data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    # --------------------------------------------------------
    # 5. 资金费率历史 (Funding Rate)
    # --------------------------------------------------------
    def fetch_funding_rate_history(self) -> pd.DataFrame:
        """获取历史资金费率"""
        logger.info(f'获取 {self.symbol} 资金费率历史...')
        since = int(
            (datetime.now() - timedelta(days=CONFIG['days'])).timestamp() * 1000
        )
        all_data = []
        current_since = since

        while True:
            params = {
                'symbol': self.raw_symbol,
                'startTime': current_since,
                'limit': 1000,
            }
            try:
                batch = self._safe_request(
                    self.exchange.fapiPublic_get_fundingRate, params
                )
                if not batch or len(batch) == 0:
                    break
                all_data.extend(batch)
                current_since = int(batch[-1]['fundingTime']) + 1
                if len(batch) < 1000:
                    break
            except Exception as e:
                logger.error(f'获取资金费率失败: {e}')
                break

        df = pd.DataFrame(all_data)
        if not df.empty:
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        return df

    # --------------------------------------------------------
    # 6. 多空账户比 (Long/Short Account Ratio)
    # --------------------------------------------------------
    def fetch_long_short_ratio(self) -> pd.DataFrame:
        """获取全局多空账户比"""
        logger.info(f'获取 {self.symbol} 多空账户比...')
        period = CONFIG['timeframe']
        since = int(
            (datetime.now() - timedelta(days=CONFIG['days'])).timestamp() * 1000
        )
        all_data = []
        current_since = since

        while True:
            params = {
                'symbol': self.raw_symbol,
                'period': period,
                'limit': 500,
                'startTime': current_since,
            }
            try:
                batch = self._safe_request(
                    self.exchange.fapiData_get_globalLongShortAccountRatio,
                    params,
                )
                if not batch or len(batch) == 0:
                    break
                all_data.extend(batch)
                current_since = int(batch[-1]['timestamp']) + 1
                if len(batch) < 500:
                    break
            except Exception as e:
                logger.error(f'获取多空比失败: {e}')
                break

        df = pd.DataFrame(all_data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    # --------------------------------------------------------
    # 7. 主动买卖量 (Taker Buy/Sell Volume)
    # --------------------------------------------------------
    def fetch_taker_volume(self) -> pd.DataFrame:
        """获取主动买卖量"""
        logger.info(f'获取 {self.symbol} 主动买卖量...')
        period = CONFIG['timeframe']
        since = int(
            (datetime.now() - timedelta(days=CONFIG['days'])).timestamp() * 1000
        )
        all_data = []
        current_since = since

        while True:
            params = {
                'symbol': self.raw_symbol,
                'period': period,
                'limit': 500,
                'startTime': current_since,
            }
            try:
                batch = self._safe_request(
                    self.exchange.fapiData_get_takerBuySellVol, params
                )
                if not batch or len(batch) == 0:
                    break
                all_data.extend(batch)
                current_since = int(batch[-1]['timestamp']) + 1
                if len(batch) < 500:
                    break
            except Exception as e:
                logger.error(f'获取主动买卖量失败: {e}')
                break

        df = pd.DataFrame(all_data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df


# ============================================================
# 数据清洗模块
# ============================================================
class DataCleaner:
    """对下载的原始数据进行检查与清洗"""

    @staticmethod
    def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """清洗 K 线数据"""
        logger.info(f'清洗 K 线数据，原始 {len(df)} 行')

        if df.empty:
            return df

        # 1. 去重
        df = df.drop_duplicates(subset=['timestamp'], keep='first')

        # 2. 排序
        df = df.sort_values('timestamp').reset_index(drop=True)

        # 3. 缺失值检查与插值
        for col in ['open', 'high', 'low', 'close', 'volume']:
            missing = df[col].isnull().sum()
            if missing > 0:
                logger.warning(f'{col} 缺失 {missing} 个值，使用线性插值')
                df[col] = df[col].interpolate(method='linear')

        # 4. 异常值检查（价格 <= 0）
        for col in ['open', 'high', 'low', 'close']:
            invalid = (df[col] <= 0).sum()
            if invalid > 0:
                logger.warning(f'{col} 存在 {invalid} 个 <=0 的异常值，前向填充')
                df[col] = df[col].mask(df[col] <= 0).ffill()

        # 5. OHLC 逻辑一致性修正
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

        # 6. 时间连续性检查
        df = DataCleaner._check_timestamp_continuity(df)

        logger.info(f'清洗后 K 线数据: {len(df)} 行')
        return df

    @staticmethod
    def _check_timestamp_continuity(df: pd.DataFrame) -> pd.DataFrame:
        """检查时间戳是否连续，并报告缺口"""
        if df.empty or 'timestamp' not in df.columns:
            return df

        expected = pd.Timedelta(CONFIG['timeframe'])
        df['time_diff'] = df['timestamp'].diff()
        gaps = df[df['time_diff'] > expected * 1.5]

        if len(gaps) > 0:
            logger.warning(f'发现 {len(gaps)} 个时间缺口')
            for _, g in gaps.iterrows():
                logger.warning(
                    f'  缺口 @ {g["timestamp"]} 间隔={g["time_diff"]}'
                )

        df = df.drop(columns=['time_diff'], errors='ignore')
        return df

    @staticmethod
    def clean_metric_df(
        df: pd.DataFrame, timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """清洗通用指标型 DataFrame"""
        logger.info(f'清洗指标数据，原始 {len(df)} 行')

        if df.empty or timestamp_col not in df.columns:
            return df

        # 去重
        df = df.drop_duplicates(subset=[timestamp_col], keep='first')
        # 排序
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        # 数值列缺失值处理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == timestamp_col:
                continue
            missing = df[col].isnull().sum()
            if missing > 0:
                logger.warning(f'{col} 缺失 {missing} 个值，插值填充')
                df[col] = df[col].interpolate(method='linear').ffill().bfill()

        logger.info(f'清洗后指标数据: {len(df)} 行')
        return df


# ============================================================
# 存储与报告
# ============================================================
def save_data(df: pd.DataFrame, name: str):
    """同时保存 CSV 和 JSON"""
    os.makedirs(CONFIG['data_dir'], exist_ok=True)

    csv_path = os.path.join(CONFIG['data_dir'], f'{name}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f'保存 CSV: {csv_path}')

    json_path = os.path.join(CONFIG['data_dir'], f'{name}.json')
    df.to_json(
        json_path, orient='records', date_format='iso', indent=2
    )
    logger.info(f'保存 JSON: {json_path}')

    return csv_path, json_path


def generate_quality_report(data_dict: Dict[str, pd.DataFrame]):
    """生成 JSON 格式的数据质量报告"""
    report = {
        'generated_at': datetime.now().isoformat(),
        'symbol': CONFIG['symbol'],
        'period_days': CONFIG['days'],
        'timeframe': CONFIG['timeframe'],
        'datasets': {},
    }

    for name, df in data_dict.items():
        if df.empty:
            report['datasets'][name] = {
                'status': 'EMPTY',
                'rows': 0,
                'columns': 0,
                'issues': ['No data retrieved'],
            }
            continue

        issues = []
        missing = int(df.isnull().sum().sum())
        if missing > 0:
            issues.append(f'Total {missing} missing values')

        if 'timestamp' in df.columns:
            dupes = int(df['timestamp'].duplicated().sum())
            if dupes > 0:
                issues.append(f'{dupes} duplicate timestamps')

        ts_col = 'timestamp' if 'timestamp' in df.columns else 'fundingTime'
        if ts_col in df.columns:
            start = str(df[ts_col].iloc[0])
            end = str(df[ts_col].iloc[-1])
        else:
            start = end = None

        report['datasets'][name] = {
            'status': 'OK' if not issues else 'WARNING',
            'rows': len(df),
            'columns': len(df.columns),
            'columns_list': list(df.columns),
            'start_time': start,
            'end_time': end,
            'issues': issues if issues else ['None'],
        }

    report_path = os.path.join(CONFIG['data_dir'], 'DATA_QUALITY_REPORT.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f'数据质量报告: {report_path}')
    return report


# ============================================================
# 主流程
# ============================================================
def main():
    logger.info('=' * 60)
    logger.info('Binance SOL 永续合约数据下载程序启动')
    logger.info('=' * 60)

    os.makedirs(CONFIG['data_dir'], exist_ok=True)
    fetcher = SafeBinanceFetcher()
    cleaner = DataCleaner()
    data_dict: Dict[str, pd.DataFrame] = {}

    # 1. K 线
    try:
        df = fetcher.fetch_ohlcv()
        df = cleaner.clean_ohlcv(df)
        save_data(df, 'ohlcv')
        data_dict['ohlcv'] = df
    except Exception as e:
        logger.error(f'K 线数据获取失败: {e}')
        data_dict['ohlcv'] = pd.DataFrame()

    # 2. Ticker
    try:
        ticker = fetcher.fetch_ticker()
        path = os.path.join(CONFIG['data_dir'], 'ticker.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(ticker, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f'保存 Ticker: {path}')
        data_dict['ticker'] = pd.DataFrame([ticker])
    except Exception as e:
        logger.error(f'Ticker 获取失败: {e}')
        data_dict['ticker'] = pd.DataFrame()

    # 3. 订单簿
    try:
        ob = fetcher.fetch_order_book()
        path = os.path.join(CONFIG['data_dir'], 'order_book.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(ob, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f'保存订单簿: {path}')
        data_dict['order_book'] = pd.DataFrame([ob])
    except Exception as e:
        logger.error(f'订单簿获取失败: {e}')
        data_dict['order_book'] = pd.DataFrame()

    # 4. 持仓量
    try:
        df = fetcher.fetch_open_interest_history()
        df = cleaner.clean_metric_df(df)
        save_data(df, 'open_interest')
        data_dict['open_interest'] = df
    except Exception as e:
        logger.error(f'持仓量获取失败: {e}')
        data_dict['open_interest'] = pd.DataFrame()

    # 5. 资金费率
    try:
        df = fetcher.fetch_funding_rate_history()
        df = cleaner.clean_metric_df(df, 'fundingTime')
        save_data(df, 'funding_rate')
        data_dict['funding_rate'] = df
    except Exception as e:
        logger.error(f'资金费率获取失败: {e}')
        data_dict['funding_rate'] = pd.DataFrame()

    # 6. 多空比
    try:
        df = fetcher.fetch_long_short_ratio()
        df = cleaner.clean_metric_df(df)
        save_data(df, 'long_short_ratio')
        data_dict['long_short_ratio'] = df
    except Exception as e:
        logger.error(f'多空比获取失败: {e}')
        data_dict['long_short_ratio'] = pd.DataFrame()

    # 7. 主动买卖量
    try:
        df = fetcher.fetch_taker_volume()
        df = cleaner.clean_metric_df(df)
        save_data(df, 'taker_volume')
        data_dict['taker_volume'] = df
    except Exception as e:
        logger.error(f'主动买卖量获取失败: {e}')
        data_dict['taker_volume'] = pd.DataFrame()

    # 生成质量报告
    generate_quality_report(data_dict)

    logger.info('=' * 60)
    logger.info('数据下载程序执行完毕')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
