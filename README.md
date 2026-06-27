# Binance SOL 永续合约数据下载项目

本项目使用 CCXT 库接入 Binance 交易所，下载 **SOL/USDT 永续合约** 最近 **15 天** 的多维度数据，并内置完整的防封、数据清洗与质量报告机制。

---

## 功能概览

| 数据类型 | 说明 | 输出文件 |
|---|---|---|
| **K 线数据** | 1 小时 OHLCV | `DATA/ohlcv.csv` / `.json` |
| **Ticker 数据** | 24h 统计（价格、成交量、涨跌等） | `DATA/ticker.json` |
| **订单簿数据** | 深度 100 档买卖盘 | `DATA/order_book.json` |
| **持仓量** | 历史 Open Interest | `DATA/open_interest.csv` / `.json` |
| **资金费率** | 历史 Funding Rate | `DATA/funding_rate.csv` / `.json` |
| **多空比** | 全局 Long/Short Account Ratio | `DATA/long_short_ratio.csv` / `.json` |
| **主动买卖量** | Taker Buy/Sell Volume | `DATA/taker_volume.csv` / `.json` |
| **质量报告** | 各数据集的清洗与检查结果 | `DATA/DATA_QUALITY_REPORT.json` |

---

## 防封机制

Binance 对频繁请求有严格的 IP 限流（429/418），脚本内置了多层防封策略：

1. **CCXT 原生速率限制** (`enableRateLimit=True`)
2. **指数退避重试** — 每次失败后延迟 `2^attempt` 秒
3. **随机抖动** — 在退避基础上增加 `0~2` 秒随机量，避免请求节奏过于规律
4. **限流特殊等待** — 检测到 429/418 时额外等待 `30~60` 秒
5. **备用域名自动切换** — 主域名 `fapi.binance.com` 被封后，自动轮询 `fapi1/2/3.binance.com`
6. **单次接口最大重试 5 次**，全部域名耗尽后才会抛出异常

---

## 数据清洗流程

每类数据下载后均经过以下检查与清洗：

- **去重**：按时间戳去重，保留最早记录
- **排序**：按时间升序排列
- **缺失值插值**：线性插值 + 前后向填充
- **异常值检测**：价格 `<= 0` 时前向填充；修正 `high/low` 逻辑（确保 high >= all, low <= all）
- **时间连续性检查**：检测并报告时间缺口

---

## 快速开始

### 安装依赖

```bash
pip install ccxt pandas numpy
```

### 运行脚本

```bash
python fetch_data.py
```

脚本会在项目根目录生成 `fetch_data.log` 日志，并在 `DATA/` 目录下保存所有数据与质量报告。

---

## 项目结构

```
project/
├── fetch_data.py           # 主下载脚本（含防封 + 清洗逻辑）
├── DATA/                   # 数据输出目录
│   ├── ohlcv.csv / .json
│   ├── open_interest.csv / .json
│   ├── funding_rate.csv / .json
│   ├── long_short_ratio.csv / .json
│   ├── taker_volume.csv / .json
│   ├── ticker.json
│   ├── order_book.json
│   └── DATA_QUALITY_REPORT.json
├── fetch_data.log          # 运行日志（运行时生成）
└── README.md               # 本说明文件
```

---

## 配置说明

如需调整参数，修改 `fetch_data.py` 顶部的 `CONFIG` 字典：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `symbol` | `'SOL/USDT:USDT'` | 交易对（CCXT 永续格式） |
| `raw_symbol` | `'SOLUSDT'` | Binance API 原始符号 |
| `days` | `15` | 回溯天数 |
| `timeframe` | `'1h'` | K 线周期 |
| `data_dir` | `'DATA'` | 输出目录 |
| `max_retries` | `5` | 单接口最大重试次数 |
| `base_delay` | `1.0` | 基础退避秒数 |
| `request_jitter` | `2.0` | 随机抖动上限 |

---

## 注意事项

- **本脚本无需 API Key**，所有接口均为 Binance 公开接口。
- 首次运行可能因数据量大耗时较长（建议保持网络稳定）。
- 若 Binance 对当前 IP 限制较严，脚本会自动切换备用域名并重试；如全部域名受限，请等待 `10~30` 分钟后再试。
- 日志文件 `fetch_data.log` 会记录每次请求、重试、异常及清洗细节，便于排查问题。

---

## 技术栈

- Python 3.8+
- [CCXT](https://github.com/ccxt/ccxt) — 加密货币交易所统一 API
- [Pandas](https://pandas.pydata.org/) — 数据清洗与存储
- [NumPy](https://numpy.org/) — 数值计算

---

> 数据仅供研究分析使用，不构成任何投资建议。
