# OKX SOL‑USDT 永续合约 行情分析与交易计划系统

通过 CCXT 拉取 OKX `SOL-USDT-SWAP` 最近 **15 天**的完整行情与衍生品数据，对数据进行**清洗、质量检查**，再基于多维度技术与量化分析自动生成**带交易计划**的行情报告。

---

## 📁 项目结构

```
okx-sol-perp/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── fetch_data.py        # 第一部分：数据下载 + 清洗 + 质量报告
│   └── analyze_report.py    # 第二部分：多维度分析 + 交易计划报告
├── data/                    # 所有原始/清洗后数据 + 数据质量报告
│   ├── ohlcv_1m.csv
│   ├── ohlcv_5m.csv
│   ├── ohlcv_15m.csv
│   ├── ohlcv_1h.csv
│   ├── ohlcv_4h.csv
│   ├── ohlcv_1d.csv
│   ├── funding_rate.csv
│   ├── funding_current.json
│   ├── open_interest.csv
│   ├── long_short_ratio.csv
│   ├── taker_volume.csv
│   ├── ticker.json
│   └── DATA_QUALITY_REPORT.md   # ← 数据质量报告
└── REPORT.md                # ← 行情分析与交易计划报告（最终交付）
```

---

## ⚙️ 环境依赖

- Python 3.9+
- 依赖：`ccxt`, `pandas`, `numpy`

```bash
pip install -r requirements.txt
```

---

## 🚀 使用方式

```bash
# 第一步：下载最近 15 天数据，自动清洗并生成数据质量报告
python scripts/fetch_data.py

# 第二步：基于 data/ 中的数据生成行情分析与交易计划
python scripts/analyze_report.py
```

执行完毕后：
- `data/DATA_QUALITY_REPORT.md` — 数据质量报告
- `REPORT.md` — 行情分析与交易计划（仓库根目录）

---

## 🛡️ OKX 防封 / 反限流机制

`fetch_data.py` 内置以下保护机制，避免被 OKX 公共 API 限流或封 IP：

| 机制 | 说明 |
|---|---|
| **CCXT `enableRateLimit`** | 启用 ccxt 自带的请求节流（OKX 默认 100 ms / request 估算） |
| **请求间随机抖动 (jitter)** | 每次请求后 `sleep(base + uniform(0, jitter))`，避免固定节奏被风控识别 |
| **指数退避重试** | 捕获 `RateLimitExceeded` / `DDoSProtection` / `NetworkError` / `ExchangeNotAvailable`，最多重试 6 次，退避序列 1s / 2s / 4s / 8s / 16s / 32s |
| **批量大小控制** | OHLCV 单次 `limit ≤ 100`（OKX 公共行情上限 300，但小批量更安全） |
| **进度断点续传** | 已下载的 bar 范围会被合并去重，意外中断重跑只补缺失段 |
| **User‑Agent 伪装** | 启用浏览器风格 UA |
| **错误兜底** | 衍生品类接口失败不影响 OHLCV 主流程 |

---

## 🌏 时区

- 所有写入磁盘的 `dt` 列采用 **Asia/Shanghai (UTC+8)**；
- 内部时间戳 (`ts`) 仍为 UTC 毫秒，可双向解析；
- 报告内所有时间均显示北京时间。

---

## 📊 数据清洗规则

`fetch_data.py` 在落盘前会对每个数据集执行：

1. **去重**：相同时间戳保留最后一条；
2. **排序**：升序按 `ts`；
3. **OHLC 完整性**：丢弃任意 `open/high/low/close` 为 NaN/≤0 的行；
4. **OHLC 关系合理性**：标记 `high < max(open, close)` 或 `low > min(open, close)` 的异常 bar；
5. **缺口检测**：基于周期粒度统计实际 bar 数 / 期望 bar 数，输出**完整度 %**；
6. **价格离群点**：基于 `|return| > 8σ` 标记潜在异常（仅记录，不删除）；
7. **资金费率范围合规**：标记 `|rate| > 0.75%`（OKX 单期上限）的异常。

所有清洗结果汇总到 `data/DATA_QUALITY_REPORT.md`。

---

## ⚠️ 免责声明

- 本项目仅用于研究与教学目的，不构成投资建议；
- 加密永续合约具有高杠杆与极端波动风险，请自负盈亏。
