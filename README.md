# 📊 Bitcoin Sentiment × Hyperliquid Trader Analysis

> Exploring the relationship between Bitcoin market sentiment (Fear & Greed Index) and real trader performance on Hyperliquid — uncovering hidden patterns that drive smarter trading strategies.

---

## 📁 Dataset

| Dataset | Rows | Period |
|---|---|---|
| Bitcoin Fear & Greed Index | 2,644 | Feb 2018 – May 2025 |
| Hyperliquid Historical Trades | 211,224 | Full Year 2024 |
| Merged Dataset | 211,218 | 2024 (99.99% coverage) |

---

## 🔍 Key Insights Uncovered

- **Short selling during Fear = $207.68 avg PnL** — the single best direction-sentiment combo in the dataset
- **Greed streak effect** — entering in days 1-3 of a Greed streak yields $92.20 avg PnL vs just $3.54 after 30+ consecutive days
- **Whales accumulate in Fear** — 830 whale trades (>$100k) during Fear vs only 61 during Extreme Greed
- **Maker orders in Fear earn 120% more** than panic market orders — execution quality matters as much as direction
- **Position flips in Extreme Fear lose $1,932 avg** — panic-switching direction at sentiment extremes is the worst move
- **Coin rotation is real** — @107 dominates Extreme Greed, ETH/MELANIA lead in Extreme Fear
- **All findings statistically validated** — Kruskal-Wallis H=1,227, p < 0.000001

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy, matplotlib, seaborn, plotly
- **Statistics** — scipy (Kruskal-Wallis, Mann-Whitney U, Spearman correlation)
- **Environment** — Jupyter Notebook

---

## 📂 Repository Structure

```
├── analysis.ipynb          # Main Jupyter notebook — all analysis & charts
├── trading_analysis_report.docx  # Full insights report with strategy framework
├── charts/
│   ├── pnl_by_sentiment.png
│   ├── winrate_by_sentiment.png
│   ├── long_short_pnl.png
│   ├── maker_taker_pnl.png
│   ├── whale_trades.png
│   ├── streak_effect.png
│   ├── coin_rotation.png
│   ├── sharpe_scatter.png
│   └── smart_money.png
├── data/
│   ├── fear_greed_index.csv
│   └── historical_data.csv
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/bitcoin-sentiment-trader-analysis
cd bitcoin-sentiment-trader-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn plotly scipy statsmodels scikit-learn jupyter

# Launch Jupyter
jupyter notebook analysis.ipynb
```

---

## 📈 Analysis Modules

| Module | What It Covers |
|---|---|
| A — Sentiment Overview | Distribution, trade activity, FG timeline |
| B — PnL by Sentiment | Avg PnL, win rate, net PnL after fees |
| C — Long vs Short | Direction performance, contrarian strategy, buy/sell ratio |
| D — Trader Analysis | Account rankings, Sharpe score, smart vs dumb money |
| E — Hidden Insights | Streak effect, whale behavior, maker/taker, coin rotation, position flips |
| F — Statistical Tests | Kruskal-Wallis, Mann-Whitney U, Spearman correlation |

---

## 📄 License

MIT
