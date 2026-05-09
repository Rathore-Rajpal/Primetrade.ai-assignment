# 🧠 Master Prompt: Bitcoin Sentiment × Hyperliquid Trader Analysis
> Use this as a direct prompt for your coding agent (Cursor, GitHub Copilot, GPT-4o, Claude, etc.)

---

## CONTEXT & DATASETS

You have two datasets:

**Dataset 1 – Fear & Greed Index** (`fear_greed_index.csv`)
- 2,644 rows, daily from 2018-02-01 to 2025-05-02
- Columns: `timestamp`, `value` (0–100), `classification` (Extreme Fear / Fear / Neutral / Greed / Extreme Greed), `date`

**Dataset 2 – Hyperliquid Historical Trades** (`historical_data.csv`)
- 211,224 rows, all trades within calendar year 2024
- Columns: `Account`, `Coin`, `Execution Price`, `Size Tokens`, `Size USD`, `Side` (BUY/SELL), `Timestamp IST` (format: DD-MM-YYYY HH:MM), `Start Position`, `Direction`, `Closed PnL`, `Transaction Hash`, `Order ID`, `Crossed`, `Fee`, `Trade ID`, `Timestamp`
- 32 unique trader accounts (wallet addresses), 246 unique coins
- Top coins: HYPE, @107, BTC, ETH, SOL, FARTCOIN, MELANIA, PURR/USDC
- Direction values: Open Long, Close Long, Open Short, Close Short, Buy, Sell, Short > Long, Long > Short, Spot Dust Conversion, Auto-Deleveraging, Settlement

---

## ENVIRONMENT SETUP

```python
# Install required libraries
pip install pandas numpy matplotlib seaborn plotly scipy statsmodels scikit-learn jupyter

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Plot style
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12
sns.set_theme(style='darkgrid', palette='husl')
```

---

## STEP 1 — DATA LOADING & CLEANING

```python
# Load
fg = pd.read_csv('fear_greed_index.csv')
ht = pd.read_csv('historical_data.csv')

# Parse dates
ht['datetime'] = pd.to_datetime(ht['Timestamp IST'], format='%d-%m-%Y %H:%M')
ht['date'] = ht['datetime'].dt.date.astype(str)
ht['hour'] = ht['datetime'].dt.hour
ht['dayofweek'] = ht['datetime'].dt.dayofweek  # 0=Monday
ht['month'] = ht['datetime'].dt.month
ht['week'] = ht['datetime'].dt.isocalendar().week.astype(int)

fg['date'] = pd.to_datetime(fg['date']).dt.date.astype(str)

# Merge on date
merged = ht.merge(fg[['date', 'value', 'classification']], on='date', how='left')
merged.rename(columns={'value': 'fg_value', 'classification': 'sentiment'}, inplace=True)

# Derived columns
merged['net_pnl'] = merged['Closed PnL'] - merged['Fee']
merged['is_win'] = merged['Closed PnL'] > 0
merged['is_loss'] = merged['Closed PnL'] < 0
merged['is_long'] = merged['Direction'].isin(['Open Long', 'Close Long'])
merged['is_short'] = merged['Direction'].isin(['Open Short', 'Close Short'])
merged['is_open'] = merged['Direction'].isin(['Open Long', 'Open Short'])
merged['is_close'] = merged['Direction'].isin(['Close Long', 'Close Short'])
merged['fee_pct'] = merged['Fee'] / (merged['Size USD'] + 1e-9) * 100

# Sentiment ordering for plots
SENTIMENT_ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
SENTIMENT_COLORS = {
    'Extreme Fear': '#d73027',
    'Fear': '#f46d43',
    'Neutral': '#fee090',
    'Greed': '#74add1',
    'Extreme Greed': '#313695'
}

print(f"Merged dataset: {merged.shape}")
print(f"Sentiment coverage: {merged['sentiment'].notna().sum()} / {len(merged)}")
print(merged['sentiment'].value_counts())
```

---

## STEP 2 — CORE ANALYSIS MODULES

Build each of the following as a clearly labeled section in your Jupyter notebook.

---

### MODULE A — Sentiment Overview & Market Context

**A1. Distribution of Sentiment in 2024 trade period**
- Bar chart: number of trading days per sentiment class
- Pie chart: proportion of total trades per sentiment

**A2. Fear/Greed timeline overlay**
- Plot FG index value over time (line chart)
- Color-shade background by sentiment zone
- Overlay total daily trade count as bar chart (secondary axis)

**A3. Trade activity vs Sentiment**
```python
# Key finding to visualize:
# Extreme Fear has LOWEST trades per day (42.1), Extreme Greed has HIGHEST (122.7)
# This means traders are paralyzed by fear but hyperactive in greed

fg_day_counts = fg['sentiment'].value_counts()  # days per sentiment (use merged fg)
trade_sent_counts = merged['sentiment'].value_counts()
activity_rate = trade_sent_counts / fg_day_counts  # trades per day by sentiment
# Plot as horizontal bar chart sorted by activity_rate
```

---

### MODULE B — PnL Performance by Sentiment

**B1. Average Closed PnL by Sentiment**
```python
pnl_by_sent = merged.groupby('sentiment')['Closed PnL'].agg(['mean', 'median', 'sum', 'std', 'count'])
pnl_by_sent = pnl_by_sent.reindex(SENTIMENT_ORDER)
# Plot grouped bar: mean PnL per sentiment with error bars (std)
```

**B2. Win Rate by Sentiment**
```python
# Only use trades where PnL != 0 (closed trades)
closed_trades = merged[merged['Closed PnL'] != 0]
win_rate = closed_trades.groupby('sentiment')['is_win'].mean().reindex(SENTIMENT_ORDER)
# Key finding: Extreme Greed = 89.2% win rate, Extreme Fear = 76.2%
# Plot: bar chart with win rate % labels
```

**B3. Net PnL After Fees**
```python
net_by_sent = merged.groupby('sentiment')['net_pnl'].mean().reindex(SENTIMENT_ORDER)
# Fee drag insight: Extreme Greed traders pay highest fee % of trade size (0.025%)
# Fear traders pay highest absolute fee % (0.049%) — sign of rushed/taker orders
```

**B4. PnL Distribution Boxplot**
```python
# Cap outliers for visualization: clip PnL between -5000 and 10000
merged['pnl_clipped'] = merged['Closed PnL'].clip(-5000, 10000)
# Plot: boxplot of pnl_clipped grouped by sentiment (ordered)
# Key finding: Extreme Fear has HIGHEST std dev (1136) = most volatile/risky sentiment
# Neutral has LOWEST std dev (517) = most consistent outcome period
```

---

### MODULE C — Long vs Short Strategy by Sentiment

**C1. Close Long PnL vs Close Short PnL by Sentiment**
```python
long_pnl = merged[merged['Direction']=='Close Long'].groupby('sentiment')['Closed PnL'].mean().reindex(SENTIMENT_ORDER)
short_pnl = merged[merged['Direction']=='Close Short'].groupby('sentiment')['Closed PnL'].mean().reindex(SENTIMENT_ORDER)

# KEY FINDING — UNIQUE INSIGHT:
# SHORT positions during FEAR yield avg $207.7 PnL — highest of any combo
# SHORT positions during EXTREME GREED yield only $28.9 — worst short combo
# LONG positions during GREED yield $89 — best long sentiment
# Plot: grouped bar with Long vs Short side by side per sentiment
```

**C2. Contrarian Strategy Analysis**
```python
# Contrarian Long: Open Long during Fear/Extreme Fear
contrarian_long = merged[
    (merged['Direction']=='Open Long') &
    (merged['sentiment'].isin(['Fear', 'Extreme Fear']))
]
# Contrarian Short: Open Short during Greed/Extreme Greed
contrarian_short = merged[
    (merged['Direction']=='Open Short') &
    (merged['sentiment'].isin(['Greed', 'Extreme Greed']))
]
# Count: 24,829 contrarian longs, 19,327 contrarian shorts
# These are the bold moves — analyze which accounts do this and their outcomes
```

**C3. Buy/Sell Ratio by Sentiment**
```python
buy_sell = merged.groupby(['sentiment','Side']).size().unstack(fill_value=0)
buy_sell['buy_ratio'] = buy_sell['BUY'] / (buy_sell['BUY'] + buy_sell['SELL'])
# KEY FINDING: Extreme Greed has LOWEST buy ratio (0.449) — traders actually SELL into greed
# Neutral and Extreme Fear have highest buy ratios (~0.51) — accumulation phases
# Plot: stacked bar or ratio line
```

**C4. Position Size by Sentiment (Risk Appetite)**
```python
size_by_sent = merged[merged['is_open']].groupby(['sentiment','Direction'])['Size USD'].mean().unstack()
# KEY FINDING: Traders open LARGEST long positions during GREED ($11,019 avg)
# But open SMALLEST longs during Extreme Greed ($4,664) — smart sizing pullback
# Shorts remain relatively consistent (~$3-6k) regardless of sentiment
# Plot: grouped bar — Open Long vs Open Short avg size by sentiment
```

---

### MODULE D — Trader-Level Analysis (Smart Money vs Crowd)

**D1. Account Performance Ranking**
```python
acct_summary = merged.groupby('Account').agg(
    total_pnl=('Closed PnL', 'sum'),
    total_trades=('Closed PnL', 'count'),
    avg_pnl=('Closed PnL', 'mean'),
    win_rate=('is_win', 'mean'),
    total_volume=('Size USD', 'sum'),
    total_fees=('Fee', 'sum'),
    net_pnl=('net_pnl', 'sum')
).sort_values('total_pnl', ascending=False)

# Top earner: 0xb123... → $2.14M total PnL
# #2: 0x0833... → $1.60M
# Plot: horizontal bar of top 15 accounts by total PnL
```

**D2. Sharpe-like Consistency Score**
```python
daily_pnl = merged.groupby(['Account','date'])['Closed PnL'].sum().reset_index()
sharpe_df = daily_pnl.groupby('Account')['Closed PnL'].agg(['mean','std','count'])
sharpe_df['sharpe'] = sharpe_df['mean'] / (sharpe_df['std'] + 1e-9)
sharpe_df = sharpe_df.sort_values('sharpe', ascending=False)

# KEY FINDING: Highest Sharpe accounts are NOT the highest earners
# 0x7f4f... has best Sharpe (0.656) but is not top-5 in absolute PnL
# This reveals: some traders are high-risk high-reward, others are consistent
# Plot: scatter plot of total_pnl vs sharpe — quadrant analysis
```

**D3. Smart Money vs Dumb Money by Sentiment**
```python
# Quartile traders by total PnL
acct_tiers = acct_summary['total_pnl'].quantile([0.25, 0.5, 0.75])
# Map accounts to Q1/Q2/Q3/Q4
# Then analyze: which sentiment did each tier perform best in?

# KEY FINDING:
# Q4 winners crush it in Extreme Greed ($172 avg PnL) and Greed ($102)
# Q1 losers LOSE money during Greed (-$50.8) and Extreme Fear (-$22.6)
# Q3 traders perform BEST in Extreme Fear ($158.9) — the hidden contrarians
# Plot: heatmap of tier vs sentiment → avg PnL
```

**D4. Best Account Deep Dive (0xb123...)**
```python
best = merged[merged['Account']=='0xb1231a4a2dd02f2276fa3c5e2a2f3436e6bfed23']
# Interesting findings:
# Makes $1.1M during Extreme Greed (672 avg per trade)
# But only $9.5k during Extreme Fear (12.9 avg) — greed-follower strategy
# Main coins: @107 ($1.4M), HYPE ($380k), DOGE, SOL, USUAL
# Main edge: Short selling (Close Short = $656k total)
# Plot: pie of PnL contribution by coin, bar of PnL by sentiment
```

---

### MODULE E — Hidden & Non-Obvious Insights

> These are the insights that will differentiate your submission. Most candidates will stop at basic PnL by sentiment. Go deeper.

---

**E1. 🔥 Sentiment Streak Effect (Diminishing Returns)**
```python
# How long has the current sentiment been running? Does streak length affect PnL?
fg_sorted = fg.sort_values('date').copy()
fg_sorted['streak_id'] = (fg_sorted['classification'] != fg_sorted['classification'].shift()).cumsum()
streak_info = fg_sorted.groupby('streak_id').agg(
    sentiment=('classification','first'),
    streak_len=('classification','count')
)
fg_sorted = fg_sorted.merge(streak_info, on='streak_id')
merged_streak = merged.merge(fg_sorted[['date','streak_len','streak_id']], on='date', how='left')
merged_streak['streak_bin'] = pd.cut(merged_streak['streak_len'], 
    bins=[0,3,7,14,30,200], labels=['1-3d','4-7d','8-14d','15-30d','30d+'])

streak_pnl = merged_streak.groupby(['sentiment','streak_bin'])['Closed PnL'].mean().unstack()

# KEY FINDING: During GREED, returns COLLAPSE as streak extends:
# 1-3 days of Greed: $92.2 avg PnL
# 4-7 days: $23.9 | 8-14 days: $27.9 | 30d+: $3.5
# Interpretation: The EARLY days of a Greed phase are most profitable
# Traders who enter late into prolonged greed get squeezed

# During FEAR: 1-3 day fear periods yield $80.5 avg
# 4-7 day: drops to $36 — fear drag compounds
# Plot: heatmap of streak_bin vs sentiment → avg PnL
```

**E2. 🔥 Position Flip Behavior (Short>Long, Long>Short)**
```python
flips = merged[merged['Direction'].isin(['Short > Long', 'Long > Short'])]
flip_analysis = flips.groupby(['sentiment','Direction'])['Closed PnL'].agg(['count','mean'])

# KEY FINDING:
# Short > Long flips during Extreme Fear LOSE $1,932 on average — worst move possible
# Short > Long flips during Fear make $1,115 — high-risk contrarian payoff
# Long > Short flips during Extreme Greed make $120 — smart exit signal
# Interpretation: Flipping short to long in EXTREME fear = panic capitulation
# Flipping long to short in extreme greed = disciplined profit taking

# Plot: grouped bar with Extreme Fear and Extreme Greed highlighted
```

**E3. 🔥 Maker vs Taker Order Quality by Sentiment**
```python
# 'Crossed' = True means taker order (paid more fees, more urgent)
maker_taker = merged.groupby(['sentiment','Crossed'])['Closed PnL'].mean().unstack()
maker_taker.columns = ['Maker (not crossed)', 'Taker (crossed)']

# KEY FINDING:
# During Extreme Greed: Maker orders earn $87.5 vs Taker $56.0 — 56% premium for patience
# During Fear: Maker $80.8 vs Taker $36.7 — HUGE gap (120% premium)
# Interpretation: In fearful markets, patient limit orders massively outperform
# Traders who panic-buy at market during fear destroy their edge
# Plot: grouped bar — Maker vs Taker PnL across sentiments
```

**E4. 🔥 Time-of-Day Alpha by Sentiment**
```python
hourly_sentiment = merged.groupby(['sentiment','hour'])['Closed PnL'].mean().unstack(level=0)

# KEY FINDING:
# During Extreme Fear: Hours 8-9 IST (early morning) yield $231 avg — best window
# During Extreme Greed: Hours 13-15 IST (afternoon) yield $135-177 avg
# These map to: Fear peaks coincide with Asian/early EU session opens
# Greed peaks coincide with EU afternoon / pre-US open

# For each sentiment, find top 3 most profitable hours
for sent in SENTIMENT_ORDER:
    top_hours = merged[merged['sentiment']==sent].groupby('hour')['Closed PnL'].mean().nlargest(3)
    print(f"\n{sent} top hours: {top_hours.to_dict()}")

# Plot: heatmap of hour (x) vs sentiment (y) → avg PnL — like a trading heatmap
```

**E5. 🔥 Coin Selection Intelligence by Sentiment**
```python
coin_sent_pnl = merged.groupby(['Coin','sentiment'])['Closed PnL'].agg(['mean','sum','count']).reset_index()

# KEY FINDING:
# In Extreme Fear: MELANIA ($218 avg), ETH ($196), SUI ($179), TRUMP ($174) outperform
# In Extreme Greed: @107 ($191 avg, $1.98M total) dominates completely
# HYPE works well in Extreme Fear ($46) but underperforms in Extreme Greed ($28)
# Interpretation: There is a clear "sentiment-coin rotation" pattern
# Smart traders should switch coin focus as sentiment changes

# Plot: Top 8 coins by avg PnL in each sentiment as a grouped bar or small multiples
```

**E6. 🔥 Whale Trade Behavior (Size > $100k)**
```python
whales = merged[merged['Size USD'] > 100000]
# Count: 1,706 whale trades (0.8% of all trades but high PnL impact)
# KEY FINDING:
# Extreme Fear: 170 whale trades, avg PnL = $1,629 — highest return per whale trade
# Extreme Greed: Only 61 whale trades, avg PnL = $1,134
# FEAR has 2.8x more whale trades than Extreme Greed
# Interpretation: Big money enters DURING fear, not greed — classic institutional behavior
# Most retail traders do the opposite

whale_by_sent = whales.groupby('sentiment')['Closed PnL'].agg(['count','mean','sum'])
# Plot: dual-axis — bar for count, line for avg PnL
```

**E7. 🔥 Sentiment Transition Signal (The Day After)**
```python
fg_sorted = fg.sort_values('date').copy()
fg_sorted['next_date'] = fg_sorted['date'].shift(-1)
fg_sorted['prev_class'] = fg_sorted['classification'].shift(1)
fg_sorted['transitioned'] = fg_sorted['classification'] != fg_sorted['prev_class']
fg_sorted['transition_label'] = fg_sorted['prev_class'] + ' → ' + fg_sorted['classification']

# Join trade data on the day of transition
transition_dates = fg_sorted[fg_sorted['transitioned']]['date'].tolist()
transition_trades = merged[merged['date'].isin(transition_dates)]
transition_trades = transition_trades.merge(
    fg_sorted[fg_sorted['transitioned']][['date','transition_label']], on='date', how='left'
)
transition_pnl = transition_trades.groupby('transition_label')['Closed PnL'].agg(['mean','count'])
transition_pnl = transition_pnl[transition_pnl['count'] > 50].sort_values('mean', ascending=False)

# KEY FINDING: The day sentiment CHANGES is a high-signal event
# Fear → Neutral transitions and Neutral → Greed transitions
# are the most profitable entry signals
# Plot: horizontal bar of top/bottom transition labels by avg PnL
```

**E8. 🔥 Trader Diversification vs Concentration**
```python
coin_diversity = merged.groupby(['Account','sentiment'])['Coin'].nunique().reset_index()
coin_diversity.columns = ['Account','sentiment','unique_coins']

# KEY FINDING:
# During Extreme Greed: traders use avg 13.5 unique coins — max diversification
# During Extreme Fear: only 6 coins — they flee to safety in blue chips
# This means: In fear, traders concentrate into known assets (ETH, BTC, HYPE, SOL)
# In greed, they scatter into memecoins and altcoins (risk-on behavior)

# Correlate coin diversity with PnL: do concentrated traders outperform in fear?
diversity_pnl = merged.groupby(['Account','date','sentiment']).agg(
    pnl=('Closed PnL','sum'),
    coins=('Coin','nunique')
).reset_index()
print(diversity_pnl.groupby('sentiment')[['pnl','coins']].corr())
```

---

### MODULE F — Statistical Validation

```python
# Don't just show averages — prove they're real differences

# F1. Kruskal-Wallis test: Is PnL significantly different across sentiments?
from scipy.stats import kruskal
groups = [merged[merged['sentiment']==s]['Closed PnL'].dropna() for s in SENTIMENT_ORDER]
stat, p = kruskal(*groups)
print(f"Kruskal-Wallis H={stat:.2f}, p={p:.4f}")
# If p < 0.05: sentiment DOES significantly affect PnL distribution

# F2. Mann-Whitney U test: Extreme Fear vs Extreme Greed PnL
from scipy.stats import mannwhitneyu
ef_pnl = merged[merged['sentiment']=='Extreme Fear']['Closed PnL'].dropna()
eg_pnl = merged[merged['sentiment']=='Extreme Greed']['Closed PnL'].dropna()
stat, p = mannwhitneyu(ef_pnl, eg_pnl, alternative='two-sided')
print(f"Mann-Whitney U (EFear vs EGreed): stat={stat}, p={p:.4f}")

# F3. Chi-square test: Is trade direction (Long/Short) dependent on sentiment?
from scipy.stats import chi2_contingency
ct = pd.crosstab(merged['sentiment'], merged['Direction'])
chi2, p, dof, expected = chi2_contingency(ct)
print(f"Chi-square (Direction vs Sentiment): chi2={chi2:.2f}, p={p:.4f}")

# F4. Pearson/Spearman correlation: FG value vs PnL
from scipy.stats import spearmanr
corr, p = spearmanr(merged['fg_value'].dropna(), 
                     merged.loc[merged['fg_value'].notna(), 'Closed PnL'])
print(f"Spearman correlation (FG value vs PnL): r={corr:.4f}, p={p:.4f}")
```

---

### MODULE G — Final Dashboard (Plotly)

Build a single interactive Plotly dashboard with subplots:

```python
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=[
        'Avg PnL by Sentiment', 'Win Rate by Sentiment', 'Trade Activity by Sentiment',
        'Long vs Short PnL by Sentiment', 'Maker vs Taker PnL', 'Whale Trade Behavior',
        'Sentiment Streak Effect', 'Time-of-Day Heatmap', 'Trader Tier Performance'
    ],
    specs=[[{},{},{}],[{},{},{}],[{},{},{}]]
)
# Add each subplot as a go.Bar, go.Heatmap, go.Scatter as appropriate
fig.update_layout(title_text="Bitcoin Sentiment × Hyperliquid Trader Intelligence Dashboard",
                  height=1200, showlegend=False, template='plotly_dark')
fig.write_html('trading_dashboard.html')
```

---

## STEP 3 — KEY INSIGHTS SUMMARY (Write This Section Last)

After running all analysis, write a structured insight report. Use this template:

```markdown
## Executive Summary

### 1. Sentiment-Performance Paradox
Extreme Greed has the HIGHEST win rate (89.2%) but Extreme Fear produces the HIGHEST 
per-trade PnL for whale trades ($1,629 avg). Win rate ≠ profitability.

### 2. The Diminishing Greed Effect  
Traders who enter in the FIRST 1-3 days of a Greed streak earn $92.2 avg PnL.
Those who enter after 30+ consecutive Greed days earn just $3.5. 
Strategy: Monitor streak length, not just sentiment label.

### 3. Short Selling in Fear is the Hidden Alpha
Close Short trades during Fear periods yield $207.7 avg PnL — the single best 
direction-sentiment combination in the dataset. Most traders avoid shorting when 
fearful, making this an underutilized edge.

### 4. Patience Pays (Maker vs Taker)
During Fear, maker (limit) orders earn 120% more than taker (market) orders. 
In volatile sentiment periods, execution quality is as important as direction.

### 5. Whales Accumulate in Fear
Only 61 whale trades ($100k+) occurred during Extreme Greed vs 170 during Extreme Fear.
Big money consistently buys when retail is most scared — the classic smart money signal.

### 6. Coin Rotation is Real
MELANIA, ETH, SUI outperform during Extreme Fear. @107 dominates during Extreme Greed.
Successful traders rotate their coin selection with sentiment, not against it.

### 7. Sentiment Transitions are Entry Signals
The day sentiment CHANGES (Fear → Neutral, Neutral → Greed) carries higher PnL than 
the middle of sustained sentiment streaks. Transitions are the signal; streaks are noise.

### 8. Position Flips in Extreme Fear = Capitulation Signal
Short → Long flips during Extreme Fear lose $1,932 avg. This is panic behavior. 
The traders who HELD their shorts during Extreme Fear instead made $123 avg.
```

---

## DELIVERABLES CHECKLIST

- [ ] `analysis.ipynb` — Full Jupyter notebook with all code, charts, and markdown commentary
- [ ] `trading_dashboard.html` — Interactive Plotly dashboard
- [ ] `insights_report.pdf` — 1-2 page write-up of key findings
- [ ] All charts exported as `.png` for report inclusion

---

## PRO TIPS FOR STANDING OUT

1. **Don't just describe — prescribe.** Every insight should end with "therefore, a trader should..."
2. **Use statistical tests** (Kruskal-Wallis, Mann-Whitney) — most candidates won't
3. **The streak analysis (E1)** is likely unique — no standard tutorial covers this
4. **The maker/taker split (E3)** requires using the `Crossed` column which most people will ignore
5. **Coin rotation (E5)** requires cross-referencing two groupby dimensions — elegant and deep
6. **Title every chart clearly** with the insight, not just the variable names
7. Keep the Plotly dashboard interactive — it's 10x more impressive than static matplotlib
