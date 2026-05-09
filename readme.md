# 📈 News Sentiment Analysis for Stock Market Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0.3-green.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

### A Comprehensive Analysis of Financial News Sentiment and Stock Price Movements

</div>

---

# 📌 Project Overview

This project was developed for **Nova Financial Solutions** to investigate the relationship between financial news sentiment and stock market movements.

The project combines:

- 📊 Exploratory Data Analysis (EDA)
- 🧠 Natural Language Processing (NLP)
- 📈 Technical Indicators
- 🔬 Statistical Correlation Analysis

to determine whether financial news sentiment can help predict stock returns.

---

# 🎯 Objectives

The major objectives of this project are:

- Analyze financial news headlines using sentiment analysis
- Compute technical indicators from historical stock data
- Measure correlation between sentiment and stock returns
- Identify whether sentiment can improve trading decisions

---

# 💼 Business Problem

Financial markets generate thousands of news headlines daily.

Some headlines strongly affect stock prices, while others create noise.

The challenge is to:

1. Quantify sentiment from financial headlines
2. Combine sentiment with technical indicators
3. Measure how strongly sentiment relates to stock price movements

---

# 📊 Dataset Description

## 1. Financial News Dataset

| Column    | Description             |
| --------- | ----------------------- |
| headline  | Financial news headline |
| url       | Article URL             |
| publisher | News publisher          |
| date      | Publication timestamp   |
| stock     | Stock ticker symbol     |

### Dataset Size

- **55,987 financial news articles**

---

## 2. Historical Stock Dataset

| Column    | Description            |
| --------- | ---------------------- |
| Date      | Trading day            |
| Open      | Opening price          |
| High      | Highest price          |
| Low       | Lowest price           |
| Close     | Closing price          |
| Adj Close | Adjusted closing price |
| Volume    | Trading volume         |

### Stocks Analyzed

- AAPL
- GOOG
- AMZN
- META
- NVDA

---

# 📁 Project Structure

```bash
news-sentiment-analysis/
│
├── .github/
│   └── workflows/
│       └── unittests.yml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Technical_Indicators.ipynb
│   └── 03_Correlation_Analysis.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── sentiment_analyzer.py
│   └── indicators.py
│
├── tests/
│
├── scripts/
│   └── run_analysis.py
│
├── requirements.txt
├── README.md
└── FINAL_REPORT.md
```

---

# 🚀 Installation & Setup

## Prerequisites

- Python 3.10+
- Git
- Jupyter Notebook

---

## Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/news-sentiment-analysis.git

cd news-sentiment-analysis
```

---

## Step 2: Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open notebooks in this order:

1. `01_Exploratory_Data_Analysis.ipynb`
2. `02_Technical_Indicators.ipynb`
3. `03_Correlation_Analysis.ipynb`

---

# 📝 Task 1: Exploratory Data Analysis

## What Was Done

- Explored financial news dataset
- Analyzed headline lengths
- Identified active publishers
- Examined publication trends by hour/day
- Performed keyword analysis using TF-IDF

---

## Key Insights

| Metric                  | Result        |
| ----------------------- | ------------- |
| Total Articles          | 55,987        |
| Unique Publishers       | 225           |
| Unique Stocks           | 6,204         |
| Average Headline Length | 80 characters |
| Peak Publishing Hour    | 2 PM UTC-4    |
| Busiest Day             | Thursday      |

---

# 📈 Task 2: Technical Indicators

## Indicators Calculated

| Indicator       | Purpose               |
| --------------- | --------------------- |
| SMA             | Trend Identification  |
| EMA             | Weighted Trend        |
| RSI             | Overbought/Oversold   |
| MACD            | Momentum Analysis     |
| Bollinger Bands | Volatility            |
| ATR             | True Range Volatility |

---

## Example Code

```python
import talib

df['RSI_14'] = talib.RSI(df['Close'].values, timeperiod=14)

df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
    df['Close'].values,
    fastperiod=12,
    slowperiod=26,
    signalperiod=9
)
```

---

# 🔬 Task 3: Correlation Analysis

## Sentiment Analysis Tool

The project used **VADER Sentiment Analyzer** because it:

- Handles financial language effectively
- Detects mixed sentiment
- Supports fast real-time analysis

---

## Sentiment Distribution

| Sentiment | Percentage |
| --------- | ---------- |
| Neutral   | 47.1%      |
| Positive  | 29.3%      |
| Negative  | 23.6%      |

---

## Correlation Results

| Stock | Correlation | Significant |
| ----- | ----------- | ----------- |
| AAPL  | +0.08       | ✅ Yes      |
| GOOG  | +0.04       | ❌ No       |
| AMZN  | +0.06       | ❌ No       |
| META  | +0.03       | ❌ No       |
| NVDA  | +0.12       | ✅ Yes      |

---

# 💡 Key Findings

| Finding                                  | Interpretation                               |
| ---------------------------------------- | -------------------------------------------- |
| Weak overall correlation                 | Sentiment alone is not enough                |
| NVDA strongest correlation               | Semiconductor sector reacts strongly to news |
| Positive sentiment outperformed negative | Sentiment adds predictive value              |
| Peak news hour at 2 PM                   | Important timing factor                      |

---

# 📊 Recommended Investment Strategies

## 1. Sentiment Confirmation Strategy

```python
def generate_signal(rsi, sentiment_score):

    if rsi < 30 and sentiment_score > 0:
        return "BUY"

    elif rsi > 70 and sentiment_score < 0:
        return "SELL"

    else:
        return "HOLD"
```

---

## 2. Mean Reversion Strategy

| Sentiment Score | Action     |
| --------------- | ---------- |
| > +0.70         | Short Sell |
| < -0.70         | Long Buy   |

---

# ⚠️ Limitations

- Same-day correlation only
- Correlation does not imply causation
- Time-zone inconsistencies
- Single sentiment model used

---

# 🔮 Future Improvements

- Add lag-based correlation analysis
- Use FinBERT or transformer models
- Build real-time news pipeline
- Add event classification
- Expand to more stocks and sectors

---

# 🛠 Technologies Used

## Core Libraries

```python
pandas
numpy
matplotlib
seaborn
nltk
vaderSentiment
ta-lib
scikit-learn
scipy
pytest
```

---

# 👥 Contributors

| Role         | Name      |
| ------------ | --------- |
| Data Analyst | Your Name |
| Facilitator  | Kerod     |
| Facilitator  | Mahbubah  |
| Facilitator  | Feven     |

---

# 📅 Project Duration

**May 6 – May 12, 2026**

---

# 📄 License

This project is licensed under the MIT License.

```text
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software
and associated documentation files...
```

---

# ⭐ Final Conclusion

The project demonstrated that financial news sentiment has a measurable but relatively weak relationship with same-day stock returns.

While sentiment alone is insufficient for accurate market prediction, combining sentiment analysis with technical indicators can improve trading decision-making and risk management.

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

</div>
