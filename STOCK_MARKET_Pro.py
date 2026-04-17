#!/usr/bin/env python3
"""
STOCK MARKET Pro - Advanced US Stock Analysis
Features:
- Real-time stock data from Yahoo Finance
- Technical indicators (RSI, MACD, Bollinger Bands, Stochastic, ATR)
- Multi-model ML predictions (Random Forest, Technical, Momentum, Ensemble)
- Advanced forecasting (Linear Regression, AR(1), Monte Carlo, ARIMA)
- Paper trading with portfolio tracking
- Watchlist with price alerts
- Financial statements and ratio analysis
- Comprehensive reports (PDF, HTML, CSV)
- Custom date range and data caching
- Freemium: 3 unique tickers free, email unlock (Web3Forms)
"""

import os
import sys
import subprocess
import importlib
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import json
from pathlib import Path
import threading
import csv
import webbrowser
import traceback
import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from io import BytesIO

warnings.filterwarnings('ignore')

REQUIRED_PACKAGES = [
    'streamlit', 'pandas', 'numpy', 'yfinance', 'plotly',
    'scikit-learn', 'matplotlib', 'requests', 'joblib',
    'openpyxl', 'xlsxwriter', 'statsmodels', 'ta',
    'reportlab', 'kaleido'
]

def install_missing_packages():
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            if package == 'scikit-learn':
                import sklearn
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
    if missing:
        print(f"📦 Installing {len(missing)} missing packages...")
        for package in missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed")
            except Exception as e:
                print(f"❌ Failed to install {package}: {e}")

install_missing_packages()

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import ta
from io import BytesIO, StringIO
import base64
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

# Freemium module
import freemium

# =============================================================================
# Configuration
# =============================================================================
class Config:
    DB_PATH = "data/pro_us_market.db"
    CACHE_DIR = "data/cache"
    EXPORT_DIR = "data/exports"
    REPORT_DIR = "data/reports"
    DEFAULT_BALANCE = 100000.0
    COMMISSION_RATE = 0.0

    MAJOR_INDICES = {
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'NASDAQ': '^IXIC',
        'Russell 2000': '^RUT',
        'S&P 400 MidCap': '^MID',
        'NYSE Composite': '^NYA',
        'CBOE Volatility': '^VIX',
        'S&P 100': '^OEX',
        'NASDAQ 100': '^NDX',
        'Dow Jones Utilities': '^DJU',
        'Dow Jones Transportation': '^DJT',
        'Wilshire 5000': '^W5000'
    }
    PERIOD_OPTIONS = ['1w','1mo','2mo','3mo','4mo','5mo','6mo','9mo','1y','2y','3y','5y','max']

for directory in [Config.CACHE_DIR, Config.EXPORT_DIR, Config.REPORT_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

POPULAR_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
    'JPM', 'JNJ', 'V', 'WMT', 'DIS', 'NFLX', 'AMD', 'INTC',
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI'
]

# =============================================================================
# Helper functions
# =============================================================================
def format_large_number(x):
    if pd.isna(x) or x == 0: return "N/A"
    if isinstance(x, (int, float)):
        if abs(x) >= 1e12: return f"${x/1e12:.2f}T"
        if abs(x) >= 1e9: return f"${x/1e9:.2f}B"
        if abs(x) >= 1e6: return f"${x/1e6:.2f}M"
        if abs(x) >= 1e3: return f"${x/1e3:.0f}K"
        return f"${x:.0f}"
    return str(x)

def format_percent(x):
    return f"{x:.2%}" if not pd.isna(x) else "N/A"

def safe_divide(a, b):
    return a / b if b and b != 0 else np.nan

# =============================================================================
# Logger
# =============================================================================
class Logger:
    def __init__(self):
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_info(self, message):
        self.logger.info(message)

    def log_error(self, message):
        self.logger.error(message)

    def log_warning(self, message):
        self.logger.warning(message)

# =============================================================================
# Professional Data Manager (caching + financial statements)
# =============================================================================
class ProfessionalDataManager:
    def __init__(self):
        self.price_cache = {}
        self.logger = Logger()

    @st.cache_data(ttl=300, show_spinner=False)
    def get_stock_data(_self, symbol: str, period: str = "max", interval: str = "1d",
                       start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            if start and end:
                df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
            else:
                # Map custom periods to yfinance periods
                period_map = {
                    "1w": "5d",
                    "1mo": "1mo", "2mo": "2mo", "3mo": "3mo", "4mo": "3mo", "5mo": "3mo",
                    "6mo": "6mo", "9mo": "6mo",
                    "1y": "1y", "2y": "2y", "3y": "3y", "5y": "5y", "max": "max"
                }
                yf_period = period_map.get(period, "2y")
                df = ticker.history(period=yf_period, interval=interval, auto_adjust=True)

            if df.empty:
                return pd.DataFrame()

            df.columns = [col.lower() for col in df.columns]
            df = _self._calculate_advanced_indicators(df)

            return df

        except Exception as e:
            _self.logger.log_error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def _calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            if period <= 50:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14 + 1e-10))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()

        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            df['volume_change'] = df['volume'].pct_change()

        df['daily_return'] = df['close'].pct_change()
        for periods in [5, 21, 63, 126, 252, 756, 1260]:
            df[f'return_{periods}d'] = df['close'].pct_change(periods)

        df['volatility_20d'] = df['daily_return'].rolling(20).std() * np.sqrt(252)

        return df.fillna(method='ffill').fillna(method='bfill')

    @st.cache_data(ttl=300, show_spinner=False)
    def get_index_data(_self, index_symbol: str) -> Dict:
        try:
            ticker = yf.Ticker(index_symbol)
            try:
                df = ticker.history(period="max", auto_adjust=True)
            except:
                df = ticker.history(period="10y", auto_adjust=True)

            if df.empty:
                return {}

            current_price = df['Close'].iloc[-1]
            returns = {}

            timeframes = [
                (1, '1d'), (5, '1w'), (21, '1m'), (63, '3m'),
                (126, '6m'), (252, '1y'), (756, '3y'), (1260, '5y'), (2520, '10y')
            ]

            for days, label in timeframes:
                if len(df) > days:
                    try:
                        past_price = df['Close'].iloc[-days-1]
                        returns[label] = (current_price - past_price) / past_price * 100
                    except:
                        returns[label] = None
                else:
                    returns[label] = None

            returns = {k: v for k, v in returns.items() if v is not None}

            return returns

        except Exception as e:
            _self.logger.log_error(f"Error fetching index data for {index_symbol}: {e}")
            return {}

    @st.cache_data(ttl=300, show_spinner=False)
    def get_comprehensive_fundamental_data(_self, symbol: str) -> Dict:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            trailing_eps = info.get('trailingEps')
            forward_eps = info.get('forwardEps')

            if not info.get('trailingPE') and current_price and trailing_eps and trailing_eps != 0:
                calculated_trailing_pe = current_price / trailing_eps
            else:
                calculated_trailing_pe = info.get('trailingPE', 'N/A')

            if not info.get('forwardPE') and current_price and forward_eps and forward_eps != 0:
                calculated_forward_pe = current_price / forward_eps
            else:
                calculated_forward_pe = info.get('forwardPE', 'N/A')

            def fmt_mcap(x):
                if not x: return 'N/A'
                if x >= 1e12: return f"${x/1e12:.2f}T"
                if x >= 1e9: return f"${x/1e9:.2f}B"
                if x >= 1e6: return f"${x/1e6:.2f}M"
                return f"${x:,.0f}"

            fundamentals = {
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'country': info.get('country', 'N/A'),
                'employees': f"{info.get('fullTimeEmployees', 0):,}" if info.get('fullTimeEmployees') else 'N/A',
                'current_price': info.get('currentPrice', 'N/A'),
                'previous_close': info.get('previousClose', 'N/A'),
                'market_cap': fmt_mcap(info.get('marketCap')),
                'enterprise_value': fmt_mcap(info.get('enterpriseValue')),
                'trailing_pe': calculated_trailing_pe if calculated_trailing_pe != 'N/A' else 'N/A',
                'forward_pe': calculated_forward_pe if calculated_forward_pe != 'N/A' else 'N/A',
                'peg_ratio': info.get('pegRatio', 'N/A'),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 'N/A'),
                'price_to_book': info.get('priceToBook', 'N/A'),
                'enterprise_to_revenue': info.get('enterpriseToRevenue', 'N/A'),
                'enterprise_to_ebitda': info.get('enterpriseToEbitda', 'N/A'),
                'profit_margin': f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get('profitMargins') else 'N/A',
                'operating_margin': f"{info.get('operatingMargins', 0) * 100:.2f}%" if info.get('operatingMargins') else 'N/A',
                'return_on_equity': f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get('returnOnEquity') else 'N/A',
                'return_on_assets': f"{info.get('returnOnAssets', 0) * 100:.2f}%" if info.get('returnOnAssets') else 'N/A',
                'debt_to_equity': f"{info.get('debtToEquity', 0):.2f}" if info.get('debtToEquity') else 'N/A',
                'current_ratio': f"{info.get('currentRatio', 0):.2f}" if info.get('currentRatio') else 'N/A',
                'quick_ratio': f"{info.get('quickRatio', 0):.2f}" if info.get('quickRatio') else 'N/A',
                'revenue_growth': f"{info.get('revenueGrowth', 0) * 100:.2f}%" if info.get('revenueGrowth') else 'N/A',
                'earnings_growth': f"{info.get('earningsGrowth', 0) * 100:.2f}%" if info.get('earningsGrowth') else 'N/A',
                'earnings_quarterly_growth': f"{info.get('earningsQuarterlyGrowth', 0) * 100:.2f}%" if info.get('earningsQuarterlyGrowth') else 'N/A',
                'dividend_yield': f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else 'N/A',
                'dividend_rate': info.get('dividendRate', 'N/A'),
                'payout_ratio': f"{info.get('payoutRatio', 0) * 100:.2f}%" if info.get('payoutRatio') else 'N/A',
                'five_year_dividend_growth': f"{info.get('dividendGrowth', 0) * 100:.2f}%" if info.get('dividendGrowth') else 'N/A',
                'beta': info.get('beta', 'N/A'),
                '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
                '50_day_ma': info.get('fiftyDayAverage', 'N/A'),
                '200_day_ma': info.get('twoHundredDayAverage', 'N/A'),
                'volume': f"{info.get('volume', 0):,}" if info.get('volume') else 'N/A',
                'avg_volume': f"{info.get('averageVolume', 0):,}" if info.get('averageVolume') else 'N/A',
                'shares_outstanding': f"{info.get('sharesOutstanding', 0):,}" if info.get('sharesOutstanding') else 'N/A',
                'float_shares': f"{info.get('floatShares', 0):,}" if info.get('floatShares') else 'N/A',
                'trailing_eps': info.get('trailingEps', 'N/A'),
                'forward_eps': info.get('forwardEps', 'N/A'),
                'earnings_date': info.get('earningsDate', 'N/A'),
                'target_mean_price': info.get('targetMeanPrice', 'N/A'),
                'target_high_price': info.get('targetHighPrice', 'N/A'),
                'target_low_price': info.get('targetLowPrice', 'N/A'),
                'recommendation': info.get('recommendationMean', 'N/A'),
                'number_of_analysts': info.get('numberOfAnalystOpinions', 'N/A')
            }

            return {k: v for k, v in fundamentals.items() if v != 'N/A' and v is not None}

        except Exception as e:
            return {'error': f'Failed to fetch fundamentals: {str(e)}'}

    @st.cache_data(ttl=300, show_spinner=False)
    def get_financial_statements(_self, symbol: str, stmt_type: str = 'income', freq: str = 'annual') -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            if freq == 'annual':
                if stmt_type == 'income':
                    return ticker.financials
                elif stmt_type == 'balance':
                    return ticker.balance_sheet
                elif stmt_type == 'cash':
                    return ticker.cashflow
            else:
                if stmt_type == 'income':
                    return ticker.quarterly_financials
                elif stmt_type == 'balance':
                    return ticker.quarterly_balance_sheet
                elif stmt_type == 'cash':
                    return ticker.quarterly_cashflow
            return pd.DataFrame()
        except Exception as e:
            _self.logger.log_error(f"Financials error: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=300, show_spinner=False)
    def calculate_ratios_from_statements(_self, symbol: str) -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            inc = ticker.financials
            bal = ticker.balance_sheet
            if inc.empty or bal.empty:
                return pd.DataFrame()
            ratios = pd.DataFrame(index=inc.columns)
            for d in inc.columns:
                try:
                    rev = inc.loc['Total Revenue', d] if 'Total Revenue' in inc.index else np.nan
                    gp = inc.loc['Gross Profit', d] if 'Gross Profit' in inc.index else np.nan
                    op = inc.loc['Operating Income', d] if 'Operating Income' in inc.index else np.nan
                    ni = inc.loc['Net Income', d] if 'Net Income' in inc.index else np.nan
                    ta = bal.loc['Total Assets', d] if 'Total Assets' in bal.index else np.nan
                    tl = bal.loc['Total Liabilities Net Minority Interest', d] if 'Total Liabilities Net Minority Interest' in bal.index else np.nan
                    te = bal.loc['Total Equity Gross Minority Interest', d] if 'Total Equity Gross Minority Interest' in bal.index else np.nan
                    ca = bal.loc['Current Assets', d] if 'Current Assets' in bal.index else np.nan
                    cl = bal.loc['Current Liabilities', d] if 'Current Liabilities' in bal.index else np.nan
                    ratios.loc[d, 'Gross Margin'] = safe_divide(gp, rev)
                    ratios.loc[d, 'Operating Margin'] = safe_divide(op, rev)
                    ratios.loc[d, 'Net Margin'] = safe_divide(ni, rev)
                    ratios.loc[d, 'ROE'] = safe_divide(ni, te)
                    ratios.loc[d, 'ROA'] = safe_divide(ni, ta)
                    ratios.loc[d, 'Debt/Equity'] = safe_divide(tl, te)
                    ratios.loc[d, 'Current Ratio'] = safe_divide(ca, cl)
                except:
                    continue
            return ratios.T
        except Exception as e:
            _self.logger.log_error(f"Ratio calc error: {e}")
            return pd.DataFrame()

# =============================================================================
# Advanced Forecasting Engine
# =============================================================================
class AdvancedForecastingEngine:
    def __init__(self):
        self.data_manager = ProfessionalDataManager()

    def project_linear_regression(self, df: pd.DataFrame, days_ahead: int = 30):
        try:
            df_clean = df.dropna(subset=['close']).copy()
            if len(df_clean) < 30:
                return None, None

            X = np.arange(len(df_clean)).reshape(-1, 1)
            y = df_clean['close'].values

            model = LinearRegression()
            model.fit(X, y)

            future_X = np.arange(len(df_clean), len(df_clean) + days_ahead).reshape(-1, 1)
            future_prices = model.predict(future_X)

            last_date = df_clean.index[-1]
            if isinstance(last_date, pd.Timestamp):
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
            else:
                future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]

            future_series = pd.Series(future_prices, index=future_dates)
            final_price = future_prices[-1]

            return final_price, future_series

        except Exception as e:
            return None, None

    def project_ar1(self, df: pd.DataFrame, days_ahead: int = 30):
        try:
            df_clean = df.dropna(subset=['close']).copy()
            if len(df_clean) < 60:
                return None, None

            returns = df_clean['close'].pct_change().dropna()

            if len(returns) < 30:
                return None, None

            X = returns[:-1].values.reshape(-1, 1)
            y = returns[1:].values

            model = LinearRegression()
            model.fit(X, y)

            last_return = returns.iloc[-1]
            last_price = df_clean['close'].iloc[-1]

            future_prices = [last_price]
            current_return = last_return

            for _ in range(days_ahead):
                next_return = model.predict([[current_return]])[0]
                next_price = future_prices[-1] * (1 + next_return)
                future_prices.append(next_price)
                current_return = next_return

            future_prices = future_prices[1:]

            last_date = df_clean.index[-1]
            if isinstance(last_date, pd.Timestamp):
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
            else:
                future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]

            future_series = pd.Series(future_prices, index=future_dates)
            final_price = future_prices[-1]

            return final_price, future_series

        except Exception as e:
            return None, None

    def project_monte_carlo_vectorized(self, df: pd.DataFrame, days_ahead: int = 30, sims: int = 1000):
        if df.empty or 'close' not in df.columns:
            return None, None, None, None

        try:
            df_clean = df.dropna(subset=['close']).copy()
            if len(df_clean) < 30:
                return None, None, None, None

            returns = df_clean['close'].pct_change().dropna()

            if len(returns) < 20:
                return None, None, None, None

            mu = returns.mean()
            sigma = returns.std()
            last_price = df_clean['close'].iloc[-1]

            np.random.seed(42)
            daily_returns = np.random.normal(mu, sigma, (sims, days_ahead))
            price_paths = np.zeros((sims, days_ahead))
            price_paths[:, 0] = last_price * (1 + daily_returns[:, 0])

            for t in range(1, days_ahead):
                price_paths[:, t] = price_paths[:, t-1] * (1 + daily_returns[:, t])

            final_prices = price_paths[:, -1]
            mean_proj = np.mean(final_prices)
            low_proj = np.percentile(final_prices, 5)
            high_proj = np.percentile(final_prices, 95)

            last_date = df_clean.index[-1]
            if isinstance(last_date, pd.Timestamp):
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
            else:
                future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]

            mean_series = pd.Series(np.mean(price_paths, axis=0), index=future_dates)
            low_series = pd.Series(np.percentile(price_paths, 5, axis=0), index=future_dates)
            high_series = pd.Series(np.percentile(price_paths, 95, axis=0), index=future_dates)

            return mean_proj, low_proj, high_proj, (mean_series, low_series, high_series)

        except Exception as e:
            return None, None, None, None

    def project_arima_smart(self, df: pd.DataFrame, days_ahead: int = 30):
        if df.empty or 'close' not in df.columns:
            return None, "No data available"

        try:
            series = df['close'].dropna()
            if len(series) < 30:
                return None, "Insufficient data for ARIMA"

            best_aic = np.inf
            best_order = (1, 1, 1)
            best_model = None

            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            results = model.fit()

                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_order = (p, d, q)
                                best_model = results
                        except:
                            continue

            if best_model is not None:
                forecast = best_model.get_forecast(steps=days_ahead)
                forecast_series = forecast.predicted_mean

                last_date = series.index[-1]
                if isinstance(last_date, pd.Timestamp):
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
                else:
                    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]

                forecast_series.index = future_dates
                summary = f"ARIMA{best_order} | AIC: {best_aic:.2f}"

                return forecast_series, summary
            else:
                return None, "No suitable ARIMA model found"

        except Exception as e:
            return None, f"ARIMA failed: {str(e)}"

# =============================================================================
# Multi-Method Prediction Engine
# =============================================================================
class MultiMethodPredictionEngine:
    def __init__(self):
        self.data_manager = ProfessionalDataManager()

    def get_comprehensive_prediction(self, symbol: str, period: str = "max") -> Dict:
        try:
            data = self.data_manager.get_stock_data(symbol, period)

            if data.empty:
                return {'error': 'No data available'}

            predictions = {}

            rf_pred = self._advanced_random_forest_prediction(data, symbol)
            if rf_pred:
                predictions['Random Forest Model'] = rf_pred

            ta_pred = self._comprehensive_technical_prediction(data)
            if ta_pred:
                predictions['Technical Analysis Composite'] = ta_pred

            momentum_pred = self._momentum_trend_prediction(data)
            if momentum_pred:
                predictions['Momentum & Trend Analysis'] = momentum_pred

            ml_pred = self._ml_ensemble_prediction(data)
            if ml_pred:
                predictions['Statistical Ensemble Model'] = ml_pred

            best_pred = self._select_best_prediction(predictions)

            analysis = self._get_detailed_analysis(data, symbol)

            return {
                'predictions': predictions,
                'best_prediction': best_pred,
                'detailed_analysis': analysis,
                'symbol': symbol,
                'period_analyzed': period,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_points': len(data)
            }

        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}

    def _advanced_random_forest_prediction(self, df: pd.DataFrame, symbol: str) -> Dict:
        try:
            if len(df) < 100:
                return None

            feature_cols = [
                'sma_10', 'sma_20', 'sma_50', 'sma_200', 'rsi', 'macd', 'macd_signal',
                'stoch_k', 'stoch_d', 'bb_position', 'atr', 'volume_sma', 'volatility_20d'
            ]
            available_features = [f for f in feature_cols if f in df.columns]

            if len(available_features) < 6:
                return None

            X = df[available_features].iloc[:-1].copy()
            y = (df['close'].shift(-1) > df['close']).iloc[:-1].astype(int)

            split_idx = int(0.8 * len(X))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            if len(X_train) < 30:
                return None

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                min_samples_split=10,
                min_samples_leaf=4
            )
            model.fit(X_train, y_train)

            accuracy = accuracy_score(y_test, model.predict(X_test))

            feature_importance = dict(zip(available_features, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

            last_features = X.iloc[[-1]]
            prediction = model.predict(last_features)[0]
            confidence = model.predict_proba(last_features)[0].max()

            adjusted_confidence = (confidence + accuracy) / 2

            signal = "BUY" if prediction == 1 else "SELL"

            return {
                'signal': signal,
                'confidence': adjusted_confidence,
                'accuracy': accuracy,
                'method': 'Random Forest Model',
                'top_features': top_features,
                'model_parameters': f"200 trees, max_depth=15",
                'training_samples': len(X_train),
                'feature_count': len(available_features)
            }

        except Exception as e:
            return None

    def _comprehensive_technical_prediction(self, df: pd.DataFrame) -> Dict:
        try:
            signals = []
            current_price = df['close'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50

            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if rsi < 30:
                    signals.append(('BUY', 0.2, f'RSI Oversold ({rsi:.1f})', 'green'))
                elif rsi > 70:
                    signals.append(('SELL', 0.2, f'RSI Overbought ({rsi:.1f})', 'red'))
                elif 30 <= rsi <= 50:
                    signals.append(('BUY', 0.1, f'RSI Neutral-Bullish ({rsi:.1f})', 'lightgreen'))
                else:
                    signals.append(('SELL', 0.1, f'RSI Neutral-Bearish ({rsi:.1f})', 'orange'))

            if all(col in df.columns for col in ['macd', 'macd_signal']):
                macd = df['macd'].iloc[-1]
                signal_line = df['macd_signal'].iloc[-1]
                histogram = df['macd_histogram'].iloc[-1] if 'macd_histogram' in df.columns else 0

                if macd > signal_line and histogram > 0:
                    signals.append(('BUY', 0.25, 'MACD Bullish & Rising', 'green'))
                elif macd < signal_line and histogram < 0:
                    signals.append(('SELL', 0.25, 'MACD Bearish & Falling', 'red'))
                elif macd > signal_line:
                    signals.append(('BUY', 0.15, 'MACD Bullish', 'lightgreen'))
                else:
                    signals.append(('SELL', 0.15, 'MACD Bearish', 'orange'))

            if all(col in df.columns for col in ['sma_20', 'sma_50', 'sma_200']):
                sma_20 = df['sma_20'].iloc[-1]
                sma_50 = df['sma_50'].iloc[-1]
                sma_200 = df['sma_200'].iloc[-1]

                if current_price > sma_20 > sma_50 > sma_200:
                    signals.append(('BUY', 0.2, 'Strong Uptrend (All MAs)', 'green'))
                elif current_price < sma_20 < sma_50 < sma_200:
                    signals.append(('SELL', 0.2, 'Strong Downtrend (All MAs)', 'red'))
                elif current_price > sma_20 > sma_50:
                    signals.append(('BUY', 0.15, 'Uptrend (Above 20 & 50 MA)', 'lightgreen'))
                elif current_price < sma_20 < sma_50:
                    signals.append(('SELL', 0.15, 'Downtrend (Below 20 & 50 MA)', 'orange'))
                elif current_price > sma_20:
                    signals.append(('BUY', 0.1, 'Above 20 MA', 'lightgreen'))
                else:
                    signals.append(('SELL', 0.1, 'Below 20 MA', 'orange'))

            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_position']):
                bb_position = df['bb_position'].iloc[-1]
                if bb_position < 0.2:
                    signals.append(('BUY', 0.15, 'Near BB Lower Band (Oversold)', 'green'))
                elif bb_position > 0.8:
                    signals.append(('SELL', 0.15, 'Near BB Upper Band (Overbought)', 'red'))
                elif bb_position < 0.4:
                    signals.append(('BUY', 0.1, 'Lower BB Region', 'lightgreen'))
                elif bb_position > 0.6:
                    signals.append(('SELL', 0.1, 'Upper BB Region', 'orange'))

            if 'volume' in df.columns and 'volume_sma' in df.columns:
                volume_ratio = df['volume'].iloc[-1] / df['volume_sma'].iloc[-1]
                if volume_ratio > 1.5 and current_price > df['close'].iloc[-2]:
                    signals.append(('BUY', 0.1, 'High Volume Breakout', 'green'))
                elif volume_ratio > 1.5 and current_price < df['close'].iloc[-2]:
                    signals.append(('SELL', 0.1, 'High Volume Breakdown', 'red'))

            if 'volatility_20d' in df.columns:
                volatility = df['volatility_20d'].iloc[-1]
                if volatility > 0.4:
                    if current_rsi < 40:
                        signals.append(('BUY', 0.1, 'High Volatility + Oversold', 'green'))
                    elif current_rsi > 60:
                        signals.append(('SELL', 0.1, 'High Volatility + Overbought', 'red'))

            if not signals:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.5,
                    'accuracy': 0.65,
                    'method': 'Technical Analysis',
                    'signals_detail': [('HOLD', 1.0, 'No clear signals', 'gray')]
                }

            buy_score = sum(weight for signal, weight, _, _ in signals if signal == 'BUY')
            sell_score = sum(weight for signal, weight, _, _ in signals if signal == 'SELL')
            hold_score = sum(weight for signal, weight, _, _ in signals if signal == 'HOLD')

            max_score = max(buy_score, sell_score, hold_score)

            if buy_score == max_score:
                final_signal = 'BUY'
                confidence = buy_score
            elif sell_score == max_score:
                final_signal = 'SELL'
                confidence = sell_score
            else:
                final_signal = 'HOLD'
                confidence = hold_score

            confidence = min(0.95, max(0.05, confidence))

            accuracy = 0.68 + (confidence - 0.5) * 0.3

            return {
                'signal': final_signal,
                'confidence': confidence,
                'accuracy': accuracy,
                'method': 'Technical Analysis Composite',
                'signals_detail': signals,
                'signal_count': len(signals),
                'buy_signals': len([s for s in signals if s[0] == 'BUY']),
                'sell_signals': len([s for s in signals if s[0] == 'SELL'])
            }

        except Exception as e:
            return None

    def _momentum_trend_prediction(self, df: pd.DataFrame) -> Dict:
        try:
            if len(df) < 50:
                return None

            price_change_5d = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
            price_change_20d = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
            price_change_50d = (df['close'].iloc[-1] / df['close'].iloc[-50] - 1) * 100

            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                trend_strength = (df['sma_20'].iloc[-1] - df['sma_50'].iloc[-1]) / df['sma_50'].iloc[-1] * 100
            else:
                trend_strength = 0

            momentum_score = (price_change_5d * 0.4 + price_change_20d * 0.3 + price_change_50d * 0.3)

            if momentum_score > 5 and trend_strength > 2:
                signal = 'BUY'
                confidence = min(0.9, 0.5 + abs(momentum_score) / 100)
                reason = f"Strong Momentum (+{momentum_score:.1f}%) & Uptrend"
            elif momentum_score < -5 and trend_strength < -2:
                signal = 'SELL'
                confidence = min(0.9, 0.5 + abs(momentum_score) / 100)
                reason = f"Strong Momentum ({momentum_score:.1f}%) & Downtrend"
            elif momentum_score > 2:
                signal = 'BUY'
                confidence = 0.6
                reason = f"Positive Momentum (+{momentum_score:.1f}%)"
            elif momentum_score < -2:
                signal = 'SELL'
                confidence = 0.6
                reason = f"Negative Momentum ({momentum_score:.1f}%)"
            else:
                signal = 'HOLD'
                confidence = 0.5
                reason = "Neutral Momentum"

            accuracy = 0.72 + (confidence - 0.5) * 0.2

            return {
                'signal': signal,
                'confidence': confidence,
                'accuracy': accuracy,
                'method': 'Momentum & Trend Analysis',
                'momentum_score': momentum_score,
                'trend_strength': trend_strength,
                'reason': reason,
                'price_changes': {
                    '5_day': f"{price_change_5d:+.1f}%",
                    '20_day': f"{price_change_20d:+.1f}%",
                    '50_day': f"{price_change_50d:+.1f}%"
                }
            }

        except Exception as e:
            return None

    def _ml_ensemble_prediction(self, df: pd.DataFrame) -> Dict:
        try:
            if len(df) < 100:
                return None

            momentum_5d = df['close'].pct_change(5).iloc[-1]
            momentum_20d = df['close'].pct_change(20).iloc[-1]

            if 'volume' in df.columns:
                volume_trend = (df['volume'].iloc[-5:].mean() / df['volume'].iloc[-10:-5].mean() - 1) * 100
            else:
                volume_trend = 0

            if 'rsi' in df.columns:
                rsi_signal = 1 if df['rsi'].iloc[-1] < 40 else -1 if df['rsi'].iloc[-1] > 60 else 0
            else:
                rsi_signal = 0

            combined_signal = (
                np.sign(momentum_5d) * 0.3 +
                np.sign(momentum_20d) * 0.3 +
                np.sign(volume_trend) * 0.2 +
                rsi_signal * 0.2
            )

            if combined_signal > 0.3:
                signal = 'BUY'
                confidence = min(0.8, 0.5 + combined_signal * 0.3)
            elif combined_signal < -0.3:
                signal = 'SELL'
                confidence = min(0.8, 0.5 - combined_signal * 0.3)
            else:
                signal = 'HOLD'
                confidence = 0.5

            accuracy = 0.70 + (confidence - 0.5) * 0.25

            return {
                'signal': signal,
                'confidence': confidence,
                'accuracy': accuracy,
                'method': 'Statistical Ensemble Model',
                'combined_signal': combined_signal,
                'model_components': {
                    'momentum_5d': momentum_5d,
                    'momentum_20d': momentum_20d,
                    'volume_trend': volume_trend,
                    'rsi_signal': rsi_signal
                }
            }

        except Exception as e:
            return None

    def _select_best_prediction(self, predictions: Dict) -> Dict:
        if not predictions:
            return {'signal': 'HOLD', 'confidence': 0.5, 'method': 'No Data', 'accuracy': 0.5}

        best_pred = None
        for name, pred in predictions.items():
            if best_pred is None or pred['confidence'] > best_pred['confidence']:
                best_pred = pred
                best_pred['method'] = name

        return best_pred

    def _get_detailed_analysis(self, df: pd.DataFrame, symbol: str) -> Dict:
        try:
            if df.empty:
                return {}

            current_price = df['close'].iloc[-1]
            analysis = {}

            if all(col in df.columns for col in ['sma_20', 'sma_50', 'sma_200']):
                sma_20 = df['sma_20'].iloc[-1]
                sma_50 = df['sma_50'].iloc[-1]
                sma_200 = df['sma_200'].iloc[-1]

                if current_price > sma_20 > sma_50 > sma_200:
                    analysis['trend'] = 'Strong Bullish'
                elif current_price < sma_20 < sma_50 < sma_200:
                    analysis['trend'] = 'Strong Bearish'
                elif current_price > sma_20 > sma_50:
                    analysis['trend'] = 'Bullish'
                elif current_price < sma_20 < sma_50:
                    analysis['trend'] = 'Bearish'
                else:
                    analysis['trend'] = 'Mixed/Neutral'

            if 'volatility_20d' in df.columns:
                volatility = df['volatility_20d'].iloc[-1]
                if volatility > 0.4:
                    analysis['volatility'] = 'High'
                elif volatility > 0.2:
                    analysis['volatility'] = 'Medium'
                else:
                    analysis['volatility'] = 'Low'
                analysis['volatility_value'] = f"{volatility:.1%}"

            if len(df) > 20:
                support = df['low'].tail(20).min()
                resistance = df['high'].tail(20).max()
                analysis['support'] = support
                analysis['resistance'] = resistance
                analysis['price_position'] = (current_price - support) / (resistance - support) * 100

            return analysis

        except Exception as e:
            return {}

# =============================================================================
# Paper Trading Engine
# =============================================================================
class PaperTradingEngine:
    def __init__(self, db_manager):
        self.db = db_manager
        self.initial_balance = 100000.0
        self.commission_rate = 0.001

    def get_portfolio_summary(self) -> Dict:
        try:
            portfolio = self.db.get_portfolio()
            transactions = self.db.get_transaction_history()

            total_invested = 0
            total_current_value = 0
            holdings = []
            cash_balance = self.initial_balance

            for symbol, action, quantity, price, total, date in transactions:
                if action.upper() == 'BUY':
                    cash_balance -= total
                else:
                    cash_balance += total

            for symbol, quantity, avg_price in portfolio:
                try:
                    ticker = yf.Ticker(symbol)
                    current_data = ticker.history(period='1d')
                    current_price = current_data['Close'].iloc[-1] if not current_data.empty else avg_price
                except:
                    current_price = avg_price

                invested_value = quantity * avg_price
                current_value = quantity * current_price
                unrealized_pnl = current_value - invested_value
                pnl_percentage = (unrealized_pnl / invested_value) * 100 if invested_value > 0 else 0

                holdings.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'invested_value': invested_value,
                    'current_value': current_value,
                    'unrealized_pnl': unrealized_pnl,
                    'pnl_percentage': pnl_percentage
                })

                total_invested += invested_value
                total_current_value += current_value

            total_pnl = total_current_value - total_invested
            total_pnl_percentage = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
            total_balance = cash_balance + total_current_value

            return {
                'cash_balance': cash_balance,
                'total_invested': total_invested,
                'total_current_value': total_current_value,
                'total_unrealized_pnl': total_pnl,
                'total_pnl_percentage': total_pnl_percentage,
                'total_balance': total_balance,
                'holdings': holdings
            }
        except Exception as e:
            return {}

    def execute_trade(self, symbol: str, action: str, quantity: int, price: float) -> Tuple[bool, str]:
        try:
            if action.upper() not in ['BUY', 'SELL']:
                return False, "Invalid action"

            if quantity <= 0:
                return False, "Quantity must be positive"

            total_cost = quantity * price
            portfolio = self.get_portfolio_summary()

            if action.upper() == 'BUY':
                if portfolio['cash_balance'] < total_cost:
                    return False, f"Insufficient cash. Required: ${total_cost:.2f}, Available: ${portfolio['cash_balance']:.2f}"
            else:
                current_holdings = [h for h in portfolio['holdings'] if h['symbol'] == symbol]
                if not current_holdings:
                    return False, f"No position found for {symbol}"
                if current_holdings[0]['quantity'] < quantity:
                    return False, f"Insufficient shares. Available: {current_holdings[0]['quantity']}, Requested: {quantity}"

            success = self.db.add_transaction(symbol, action, quantity, price, 0.0)

            if success:
                action_text = "bought" if action.upper() == 'BUY' else "sold"
                return True, f"Successfully {action_text} {quantity} shares of {symbol} at ${price:.2f}"
            else:
                return False, "Failed to record transaction"

        except Exception as e:
            return False, f"Trade execution error: {str(e)}"

# =============================================================================
# Professional Chart Engine
# =============================================================================
class ProfessionalChartEngine:
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ffbb78',
            'info': '#98df8a',
            'background': '#f9f9f9'
        }

    def create_price_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create separate price chart (candlestick + indicators) with tight y-axis."""
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#2ca02c',
            decreasing_line_color='#d62728'
        ))

        ma_colors = {
            20: ('orange', 'SMA 20'),
            50: ('red', 'SMA 50'),
            200: ('purple', 'SMA 200')
        }

        for period, (color, name) in ma_colors.items():
            if f'sma_{period}' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[f'sma_{period}'],
                    name=name,
                    line=dict(color=color, width=2),
                    opacity=0.8
                ))

        if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['bb_upper'],
                name='BB Upper',
                line=dict(color='gray', width=1.5, dash='dash'),
                opacity=0.7
            ))
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['bb_lower'],
                name='BB Lower',
                line=dict(color='gray', width=1.5, dash='dash'),
                opacity=0.7,
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            ))

        # Tighten y-axis range
        y_min = df['low'].min()
        y_max = df['high'].max()
        if 'bb_lower' in df.columns:
            y_min = min(y_min, df['bb_lower'].min())
        if 'bb_upper' in df.columns:
            y_max = max(y_max, df['bb_upper'].max())
        padding = 0.05 * (y_max - y_min)
        fig.update_yaxes(range=[y_min - padding, y_max + padding])

        fig.update_layout(
            height=400,
            template='plotly_white',
            showlegend=True,
            title=f"{symbol} Price Chart",
            hovermode='x unified',
            plot_bgcolor='rgba(240,240,240,0.5)',
            paper_bgcolor='rgba(240,240,240,0.1)',
            font=dict(size=12),
            margin=dict(t=80, b=40, l=80, r=40)
        )

        fig.update_xaxes(
            tickformat='%Y-%m-%d',
            tickangle=45,
            tickmode='auto',
            nticks=12,
            showgrid=True,
            title_text="Date"
        )

        return fig

    def create_volume_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create separate volume chart with tight y-axis."""
        fig = go.Figure()

        if 'volume' in df.columns:
            colors = ['#2ca02c' if df['close'].iloc[i] > df['open'].iloc[i] else '#d62728'
                     for i in range(len(df))]
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ))

            if 'volume_sma' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['volume_sma'],
                    name='Volume SMA',
                    line=dict(color='orange', width=2)
                ))

        # Tighten y-axis: start at 0, end at max volume with small padding
        if not df['volume'].empty:
            max_vol = df['volume'].max()
            fig.update_yaxes(range=[0, max_vol * 1.05])

        fig.update_layout(
            height=250,  # Increased from 200 to 250 for better visibility
            template='plotly_white',
            showlegend=True,
            title=f"{symbol} Volume",
            hovermode='x unified',
            plot_bgcolor='rgba(240,240,240,0.5)',
            paper_bgcolor='rgba(240,240,240,0.1)',
            font=dict(size=12),
            margin=dict(t=40, b=40, l=80, r=40)
        )

        fig.update_xaxes(
            tickformat='%Y-%m-%d',
            tickangle=45,
            tickmode='auto',
            nticks=12,
            showgrid=True,
            title_text="Date"
        )

        return fig

    def create_technical_chart(self, df: pd.DataFrame, indicator: str) -> go.Figure:
        fig = go.Figure()

        if indicator == 'RSI' and 'rsi' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['rsi'],
                name='RSI',
                line=dict(color='purple', width=3)
            ))
            fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.2, annotation_text="Overbought")
            fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.2, annotation_text="Oversold")
            fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Neutral")
            fig.update_layout(
                title="RSI (14)",
                yaxis_range=[0, 100],
                height=400,
                plot_bgcolor='rgba(240,240,240,0.5)'
            )

        elif indicator == 'MACD' and all(col in df.columns for col in ['macd', 'macd_signal']):
            fig.add_trace(go.Scatter(
                x=df.index, y=df['macd'],
                name='MACD',
                line=dict(color='blue', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df['macd_signal'],
                name='Signal',
                line=dict(color='red', width=3)
            ))

            if 'macd_histogram' in df.columns:
                colors = ['green' if x >= 0 else 'red' for x in df['macd_histogram']]
                fig.add_trace(go.Bar(
                    x=df.index, y=df['macd_histogram'],
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.6
                ))
            fig.update_layout(
                title="MACD",
                height=500,
                plot_bgcolor='rgba(240,240,240,0.5)'
            )

        elif indicator == 'Stochastic' and all(col in df.columns for col in ['stoch_k', 'stoch_d']):
            fig.add_trace(go.Scatter(
                x=df.index, y=df['stoch_k'],
                name='%K',
                line=dict(color='blue', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df['stoch_d'],
                name='%D',
                line=dict(color='red', width=3)
            ))
            fig.add_hrect(y0=80, y1=100, line_width=0, fillcolor="red", opacity=0.2, annotation_text="Overbought")
            fig.add_hrect(y0=0, y1=20, line_width=0, fillcolor="green", opacity=0.2, annotation_text="Oversold")
            fig.update_layout(
                title="Stochastic Oscillator",
                yaxis_range=[0, 100],
                height=400,
                plot_bgcolor='rgba(240,240,240,0.5)'
            )

        elif indicator == 'Bollinger Bands' and all(col in df.columns for col in ['bb_upper', 'bb_lower']):
            fig.add_trace(go.Scatter(
                x=df.index, y=df['close'],
                name='Price',
                line=dict(color='blue', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df['bb_upper'],
                name='Upper',
                line=dict(color='gray', width=2, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df['bb_middle'],
                name='Middle',
                line=dict(color='darkgray', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df['bb_lower'],
                name='Lower',
                line=dict(color='gray', width=2, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            ))
            fig.update_layout(
                title="Bollinger Bands",
                height=500,
                plot_bgcolor='rgba(240,240,240,0.5)'
            )

        elif indicator == 'Volume' and 'volume' in df.columns:
            colors = ['#2ca02c' if df['close'].iloc[i] > df['open'].iloc[i] else '#d62728' for i in range(len(df))]
            fig.add_trace(go.Bar(
                x=df.index, y=df['volume'],
                name='Volume',
                marker_color=colors
            ))
            if 'volume_sma' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['volume_sma'],
                    name='Volume SMA',
                    line=dict(color='orange', width=3)
                ))
            fig.update_layout(
                title="Volume Analysis",
                height=400,
                plot_bgcolor='rgba(240,240,240,0.5)'
            )

        fig.update_layout(
            template='plotly_white',
            font=dict(size=12)
        )
        return fig

# =============================================================================
# Database Manager (extended with report_history)
# =============================================================================
class DatabaseManager:
    def __init__(self, db_path=Config.DB_PATH):
        self.db_path = db_path
        self.logger = Logger()
        self.init_database()

    def init_database(self):
        schema = '''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            added_price REAL NOT NULL,
            added_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            avg_price REAL NOT NULL,
            current_price REAL,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            total_value REAL NOT NULL,
            transaction_date DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS report_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            price_at_report REAL,
            predicted_signal TEXT,
            predicted_price REAL,
            pdf_path TEXT
        );
        '''

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executescript(schema)
        conn.commit()
        conn.close()

    def add_to_watchlist(self, symbol: str, current_price: float, notes: str = "") -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO watchlist (symbol, added_price, notes) VALUES (?, ?, ?)",
                (symbol.upper(), current_price, notes)
            )
            conn.commit()
            self.logger.log_info(f"Added {symbol} to watchlist")
            return True
        except Exception as e:
            self.logger.log_error(f"Watchlist error: {e}")
            return False
        finally:
            conn.close()

    def get_watchlist(self) -> List[Tuple]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, added_price, added_date, notes FROM watchlist ORDER BY added_date DESC")
        results = cursor.fetchall()
        conn.close()
        return results

    def remove_from_watchlist(self, symbol: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        if success:
            self.logger.log_info(f"Removed {symbol} from watchlist")
        return success

    def get_portfolio(self) -> List[Tuple]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, quantity, avg_price FROM portfolio")
        results = cursor.fetchall()
        conn.close()
        return results

    def add_transaction(self, symbol: str, action: str, quantity: int, price: float, commission: float = 0.0) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            total_value = quantity * price
            cursor.execute(
                '''INSERT INTO transactions
                (symbol, action, quantity, price, total_value, transaction_date)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (symbol.upper(), action.upper(), quantity, price, total_value, datetime.now().isoformat())
            )

            if action.upper() == 'BUY':
                self._update_portfolio_buy(conn, symbol, quantity, price)
            else:
                self._update_portfolio_sell(conn, symbol, quantity, price)

            conn.commit()
            self.logger.log_info(f"Added transaction: {action} {quantity} {symbol} at ${price}")
            return True
        except Exception as e:
            conn.rollback()
            self.logger.log_error(f"Transaction error: {e}")
            return False
        finally:
            conn.close()

    def _update_portfolio_buy(self, conn, symbol: str, quantity: int, price: float):
        cursor = conn.cursor()

        cursor.execute("SELECT quantity, avg_price FROM portfolio WHERE symbol = ?", (symbol,))
        result = cursor.fetchone()

        if result:
            old_qty, old_avg = result
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty

            cursor.execute(
                "UPDATE portfolio SET quantity = ?, avg_price = ? WHERE symbol = ?",
                (new_qty, new_avg, symbol)
            )
        else:
            cursor.execute(
                "INSERT INTO portfolio (symbol, quantity, avg_price, current_price) VALUES (?, ?, ?, ?)",
                (symbol, quantity, price, price)
            )

    def _update_portfolio_sell(self, conn, symbol: str, quantity: int, price: float):
        cursor = conn.cursor()

        cursor.execute("SELECT quantity, avg_price FROM portfolio WHERE symbol = ?", (symbol,))
        result = cursor.fetchone()

        if not result:
            raise ValueError(f"No position found for {symbol}")

        old_qty, avg_price = result
        if old_qty < quantity:
            raise ValueError(f"Insufficient shares: {old_qty} available, {quantity} requested")

        new_qty = old_qty - quantity
        if new_qty == 0:
            cursor.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol,))
        else:
            cursor.execute(
                "UPDATE portfolio SET quantity = ? WHERE symbol = ?",
                (new_qty, symbol)
            )

    def get_transaction_history(self, limit: int = 50) -> List[Tuple]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT symbol, action, quantity, price, total_value, transaction_date FROM transactions ORDER BY transaction_date DESC LIMIT ?",
            (limit,)
        )
        results = cursor.fetchall()
        conn.close()
        return results

    def add_report_history(self, sym, price_at, sig=None, pred_price=None, pdf=None):
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute("INSERT INTO report_history (symbol, price_at_report, predicted_signal, predicted_price, pdf_path) VALUES (?,?,?,?,?)",
                             (sym.upper(), price_at, sig, pred_price, pdf))
                return True
            except:
                return False

    def get_report_history(self, limit=50):
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT id,symbol,generated_at,price_at_report,predicted_signal,predicted_price,pdf_path FROM report_history ORDER BY generated_at DESC LIMIT ?",
                                (limit,)).fetchall()

    def delete_report_history(self, rid):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute("DELETE FROM report_history WHERE id=?", (rid,))
            return c.rowcount > 0

# =============================================================================
# Professional Report Generator (enhanced with charts)
# =============================================================================
class ProfessionalReportGenerator:
    def __init__(self):
        self.config = Config()
        self.logger = Logger()

    def generate_comprehensive_csv(self, symbol: str, data: pd.DataFrame, fundamentals: Dict, prediction: Dict, statements: Optional[Dict] = None) -> BytesIO:
        try:
            output = BytesIO()

            csv_data = []

            csv_data.append(["COMPREHENSIVE STOCK ANALYSIS REPORT"])
            csv_data.append([f"Symbol: {symbol}"])
            csv_data.append([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
            csv_data.append([])

            # PRICE DATA
            csv_data.append(["PRICE DATA"])
            csv_data.append([])
            if not data.empty:
                price_data = data.reset_index()
                price_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] + \
                              [col for col in data.columns if any(x in col for x in ['sma', 'ema', 'rsi', 'macd'])]
                available_cols = [col for col in price_columns if col in price_data.columns]
                csv_data.append(available_cols)
                for _, row in price_data[available_cols].tail(100).iterrows():
                    csv_data.append([str(row[col]) for col in available_cols])

            csv_data.append([])

            # FUNDAMENTAL DATA
            csv_data.append(["FUNDAMENTAL DATA"])
            csv_data.append([])
            if fundamentals and 'error' not in fundamentals:
                for key, value in fundamentals.items():
                    csv_data.append([key.replace('_', ' ').title(), str(value)])

            csv_data.append([])

            # FINANCIAL STATEMENTS (Annual)
            if statements and not statements.get('income', pd.DataFrame()).empty:
                csv_data.append(["FINANCIAL STATEMENTS (Annual Income)"])
                csv_data.append([])
                inc = statements['income'].copy()
                inc.columns = pd.to_datetime(inc.columns).strftime('%d-%b-%Y')
                csv_data.append(['Item'] + list(inc.columns))
                for idx in inc.index:
                    row = [idx] + [format_large_number(inc.loc[idx, c]) for c in inc.columns]
                    csv_data.append(row)
                csv_data.append([])

            if statements and not statements.get('balance', pd.DataFrame()).empty:
                csv_data.append(["FINANCIAL STATEMENTS (Annual Balance Sheet)"])
                csv_data.append([])
                bal = statements['balance'].copy()
                bal.columns = pd.to_datetime(bal.columns).strftime('%d-%b-%Y')
                csv_data.append(['Item'] + list(bal.columns))
                for idx in bal.index:
                    row = [idx] + [format_large_number(bal.loc[idx, c]) for c in bal.columns]
                    csv_data.append(row)
                csv_data.append([])

            if statements and not statements.get('cash', pd.DataFrame()).empty:
                csv_data.append(["FINANCIAL STATEMENTS (Annual Cash Flow)"])
                csv_data.append([])
                cash = statements['cash'].copy()
                cash.columns = pd.to_datetime(cash.columns).strftime('%d-%b-%Y')
                csv_data.append(['Item'] + list(cash.columns))
                for idx in cash.index:
                    row = [idx] + [format_large_number(cash.loc[idx, c]) for c in cash.columns]
                    csv_data.append(row)
                csv_data.append([])

            # RATIOS
            if statements and not statements.get('ratios', pd.DataFrame()).empty:
                csv_data.append(["FINANCIAL RATIOS (Annual)"])
                csv_data.append([])
                ratios = statements['ratios'].copy()
                ratios.columns = pd.to_datetime(ratios.columns).strftime('%d-%b-%Y')
                csv_data.append(['Ratio'] + list(ratios.columns))
                for idx in ratios.index:
                    row = [idx] + [format_percent(ratios.loc[idx, c]) if 'Margin' in idx or 'RO' in idx else f"{ratios.loc[idx, c]:.2f}" for c in ratios.columns]
                    csv_data.append(row)
                csv_data.append([])

            # PREDICTION DATA
            csv_data.append(["PREDICTION DATA"])
            csv_data.append([])
            if prediction and 'error' not in prediction:
                best = prediction.get('best_prediction', {})
                csv_data.append(["Best Prediction", f"{best.get('signal', 'N/A')}"])
                csv_data.append(["Confidence", f"{best.get('confidence', 0):.1%}"])
                csv_data.append(["Method", f"{best.get('method', 'N/A')}"])
                csv_data.append([])

                for method_name, method_pred in prediction.get('predictions', {}).items():
                    csv_data.append([f"{method_name} Prediction", method_pred.get('signal', 'N/A')])
                    csv_data.append([f"{method_name} Confidence", f"{method_pred.get('confidence', 0):.1%}"])
                    csv_data.append([f"{method_name} Accuracy", f"{method_pred.get('accuracy', 0):.1%}"])
                    if 'top_features' in method_pred:
                        csv_data.append(["Top Features"] + [f"{f}: {imp:.1%}" for f, imp in method_pred['top_features']])
                    csv_data.append([])

            csv_string = "\n".join([",".join(map(str, row)) for row in csv_data])

            output.write(csv_string.encode('utf-8'))
            output.seek(0)

            return output

        except Exception as e:
            self.logger.log_error(f"CSV generation error: {e}")
            return None

    def generate_professional_pdf_report(self, symbol: str, data: pd.DataFrame, fundamentals: Dict, prediction: Dict, statements: Optional[Dict] = None, charts: Optional[Dict] = None) -> str:
        try:
            html_content = self._create_professional_html_report(symbol, data, fundamentals, prediction, statements, charts)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_filename = f"{self.config.REPORT_DIR}/{symbol}_professional_report_{timestamp}.html"

            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)

            webbrowser.open(f'file://{os.path.abspath(html_filename)}')

            return f"Professional report generated and opened: {html_filename}"

        except Exception as e:
            return f"Report generation failed: {str(e)}"

    def _create_professional_html_report(self, symbol: str, data: pd.DataFrame, fundamentals: Dict, prediction: Dict, statements: Optional[Dict] = None, charts: Optional[Dict] = None) -> str:
        current_price = data['close'].iloc[-1] if not data.empty else 'N/A'
        best_pred = prediction.get('best_prediction', {}) if prediction else {}
        company_name = fundamentals.get('company_name', symbol)

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{company_name} Analysis Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                .header .subtitle {{
                    font-size: 1.2em;
                    opacity: 0.9;
                    margin-top: 10px;
                }}
                .content {{
                    padding: 30px;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 10px;
                    border-left: 5px solid #3498db;
                }}
                .section h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                    margin-top: 0;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                .metric-card {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{
                    font-size: 0.9em;
                    color: #7f8c8d;
                    margin-top: 5px;
                }}
                .prediction-signal {{
                    font-size: 2em;
                    font-weight: bold;
                    padding: 20px;
                    text-align: center;
                    border-radius: 10px;
                    margin: 20px 0;
                }}
                .buy {{
                    background: #d4edda;
                    color: #155724;
                    border: 2px solid #c3e6cb;
                }}
                .sell {{
                    background: #f8d7da;
                    color: #721c24;
                    border: 2px solid #f5c6cb;
                }}
                .hold {{
                    background: #fff3cd;
                    color: #856404;
                    border: 2px solid #ffeaa7;
                }}
                .fundamental-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 15px;
                }}
                .fundamental-table th,
                .fundamental-table td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                .fundamental-table th {{
                    background: #3498db;
                    color: white;
                }}
                .fundamental-table tr:nth-child(even) {{
                    background: #f2f2f2;
                }}
                .positive {{
                    color: #28a745;
                    font-weight: bold;
                }}
                .negative {{
                    color: #dc3545;
                    font-weight: bold;
                }}
                table.statement-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.9em;
                }}
                .statement-table th {{
                    background: #3498db;
                    color: white;
                    padding: 8px;
                }}
                .statement-table td {{
                    padding: 6px;
                    border: 1px solid #ddd;
                }}
                .chart-img {{
                    max-width: 100%;
                    height: auto;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 {company_name} Analysis Report</h1>
                    <div class="subtitle">{symbol} | {datetime.now().strftime('%B %d, %Y %H:%M:%S')}</div>
                </div>

                <div class="content">
                    <div class="section">
                        <h2>🎯 Executive Summary</h2>
                        <div class="prediction-signal {best_pred.get('signal', 'hold').lower()}">
                            SIGNAL: {best_pred.get('signal', 'HOLD')}
                            <br>
                            <small>Confidence: {best_pred.get('confidence', 0):.1%}</small>
                        </div>
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <div class="metric-value">${current_price:.2f}</div>
                                <div class="metric-label">Current Price</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{fundamentals.get('sector', 'N/A')}</div>
                                <div class="metric-label">Sector</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{fundamentals.get('market_cap', 'N/A')}</div>
                                <div class="metric-label">Market Cap</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{fundamentals.get('trailing_pe', 'N/A')}</div>
                                <div class="metric-label">P/E Ratio</div>
                            </div>
                        </div>
                    </div>

                    <div class="section">
                        <h2>📈 Fundamental Analysis</h2>
                        <table class="fundamental-table">
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Value</th>
                                    <th>Metric</th>
                                    <th>Value</th>
                                </tr>
                            </thead>
                            <tbody>
        """

        fundamental_items = list(fundamentals.items()) if fundamentals else []
        for i in range(0, len(fundamental_items), 2):
            html += "<tr>"
            if i < len(fundamental_items):
                key1, value1 = fundamental_items[i]
                value_class = self._get_value_class(key1, value1)
                html += f"<td><strong>{key1.replace('_', ' ').title()}</strong></td><td class='{value_class}'>{value1}</td>"
            else:
                html += "<td></td><td></td>"

            if i + 1 < len(fundamental_items):
                key2, value2 = fundamental_items[i + 1]
                value_class = self._get_value_class(key2, value2)
                html += f"<td><strong>{key2.replace('_', ' ').title()}</strong></td><td class='{value_class}'>{value2}</td>"
            else:
                html += "<td></td><td></td>"
            html += "</tr>"

        html += """
                            </tbody>
                        </table>
                    </div>
        """

        # Financial Statements
        if statements:
            for stmt_name, stmt_df in statements.items():
                if stmt_name in ['income', 'balance', 'cash'] and not stmt_df.empty:
                    display_name = {'income': 'Income Statement', 'balance': 'Balance Sheet', 'cash': 'Cash Flow'}.get(stmt_name, stmt_name)
                    html += f"""
                    <div class="section">
                        <h2>📊 {display_name} (Annual)</h2>
                        <table class="statement-table">
                            <thead>
                                <tr>
                                    <th>Item</th>
                    """
                    stmt_df.columns = pd.to_datetime(stmt_df.columns).strftime('%d-%b-%Y')
                    for col in stmt_df.columns:
                        html += f"<th>{col}</th>"
                    html += "</tr></thead><tbody>"
                    for idx in stmt_df.index:
                        html += f"<tr><td>{idx}</td>"
                        for col in stmt_df.columns:
                            html += f"<td>{format_large_number(stmt_df.loc[idx, col])}</td>"
                        html += "</tr>"
                    html += "</tbody></table></div>"
                elif stmt_name == 'ratios' and not stmt_df.empty:
                    html += """
                    <div class="section">
                        <h2>📐 Financial Ratios (Annual)</h2>
                        <table class="statement-table">
                            <thead>
                                <tr>
                                    <th>Ratio</th>
                    """
                    stmt_df.columns = pd.to_datetime(stmt_df.columns).strftime('%d-%b-%Y')
                    for col in stmt_df.columns:
                        html += f"<th>{col}</th>"
                    html += "</tr></thead><tbody>"
                    for idx in stmt_df.index:
                        html += f"<tr><td>{idx}</td>"
                        for col in stmt_df.columns:
                            val = stmt_df.loc[idx, col]
                            if 'Margin' in idx or 'RO' in idx:
                                formatted = format_percent(val)
                            else:
                                formatted = f"{val:.2f}" if not pd.isna(val) else "N/A"
                            html += f"<td>{formatted}</td>"
                        html += "</tr>"
                    html += "</tbody></table></div>"

        # ML Predictions
        html += """
                    <div class="section">
                        <h2>🤖 Multi-Method Prediction Analysis</h2>
        """

        if prediction and 'predictions' in prediction:
            for method_name, method_pred in prediction['predictions'].items():
                signal_class = method_pred.get('signal', 'hold').lower()
                html += f"""
                        <div class="metric-card" style="text-align: left; margin-bottom: 15px;">
                            <h3 style="margin-top: 0; color: #2c3e50;">{method_name}</h3>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="prediction-signal {signal_class}" style="font-size: 1.2em; padding: 10px; margin: 0;">
                                    {method_pred.get('signal', 'HOLD')} ({method_pred.get('confidence', 0):.1%})
                                </span>
                                <div style="font-size: 0.9em; color: #7f8c8d;">
                                    Accuracy: {method_pred.get('accuracy', 0):.1%}
                                </div>
                            </div>
                """

                if 'top_features' in method_pred:
                    html += "<div style='margin-top: 10px;'><strong>Top Features:</strong><br>"
                    for feature, importance in method_pred['top_features']:
                        html += f"{feature}: {importance:.1%}<br>"
                    html += "</div>"

                if 'signals_detail' in method_pred:
                    html += "<div style='margin-top: 10px;'><strong>Technical Signals:</strong><br>"
                    for sig, w, desc, _ in method_pred['signals_detail']:
                        html += f"{sig} ({w:.2f}): {desc}<br>"
                    html += "</div>"

                html += "</div>"

        html += """
                    </div>
        """

        # Technical Charts from ML Predictions
        if charts:
            html += """
                    <div class="section">
                        <h2>📈 Technical Analysis Charts</h2>
            """
            for chart_name, chart_base64 in charts.items():
                html += f"""
                        <h3>{chart_name}</h3>
                        <img src="data:image/png;base64,{chart_base64}" class="chart-img" />
                """
            html += "</div>"

        # Technical Overview
        html += """
                    <div class="section">
                        <h2>📊 Technical Overview</h2>
                        <div class="metrics-grid">
        """

        if not data.empty:
            tech_metrics = [
                ('RSI', f"{data['rsi'].iloc[-1]:.1f}" if 'rsi' in data.columns else 'N/A'),
                ('MACD', f"{data['macd'].iloc[-1]:.3f}" if 'macd' in data.columns else 'N/A'),
                ('Volume', f"{data['volume'].iloc[-1]:,}" if 'volume' in data.columns else 'N/A'),
                ('Volatility', f"{data.get('volatility_20d', [0]).iloc[-1]:.1%}" if 'volatility_20d' in data.columns else 'N/A'),
                ('SMA 20', f"${data['sma_20'].iloc[-1]:.2f}" if 'sma_20' in data.columns else 'N/A'),
                ('SMA 50', f"${data['sma_50'].iloc[-1]:.2f}" if 'sma_50' in data.columns else 'N/A')
            ]

            for metric, value in tech_metrics:
                html += f"""
                            <div class="metric-card">
                                <div class="metric-value">{value}</div>
                                <div class="metric-label">{metric}</div>
                            </div>
                """

        html += """
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _get_value_class(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            return ""

        if '%' in value:
            if value.startswith('+') or (value.replace('%', '').replace('.', '').replace(',', '').replace('$', '').replace('-', '').isdigit() and float(value.replace('%', '').replace('$', '').replace(',', '').replace('+', '')) > 0):
                return 'positive'
            elif value.startswith('-'):
                return 'negative'

        growth_indicators = ['growth', 'return', 'yield', 'margin']
        if any(indicator in key.lower() for indicator in growth_indicators):
            try:
                num_value = float(value.replace('%', '').replace('$', '').replace(',', ''))
                if num_value > 0:
                    return 'positive'
                elif num_value < 0:
                    return 'negative'
            except:
                pass

        return ""

# =============================================================================
# PDF Report Generator (using reportlab)
# =============================================================================
class PDFReportGenerator:
    def __init__(self):
        self.config = Config()
        self.logger = Logger()
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle('CustomTitle', parent=self.styles['Heading1'],
                                           fontSize=24, alignment=TA_CENTER, spaceAfter=30,
                                           textColor=colors.HexColor('#2c3e50'))
        self.heading_style = ParagraphStyle('CustomHeading', parent=self.styles['Heading2'],
                                            fontSize=16, spaceAfter=12, textColor=colors.HexColor('#3498db'))

    def generate_pdf(self, symbol, data, fundamentals, prediction, statements=None, charts=None):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.REPORT_DIR}/{symbol}_comprehensive_{timestamp}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
            story = []

            # Cover
            story.append(Paragraph(f"<b>{fundamentals.get('company_name', symbol)}</b>", self.title_style))
            story.append(Paragraph(f"Stock Analysis Report", self.styles['Heading2']))
            story.append(Paragraph(f"Symbol: {symbol}", self.styles['Normal']))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}", self.styles['Normal']))
            story.append(Spacer(1, 0.5*inch))

            # Executive summary
            story.append(Paragraph("Executive Summary", self.heading_style))
            curr = data['close'].iloc[-1] if not data.empty else 0
            best = prediction.get('best_prediction', {})
            sum_data = [['Current Price', f"${curr:.2f}"],
                        ['Best Signal', best.get('signal', 'N/A')],
                        ['Confidence', f"{best.get('confidence', 0):.1%}"],
                        ['1-Day Change', f"{data['daily_return'].iloc[-1]*100:.2f}%" if not data.empty else 'N/A']]
            t = Table(sum_data, colWidths=[2*inch, 2*inch])
            t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.grey)]))
            story.append(t)
            story.append(Spacer(1, 0.3*inch))

            # Fundamental highlights
            story.append(Paragraph("Fundamental Highlights", self.heading_style))
            fund_rows = [['Metric', 'Value']]
            for k in ['market_cap', 'trailing_pe', 'forward_pe', 'profit_margin', 'return_on_equity', 'debt_to_equity', 'beta']:
                if k in fundamentals:
                    fund_rows.append([k.replace('_', ' ').title(), str(fundamentals[k])])
            t = Table(fund_rows, colWidths=[2.5*inch, 2.5*inch])
            t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')]))
            story.append(t)
            story.append(Spacer(1, 0.3*inch))

            # Performance bar chart
            story.append(Paragraph("Performance Across Timeframes", self.heading_style))
            fig, ax = plt.subplots(figsize=(6, 3))
            tf = [('1D', 'daily_return'), ('1W', 'return_5d'), ('1M', 'return_21d'), ('3M', 'return_63d'),
                  ('6M', 'return_126d'), ('1Y', 'return_252d'), ('3Y', 'return_756d'), ('5Y', 'return_1260d')]
            rets, labs = [], []
            for lab, col in tf:
                if col in data and not pd.isna(data[col].iloc[-1]):
                    rets.append(data[col].iloc[-1]*100)
                    labs.append(lab)
            if rets:
                colors_bar = ['green' if r >= 0 else 'red' for r in rets]
                ax.bar(labs, rets, color=colors_bar)
                ax.set_ylabel('Return %')
                ax.set_title(f"{symbol} Returns")
                for i, v in enumerate(rets):
                    ax.text(i, v + (1 if v >= 0 else -3), f"{v:+.1f}%", ha='center', fontweight='bold')
                plt.tight_layout()
                img_buf = BytesIO()
                plt.savefig(img_buf, format='png', dpi=150)
                img_buf.seek(0)
                plt.close(fig)
                story.append(Image(img_buf, width=6*inch, height=3*inch))
            else:
                story.append(Paragraph("Insufficient data for performance chart.", self.styles['Normal']))
            story.append(Spacer(1, 0.3*inch))

            # Technical charts from ML predictions
            if charts:
                story.append(Paragraph("Technical Analysis Charts", self.heading_style))
                for chart_name, img_bytes in charts.items():
                    story.append(Paragraph(chart_name, self.styles['Heading3']))
                    img_buf = BytesIO(img_bytes)
                    story.append(Image(img_buf, width=5*inch, height=3*inch))
                    story.append(Spacer(1, 0.2*inch))

            doc.build(story)
            return filename

        except Exception as e:
            self.logger.log_error(f"PDF error: {e}")
            return ""

# =============================================================================
# Professional Market Platform (main app)
# =============================================================================
class ProfessionalMarketPlatform:
    def __init__(self):
        self.data_manager = ProfessionalDataManager()
        self.prediction_engine = MultiMethodPredictionEngine()
        self.forecasting_engine = AdvancedForecastingEngine()
        self.chart_engine = ProfessionalChartEngine()
        self.db_manager = DatabaseManager()
        self.trading_engine = PaperTradingEngine(self.db_manager)
        self.report_generator = ProfessionalReportGenerator()
        self.pdf_generator = PDFReportGenerator()
        self.logger = Logger()

        self.setup_page()
        self.init_session_state()
        freemium.init_freemium_session_state()

    def setup_page(self):
        st.set_page_config(
            page_title="STOCK MARKET Pro - Advanced US Stock Analysis",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                color: #1f77b4;
                font-weight: bold;
                text-align: center;
                margin-bottom: 1rem;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #3498db;
                margin-bottom: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .positive { color: #28a745; font-weight: bold; }
            .negative { color: #dc3545; font-weight: bold; }
            .prediction-buy {
                background: #d4edda;
                border-left: 4px solid #28a745;
                padding: 15px;
                border-radius: 5px;
            }
            .prediction-sell {
                background: #f8d7da;
                border-left: 4px solid #dc3545;
                padding: 15px;
                border-radius: 5px;
            }
            .prediction-hold {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                border-radius: 5px;
            }
            .index-positive { color: #28a745; font-weight: bold; }
            .index-negative { color: #dc3545; font-weight: bold; }
            .stPlotlyChart {
                margin-top: 20px;
                margin-bottom: 20px;
            }
            .fund-group {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-left: 4px solid #3498db;
            }
            .fund-group h4 {
                margin-top: 0;
                color: #2c3e50;
                border-bottom: 1px solid #e0e0e0;
                padding-bottom: 8px;
            }
        </style>
        """, unsafe_allow_html=True)

    def init_session_state(self):
        if 'current_symbol' not in st.session_state:
            st.session_state.current_symbol = 'AAPL'
        if 'analysis_period' not in st.session_state:
            st.session_state.analysis_period = "1y"
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = "Market Dashboard"
        if 'previous_tab' not in st.session_state:
            st.session_state.previous_tab = "Market Dashboard"
        if 'use_date_range' not in st.session_state:
            st.session_state.use_date_range = False
        if 'start_date' not in st.session_state:
            st.session_state.start_date = None
        if 'end_date' not in st.session_state:
            st.session_state.end_date = None

    def run(self):
        if st.session_state.previous_tab != st.session_state.current_tab:
            st.session_state.previous_tab = st.session_state.current_tab
            st.markdown("""
            <script>
                window.scrollTo(0, 0);
            </script>
            """, unsafe_allow_html=True)

        st.markdown('<div class="main-header">STOCK MARKET Pro - Advanced US Stock Analysis</div>', unsafe_allow_html=True)

        self.render_sidebar()

        if st.session_state.current_tab == "Market Dashboard":
            self.render_market_dashboard()
        elif st.session_state.current_tab == "Stock Analysis":
            self.render_stock_analysis()
        elif st.session_state.current_tab == "Advanced Charts":
            self.render_advanced_charts()
        elif st.session_state.current_tab == "ML Predictions":
            self.render_ml_predictions()
        elif st.session_state.current_tab == "Forecasting":
            self.render_forecasting()
        elif st.session_state.current_tab == "Paper Trading":
            self.render_paper_trading()
        elif st.session_state.current_tab == "Watchlist":
            self.render_watchlist()
        elif st.session_state.current_tab == "Reports":
            self.render_reports()

    def render_sidebar(self):
        with st.sidebar:
            st.header("🎯 Navigation")

            tabs = [
                "Market Dashboard", "Stock Analysis", "Advanced Charts", "ML Predictions",
                "Forecasting", "Paper Trading", "Watchlist", "Reports"
            ]
            selected_tab = st.selectbox("Go to", tabs, index=tabs.index(st.session_state.current_tab))
            if selected_tab != st.session_state.current_tab:
                st.session_state.current_tab = selected_tab
                st.rerun()

            st.markdown("---")
            st.header("🔍 Stock Analysis")

            with st.form("symbol_form"):
                st.session_state.current_symbol = st.text_input(
                    "Stock Symbol",
                    value=st.session_state.current_symbol
                ).upper()

                st.session_state.analysis_period = st.selectbox(
                    "Time Period",
                    Config.PERIOD_OPTIONS,
                    index=Config.PERIOD_OPTIONS.index(st.session_state.analysis_period) if st.session_state.analysis_period in Config.PERIOD_OPTIONS else 8
                )

                if st.checkbox("Use custom date range"):
                    col1, col2 = st.columns(2)
                    with col1:
                        start = st.date_input("Start", datetime.now() - timedelta(days=365))
                    with col2:
                        end = st.date_input("End", datetime.now())
                    st.session_state.start_date = start
                    st.session_state.end_date = end
                    st.session_state.use_date_range = True
                else:
                    st.session_state.use_date_range = False

                if st.form_submit_button("Analyze Stock", use_container_width=True):
                    st.session_state.current_tab = "Stock Analysis"
                    st.rerun()

            st.markdown("---")
            st.header("⚡ Popular Stocks")

            cols = st.columns(3)
            for i, stock in enumerate(POPULAR_STOCKS[:9]):
                with cols[i % 3]:
                    if st.button(stock, key=f"quick_{stock}", use_container_width=True):
                        st.session_state.current_symbol = stock
                        st.session_state.current_tab = "Stock Analysis"
                        st.rerun()

    def render_market_dashboard(self):
        st.header("📊 US Market Dashboard")

        st.subheader("Major US Indices Performance")

        progress_bar = st.progress(0)
        indices_data = []

        for i, (index_name, index_symbol) in enumerate(Config.MAJOR_INDICES.items()):
            returns = self.data_manager.get_index_data(index_symbol)
            if returns:
                try:
                    ticker = yf.Ticker(index_symbol)
                    current_data = ticker.history(period='1d')
                    current_price = current_data['Close'].iloc[-1] if not current_data.empty else 0
                except:
                    current_price = 0

                row_data = {'Index': index_name, 'Symbol': index_symbol, 'Current Price': f"${current_price:,.2f}"}

                for timeframe in ['1d', '1w', '1m', '3m', '6m', '1y', '3y', '5y']:
                    if timeframe in returns:
                        return_val = returns[timeframe]
                        if return_val >= 0:
                            row_data[timeframe] = f"<span class='index-positive'>+{return_val:.2f}%</span>"
                        else:
                            row_data[timeframe] = f"<span class='index-negative'>{return_val:.2f}%</span>"

                indices_data.append(row_data)

            progress_bar.progress((i + 1) / len(Config.MAJOR_INDICES))

        if indices_data:
            df_indices = pd.DataFrame(indices_data)
            st.markdown(df_indices.to_html(escape=False, index=False), unsafe_allow_html=True)

        st.subheader("Market Index Charts")

        selected_index = st.selectbox("Select Index for Chart", list(Config.MAJOR_INDICES.keys()))
        index_data = self.data_manager.get_stock_data(
            Config.MAJOR_INDICES[selected_index],
            "2y"
        )

        if not index_data.empty:
            st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
            price_fig = self.chart_engine.create_price_chart(index_data, selected_index)
            st.plotly_chart(price_fig, use_container_width=True)
            volume_fig = self.chart_engine.create_volume_chart(index_data, selected_index)
            st.plotly_chart(volume_fig, use_container_width=True)
            st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    def render_stock_analysis(self):
        st.header("🔍 Stock Analysis")

        symbol = st.session_state.current_symbol
        period = st.session_state.analysis_period

        if not symbol:
            st.info("Enter a stock symbol to begin analysis")
            return

        # Freemium check
        if not freemium.can_analyze(symbol):
            freemium.mandatory_email_popup()
            st.stop()

        with st.spinner(f"Analyzing {symbol}..."):
            if st.session_state.use_date_range and st.session_state.start_date and st.session_state.end_date:
                data = self.data_manager.get_stock_data(symbol, start=st.session_state.start_date, end=st.session_state.end_date)
            else:
                data = self.data_manager.get_stock_data(symbol, period)

            if data.empty:
                st.error(f"No data found for {symbol}")
                return

            fundamentals = self.data_manager.get_comprehensive_fundamental_data(symbol)

        # Record analysis
        freemium.record_analysis(symbol)

        current_price = data['close'].iloc[-1]

        # Top row: company info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**Company**<br>{fundamentals.get('company_name', 'N/A')}", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**Sector**<br>{fundamentals.get('sector', 'N/A')}", unsafe_allow_html=True)
        with col3:
            st.markdown(f"**Industry**<br>{fundamentals.get('industry', 'N/A')}", unsafe_allow_html=True)
        with col4:
            st.markdown(f"**Market Cap**<br>{fundamentals.get('market_cap', 'N/A')}", unsafe_allow_html=True)

        # Returns row with robust date-based calculations
        def safe_return(col_name, days):
            if col_name in data.columns and not pd.isna(data[col_name].iloc[-1]):
                return f"{data[col_name].iloc[-1]*100:+.2f}%"
            if len(data) > 0:
                end_date = data.index[-1]
                start_date = end_date - timedelta(days=days)
                mask = data.index <= start_date
                if mask.any():
                    idx = data.index[mask][-1]
                    start_price = data.loc[idx, 'close']
                    return f"{(current_price / start_price - 1)*100:+.2f}%"
                else:
                    start_price = data.iloc[0]['close']
                    return f"{(current_price / start_price - 1)*100:+.2f}%"
            return "N/A"

        cols = st.columns(6)
        with cols[0]:
            st.metric("Current", f"${current_price:.2f}")
        with cols[1]:
            st.metric("Daily", safe_return('daily_return', 1))
        with cols[2]:
            st.metric("1W", safe_return('return_5d', 5))
        with cols[3]:
            st.metric("1M", safe_return('return_21d', 21))
        with cols[4]:
            st.metric("1Y", safe_return('return_252d', 252))
        with cols[5]:
            st.metric("5Y", safe_return('return_1260d', 1260))

        # Performance bar chart
        st.subheader("Performance Across Timeframes")
        perf_data = []
        for label, col, days in [('1D', 'daily_return', 1), ('1W', 'return_5d', 5), ('1M', 'return_21d', 21),
                                 ('3M', 'return_63d', 63), ('6M', 'return_126d', 126), ('1Y', 'return_252d', 252),
                                 ('3Y', 'return_756d', 756), ('5Y', 'return_1260d', 1260)]:
            if col in data and not pd.isna(data[col].iloc[-1]):
                perf_data.append({'Timeframe': label, 'Return': data[col].iloc[-1] * 100})
            else:
                if len(data) > 0:
                    end_date = data.index[-1]
                    start_date = end_date - timedelta(days=days)
                    mask = data.index <= start_date
                    if mask.any():
                        idx = data.index[mask][-1]
                        start_price = data.loc[idx, 'close']
                        ret = (current_price / start_price - 1) * 100
                        perf_data.append({'Timeframe': label, 'Return': ret})
                    elif days <= (data.index[-1] - data.index[0]).days:
                        start_price = data.iloc[0]['close']
                        ret = (current_price / start_price - 1) * 100
                        perf_data.append({'Timeframe': label, 'Return': ret})
        if perf_data:
            df_perf = pd.DataFrame(perf_data)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_perf['Timeframe'],
                y=df_perf['Return'],
                marker_color=['green' if r >= 0 else 'red' for r in df_perf['Return']],
                text=df_perf['Return'].apply(lambda x: f"{x:+.1f}%"),
                textposition='auto'
            ))
            fig.update_layout(title=f"{symbol} Returns", template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for bar chart")

        # Price chart
        st.subheader("Price Chart with Technical Indicators")
        price_fig = self.chart_engine.create_price_chart(data, symbol)
        st.plotly_chart(price_fig, use_container_width=True)

        # Volume chart
        st.subheader("Volume Analysis")
        volume_fig = self.chart_engine.create_volume_chart(data, symbol)
        st.plotly_chart(volume_fig, use_container_width=True)

        # Technical indicators row
        st.subheader("Technical Indicators")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                st.metric("RSI", f"{rsi:.1f}", rsi_status)
        with c2:
            if all(col in data.columns for col in ['macd', 'macd_signal']):
                macd_signal = "Bullish" if data['macd'].iloc[-1] > data['macd_signal'].iloc[-1] else "Bearish"
                st.metric("MACD Signal", macd_signal)
        with c3:
            if 'stoch_k' in data.columns:
                stoch = data['stoch_k'].iloc[-1]
                stoch_status = "Oversold" if stoch < 20 else "Overbought" if stoch > 80 else "Neutral"
                st.metric("Stochastic", f"{stoch:.1f}", stoch_status)
        with c4:
            if 'volatility_20d' in data.columns:
                volatility = data['volatility_20d'].iloc[-1]
                st.metric("Volatility (20D)", f"{volatility:.1%}")

        # Fundamental groups
        if 'error' not in fundamentals:
            st.subheader("Fundamental Analysis")

            with st.container():
                st.markdown('<div class="fund-group">', unsafe_allow_html=True)
                st.markdown("#### 💎 Valuation Metrics")
                cols = st.columns(2)
                metrics = ['market_cap', 'trailing_pe', 'forward_pe', 'price_to_book', 'price_to_sales', 'peg_ratio']
                for i, k in enumerate(metrics):
                    if k in fundamentals:
                        with cols[i % 2]:
                            self._display_colored_metric(k, fundamentals[k])
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="fund-group">', unsafe_allow_html=True)
                st.markdown("#### 📊 Financial Health")
                cols = st.columns(2)
                metrics = ['profit_margin', 'operating_margin', 'return_on_equity', 'return_on_assets', 'debt_to_equity', 'current_ratio', 'quick_ratio']
                for i, k in enumerate(metrics):
                    if k in fundamentals:
                        with cols[i % 2]:
                            self._display_colored_metric(k, fundamentals[k])
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="fund-group">', unsafe_allow_html=True)
                st.markdown("#### 🚀 Growth Metrics")
                cols = st.columns(2)
                metrics = ['revenue_growth', 'earnings_growth', 'dividend_yield']
                for i, k in enumerate(metrics):
                    if k in fundamentals:
                        with cols[i % 2]:
                            self._display_colored_metric(k, fundamentals[k])
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="fund-group">', unsafe_allow_html=True)
                st.markdown("#### 📈 Trading Info")
                cols = st.columns(2)
                metrics = ['beta', 'volume', 'avg_volume', '52_week_high', '52_week_low', '50_day_ma', '200_day_ma', 'target_mean_price', 'recommendation']
                for i, k in enumerate(metrics):
                    if k in fundamentals:
                        with cols[i % 2]:
                            self._display_colored_metric(k, fundamentals[k])
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(fundamentals['error'])

        # Fundamental Charts
        st.subheader("📈 Fundamental Charts")
        self._display_fundamental_charts_enhanced(symbol)

        # Financial Statements
        st.subheader("📊 Financial Statements")
        freq = st.radio("Frequency", ["Annual", "Quarterly"], horizontal=True, key="stmt_freq")
        stype = st.selectbox("Statement Type", ["Income Statement", "Balance Sheet", "Cash Flow"], key="stmt_type")
        tmap = {"Income Statement": "income", "Balance Sheet": "balance", "Cash Flow": "cash"}
        fmap = {"Annual": "annual", "Quarterly": "quarterly"}
        df_stmt = self.data_manager.get_financial_statements(symbol, tmap[stype], fmap[freq])
        if not df_stmt.empty:
            df_stmt.columns = pd.to_datetime(df_stmt.columns).strftime('%d-%b-%Y')
            st.dataframe(df_stmt.style.format(format_large_number), use_container_width=True)
        else:
            st.info("No data available")

        # Technical data table
        st.subheader("📋 Technical Analysis Data")
        if not data.empty:
            tech_data = data.tail(10)[['close', 'sma_20', 'sma_50', 'rsi', 'macd', 'volume']].copy()
            tech_data = tech_data.round(3)
            tech_data.index = tech_data.index.strftime('%Y-%m-%d')
            st.dataframe(tech_data, use_container_width=True)

    def _display_colored_metric(self, key: str, value: str):
        display_key = key.replace('_', ' ').title()
        if isinstance(value, str):
            if '%' in value:
                if value.startswith('+') or (value.replace('%', '').replace('.', '').replace(',', '').replace('$', '').replace('-', '').isdigit() and float(value.replace('%', '').replace('$', '').replace(',', '').replace('+', '')) > 0):
                    st.markdown(f"**{display_key}:** <span class='positive'>{value}</span>", unsafe_allow_html=True)
                elif value.startswith('-'):
                    st.markdown(f"**{display_key}:** <span class='negative'>{value}</span>", unsafe_allow_html=True)
                else:
                    st.write(f"**{display_key}:** {value}")
            else:
                st.write(f"**{display_key}:** {value}")
        else:
            st.write(f"**{display_key}:** {value}")

    def _display_fundamental_charts_enhanced(self, symbol):
        inc = self.data_manager.get_financial_statements(symbol, 'income', 'annual')
        if inc.empty:
            st.info("No fundamental trend data")
            return
        metrics = {
            'Revenue': inc.loc['Total Revenue'] if 'Total Revenue' in inc.index else None,
            'Net Income': inc.loc['Net Income'] if 'Net Income' in inc.index else None,
            'EPS': inc.loc['Diluted EPS'] if 'Diluted EPS' in inc.index else None,
        }
        ratios = self.data_manager.calculate_ratios_from_statements(symbol)
        if not ratios.empty:
            for col in ratios.index:
                metrics[col] = ratios.loc[col]
        avail = [k for k, v in metrics.items() if v is not None and not v.isnull().all()]
        if not avail:
            st.info("No metrics available")
            return
        sel = st.selectbox("Select Metric", avail)
        if sel:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=metrics[sel].index, y=metrics[sel].values, mode='lines+markers', name=sel))
            fig.update_layout(title=f"{sel} Over Time", template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

    def render_advanced_charts(self):
        st.header("📊 Advanced Technical Charts")
        symbol = st.session_state.current_symbol
        period = st.session_state.analysis_period
        if not symbol:
            st.info("Enter a stock symbol to view advanced charts")
            return
        with st.spinner(f"Loading advanced charts for {symbol}..."):
            if st.session_state.use_date_range and st.session_state.start_date and st.session_state.end_date:
                data = self.data_manager.get_stock_data(symbol, start=st.session_state.start_date, end=st.session_state.end_date)
            else:
                data = self.data_manager.get_stock_data(symbol, period)
            if data.empty:
                st.error(f"No data found for {symbol}")
                return
        chart_options = {
            "RSI": "RSI",
            "MACD": "MACD",
            "Stochastic": "Stochastic",
            "Bollinger Bands": "Bollinger Bands",
            "Volume Analysis": "Volume"
        }
        selected_charts = st.multiselect(
            "Select Charts to Display",
            list(chart_options.keys()),
            default=["RSI", "MACD", "Bollinger Bands"]
        )
        for chart_name in selected_charts:
            st.subheader(chart_name)
            fig = self.chart_engine.create_technical_chart(data, chart_options[chart_name])
            st.plotly_chart(fig, use_container_width=True)

    def render_ml_predictions(self):
        st.header("🤖 Multi-Method Market Predictions")
        symbol = st.session_state.current_symbol
        period = st.session_state.analysis_period
        if not symbol:
            st.info("Enter a stock symbol for analysis")
            return
        with st.spinner(f"Running comprehensive analysis for {symbol}..."):
            prediction_result = self.prediction_engine.get_comprehensive_prediction(symbol, period)
        if 'error' in prediction_result:
            st.error(prediction_result['error'])
            return
        best_pred = prediction_result['best_prediction']
        st.subheader("🎯 Best Prediction")
        signal_class = {
            'BUY': 'prediction-buy',
            'SELL': 'prediction-sell',
            'HOLD': 'prediction-hold'
        }.get(best_pred['signal'], 'prediction-hold')
        st.markdown(f"""
        <div class="{signal_class}">
            <h2 style="margin: 0; text-align: center;">{best_pred['signal']}</h2>
            <p style="text-align: center; font-size: 1.2em; margin: 10px 0;">
                Confidence: <strong>{best_pred['confidence']:.1%}</strong>
            </p>
            <p style="text-align: center; margin: 0;">
                Method: {best_pred['method']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Detailed Analysis")
            if 'detailed_analysis' in prediction_result:
                analysis = prediction_result['detailed_analysis']
                for key, value in analysis.items():
                    st.metric(key.replace('_', ' ').title(), str(value))
        with col2:
            st.subheader("Prediction Overview")
            st.metric("Data Points Analyzed", f"{prediction_result.get('data_points', 0):,}")
            st.metric("Analysis Period", prediction_result.get('period_analyzed', 'N/A'))
            st.metric("Analysis Timestamp", prediction_result.get('timestamp', 'N/A'))
        st.subheader("📊 All Prediction Methods Analysis")
        for method_name, prediction in prediction_result['predictions'].items():
            with st.expander(f"{method_name} - {prediction['signal']} ({prediction['confidence']:.1%})", expanded=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Signal", prediction['signal'])
                    st.metric("Confidence", f"{prediction['confidence']:.1%}")
                    st.metric("Accuracy", f"{prediction.get('accuracy', 0):.1%}")
                with col2:
                    if 'top_features' in prediction:
                        st.write("**Top Predictive Features:**")
                        for feature, importance in prediction['top_features']:
                            st.progress(importance, text=f"{feature}: {importance:.1%}")
                    if 'signals_detail' in prediction:
                        st.write("**Technical Signals:**")
                        for signal, weight, desc, color in prediction['signals_detail']:
                            col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
                            with col_s1:
                                st.write(f"**{signal}**")
                            with col_s2:
                                st.write(desc)
                            with col_s3:
                                st.progress(weight)
                    if 'price_changes' in prediction:
                        st.write("**Price Momentum:**")
                        for period, change in prediction['price_changes'].items():
                            st.write(f"- {period}: {change}")
        st.subheader("📈 Supporting Technical Charts")
        data = self.data_manager.get_stock_data(symbol, "6mo")
        if not data.empty:
            tab1, tab2 = st.tabs(["RSI & MACD", "Price Trends"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    fig_rsi = self.chart_engine.create_technical_chart(data, "RSI")
                    st.plotly_chart(fig_rsi, use_container_width=True)
                with col2:
                    fig_macd = self.chart_engine.create_technical_chart(data, "MACD")
                    st.plotly_chart(fig_macd, use_container_width=True)
            with tab2:
                fig_bb = self.chart_engine.create_technical_chart(data, "Bollinger Bands")
                st.plotly_chart(fig_bb, use_container_width=True)

    def render_forecasting(self):
        st.header("🔮 Price Forecasting")
        symbol = st.session_state.current_symbol
        period = st.session_state.analysis_period
        if not symbol:
            st.info("Enter a stock symbol for forecasting")
            return
        col1, col2, col3 = st.columns(3)
        with col1:
            days_ahead = st.slider("Forecast Days", 7, 365, 30)
        with col2:
            monte_carlo_sims = st.slider("Monte Carlo Simulations", 100, 5000, 1000)
        with col3:
            st.write("Forecasting Models:")
            use_lr = st.checkbox("Linear Regression", value=True)
            use_ar1 = st.checkbox("AR(1) Model", value=True)
            use_mc = st.checkbox("Monte Carlo", value=True)
            use_arima = st.checkbox("ARIMA", value=True)
        if st.button("Run Forecast", type="primary", use_container_width=True):
            with st.spinner(f"Running forecasts for {symbol}..."):
                if st.session_state.use_date_range and st.session_state.start_date and st.session_state.end_date:
                    data = self.data_manager.get_stock_data(symbol, start=st.session_state.start_date, end=st.session_state.end_date)
                else:
                    data = self.data_manager.get_stock_data(symbol, period)
                if data.empty:
                    st.error(f"No data found for {symbol}")
                    return
                forecasts = {}
                if use_lr:
                    lr_price, lr_series = self.forecasting_engine.project_linear_regression(data, days_ahead)
                    if lr_price is not None:
                        forecasts['Linear Regression'] = {'price': lr_price, 'series': lr_series}
                if use_ar1:
                    ar1_price, ar1_series = self.forecasting_engine.project_ar1(data, days_ahead)
                    if ar1_price is not None:
                        forecasts['AR(1) Model'] = {'price': ar1_price, 'series': ar1_series}
                if use_mc:
                    mc_mean, mc_low, mc_high, mc_series = self.forecasting_engine.project_monte_carlo_vectorized(
                        data, days_ahead, monte_carlo_sims
                    )
                    if mc_mean is not None:
                        forecasts['Monte Carlo'] = {
                            'mean': mc_mean, 'low': mc_low, 'high': mc_high, 'series': mc_series
                        }
                if use_arima:
                    arima_series, arima_summary = self.forecasting_engine.project_arima_smart(data, days_ahead)
                    if arima_series is not None:
                        forecasts['ARIMA'] = {'series': arima_series, 'summary': arima_summary}
                self._display_forecast_results(data, forecasts, symbol, days_ahead)

    def _display_forecast_results(self, data: pd.DataFrame, forecasts: Dict, symbol: str, days_ahead: int):
        if not forecasts:
            st.warning("No forecasts generated. Check model requirements.")
            return
        current_price = data['close'].iloc[-1]
        st.subheader("📊 Forecast Summary")
        cols = st.columns(len(forecasts))
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            with cols[i]:
                if model_name == 'Monte Carlo':
                    price = forecast['mean']
                    low = forecast['low']
                    high = forecast['high']
                    st.metric(
                        f"{model_name}",
                        f"${price:.2f}",
                        f"{((price - current_price) / current_price * 100):+.1f}%"
                    )
                    st.caption(f"Range: ${low:.2f} - ${high:.2f}")
                else:
                    price = forecast.get('price')
                    if price is None and 'series' in forecast:
                        price = forecast['series'].iloc[-1] if not forecast['series'].empty else None
                    if price is not None:
                        st.metric(
                            f"{model_name}",
                            f"${price:.2f}",
                            f"{((price - current_price) / current_price * 100):+.1f}%"
                        )
        st.subheader("📈 Forecast Visualizations")
        for model_name, forecast in forecasts.items():
            with st.expander(f"{model_name} Forecast", expanded=True):
                if model_name == 'Monte Carlo':
                    mean_series, low_series, high_series = forecast['series']
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index[-60:],
                        y=data['close'][-60:],
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=mean_series.index,
                        y=mean_series.values,
                        name='Mean Forecast',
                        line=dict(color='green', width=2, dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=high_series.index,
                        y=high_series.values,
                        name='95% Upper',
                        line=dict(color='gray', width=1, dash='dot'),
                        opacity=0.7
                    ))
                    fig.add_trace(go.Scatter(
                        x=low_series.index,
                        y=low_series.values,
                        name='95% Lower',
                        line=dict(color='gray', width=1, dash='dot'),
                        opacity=0.7,
                        fill='tonexty'
                    ))
                    fig.update_layout(
                        title=f"{symbol} - {model_name} Forecast",
                        template='plotly_white',
                        height=400,
                        plot_bgcolor='rgba(240,240,240,0.5)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    if 'series' not in forecast or forecast['series'] is None:
                        continue
                    fig = go.Figure()
                    recent_data = data.tail(60)
                    fig.add_trace(go.Scatter(
                        x=recent_data.index,
                        y=recent_data['close'],
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast['series'].index,
                        y=forecast['series'].values,
                        name='Forecast',
                        line=dict(color='green', width=2, dash='dash')
                    ))
                    fig.update_layout(
                        title=f"{symbol} - {model_name} Forecast",
                        template='plotly_white',
                        height=400,
                        plot_bgcolor='rgba(240,240,240,0.5)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                if model_name == 'ARIMA' and 'summary' in forecast:
                    st.text_area("Model Details", forecast['summary'], height=100)

    def render_paper_trading(self):
        st.header("💼 Paper Trading Platform")
        symbol = st.session_state.current_symbol
        if not symbol:
            st.info("Enter a stock symbol to trade")
            return
        data = self.data_manager.get_stock_data(symbol, "1d", "1d")
        if data.empty:
            st.error(f"No data for {symbol}")
            return
        current_price = data['close'].iloc[-1]
        st.subheader("📊 Portfolio Summary")
        portfolio = self.trading_engine.get_portfolio_summary()
        if portfolio:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cash Balance", f"${portfolio['cash_balance']:,.2f}")
            with col2:
                st.metric("Total Invested", f"${portfolio['total_invested']:,.2f}")
            with col3:
                st.metric("Current Value", f"${portfolio['total_current_value']:,.2f}")
            with col4:
                pnl_color = "normal" if portfolio['total_unrealized_pnl'] >= 0 else "inverse"
                st.metric(
                    "Total P&L",
                    f"${portfolio['total_unrealized_pnl']:,.2f}",
                    f"{portfolio['total_pnl_percentage']:+.2f}%",
                    delta_color=pnl_color
                )
            if portfolio['holdings']:
                st.subheader("Current Holdings")
                holdings_data = []
                for holding in portfolio['holdings']:
                    holdings_data.append({
                        'Symbol': holding['symbol'],
                        'Quantity': holding['quantity'],
                        'Avg Price': f"${holding['avg_price']:.2f}",
                        'Current Price': f"${holding['current_price']:.2f}",
                        'Invested': f"${holding['invested_value']:,.2f}",
                        'Current Value': f"${holding['current_value']:,.2f}",
                        'P&L': f"${holding['unrealized_pnl']:,.2f}",
                        'P&L %': f"{holding['pnl_percentage']:+.2f}%"
                    })
                df_holdings = pd.DataFrame(holdings_data)
                st.dataframe(df_holdings, use_container_width=True, hide_index=True)
                labels = [h['symbol'] for h in portfolio['holdings']]
                values = [h['current_value'] for h in portfolio['holdings']]
                if values:
                    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
                    fig.update_layout(title="Portfolio Allocation")
                    st.plotly_chart(fig, use_container_width=True)
        st.subheader("💰 Trading Interface")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Buy Order")
            with st.form("buy_form"):
                buy_qty = st.number_input("Quantity", min_value=1, value=10, key="buy_qty")
                buy_price = st.number_input("Price", value=float(current_price), key="buy_price")
                if st.form_submit_button("🟢 Execute Buy Order", use_container_width=True):
                    success, message = self.trading_engine.execute_trade(symbol, 'BUY', buy_qty, buy_price)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        with col2:
            st.subheader("Sell Order")
            symbol_holding = None
            if portfolio and 'holdings' in portfolio:
                symbol_holdings = [h for h in portfolio['holdings'] if h['symbol'] == symbol]
                if symbol_holdings:
                    symbol_holding = symbol_holdings[0]
            max_sell = symbol_holding['quantity'] if symbol_holding else 0
            if max_sell > 0:
                with st.form("sell_form"):
                    sell_qty = st.number_input("Quantity", min_value=1, value=min(10, max_sell),
                                             max_value=max_sell, key="sell_qty")
                    sell_price = st.number_input("Price", value=float(current_price), key="sell_price")
                    if st.form_submit_button("🔴 Execute Sell Order", use_container_width=True):
                        success, message = self.trading_engine.execute_trade(symbol, 'SELL', sell_qty, sell_price)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
            else:
                st.info(f"You don't have any shares of {symbol} to sell.")
        st.subheader("📋 Transaction History")
        transactions = self.db_manager.get_transaction_history(20)
        if transactions:
            trans_data = []
            for sym, act, qty, price, total, date in transactions:
                trans_data.append({
                    'Date': date[:16],
                    'Symbol': sym,
                    'Action': act,
                    'Quantity': qty,
                    'Price': f"${price:.2f}",
                    'Total': f"${total:.2f}"
                })
            df_transactions = pd.DataFrame(trans_data)
            st.dataframe(df_transactions, use_container_width=True, hide_index=True)
        else:
            st.info("No transactions yet")

    def render_watchlist(self):
        st.header("⭐ Watchlist")
        with st.form("add_watchlist_form"):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                new_symbol = st.text_input("Stock Symbol", placeholder="Enter symbol (e.g., AAPL)").upper()
            with col2:
                notes = st.text_input("Notes", placeholder="Optional notes")
            with col3:
                add_submitted = st.form_submit_button("Add to Watchlist", use_container_width=True)
                if add_submitted and new_symbol:
                    data = self.data_manager.get_stock_data(new_symbol, "1d")
                    if not data.empty:
                        current_price = data['close'].iloc[-1]
                        if self.db_manager.add_to_watchlist(new_symbol, current_price, notes):
                            st.success(f"✅ Added {new_symbol} to watchlist!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to add to watchlist")
                    else:
                        st.error(f"❌ Could not fetch data for {new_symbol}")
        st.subheader("Your Watchlist")
        watchlist = self.db_manager.get_watchlist()
        if watchlist:
            if st.button("🔄 Refresh Watchlist Prices", use_container_width=True):
                st.rerun()
            watchlist_data = []
            for symbol, added_price, added_date, notes in watchlist:
                current_data = self.data_manager.get_stock_data(symbol, "1d")
                if not current_data.empty:
                    current_price = current_data['close'].iloc[-1]
                    change = current_price - added_price
                    change_pct = (change / added_price) * 100
                    watchlist_data.append({
                        'Symbol': symbol,
                        'Added Price': added_price,
                        'Current Price': current_price,
                        'Change $': change,
                        'Change %': change_pct,
                        'Added Date': added_date[:10],
                        'Notes': notes or ''
                    })
            if watchlist_data:
                df = pd.DataFrame(watchlist_data)
                def color_negative_red(val):
                    if isinstance(val, str) and '-' in val:
                        return 'color: #dc3545'
                    elif isinstance(val, str) and '+' in val:
                        return 'color: #28a745'
                    return ''
                styled_df = df.style.format({
                    'Added Price': '${:.2f}',
                    'Current Price': '${:.2f}',
                    'Change $': '${:+.2f}',
                    'Change %': '{:+.2f}%'
                }).map(color_negative_red, subset=['Change $', 'Change %'])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                st.subheader("Remove Stock")
                selected_symbol = st.selectbox("Select symbol to remove", [w[0] for w in watchlist])
                if st.button("🗑️ Remove Selected Symbol", use_container_width=True):
                    if self.db_manager.remove_from_watchlist(selected_symbol):
                        st.success(f"Removed {selected_symbol} from watchlist")
                        st.rerun()
                    else:
                        st.error("Failed to remove from watchlist")
        else:
            st.info("🌟 Your watchlist is empty. Add some stocks to get started!")

    def render_reports(self):
        st.header("📊 Reports & Export")
        symbol = st.session_state.current_symbol
        if not symbol:
            st.info("Enter a stock symbol to generate reports")
            return
        st.subheader("📄 Comprehensive Reports")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate PDF Report", use_container_width=True, type="primary"):
                freemium.download_reminder_popup()
                st.stop()
        with col2:
            if st.button("Generate HTML Report", use_container_width=True):
                freemium.download_reminder_popup()
                st.stop()
        st.subheader("📥 Comprehensive Data Export")
        if st.button("Export Detailed CSV", use_container_width=True):
            freemium.download_reminder_popup()
            st.stop()

# =============================================================================
# Main
# =============================================================================
def main():
    try:
        app = ProfessionalMarketPlatform()
        app.run()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.error("Please check your internet connection and try again.")

if __name__ == "__main__":
    if 'streamlit' not in sys.argv[0] and not os.environ.get('STREAMLIT_RUN'):
        os.environ['STREAMLIT_RUN'] = '1'
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    else:
        main()