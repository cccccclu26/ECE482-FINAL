"""
Configuration for ECE482-FINAL LLM-Driven Stock Prediction System
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM Model (OpenAI GPT-5.4)
OPENAI_MODEL = "gpt-5.4"

# Stock Universe
TICKERS = ["AAPL", "NVDA", "META", "JPM", "TSLA", "MSFT", "AMZN", "GOOGL", "AVGO", "LLY"]

# Prediction Config
PREDICTION_HORIZON_DAYS = 63   # ~3 months forward
REBALANCE_DAYS = 21            # ~monthly rebalance
SAMPLE_INTERVAL_DAYS = 3       # collect training sample every 2-3 trading days
LOOKBACK_SEQUENCE = 60         # LSTM input sequence length (trading days)

# Portfolio Config
INITIAL_CAPITAL = 10000.0
MIN_PROBABILITY = 0.52         # Min P(up) to include in portfolio

# Data Config
POLYGON_BASE_URL = "https://api.polygon.io"
NEWS_LOOKBACK_DAYS = 7
DEFAULT_NEWS_LIMIT = 10

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
