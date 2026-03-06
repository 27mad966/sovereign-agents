"""
Sovereign Trading System — Configuration
عدّل هذا الملف لتغيير الإعدادات
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    # ══════════════════════════════════════════════
    # Binance API
    # ══════════════════════════════════════════════
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET: str  = os.getenv("BINANCE_SECRET", "")

    # ══════════════════════════════════════════════
    # Telegram
    # ══════════════════════════════════════════════
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str   = os.getenv("TELEGRAM_CHAT_ID", "")

    # ══════════════════════════════════════════════
    # ⬇️ العملات — أضف أو احذف كما تريد
    # ══════════════════════════════════════════════
    SYMBOLS: list = None

    # ══════════════════════════════════════════════
    # ⬇️ الفريم الزمني للتحليل
    # الخيارات: "5m" "10m" "15m" "30m" "1h"
    # ══════════════════════════════════════════════
    ANALYSIS_TIMEFRAME: str = os.getenv("ANALYSIS_TIMEFRAME", "15m")
    HTF_TIMEFRAME: str      = os.getenv("HTF_TIMEFRAME", "1h")

    # ══════════════════════════════════════════════
    # Risk limits
    # ══════════════════════════════════════════════
    MAX_DAILY_DRAWDOWN_PCT: float = 3.0
    MAX_POSITION_SIZE_PCT: float  = 10.0
    VAR_CONFIDENCE: float         = 0.95

    # Agent intervals
    EXECUTION_CHECK_INTERVAL: int = 30
    RISK_CHECK_INTERVAL: int      = 60
    AUDIT_INTERVAL: int           = 3600
    META_CHECK_INTERVAL: int      = 120

    def __post_init__(self):
        if self.SYMBOLS is None:
            symbols_env = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT")
            self.SYMBOLS = [s.strip() for s in symbols_env.split(",")]

        tf_map = {
            "1m": 60, "3m": 180, "5m": 300, "10m": 600,
            "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400,
        }
        self.MARKET_INTEL_INTERVAL = tf_map.get(self.ANALYSIS_TIMEFRAME, 900)


config = Config()
