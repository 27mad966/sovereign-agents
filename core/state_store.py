"""
Shared State Store — حالة النظام المشتركة بين كل الـ agents
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import threading


@dataclass
class Trade:
    symbol: str
    side: str            # BUY / SELL
    qty: float
    entry_price: float
    exit_price: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    slippage_pct: float = 0.0
    pnl: float = 0.0
    closed: bool = False


@dataclass
class PerformanceSnapshot:
    timestamp: float
    total_pnl: float
    win_rate: float
    total_trades: int
    winning_trades: int
    max_drawdown: float
    sharpe: float
    daily_pnl: float


@dataclass
class MarketRegime:
    symbol: str
    regime: str          # "trending_up" | "trending_down" | "ranging" | "volatile"
    volatility: float    # ATR %
    volume_zscore: float
    sentiment: str       # "bullish" | "bearish" | "neutral"
    timestamp: float = field(default_factory=time.time)


@dataclass
class RiskState:
    daily_pnl: float = 0.0
    daily_drawdown_pct: float = 0.0
    current_var_95: float = 0.0
    open_positions_count: int = 0
    total_exposure_pct: float = 0.0
    trading_halted: bool = False
    halt_reason: str = ""


class StateStore:
    """
    Thread-safe shared state لكل الـ agents
    """
    def __init__(self):
        self._lock = threading.RLock()

        # Trading history
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}

        # Performance
        self.performance: PerformanceSnapshot = PerformanceSnapshot(
            timestamp=time.time(), total_pnl=0, win_rate=0,
            total_trades=0, winning_trades=0, max_drawdown=0,
            sharpe=0, daily_pnl=0
        )

        # Market
        self.market_regimes: Dict[str, MarketRegime] = {}
        self.orderbook_depth: Dict[str, Dict] = {}

        # Risk
        self.risk: RiskState = RiskState()

        # Strategy params (live, قابلة للتغيير)
        self.strategy_params: Dict[str, Any] = {
            "fast_ma": 9,
            "slow_ma": 21,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "atr_multiplier": 2.0,
        }
        self.param_history: List[Dict] = []

        # Audit log
        self.audit_log: List[Dict] = []

        # Agent metrics
        self.agent_metrics: Dict[str, Dict] = {}

    # ── Trade Management ──────────────────────────────
    def add_trade(self, trade: Trade):
        with self._lock:
            self.trades.append(trade)
            if not trade.closed:
                self.open_positions[trade.symbol] = trade
            self._recalc_performance()

    def close_trade(self, symbol: str, exit_price: float):
        with self._lock:
            if symbol in self.open_positions:
                t = self.open_positions.pop(symbol)
                t.exit_price = exit_price
                t.closed = True
                t.pnl = (exit_price - t.entry_price) / t.entry_price * 100
                if t.side == "SELL":
                    t.pnl = -t.pnl
                self._recalc_performance()
                return t
        return None

    def _recalc_performance(self):
        closed = [t for t in self.trades if t.closed]
        if not closed:
            return
        wins = [t for t in closed if t.pnl > 0]
        total_pnl = sum(t.pnl for t in closed)
        win_rate = len(wins) / len(closed) * 100

        # Drawdown
        cumulative = []
        cum = 0
        peak = 0
        max_dd = 0
        for t in closed:
            cum += t.pnl
            if cum > peak:
                peak = cum
            dd = (peak - cum)
            if dd > max_dd:
                max_dd = dd
            cumulative.append(cum)

        # Daily PnL
        today = time.time() - 86400
        daily_pnl = sum(t.pnl for t in closed if t.timestamp > today)

        self.performance = PerformanceSnapshot(
            timestamp=time.time(),
            total_pnl=round(total_pnl, 4),
            win_rate=round(win_rate, 2),
            total_trades=len(closed),
            winning_trades=len(wins),
            max_drawdown=round(max_dd, 4),
            sharpe=self._calc_sharpe(closed),
            daily_pnl=round(daily_pnl, 4),
        )
        self.risk.daily_pnl = daily_pnl

    def _calc_sharpe(self, trades) -> float:
        if len(trades) < 2:
            return 0.0
        pnls = [t.pnl for t in trades]
        mean = sum(pnls) / len(pnls)
        std = (sum((p - mean) ** 2 for p in pnls) / len(pnls)) ** 0.5
        return round(mean / std * (252 ** 0.5), 3) if std > 0 else 0.0

    # ── Strategy Params ───────────────────────────────
    def update_params(self, new_params: Dict, reason: str = ""):
        with self._lock:
            old = dict(self.strategy_params)
            self.strategy_params.update(new_params)
            self.param_history.append({
                "timestamp": time.time(),
                "old": old,
                "new": dict(self.strategy_params),
                "reason": reason,
            })

    # ── Audit ─────────────────────────────────────────
    def log_audit(self, agent: str, action: str, details: Dict):
        with self._lock:
            self.audit_log.append({
                "timestamp": time.time(),
                "agent": agent,
                "action": action,
                "details": details,
            })
            if len(self.audit_log) > 10000:
                self.audit_log = self.audit_log[-5000:]

    def get_summary(self) -> Dict:
        with self._lock:
            return {
                "performance": {
                    "total_pnl": self.performance.total_pnl,
                    "win_rate": self.performance.win_rate,
                    "total_trades": self.performance.total_trades,
                    "max_drawdown": self.performance.max_drawdown,
                    "sharpe": self.performance.sharpe,
                    "daily_pnl": self.performance.daily_pnl,
                },
                "risk": {
                    "trading_halted": self.risk.trading_halted,
                    "daily_drawdown_pct": self.risk.daily_drawdown_pct,
                    "var_95": self.risk.current_var_95,
                    "open_positions": len(self.open_positions),
                },
                "market": {
                    sym: {
                        "regime": r.regime,
                        "sentiment": r.sentiment,
                        "volatility": r.volatility,
                    }
                    for sym, r in self.market_regimes.items()
                },
                "strategy_params": self.strategy_params,
            }


# Singleton
state = StateStore()
