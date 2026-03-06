"""
Agent 2 — Market Intelligence
مع Binance Testnet API + مؤشرات الاستراتيجية الفعلية
"""

import asyncio
import time
import logging
import random
from typing import Dict, Optional
import pandas as pd
import numpy as np

from core.message_bus import bus, AgentID, Event, EventType
from core.state_store import state, MarketRegime
from core.strategy_engine import SovereignStrategy, StrategyParams
from config.settings import config

logger = logging.getLogger("MarketIntelAgent")


class MarketIntelligenceAgent:
    def __init__(self):
        self.agent_id = AgentID.MARKET
        self.name = "Market Intelligence"
        self.running = False
        self.strategies: Dict[str, SovereignStrategy] = {
            sym: SovereignStrategy(StrategyParams()) for sym in config.SYMBOLS
        }
        self.metrics = {
            "sentiment_calls": 0,
            "whale_alerts": 0,
            "regime_changes": 0,
            "strategy_signals": {},
        }
        self._prev_regimes: Dict[str, str] = {}

    async def start(self):
        self.running = True
        logger.info(f"✅ {self.name} started")
        await asyncio.gather(
            self._strategy_monitor_loop(),
            self._onchain_loop(),
        )

    async def stop(self):
        self.running = False

    async def _strategy_monitor_loop(self):
        while self.running:
            try:
                bus.heartbeat(self.agent_id)
                for symbol in config.SYMBOLS:
                    await self._analyze_symbol(symbol)
                self.metrics["sentiment_calls"] += 1
                state.agent_metrics[self.agent_id.value] = self.metrics.copy()
                await asyncio.sleep(config.MARKET_INTEL_INTERVAL)
            except Exception as e:
                logger.error(f"Strategy monitor error: {e}")
                await asyncio.sleep(30)

    async def _onchain_loop(self):
        while self.running:
            try:
                moves = await self._check_whale_activity()
                for move in moves:
                    if move["amount_usd"] >= 1_000_000:
                        self.metrics["whale_alerts"] += 1
                        await bus.publish(Event(
                            type=EventType.ONCHAIN_SIGNAL,
                            source=self.agent_id,
                            payload={
                                "symbol": move["symbol"],
                                "message": f"🐋 whale: {move['direction']} ${move['amount_usd']:,.0f} في {move['symbol']}",
                            },
                            priority=2
                        ))
                await asyncio.sleep(600)
            except Exception as e:
                logger.error(f"Onchain error: {e}")
                await asyncio.sleep(60)

    async def _analyze_symbol(self, symbol: str):
        df = await self._fetch_ohlcv(symbol, "15m", 200)
        htf_df = await self._fetch_ohlcv(symbol, "1h", 100)
        if df is None or len(df) < 50:
            return

        strategy = self.strategies[symbol]
        dashboard = strategy.get_dashboard_data(df, htf_df)
        signals = dashboard["signal"]
        indicators = dashboard["indicators"]
        position = dashboard["position"]

        self.metrics["strategy_signals"][symbol] = {
            "supertrend": indicators["supertrend"],
            "squeeze": indicators["squeeze"],
            "mfi": indicators["mfi"],
            "htf": indicators["htf_filter"],
            "pnl_now": position["pnl_now"],
        }

        # Regime
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        vol = float(np.mean(high[-14:] - low[-14:]) / np.mean(close[-14:]) * 100)
        is_bullish = "🟢" in indicators["supertrend"]
        htf_ok = "✅" in indicators["htf_filter"]

        if vol > 4.0:
            regime = "volatile"
        elif is_bullish and htf_ok:
            regime = "trending_up"
        elif not is_bullish and not htf_ok:
            regime = "trending_down"
        else:
            regime = "ranging"

        mr = MarketRegime(
            symbol=symbol, regime=regime, volatility=round(vol, 2),
            volume_zscore=0.0,
            sentiment="bullish" if is_bullish and htf_ok else "bearish" if not is_bullish else "neutral"
        )
        state.market_regimes[symbol] = mr

        prev = self._prev_regimes.get(symbol)
        if prev and prev != regime:
            self.metrics["regime_changes"] += 1
            emoji = {"trending_up": "📈", "trending_down": "📉", "ranging": "↔️", "volatile": "⚡"}.get(regime, "🔄")
            await bus.publish(Event(
                type=EventType.REGIME_CHANGE,
                source=self.agent_id,
                payload={
                    "symbol": symbol, "from_regime": prev, "to_regime": regime,
                    "message": f"{emoji} تغيير Regime على {symbol}: {prev} → {regime}",
                },
                priority=2
            ))
        self._prev_regimes[symbol] = regime

        if signals["long_entry"]:
            await bus.publish(Event(
                type=EventType.SENTIMENT_UPDATE,
                source=self.agent_id,
                payload={
                    "symbol": symbol, "signal": "LONG_ENTRY",
                    "price": position["current"], "sl": position["sl_price"],
                    "message": f"🟢 إشارة LONG على {symbol} @ {position['current']:.4f} | SL: {position['sl_price']:.4f}",
                },
                priority=2
            ))

        if signals["exit_peak"]:
            await bus.publish(Event(
                type=EventType.SENTIMENT_UPDATE,
                source=self.agent_id,
                payload={
                    "symbol": symbol, "signal": "PEAK_EXIT",
                    "message": f"🟣 PEAK EXIT على {symbol} — PnL: {position['pnl_now']:+.2f}%",
                },
                priority=2
            ))

        if signals["exit_sl"]:
            await bus.publish(Event(
                type=EventType.DRAWDOWN_WARNING,
                source=self.agent_id,
                payload={
                    "symbol": symbol, "signal": "STOP_LOSS",
                    "message": f"🔴 STOP LOSS على {symbol} — PnL: {position['pnl_now']:+.2f}%",
                },
                priority=3
            ))

    async def _fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """جلب بيانات من Binance Testnet"""
        try:
            from binance.client import Client
            client = Client(
                config.BINANCE_API_KEY,
                config.BINANCE_SECRET,
                testnet=True
            )

            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

            df = pd.DataFrame(klines, columns=[
                'time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_vol', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            df.index.name = symbol
            logger.info(f"✅ Binance Testnet: {symbol} {interval} — {len(df)} candles — آخر سعر: {df['close'].iloc[-1]:.2f}")
            return df

        except Exception as e:
            logger.error(f"Binance Testnet fetch error {symbol} {interval}: {e}")
            return None

    async def _check_whale_activity(self) -> list:
        await asyncio.sleep(0.1)
        if random.random() < 0.15:
            return [{"symbol": random.choice(config.SYMBOLS),
                     "direction": random.choice(["to_exchange", "from_exchange"]),
                     "amount_usd": random.uniform(500_000, 5_000_000)}]
        return []

    def get_report(self) -> dict:
        return {
            "agent": self.name,
            "metrics": self.metrics,
            "strategy_signals": self.metrics.get("strategy_signals", {}),
            "regimes": {
                sym: {"regime": r.regime, "sentiment": r.sentiment, "volatility": r.volatility}
                for sym, r in state.market_regimes.items()
            },
            "timestamp": time.time(),
        }