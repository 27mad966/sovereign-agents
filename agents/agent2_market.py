"""
Agent 2 — Market Intelligence
الأولوية الثانية — عيون النظام على السوق

مسؤوليات:
- تحليل Sentiment من بيانات Binance (funding rate, long/short ratio)
- On-chain signals: whale movements, exchange flows
- تصنيف Market Regime (trending/ranging/volatile)
- اكتشاف Macro events مؤثرة
"""

import asyncio
import time
import random
import logging
from typing import Dict, List, Tuple

from core.message_bus import bus, AgentID, Event, EventType
from core.state_store import state, MarketRegime
from config.settings import config

logger = logging.getLogger("MarketIntelAgent")


class MarketIntelligenceAgent:
    def __init__(self):
        self.agent_id = AgentID.MARKET
        self.name = "Market Intelligence"
        self.running = False

        # Thresholds
        self.FUNDING_RATE_EXTREME = 0.05     # 0.05% = extremamente bullish/bearish
        self.WHALE_MOVE_THRESHOLD = 1_000_000  # $1M+
        self.VOLATILITY_HIGH = 4.0            # ATR% > 4 = volatile regime

        self.metrics = {
            "regimes_detected": {},
            "sentiment_calls": 0,
            "whale_alerts": 0,
            "regime_changes": 0,
        }

        self._prev_regimes: Dict[str, str] = {}

    async def start(self):
        self.running = True
        logger.info(f"✅ {self.name} started")
        await asyncio.gather(
            self._sentiment_loop(),
            self._regime_loop(),
            self._onchain_loop(),
        )

    async def stop(self):
        self.running = False

    # ── Loops ─────────────────────────────────────────

    async def _sentiment_loop(self):
        """تحليل sentiment كل 5 دقائق"""
        while self.running:
            try:
                bus.heartbeat(self.agent_id)
                for symbol in config.SYMBOLS:
                    sentiment_data = await self._fetch_sentiment(symbol)
                    await self._process_sentiment(symbol, sentiment_data)
                self.metrics["sentiment_calls"] += 1
                state.agent_metrics[self.agent_id.value] = self.metrics.copy()
                await asyncio.sleep(config.MARKET_INTEL_INTERVAL)
            except Exception as e:
                logger.error(f"Sentiment loop error: {e}")
                await asyncio.sleep(30)

    async def _regime_loop(self):
        """تحديد Market Regime كل 5 دقائق"""
        while self.running:
            try:
                for symbol in config.SYMBOLS:
                    regime = await self._classify_regime(symbol)
                    state.market_regimes[symbol] = regime

                    # كشف تغيير الـ regime
                    prev = self._prev_regimes.get(symbol)
                    if prev and prev != regime.regime:
                        self.metrics["regime_changes"] += 1
                        await self._emit_regime_change(symbol, prev, regime)

                    self._prev_regimes[symbol] = regime.regime

                await asyncio.sleep(config.MARKET_INTEL_INTERVAL)
            except Exception as e:
                logger.error(f"Regime loop error: {e}")
                await asyncio.sleep(30)

    async def _onchain_loop(self):
        """مراقبة on-chain كل 10 دقائق"""
        while self.running:
            try:
                whale_moves = await self._check_whale_activity()
                for move in whale_moves:
                    if move["amount_usd"] >= self.WHALE_MOVE_THRESHOLD:
                        self.metrics["whale_alerts"] += 1
                        await bus.publish(Event(
                            type=EventType.ONCHAIN_SIGNAL,
                            source=self.agent_id,
                            payload={
                                "type": "whale_move",
                                "symbol": move["symbol"],
                                "direction": move["direction"],
                                "amount_usd": move["amount_usd"],
                                "message": (
                                    f"🐋 حركة whale: {move['direction']} "
                                    f"${move['amount_usd']:,.0f} في {move['symbol']}"
                                ),
                            },
                            priority=2
                        ))

                await asyncio.sleep(600)
            except Exception as e:
                logger.error(f"Onchain loop error: {e}")
                await asyncio.sleep(60)

    # ── Core Analysis ─────────────────────────────────

    async def _classify_regime(self, symbol: str) -> MarketRegime:
        """
        تصنيف حالة السوق بناءً على:
        - ATR relative (volatility)
        - Price vs MA200
        - Volume Z-score
        - Funding rate
        """
        # في الإنتاج: جلب بيانات OHLCV من Binance وحساب المؤشرات
        # from binance.client import Client
        # klines = client.get_klines(symbol=symbol, interval='1h', limit=200)

        await asyncio.sleep(0.05)

        # محاكاة
        volatility = random.uniform(1.0, 6.0)
        vol_zscore = random.uniform(-2.0, 2.0)
        price_vs_ma = random.uniform(-5.0, 5.0)  # % فوق/تحت MA200
        funding = random.uniform(-0.1, 0.1)

        # منطق التصنيف
        if volatility > self.VOLATILITY_HIGH:
            regime = "volatile"
        elif abs(price_vs_ma) < 1.5 and abs(vol_zscore) < 0.5:
            regime = "ranging"
        elif price_vs_ma > 2.0:
            regime = "trending_up"
        else:
            regime = "trending_down"

        # Sentiment
        if funding > self.FUNDING_RATE_EXTREME:
            sentiment = "bullish"
        elif funding < -self.FUNDING_RATE_EXTREME:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return MarketRegime(
            symbol=symbol,
            regime=regime,
            volatility=round(volatility, 2),
            volume_zscore=round(vol_zscore, 2),
            sentiment=sentiment,
        )

    async def _process_sentiment(self, symbol: str, data: Dict):
        """معالجة بيانات الـ sentiment وإطلاق events"""
        funding_rate = data.get("funding_rate", 0)
        ls_ratio = data.get("long_short_ratio", 1.0)

        severity = "normal"
        message = ""

        if abs(funding_rate) > self.FUNDING_RATE_EXTREME:
            severity = "extreme"
            direction = "bullish 🟢" if funding_rate > 0 else "bearish 🔴"
            message = f"Funding rate متطرف ({direction}): {funding_rate:.4f}% على {symbol}"

        await bus.publish(Event(
            type=EventType.SENTIMENT_UPDATE,
            source=self.agent_id,
            payload={
                "symbol": symbol,
                "funding_rate": funding_rate,
                "long_short_ratio": ls_ratio,
                "severity": severity,
                "message": message,
                "timestamp": time.time(),
            }
        ))

    async def _emit_regime_change(self, symbol: str, old: str, new: MarketRegime):
        """تنبيه عند تغيير حالة السوق — هذا مهم جداً لـ Parameter Optimizer"""
        regime_emoji = {
            "trending_up": "📈",
            "trending_down": "📉",
            "ranging": "↔️",
            "volatile": "⚡",
        }
        emoji = regime_emoji.get(new.regime, "🔄")

        await bus.publish(Event(
            type=EventType.REGIME_CHANGE,
            source=self.agent_id,
            payload={
                "symbol": symbol,
                "from_regime": old,
                "to_regime": new.regime,
                "volatility": new.volatility,
                "message": f"{emoji} تغيير Regime على {symbol}: {old} → {new.regime}",
                "recommendation": self._regime_recommendation(new.regime),
            },
            priority=2
        ))

        state.log_audit(self.agent_id.value, "regime_change", {
            "symbol": symbol, "from": old, "to": new.regime,
        })

    def _regime_recommendation(self, regime: str) -> str:
        recs = {
            "trending_up":   "زيادة حجم الصفقات، تشديد Stop Loss",
            "trending_down": "تقليل الـ exposure، التحوط ممكن",
            "ranging":       "تفعيل استراتيجية Mean Reversion، تقليل MA periods",
            "volatile":      "تقليل حجم الصفقات 50%، زيادة ATR multiplier",
        }
        return recs.get(regime, "مراجعة الاستراتيجية")

    # ── Binance Data Fetchers ─────────────────────────

    async def _fetch_sentiment(self, symbol: str) -> Dict:
        """
        Binance Futures sentiment data
        في الإنتاج:
        GET /fapi/v1/fundingRate?symbol=BTCUSDT
        GET /futures/data/globalLongShortAccountRatio
        """
        await asyncio.sleep(0.05)
        return {
            "symbol": symbol,
            "funding_rate": random.uniform(-0.1, 0.1),
            "long_short_ratio": random.uniform(0.8, 1.5),
            "open_interest_change": random.uniform(-10, 10),
        }

    async def _check_whale_activity(self) -> List[Dict]:
        """
        مراقبة حركات الـ whales
        في الإنتاج: استخدم Glassnode API أو Whale Alert API
        """
        await asyncio.sleep(0.1)
        moves = []
        if random.random() < 0.2:  # 20% فرصة وجود حركة whale
            moves.append({
                "symbol": random.choice(config.SYMBOLS),
                "direction": random.choice(["to_exchange", "from_exchange"]),
                "amount_usd": random.uniform(500_000, 5_000_000),
            })
        return moves

    def get_report(self) -> Dict:
        return {
            "agent": self.name,
            "metrics": self.metrics,
            "regimes": {
                sym: {
                    "regime": r.regime,
                    "sentiment": r.sentiment,
                    "volatility": r.volatility,
                }
                for sym, r in state.market_regimes.items()
            },
            "timestamp": time.time(),
        }
