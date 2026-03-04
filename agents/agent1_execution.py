"""
Agent 1 — Execution Quality Monitor
الأولوية الأولى — يراقب جودة تنفيذ الصفقات لحظياً

مسؤوليات:
- قياس الـ slippage الفعلي مقابل النظري
- مراقبة عمق السوق (order book depth) قبل أي أمر
- تنبيه فوري إذا تدهورت جودة التنفيذ
- حساب market impact لكل صفقة
"""

import asyncio
import time
import random
import logging
from typing import Dict, List, Optional

from core.message_bus import bus, AgentID, Event, EventType
from core.state_store import state, Trade
from config.settings import config

logger = logging.getLogger("ExecutionAgent")


class ExecutionQualityAgent:
    def __init__(self):
        self.agent_id = AgentID.EXECUTION
        self.name = "Execution Quality Monitor"
        self.running = False

        # Thresholds
        self.SLIPPAGE_WARN_PCT = 0.05      # تنبيه عند 0.05%
        self.SLIPPAGE_CRITICAL_PCT = 0.15  # حرج عند 0.15%
        self.MIN_DEPTH_USDT = 50_000       # حد أدنى لعمق السوق

        # Metrics
        self.metrics = {
            "avg_slippage": 0.0,
            "max_slippage": 0.0,
            "liquidity_warnings": 0,
            "orders_analyzed": 0,
            "execution_score": 100.0,      # 0-100, كلما أعلى كلما أفضل
        }

        # Subscribe to relevant events
        bus.subscribe(EventType.RISK_BREACH, self._on_risk_breach)

    async def start(self):
        self.running = True
        logger.info(f"✅ {self.name} started")
        await asyncio.gather(
            self._monitor_loop(),
            self._orderbook_loop(),
        )

    async def stop(self):
        self.running = False

    # ── Main Loops ────────────────────────────────────

    async def _monitor_loop(self):
        """يراقب الصفقات المنفذة ويحسب الجودة"""
        analyzed_ids = set()
        while self.running:
            try:
                bus.heartbeat(self.agent_id)
                closed_trades = [t for t in state.trades if t.closed]

                for trade in closed_trades:
                    tid = id(trade)
                    if tid not in analyzed_ids:
                        analyzed_ids.add(tid)
                        await self._analyze_execution(trade)

                state.agent_metrics[self.agent_id.value] = self.metrics.copy()
                await asyncio.sleep(config.EXECUTION_CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Execution monitor error: {e}")
                await asyncio.sleep(5)

    async def _orderbook_loop(self):
        """يفحص عمق السوق كل 30 ثانية"""
        while self.running:
            try:
                for symbol in config.SYMBOLS:
                    depth = await self._fetch_orderbook_depth(symbol)
                    state.orderbook_depth[symbol] = depth

                    if depth["bid_depth_usdt"] < self.MIN_DEPTH_USDT:
                        await self._emit_liquidity_warning(symbol, depth)

                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Orderbook loop error: {e}")
                await asyncio.sleep(10)

    # ── Analysis ──────────────────────────────────────

    async def _analyze_execution(self, trade: Trade):
        """يحلل جودة تنفيذ صفقة واحدة"""
        # في البيئة الحقيقية: يقارن سعر الأمر بسعر التنفيذ الفعلي
        # هنا نحاكي الحساب
        theoretical_price = trade.entry_price
        actual_price = trade.entry_price * (1 + trade.slippage_pct / 100)
        slippage = abs(actual_price - theoretical_price) / theoretical_price * 100

        # تحديث الـ metrics
        self.metrics["orders_analyzed"] += 1
        self.metrics["max_slippage"] = max(self.metrics["max_slippage"], slippage)

        # Moving average للـ slippage
        n = self.metrics["orders_analyzed"]
        self.metrics["avg_slippage"] = (
            (self.metrics["avg_slippage"] * (n - 1) + slippage) / n
        )

        # Execution score (100 = مثالي)
        score_penalty = min(slippage * 20, 30)
        self.metrics["execution_score"] = max(0, 100 - score_penalty)

        # تسجيل في الـ audit
        state.log_audit(self.agent_id.value, "execution_analyzed", {
            "symbol": trade.symbol,
            "slippage_pct": round(slippage, 4),
            "score": self.metrics["execution_score"],
        })

        # تنبيه إذا الـ slippage مرتفع
        if slippage >= self.SLIPPAGE_CRITICAL_PCT:
            await bus.publish(Event(
                type=EventType.SLIPPAGE_ALERT,
                source=self.agent_id,
                payload={
                    "symbol": trade.symbol,
                    "slippage_pct": round(slippage, 4),
                    "severity": "critical",
                    "message": f"⚠️ Slippage حرج: {slippage:.3f}% على {trade.symbol}",
                },
                priority=3
            ))
        elif slippage >= self.SLIPPAGE_WARN_PCT:
            await bus.publish(Event(
                type=EventType.SLIPPAGE_ALERT,
                source=self.agent_id,
                payload={
                    "symbol": trade.symbol,
                    "slippage_pct": round(slippage, 4),
                    "severity": "warning",
                    "message": f"📊 Slippage مرتفع: {slippage:.3f}% على {trade.symbol}",
                },
                priority=2
            ))

        # تقرير جودة التنفيذ
        await bus.publish(Event(
            type=EventType.ORDER_QUALITY,
            source=self.agent_id,
            payload={
                "symbol": trade.symbol,
                "slippage_pct": round(slippage, 4),
                "execution_score": self.metrics["execution_score"],
                "avg_slippage": round(self.metrics["avg_slippage"], 4),
            }
        ))

    async def _emit_liquidity_warning(self, symbol: str, depth: Dict):
        self.metrics["liquidity_warnings"] += 1
        await bus.publish(Event(
            type=EventType.LIQUIDITY_WARNING,
            source=self.agent_id,
            payload={
                "symbol": symbol,
                "bid_depth_usdt": depth["bid_depth_usdt"],
                "ask_depth_usdt": depth["ask_depth_usdt"],
                "spread_pct": depth["spread_pct"],
                "message": f"⚠️ سيولة منخفضة على {symbol}: ${depth['bid_depth_usdt']:,.0f}",
                "recommendation": "تجنب الأوامر الكبيرة حتى تتحسن السيولة",
            },
            priority=2
        ))

    # ── Binance Data ──────────────────────────────────

    async def _fetch_orderbook_depth(self, symbol: str) -> Dict:
        """
        جلب عمق السوق من Binance
        في الإنتاج: استخدم python-binance أو ccxt
        """
        try:
            # محاكاة — استبدل بـ:
            # from binance.client import Client
            # client = Client(config.BINANCE_API_KEY, config.BINANCE_SECRET)
            # depth = client.get_order_book(symbol=symbol, limit=20)
            await asyncio.sleep(0.1)
            return {
                "symbol": symbol,
                "bid_depth_usdt": random.uniform(80_000, 500_000),
                "ask_depth_usdt": random.uniform(80_000, 500_000),
                "spread_pct": random.uniform(0.01, 0.05),
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            return {"symbol": symbol, "bid_depth_usdt": 0, "ask_depth_usdt": 0, "spread_pct": 0}

    # ── Event Handlers ────────────────────────────────

    async def _on_risk_breach(self, event: Event):
        """إذا وقف الـ Risk Agent التداول، نوقف مراقبة جودة التنفيذ مؤقتاً"""
        if event.payload.get("halt"):
            logger.warning("Trading halted — Execution agent in standby mode")

    def get_report(self) -> Dict:
        return {
            "agent": self.name,
            "metrics": self.metrics,
            "orderbook": {
                sym: {
                    "depth_usdt": d.get("bid_depth_usdt", 0),
                    "spread_pct": d.get("spread_pct", 0),
                }
                for sym, d in state.orderbook_depth.items()
            },
            "timestamp": time.time(),
        }
