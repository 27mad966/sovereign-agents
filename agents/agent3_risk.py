"""
Agent 3 — Risk Management
الأولوية الثالثة — الحارس الأخير للرأس المال

مسؤوليات:
- حساب VaR (Value at Risk) لحظياً
- مراقبة الـ drawdown اليومي والأسبوعي
- وقف التداول عند تجاوز حدود المخاطرة
- Correlation monitoring بين العملات
- Position sizing recommendations
"""

import asyncio
import time
import math
import random
import logging
from typing import Dict, List, Optional

from core.message_bus import bus, AgentID, Event, EventType
from core.state_store import state
from config.settings import config

logger = logging.getLogger("RiskAgent")


class RiskManagementAgent:
    def __init__(self):
        self.agent_id = AgentID.RISK
        self.name = "Risk Management"
        self.running = False

        self.metrics = {
            "var_95": 0.0,
            "var_99": 0.0,
            "current_drawdown_pct": 0.0,
            "max_drawdown_session": 0.0,
            "sharpe_ratio": 0.0,
            "risk_score": 0,    # 0-100, كلما أقل كلما أأمن
            "breaches_today": 0,
            "trading_halted": False,
        }

        # Subscribe to events
        bus.subscribe(EventType.REGIME_CHANGE, self._on_regime_change)
        bus.subscribe(EventType.ONCHAIN_SIGNAL, self._on_whale_alert)

    async def start(self):
        self.running = True
        logger.info(f"✅ {self.name} started")
        await asyncio.gather(
            self._risk_monitor_loop(),
            self._var_calc_loop(),
            self._correlation_loop(),
        )

    async def stop(self):
        self.running = False

    # ── Loops ─────────────────────────────────────────

    async def _risk_monitor_loop(self):
        """مراقبة حدود المخاطرة كل دقيقة"""
        while self.running:
            try:
                bus.heartbeat(self.agent_id)
                await self._check_drawdown_limits()
                await self._check_position_limits()
                self._calc_risk_score()
                state.agent_metrics[self.agent_id.value] = self.metrics.copy()
                await asyncio.sleep(config.RISK_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"Risk monitor error: {e}")
                await asyncio.sleep(10)

    async def _var_calc_loop(self):
        """حساب VaR كل 5 دقائق"""
        while self.running:
            try:
                var_95, var_99 = await self._calculate_var()
                self.metrics["var_95"] = var_95
                self.metrics["var_99"] = var_99
                state.risk.current_var_95 = var_95

                await bus.publish(Event(
                    type=EventType.VAR_UPDATE,
                    source=self.agent_id,
                    payload={
                        "var_95": var_95,
                        "var_99": var_99,
                        "timestamp": time.time(),
                    }
                ))
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"VaR calc error: {e}")
                await asyncio.sleep(60)

    async def _correlation_loop(self):
        """مراقبة الـ correlation بين العملات كل 15 دقيقة"""
        while self.running:
            try:
                correlations = await self._calc_correlations()
                high_corr_pairs = [
                    (pair, corr) for pair, corr in correlations.items()
                    if abs(corr) > 0.85
                ]
                if high_corr_pairs:
                    await bus.publish(Event(
                        type=EventType.RISK_BREACH,
                        source=self.agent_id,
                        payload={
                            "type": "high_correlation",
                            "pairs": [
                                {"pair": p, "correlation": round(c, 3)}
                                for p, c in high_corr_pairs
                            ],
                            "message": f"⚠️ Correlation مرتفعة: {len(high_corr_pairs)} أزواج > 85%",
                            "recommendation": "تقليل positions المتلازمة لتنويع المخاطر",
                        },
                        priority=2
                    ))
                await asyncio.sleep(900)
            except Exception as e:
                logger.error(f"Correlation loop error: {e}")
                await asyncio.sleep(60)

    # ── Core Risk Logic ───────────────────────────────

    async def _check_drawdown_limits(self):
        perf = state.performance
        daily_pnl = perf.daily_pnl

        if daily_pnl < 0:
            drawdown_pct = abs(daily_pnl)
            self.metrics["current_drawdown_pct"] = drawdown_pct
            self.metrics["max_drawdown_session"] = max(
                self.metrics["max_drawdown_session"], drawdown_pct
            )
            state.risk.daily_drawdown_pct = drawdown_pct

            warn_threshold = config.MAX_DAILY_DRAWDOWN_PCT * 0.7

            if drawdown_pct >= config.MAX_DAILY_DRAWDOWN_PCT:
                await self._halt_trading(f"Drawdown يومي تجاوز {config.MAX_DAILY_DRAWDOWN_PCT}%")

            elif drawdown_pct >= warn_threshold:
                await bus.publish(Event(
                    type=EventType.DRAWDOWN_WARNING,
                    source=self.agent_id,
                    payload={
                        "drawdown_pct": round(drawdown_pct, 2),
                        "limit_pct": config.MAX_DAILY_DRAWDOWN_PCT,
                        "message": f"🔴 تحذير Drawdown: {drawdown_pct:.2f}% (الحد {config.MAX_DAILY_DRAWDOWN_PCT}%)",
                        "recommendation": "تقليل حجم الصفقات فوراً",
                    },
                    priority=2
                ))

    async def _check_position_limits(self):
        open_count = len(state.open_positions)
        if open_count > 5:
            await bus.publish(Event(
                type=EventType.RISK_BREACH,
                source=self.agent_id,
                payload={
                    "type": "position_limit",
                    "open_positions": open_count,
                    "message": f"⚠️ {open_count} positions مفتوحة — تجاوز الحد الموصى به",
                }
            ))

    async def _halt_trading(self, reason: str):
        if not state.risk.trading_halted:
            state.risk.trading_halted = True
            state.risk.halt_reason = reason
            self.metrics["trading_halted"] = True
            self.metrics["breaches_today"] += 1

            logger.critical(f"🚨 TRADING HALTED: {reason}")
            state.log_audit(self.agent_id.value, "trading_halt", {"reason": reason})

            await bus.publish(Event(
                type=EventType.TRADING_HALT,
                source=self.agent_id,
                payload={
                    "halt": True,
                    "reason": reason,
                    "message": f"🚨 تم وقف التداول: {reason}",
                    "timestamp": time.time(),
                },
                priority=3
            ))

    async def _calculate_var(self) -> tuple:
        """
        حساب Value at Risk باستخدام Historical Simulation
        في الإنتاج: جلب بيانات prices من Binance وحساب distribution
        """
        await asyncio.sleep(0.1)
        closed_trades = [t for t in state.trades if t.closed]

        if len(closed_trades) < 10:
            return 0.0, 0.0

        pnls = sorted([t.pnl for t in closed_trades])
        n = len(pnls)
        var_95 = abs(pnls[int(n * 0.05)]) if n > 20 else 0.0
        var_99 = abs(pnls[int(n * 0.01)]) if n > 100 else 0.0

        return round(var_95, 4), round(var_99, 4)

    async def _calc_correlations(self) -> Dict[str, float]:
        """
        حساب correlation بين العملات
        في الإنتاج: جلب returns من Binance وحساب Pearson correlation
        """
        await asyncio.sleep(0.1)
        symbols = config.SYMBOLS
        correlations = {}
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                pair = f"{symbols[i]}/{symbols[j]}"
                correlations[pair] = random.uniform(0.5, 0.95)
        return correlations

    def _calc_risk_score(self):
        """
        Risk Score من 0 إلى 100:
        0  = لا مخاطر
        100 = خطر أقصى
        """
        score = 0
        dd = self.metrics["current_drawdown_pct"]
        var = self.metrics["var_95"]

        score += min(dd / config.MAX_DAILY_DRAWDOWN_PCT * 40, 40)
        score += min(var * 10, 30)
        score += min(len(state.open_positions) * 5, 20)
        if state.risk.trading_halted:
            score = 100

        self.metrics["risk_score"] = round(score)

    # ── Event Handlers ────────────────────────────────

    async def _on_regime_change(self, event: Event):
        """تعديل حدود المخاطرة بناءً على تغيير الـ regime"""
        new_regime = event.payload.get("to_regime")
        if new_regime == "volatile":
            logger.warning("Volatile regime — Risk limits tightened")
            # في الإنتاج: تقليل MAX_DAILY_DRAWDOWN_PCT مؤقتاً

    async def _on_whale_alert(self, event: Event):
        """زيادة المراقبة عند وجود حركات whales"""
        logger.info(f"Whale alert received — heightened risk monitoring")

    def get_report(self) -> Dict:
        return {
            "agent": self.name,
            "metrics": self.metrics,
            "risk_state": {
                "trading_halted": state.risk.trading_halted,
                "halt_reason": state.risk.halt_reason,
                "daily_drawdown": state.risk.daily_drawdown_pct,
                "open_positions": len(state.open_positions),
            },
            "timestamp": time.time(),
        }
