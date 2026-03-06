"""
Orchestrator — قائد الفريق
يجمع تقارير كل الـ agents ويرسلها لـ Telegram + Dashboard

مسؤوليات:
- استقبال events من كل الـ agents
- إرسال تنبيهات فورية على Telegram
- تجميع تقرير يومي شامل
- تغذية الـ Dashboard بالبيانات
- Priority routing: critical events → فوري، normal → مجمّع
"""

import asyncio
import time
import logging
import json
import aiohttp
from typing import Dict, List
from collections import deque

from core.message_bus import bus, AgentID, Event, EventType
from core.state_store import state
from config.settings import config

logger = logging.getLogger("Orchestrator")


class Orchestrator:
    def __init__(self):
        self.agent_id = AgentID.ORCHESTRATOR
        self.name = "Orchestrator"
        self.running = False

        # Buffer للـ events العادية (ترسل مجمّعة)
        self._normal_buffer: deque = deque(maxlen=100)
        self._last_batch_send = time.time()
        self.BATCH_INTERVAL = 300  # كل 5 دقائق

        # Subscribe to critical events (ترسل فوراً)
        critical_events = [
            EventType.TRADING_HALT,
            EventType.AGENT_DOWN,
            EventType.SLIPPAGE_ALERT,
            EventType.DRAWDOWN_WARNING,
            EventType.REGIME_CHANGE,
        ]
        for et in critical_events:
            bus.subscribe(et, self._handle_critical)

        # Subscribe to normal events (تُجمّع)
        normal_events = [
            EventType.SENTIMENT_UPDATE,
            EventType.VAR_UPDATE,
            EventType.ORDER_QUALITY,
            EventType.BACKTEST_RESULT,
            EventType.PARAM_SUGGESTION,
            EventType.AGENT_HEALTH,
            EventType.ONCHAIN_SIGNAL,
        ]
        for et in normal_events:
            bus.subscribe(et, self._handle_normal)

    async def start(self):
        self.running = True
        logger.info(f"✅ {self.name} started")
        await asyncio.gather(
            self._batch_sender_loop(),
            self._daily_report_loop(),
            self._dashboard_updater_loop(),
        )

    # ── Event Handlers ────────────────────────────────

    async def _handle_critical(self, event: Event):
        """إرسال فوري للـ Telegram"""
        msg = event.payload.get("message", str(event.payload))
        priority_prefix = {1: "ℹ️", 2: "⚠️", 3: "🚨"}.get(event.priority, "📌")
        await self._send_telegram(f"{priority_prefix} *{event.type.value.upper()}*\n{msg}")

    async def _handle_normal(self, event: Event):
        """تخزين في buffer للإرسال المجمّع"""
        self._normal_buffer.append(event)

    # ── Loops ─────────────────────────────────────────

    async def _batch_sender_loop(self):
        """إرسال الـ events المجمّعة كل 5 دقائق"""
        while self.running:
            try:
                bus.heartbeat(self.agent_id)
                now = time.time()
                if (now - self._last_batch_send >= self.BATCH_INTERVAL
                        and self._normal_buffer):
                    await self._send_batch_report()
                    self._last_batch_send = now
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Batch sender error: {e}")
                await asyncio.sleep(30)

    async def _daily_report_loop(self):
        """تقرير يومي شامل كل 24 ساعة"""
        await asyncio.sleep(3600)
        while self.running:
            try:
                await self._send_daily_report()
                await asyncio.sleep(86400)
            except Exception as e:
                logger.error(f"Daily report error: {e}")
                await asyncio.sleep(3600)

    async def _dashboard_updater_loop(self):
        """تحديث state للـ Dashboard كل 30 ثانية"""
        while self.running:
            try:
                # الـ Dashboard يقرأ من state مباشرة عبر API
                summary = state.get_summary()
                summary["agent_health"] = bus.get_agent_status()
                summary["recent_events"] = bus.get_recent_events(20)
                summary["timestamp"] = time.time()
                # حفظ في state للـ API
                state.agent_metrics["dashboard_snapshot"] = summary
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Dashboard updater error: {e}")
                await asyncio.sleep(30)

    # ── Telegram ──────────────────────────────────────

    async def _send_telegram(self, message: str):
        """إرسال رسالة Telegram"""
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            logger.debug(f"[TELEGRAM] {message}")
            return

        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
        }
        try:
            import requests
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, json=payload, timeout=10)
            )
            if response.status_code != 200:
                logger.error(f"Telegram error: {response.status_code}")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    async def _send_batch_report(self):
        """ملخص الـ events المجمّعة"""
        events = list(self._normal_buffer)
        self._normal_buffer.clear()

        if not events:
            return

        # تصنيف
        by_type: Dict[str, List] = {}
        for e in events:
            t = e.type.value
            by_type.setdefault(t, []).append(e)

        lines = [f"📊 *تحديث دوري — {len(events)} events*\n"]
        for etype, evs in by_type.items():
            last = evs[-1]
            msg = last.payload.get("message", "")
            if msg:
                lines.append(f"• {msg}")
            else:
                lines.append(f"• {etype}: {len(evs)} updates")

        await self._send_telegram("\n".join(lines))

    async def _send_daily_report(self):
        """التقرير اليومي الشامل"""
        summary = state.get_summary()
        perf = summary["performance"]
        risk = summary["risk"]
        health = bus.get_agent_status()

        healthy_agents = sum(1 for a in health.values() if a.get("healthy"))
        total_agents = len(health)

        msg = f"""
📈 *التقرير اليومي — Sovereign Trading System*

💰 *الأداء*
• PnL الإجمالي: `{perf['total_pnl']:+.2f}%`
• PnL اليوم: `{perf['daily_pnl']:+.2f}%`
• Win Rate: `{perf['win_rate']:.1f}%`
• الصفقات: `{perf['total_trades']}`
• Sharpe: `{perf['sharpe']:.2f}`
• Max Drawdown: `{perf['max_drawdown']:.2f}%`

🛡️ *المخاطر*
• الحالة: `{'🔴 موقوف' if risk['trading_halted'] else '🟢 نشط'}`
• VaR 95%: `{risk['var_95']:.3f}%`
• Positions مفتوحة: `{risk['open_positions']}`

🏥 *صحة النظام*
• Agents يعملون: `{healthy_agents}/{total_agents}`

⚙️ *البارامترات الحالية*
• MA: `{state.strategy_params.get('fast_ma')}/{state.strategy_params.get('slow_ma')}`
• RSI: `{state.strategy_params.get('rsi_oversold')}/{state.strategy_params.get('rsi_overbought')}`
• ATR Multiplier: `{state.strategy_params.get('atr_multiplier')}`
        """.strip()

        await self._send_telegram(msg)
        logger.info("Daily report sent")