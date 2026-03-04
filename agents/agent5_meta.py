"""
Agent 5 — Meta Supervisor
المراقب الأعلى — يراقب ويقيّم الـ agents الأخرى

مسؤوليات:
- مراقبة heartbeat لكل agent
- تقييم جودة تقارير كل agent
- اكتشاف agents متعطلة أو تعطي نتائج شاذة
- تنبيه فوري على Telegram إذا agent توقف
- تقرير صحة الفريق كل ساعتين
- محاولة إعادة تشغيل agent متعطل
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional

from core.message_bus import bus, AgentID, Event, EventType
from core.state_store import state
from config.settings import config

logger = logging.getLogger("MetaSupervisor")

AGENT_MAX_SILENCE = {
    AgentID.EXECUTION:   90,    # ثانية — يجب أن يرسل كل 30 ثانية
    AgentID.MARKET:      360,   # كل 5 دقائق
    AgentID.RISK:        120,   # كل دقيقة
    AgentID.AUDIT:       3700,  # كل ساعة
}


class MetaSupervisorAgent:
    def __init__(self):
        self.agent_id = AgentID.META
        self.name = "Meta Supervisor"
        self.running = False

        self.metrics = {
            "agents_monitored": len(AgentID) - 1,
            "alerts_sent": 0,
            "restarts_triggered": 0,
            "anomalies_detected": 0,
            "system_health_pct": 100.0,
        }

        self._agent_alert_sent: Dict[AgentID, bool] = {a: False for a in AgentID}
        self._agent_restart_count: Dict[AgentID, int] = {a: 0 for a in AgentID}
        self._agent_quality_scores: Dict[str, float] = {}

        # Subscribe to all events to evaluate quality
        for event_type in EventType:
            bus.subscribe(event_type, self._observe_event)

    async def start(self):
        self.running = True
        logger.info(f"✅ {self.name} started — watching all agents")
        await asyncio.gather(
            self._heartbeat_monitor_loop(),
            self._quality_eval_loop(),
            self._health_report_loop(),
        )

    async def stop(self):
        self.running = False

    # ── Loops ─────────────────────────────────────────

    async def _heartbeat_monitor_loop(self):
        """فحص heartbeat كل دقيقتين"""
        while self.running:
            try:
                bus.heartbeat(self.agent_id)
                agent_statuses = bus.get_agent_status()
                now = time.time()
                unhealthy = []

                for agent in [AgentID.EXECUTION, AgentID.MARKET, AgentID.RISK, AgentID.AUDIT]:
                    status = agent_statuses.get(agent.value, {})
                    last_beat = status.get("last_heartbeat", 0)
                    max_silence = AGENT_MAX_SILENCE.get(agent, 300)

                    if last_beat == 0:
                        # لم يبدأ بعد
                        continue

                    silence = now - last_beat
                    if silence > max_silence:
                        unhealthy.append({
                            "agent": agent.value,
                            "silence_seconds": round(silence),
                            "max_allowed": max_silence,
                        })

                        if not self._agent_alert_sent[agent]:
                            self._agent_alert_sent[agent] = True
                            self.metrics["alerts_sent"] += 1
                            await self._emit_agent_down(agent, silence)
                    else:
                        # Agent عاد — مسح التنبيه
                        if self._agent_alert_sent[agent]:
                            self._agent_alert_sent[agent] = False
                            logger.info(f"✅ {agent.value} recovered")
                            await bus.publish(Event(
                                type=EventType.AGENT_HEALTH,
                                source=self.agent_id,
                                payload={
                                    "agent": agent.value,
                                    "status": "recovered",
                                    "message": f"✅ {agent.value} عاد للعمل",
                                }
                            ))

                # حساب System Health
                total = len([a for a in AgentID if a != AgentID.META and a != AgentID.ORCHESTRATOR])
                healthy = total - len(unhealthy)
                self.metrics["system_health_pct"] = round(healthy / total * 100, 1)

                state.agent_metrics[self.agent_id.value] = self.metrics.copy()
                await asyncio.sleep(config.META_CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(30)

    async def _quality_eval_loop(self):
        """تقييم جودة مخرجات كل agent كل 10 دقائق"""
        while self.running:
            try:
                await self._evaluate_execution_quality()
                await self._evaluate_risk_quality()
                await self._evaluate_market_quality()
                await asyncio.sleep(600)
            except Exception as e:
                logger.error(f"Quality eval error: {e}")
                await asyncio.sleep(60)

    async def _health_report_loop(self):
        """تقرير شامل عن صحة الفريق كل ساعتين"""
        await asyncio.sleep(7200)
        while self.running:
            try:
                report = self._build_health_report()
                await bus.publish(Event(
                    type=EventType.AGENT_HEALTH,
                    source=self.agent_id,
                    payload=report
                ))
                state.log_audit(self.agent_id.value, "health_report", report)
                await asyncio.sleep(7200)
            except Exception as e:
                logger.error(f"Health report error: {e}")
                await asyncio.sleep(600)

    # ── Quality Evaluations ───────────────────────────

    async def _evaluate_execution_quality(self):
        """هل Execution Agent يرصد الـ slippage بدقة؟"""
        exec_metrics = state.agent_metrics.get(AgentID.EXECUTION.value, {})
        orders = exec_metrics.get("orders_analyzed", 0)
        score = exec_metrics.get("execution_score", 100)

        quality = 100.0
        if orders == 0:
            quality = 0.0  # لا يعمل
        elif score < 70:
            quality = 60.0  # أداء سيء
            self.metrics["anomalies_detected"] += 1
            await bus.publish(Event(
                type=EventType.AGENT_HEALTH,
                source=self.agent_id,
                payload={
                    "agent": AgentID.EXECUTION.value,
                    "issue": "execution_score_low",
                    "score": score,
                    "message": f"⚠️ Execution Agent: جودة تنفيذ منخفضة ({score:.0f}/100)",
                }
            ))

        self._agent_quality_scores[AgentID.EXECUTION.value] = quality

    async def _evaluate_risk_quality(self):
        """هل Risk Agent يراقب الـ drawdown بشكل صحيح؟"""
        risk_metrics = state.agent_metrics.get(AgentID.RISK.value, {})
        risk_score = risk_metrics.get("risk_score", 0)
        var = risk_metrics.get("var_95", 0)

        quality = 100.0
        if risk_score > 80 and not state.risk.trading_halted:
            # Risk مرتفع لكن لم يوقف التداول — مشكلة!
            quality = 40.0
            self.metrics["anomalies_detected"] += 1
            await bus.publish(Event(
                type=EventType.AGENT_HEALTH,
                source=self.agent_id,
                payload={
                    "agent": AgentID.RISK.value,
                    "issue": "missed_halt",
                    "risk_score": risk_score,
                    "message": f"🚨 Risk Agent: Risk Score = {risk_score} لكن لم يوقف التداول!",
                },
                priority=3
            ))

        self._agent_quality_scores[AgentID.RISK.value] = quality

    async def _evaluate_market_quality(self):
        """هل Market Intelligence Agent يحدّث الـ regimes؟"""
        last_regime_update = 0
        for regime in state.market_regimes.values():
            last_regime_update = max(last_regime_update, regime.timestamp)

        if last_regime_update > 0:
            age = time.time() - last_regime_update
            if age > 900:  # أكثر من 15 دقيقة
                quality = 30.0
                self.metrics["anomalies_detected"] += 1
                await bus.publish(Event(
                    type=EventType.AGENT_HEALTH,
                    source=self.agent_id,
                    payload={
                        "agent": AgentID.MARKET.value,
                        "issue": "stale_regime_data",
                        "age_minutes": round(age / 60, 1),
                        "message": f"⚠️ Market Agent: بيانات الـ regime قديمة ({age/60:.0f} دقيقة)",
                    }
                ))
            else:
                quality = 100.0
            self._agent_quality_scores[AgentID.MARKET.value] = quality

    # ── Alert & Report ────────────────────────────────

    async def _emit_agent_down(self, agent: AgentID, silence: float):
        logger.critical(f"🚨 AGENT DOWN: {agent.value} silent for {silence:.0f}s")
        await bus.publish(Event(
            type=EventType.AGENT_DOWN,
            source=self.agent_id,
            payload={
                "agent": agent.value,
                "silence_seconds": round(silence),
                "restart_count": self._agent_restart_count[agent],
                "message": (
                    f"🚨 {agent.value} متوقف منذ {silence/60:.1f} دقيقة — "
                    f"إعادة تشغيل #{self._agent_restart_count[agent] + 1}"
                ),
            },
            priority=3
        ))
        self._agent_restart_count[agent] += 1
        self.metrics["restarts_triggered"] += 1

    def _build_health_report(self) -> Dict:
        statuses = bus.get_agent_status()
        return {
            "system_health_pct": self.metrics["system_health_pct"],
            "agents": statuses,
            "quality_scores": self._agent_quality_scores,
            "anomalies_detected": self.metrics["anomalies_detected"],
            "alerts_sent": self.metrics["alerts_sent"],
            "message": (
                f"🏥 تقرير صحة النظام: {self.metrics['system_health_pct']}% — "
                f"{self.metrics['anomalies_detected']} شذوذات"
            ),
            "timestamp": time.time(),
        }

    async def _observe_event(self, event: Event):
        """يراقب كل event لاكتشاف أنماط غير طبيعية"""
        # كشف flood: نفس الـ event يتكرر أكثر من 10 مرات في دقيقة
        recent = bus.get_recent_events(50)
        same_type = [e for e in recent if e["type"] == event.type.value]
        if len(same_type) > 10:
            logger.warning(f"Event flood detected: {event.type.value}")

    def get_report(self) -> Dict:
        return {
            "agent": self.name,
            "metrics": self.metrics,
            "quality_scores": self._agent_quality_scores,
            "agent_statuses": bus.get_agent_status(),
            "timestamp": time.time(),
        }
