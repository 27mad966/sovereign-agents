"""
Main — نقطة الإطلاق الرئيسية
"""

import asyncio
import logging
import signal
import uvicorn
from contextlib import asynccontextmanager

from agents.agent1_execution import ExecutionQualityAgent
from agents.agent2_market import MarketIntelligenceAgent
from agents.agent3_risk import RiskManagementAgent
from agents.agent4_audit import AuditBacktestAgent
from agents.agent5_meta import MetaSupervisorAgent
from agents.orchestrator import Orchestrator
from core.daily_report import DailyReportEngine
from core.auto_optimizer import AutoOptimizer
from api_server import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Main")


async def run_all_agents():
    agents = [
        ExecutionQualityAgent(),
        MarketIntelligenceAgent(),
        RiskManagementAgent(),
        AuditBacktestAgent(),
        MetaSupervisorAgent(),
        Orchestrator(),
    ]

    daily_report  = DailyReportEngine()
    auto_optimizer = AutoOptimizer()

    logger.info("🚀 Sovereign Trading System — LAUNCHING")
    logger.info("=" * 50)
    for a in agents:
        logger.info(f"   ► {a.name}")
    logger.info("   ► Daily Report Engine")
    logger.info("   ► Auto-Optimizer")
    logger.info("=" * 50)

    tasks = [asyncio.create_task(agent.start()) for agent in agents]
    tasks.append(asyncio.create_task(daily_report.start()))
    tasks.append(asyncio.create_task(auto_optimizer.start()))

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _shutdown():
        logger.info("🛑 Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass  # Windows

    await stop_event.wait()

    for agent in agents:
        await agent.stop()
    await daily_report.stop()
    await auto_optimizer.stop()
    for task in tasks:
        task.cancel()

    logger.info("✅ All agents stopped gracefully")


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(run_all_agents())
    yield
    task.cancel()

app.router.lifespan_context = lifespan

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )