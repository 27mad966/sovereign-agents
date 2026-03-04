"""
Main — نقطة الإطلاق الرئيسية
يشغل كل الـ agents بالتوازي
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
from api_server import app

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Main")


async def run_all_agents():
    """تشغيل كل الـ agents بالتوازي"""
    agents = [
        ExecutionQualityAgent(),
        MarketIntelligenceAgent(),
        RiskManagementAgent(),
        AuditBacktestAgent(),
        MetaSupervisorAgent(),
        Orchestrator(),
    ]

    logger.info("🚀 Sovereign Trading System — LAUNCHING")
    logger.info("=" * 50)
    for a in agents:
        logger.info(f"   ► {a.name}")
    logger.info("=" * 50)

    # تشغيل كل agent في task منفصل
    tasks = [asyncio.create_task(agent.start()) for agent in agents]

    # Graceful shutdown
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _shutdown():
        logger.info("🛑 Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    await stop_event.wait()

    # إيقاف
    for agent in agents:
        await agent.stop()
    for task in tasks:
        task.cancel()

    logger.info("✅ All agents stopped gracefully")


@asynccontextmanager
async def lifespan(app):
    # Start agents in background when FastAPI starts
    task = asyncio.create_task(run_all_agents())
    yield
    task.cancel()

# Attach lifespan to FastAPI
app.router.lifespan_context = lifespan


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
