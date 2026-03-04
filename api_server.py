"""
FastAPI Server — REST API للـ Dashboard
يوفر endpoints للـ Dashboard الويب
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import time
from typing import List

from core.state_store import state
from core.message_bus import bus, EventType

app = FastAPI(title="Sovereign Trading System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket clients for live dashboard
ws_clients: List[WebSocket] = []


@app.get("/api/summary")
async def get_summary():
    return {
        **state.get_summary(),
        "agent_health": bus.get_agent_status(),
        "timestamp": time.time(),
    }

@app.get("/api/performance")
async def get_performance():
    p = state.performance
    return {
        "total_pnl": p.total_pnl,
        "daily_pnl": p.daily_pnl,
        "win_rate": p.win_rate,
        "total_trades": p.total_trades,
        "winning_trades": p.winning_trades,
        "max_drawdown": p.max_drawdown,
        "sharpe": p.sharpe,
        "trades_history": [
            {"pnl": t.pnl, "symbol": t.symbol, "timestamp": t.timestamp}
            for t in state.trades[-100:]
        ],
    }

@app.get("/api/risk")
async def get_risk():
    return {
        "trading_halted": state.risk.trading_halted,
        "halt_reason": state.risk.halt_reason,
        "daily_drawdown_pct": state.risk.daily_drawdown_pct,
        "var_95": state.risk.current_var_95,
        "open_positions": {
            sym: {"entry": t.entry_price, "side": t.side}
            for sym, t in state.open_positions.items()
        },
        "agent_metrics": state.agent_metrics.get("risk_management", {}),
    }

@app.get("/api/market")
async def get_market():
    return {
        "regimes": {
            sym: {
                "regime": r.regime,
                "sentiment": r.sentiment,
                "volatility": r.volatility,
                "volume_zscore": r.volume_zscore,
                "timestamp": r.timestamp,
            }
            for sym, r in state.market_regimes.items()
        },
        "orderbook": state.orderbook_depth,
    }

@app.get("/api/agents")
async def get_agents():
    return {
        "health": bus.get_agent_status(),
        "metrics": state.agent_metrics,
    }

@app.get("/api/events")
async def get_events(n: int = 50):
    return {"events": bus.get_recent_events(n)}

@app.get("/api/audit")
async def get_audit(n: int = 100):
    return {"log": state.audit_log[-n:]}

@app.get("/api/params")
async def get_params():
    return {
        "current": state.strategy_params,
        "history": state.param_history[-20:],
    }

@app.post("/api/params")
async def update_params(params: dict):
    """تحديث يدوي للبارامترات"""
    state.update_params(params, reason="Manual override via Dashboard")
    return {"status": "updated", "params": state.strategy_params}

@app.get("/api/execution")
async def get_execution():
    return {
        "metrics": state.agent_metrics.get("execution_quality", {}),
        "orderbook": state.orderbook_depth,
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket للـ Dashboard live updates"""
    await websocket.accept()
    ws_clients.append(websocket)
    try:
        while True:
            snapshot = state.agent_metrics.get("dashboard_snapshot", {})
            await websocket.send_text(json.dumps(snapshot))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        ws_clients.remove(websocket)

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}
