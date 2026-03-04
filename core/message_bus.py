"""
Message Bus — قناة التواصل بين الـ agents
كل agent يرسل ويستقبل events من هنا
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict, deque


class AgentID(str, Enum):
    EXECUTION   = "execution_quality"
    MARKET      = "market_intelligence"
    RISK        = "risk_management"
    AUDIT       = "audit_backtesting"
    META        = "meta_supervisor"
    ORCHESTRATOR = "orchestrator"


class EventType(str, Enum):
    # Execution
    SLIPPAGE_ALERT      = "slippage_alert"
    LIQUIDITY_WARNING   = "liquidity_warning"
    ORDER_QUALITY       = "order_quality"

    # Market Intelligence
    SENTIMENT_UPDATE    = "sentiment_update"
    ONCHAIN_SIGNAL      = "onchain_signal"
    MACRO_EVENT         = "macro_event"
    REGIME_CHANGE       = "regime_change"

    # Risk
    RISK_BREACH         = "risk_breach"
    VAR_UPDATE          = "var_update"
    DRAWDOWN_WARNING    = "drawdown_warning"
    TRADING_HALT        = "trading_halt"

    # Audit
    BACKTEST_RESULT     = "backtest_result"
    PARAM_SUGGESTION    = "param_suggestion"
    AUDIT_LOG           = "audit_log"

    # Performance
    PERFORMANCE_UPDATE  = "performance_update"

    # Meta
    AGENT_HEALTH        = "agent_health"
    AGENT_DOWN          = "agent_down"

    # Orchestrator
    DAILY_REPORT        = "daily_report"
    TELEGRAM_ALERT      = "telegram_alert"


@dataclass
class Event:
    type: EventType
    source: AgentID
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # 1=normal, 2=high, 3=critical

    def to_dict(self):
        d = asdict(self)
        d['type'] = self.type.value
        d['source'] = self.source.value
        return d


class MessageBus:
    """
    In-memory pub/sub bus لربط الـ agents ببعض
    """
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._history: deque = deque(maxlen=1000)
        self._agent_heartbeats: Dict[AgentID, float] = {}
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: EventType, handler: Callable):
        self._subscribers[event_type].append(handler)

    async def publish(self, event: Event):
        async with self._lock:
            self._history.append(event)

        handlers = self._subscribers.get(event.type, [])
        tasks = [asyncio.create_task(h(event)) for h in handlers if asyncio.iscoroutinefunction(h)]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def heartbeat(self, agent_id: AgentID):
        self._agent_heartbeats[agent_id] = time.time()

    def get_agent_status(self) -> Dict[str, Any]:
        now = time.time()
        status = {}
        for agent in AgentID:
            last_beat = self._agent_heartbeats.get(agent, 0)
            elapsed = now - last_beat if last_beat else None
            status[agent.value] = {
                "last_heartbeat": last_beat,
                "seconds_ago": round(elapsed, 1) if elapsed else None,
                "healthy": elapsed is not None and elapsed < 180,
            }
        return status

    def get_recent_events(self, n: int = 50) -> List[Dict]:
        events = list(self._history)[-n:]
        return [e.to_dict() for e in reversed(events)]

    def get_events_by_type(self, event_type: EventType, n: int = 20) -> List[Dict]:
        filtered = [e for e in self._history if e.type == event_type]
        return [e.to_dict() for e in filtered[-n:]]


# Singleton
bus = MessageBus()
