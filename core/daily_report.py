"""
Daily Analysis Report
تقرير يومي شامل يُرسل على Telegram كل 24 ساعة

يشمل:
- جودة الإشارات (Win Rate, Sharpe) لكل عملة
- أفضل بارامترات للسوق الحالي
- تقييم مخاطر الاستراتيجية
- مقارنة أداء BTC vs ETH vs SOL
"""

import asyncio
import time
import logging
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

from core.strategy_engine import SovereignStrategy, StrategyParams, SignalResult
from config.settings import config

logger = logging.getLogger("DailyReport")


@dataclass
class SymbolAnalysis:
    symbol: str
    total_signals: int
    win_rate: float
    avg_gain: float
    avg_loss: float
    sharpe: float
    max_drawdown: float
    best_params: Dict
    risk_score: float
    regime: str
    recommendation: str


class DailyReportEngine:
    def __init__(self):
        self.running = False
        self.REPORT_INTERVAL = 86400  # 24 ساعة

    async def start(self):
        self.running = True
        logger.info("✅ Daily Report Engine started")
        # أرسل تقرير أول عند البدء بعد دقيقة
        await asyncio.sleep(60)
        while self.running:
            try:
                await self.generate_and_send()
                await asyncio.sleep(self.REPORT_INTERVAL)
            except Exception as e:
                logger.error(f"Daily report error: {e}")
                await asyncio.sleep(3600)

    async def stop(self):
        self.running = False

    # ── Core ──────────────────────────────────────────

    async def generate_and_send(self):
        logger.info("📊 Generating daily report...")
        analyses = {}
        for symbol in config.SYMBOLS:
            analysis = await self._analyze_symbol(symbol)
            if analysis:
                analyses[symbol] = analysis

        if not analyses:
            return

        report = self._build_report(analyses)
        await self._send_telegram(report)
        logger.info("✅ Daily report sent to Telegram")

    async def _analyze_symbol(self, symbol: str) -> Optional[SymbolAnalysis]:
        """تحليل شامل لرمز واحد على بيانات 90 يوم"""
        try:
            df = await self._fetch_ohlcv(symbol, "1h", 500)  # ~21 يوم
            if df is None or len(df) < 100:
                return None

            # Backtest على البيانات
            results = self._run_backtest(df, StrategyParams())
            best_params = self._optimize_params(df)
            regime = self._detect_regime(df)
            risk_score = self._calc_risk_score(results)
            recommendation = self._generate_recommendation(results, regime, risk_score)

            return SymbolAnalysis(
                symbol=symbol,
                total_signals=results["total_signals"],
                win_rate=results["win_rate"],
                avg_gain=results["avg_gain"],
                avg_loss=results["avg_loss"],
                sharpe=results["sharpe"],
                max_drawdown=results["max_drawdown"],
                best_params=best_params,
                risk_score=risk_score,
                regime=regime,
                recommendation=recommendation,
            )
        except Exception as e:
            logger.error(f"Analysis error {symbol}: {e}")
            return None

    def _run_backtest(self, df: pd.DataFrame, params: StrategyParams) -> Dict:
        """Backtest كامل على البيانات المتاحة"""
        strategy = SovereignStrategy(params)
        trades = []
        pos_state = 0
        entry_price = 0.0
        peak_hit = False

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # حساب المؤشرات مرة واحدة
        st_vals, st_dir = strategy.calc_supertrend(high, low, close, params.st_len, params.st_mult)
        squeeze = strategy.calc_squeeze(high, low, close, params.bb_period, params.bb_mult, params.kc_period, params.kc_mult)
        mfi = strategy.calc_mfi(high, low, close, volume, params.mfi_period)
        lr = strategy.calc_linreg(close, high, low, params.lr_period)

        for i in range(50, len(close)):
            is_bullish = st_dir[i] < 0
            is_bearish = st_dir[i] > 0
            squeeze_fires = bool(squeeze[i])
            mfi_val = mfi[i]
            lr_val = lr[i]

            # BB للخروج
            bb_m = np.mean(close[max(0,i-params.bb_period):i+1])
            bb_std = np.std(close[max(0,i-params.bb_period):i+1], ddof=0)
            bb_u = bb_m + params.bb_mult * bb_std
            bb_l = bb_m - params.bb_mult * bb_std

            # شروط الخروج
            if pos_state == 1:
                sl_hit = close[i] < entry_price * (1 - params.stop_loss_pct)
                tp_peak = params.use_peak_sell and not peak_hit and close[i] > bb_u and mfi_val >= params.mfi_limit
                st_flip = is_bearish and close[i] < entry_price

                if sl_hit or tp_peak or st_flip:
                    pnl = (close[i] - entry_price) / entry_price * 100
                    exit_type = "SL" if sl_hit else "PEAK" if tp_peak else "FLIP"
                    trades.append({"pnl": pnl, "exit": exit_type, "entry": entry_price, "exit_price": close[i]})
                    pos_state = 0
                    peak_hit = tp_peak

            # شروط الدخول
            if pos_state == 0:
                buy = is_bullish and squeeze_fires and lr_val > 0 and mfi_val > params.mfi_min_buy
                if buy:
                    pos_state = 1
                    entry_price = close[i]
                    peak_hit = False

        if not trades:
            return {"total_signals": 0, "win_rate": 0, "avg_gain": 0, "avg_loss": 0,
                    "sharpe": 0, "max_drawdown": 0}

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Sharpe
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if len(pnls) > 1 else 1
        sharpe = mean_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0

        # Max Drawdown
        cum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        drawdown = np.max(peak - cum) if len(cum) > 0 else 0

        return {
            "total_signals": len(trades),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "avg_gain": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(drawdown, 2),
            "trades": trades,
        }

    def _optimize_params(self, df: pd.DataFrame) -> Dict:
        """
        Walk-forward optimization — يجرب مجموعة بارامترات ويختار الأفضل
        """
        best_sharpe = -999
        best_params = {}

        param_grid = [
            {"mfi_min_buy": 45, "mfi_limit": 70, "st_mult": 2.5, "stop_loss_pct": 0.025},
            {"mfi_min_buy": 50, "mfi_limit": 75, "st_mult": 3.0, "stop_loss_pct": 0.030},
            {"mfi_min_buy": 55, "mfi_limit": 80, "st_mult": 3.5, "stop_loss_pct": 0.035},
            {"mfi_min_buy": 45, "mfi_limit": 75, "st_mult": 3.0, "stop_loss_pct": 0.020},
            {"mfi_min_buy": 50, "mfi_limit": 70, "st_mult": 2.0, "stop_loss_pct": 0.030},
        ]

        for grid in param_grid:
            try:
                p = StrategyParams(**grid)
                results = self._run_backtest(df, p)
                if results["sharpe"] > best_sharpe and results["total_signals"] >= 3:
                    best_sharpe = results["sharpe"]
                    best_params = {**grid, "sharpe": round(best_sharpe, 2),
                                   "win_rate": round(results["win_rate"], 1)}
            except Exception:
                continue

        return best_params if best_params else {
            "mfi_min_buy": 50, "mfi_limit": 75, "st_mult": 3.0, "stop_loss_pct": 0.030
        }

    def _detect_regime(self, df: pd.DataFrame) -> str:
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        vol = float(np.mean(high[-14:] - low[-14:]) / np.mean(close[-14:]) * 100)
        last_20 = close[-20:]
        trend = (last_20[-1] - last_20[0]) / last_20[0] * 100
        if vol > 4.0:
            return "volatile ⚡"
        elif trend > 3:
            return "trending_up 📈"
        elif trend < -3:
            return "trending_down 📉"
        else:
            return "ranging ↔️"

    def _calc_risk_score(self, results: Dict) -> float:
        """0 = آمن جداً، 100 = خطر"""
        score = 0
        win_rate = results.get("win_rate", 0)
        sharpe = results.get("sharpe", 0)
        dd = results.get("max_drawdown", 0)

        score += max(0, (50 - win_rate))        # Win Rate أقل من 50% يرفع الخطر
        score += max(0, (1.0 - sharpe) * 20)   # Sharpe أقل من 1 يرفع الخطر
        score += min(dd * 2, 30)                # Drawdown يرفع الخطر

        return round(min(score, 100), 1)

    def _generate_recommendation(self, results: Dict, regime: str, risk_score: float) -> str:
        win_rate = results.get("win_rate", 0)
        sharpe = results.get("sharpe", 0)
        signals = results.get("total_signals", 0)

        if signals < 3:
            return "⚪ بيانات غير كافية للتقييم"
        elif risk_score > 70:
            return "🔴 استراتيجية عالية المخاطر في هذا السوق — قلل حجم الصفقات"
        elif win_rate > 60 and sharpe > 1.5:
            return "🟢 الاستراتيجية ممتازة — شروط مثالية للتداول"
        elif win_rate > 50 and sharpe > 1.0:
            return "🟡 الاستراتيجية جيدة — تداول بحجم معتدل"
        elif "volatile" in regime:
            return "🟠 سوق متقلب — انتظر Squeeze واضح قبل الدخول"
        else:
            return "🟡 أداء مقبول — راقب إشارات HTF قبل الدخول"

    # ── Report Builder ────────────────────────────────

    def _build_report(self, analyses: Dict[str, SymbolAnalysis]) -> str:
        now = time.strftime("%Y-%m-%d %H:%M", time.localtime())

        lines = [
            f"📊 *التقرير اليومي — Sovereign System*",
            f"🕐 {now}\n",
        ]

        # مقارنة الأداء
        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        lines.append("📈 *مقارنة الأداء*\n")

        sorted_by_sharpe = sorted(analyses.values(), key=lambda x: x.sharpe, reverse=True)
        for a in sorted_by_sharpe:
            sym = a.symbol.replace("USDT", "")
            medal = "🥇" if sorted_by_sharpe.index(a) == 0 else "🥈" if sorted_by_sharpe.index(a) == 1 else "🥉"
            lines.append(
                f"{medal} *{sym}* — Win Rate: `{a.win_rate:.1f}%` | Sharpe: `{a.sharpe:.2f}` | "
                f"Signals: `{a.total_signals}` | DD: `{a.max_drawdown:.1f}%`"
            )

        lines.append("")

        # تقييم كل عملة
        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        lines.append("🔍 *التحليل التفصيلي*\n")

        for a in sorted_by_sharpe:
            sym = a.symbol.replace("USDT", "")
            risk_emoji = "🟢" if a.risk_score < 30 else "🟡" if a.risk_score < 60 else "🔴"
            bp = a.best_params

            lines.append(f"*{sym}* ({a.regime})")
            lines.append(f"  • متوسط ربح: `{a.avg_gain:.2f}%` | متوسط خسارة: `{a.avg_loss:.2f}%`")
            lines.append(f"  • مخاطر: {risk_emoji} `{a.risk_score}/100`")

            if bp:
                lines.append(
                    f"  • أفضل بارامترات: MFI `{bp.get('mfi_min_buy','—')}` | "
                    f"ST `{bp.get('st_mult','—')}` | SL `{bp.get('stop_loss_pct',0)*100:.1f}%`"
                )
            lines.append(f"  • {a.recommendation}\n")

        # توصية إجمالية
        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        best = sorted_by_sharpe[0] if sorted_by_sharpe else None
        worst = sorted_by_sharpe[-1] if sorted_by_sharpe else None

        if best:
            lines.append(f"✅ *الأفضل اليوم:* {best.symbol.replace('USDT','')} — {best.recommendation}")
        if worst and worst != best:
            lines.append(f"⚠️ *الأضعف اليوم:* {worst.symbol.replace('USDT','')} — تجنب أو قلل الحجم")

        lines.append(f"\n_Sovereign Trading System — تقرير آلي_")

        return "\n".join(lines)

    # ── Binance + Telegram ────────────────────────────

    async def _fetch_ohlcv(self, symbol: str, interval: str, limit: int):
        try:
            from binance.client import Client
            client = Client(config.BINANCE_API_KEY, config.BINANCE_SECRET, testnet=True)
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'time','open','high','low','close','volume',
                'close_time','quote_vol','trades','taker_buy_base','taker_buy_quote','ignore'
            ])
            for col in ['open','high','low','close','volume']:
                df[col] = df[col].astype(float)
            df.index.name = symbol
            return df
        except Exception as e:
            logger.error(f"Fetch error {symbol}: {e}")
            return None

    async def _send_telegram(self, message: str):
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            logger.warning("Telegram not configured")
            return
        try:
            import requests
            url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, json=payload, timeout=15)
            )
            if response.status_code == 200:
                logger.info("✅ Telegram report sent")
            else:
                logger.error(f"Telegram error: {response.status_code} — {response.text}")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")