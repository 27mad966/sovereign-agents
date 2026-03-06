"""
Auto-Optimizer — المحسّن التلقائي للاستراتيجية
يقترح فقط الإعدادات الموجودة في لوحة التحكم في المؤشر
ويصلح مشكلة Sharpe السلبي
"""

import asyncio
import time
import logging
import requests
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.strategy_engine import SovereignStrategy, StrategyParams
from config.settings import config

logger = logging.getLogger("AutoOptimizer")


@dataclass
class OptimizationResult:
    symbol: str
    old_params: Dict
    new_params: Dict
    old_score: float
    new_score: float
    old_winrate: float
    new_winrate: float
    old_drawdown: float
    new_drawdown: float
    improvement_pct: float
    applied: bool
    timestamp: float


# الإعدادات الحالية المحفوظة
CURRENT_PARAMS: Dict[str, Dict] = {}


class AutoOptimizer:
    def __init__(self):
        self.running = False
        self.OPTIMIZE_INTERVAL = 86400
        self.AUTO_APPLY_THRESHOLD = 0.10
        self.history: List[OptimizationResult] = []

        # فقط الإعدادات الموجودة في لوحة التحكم
        self.PARAM_GRID = {
            "mfi_period":    [10, 14, 20],
            "mfi_min_buy":   [45, 50, 55, 60],
            "bb_period":     [15, 20, 25],
            "bb_mult":       [1.8, 2.0, 2.2],
            "kc_period":     [15, 20, 25],
            "kc_mult":       [1.2, 1.5, 1.8],
            "lr_period":     [15, 20, 25],
            "st_mult":       [2.5, 3.0, 3.5],
            "st_len":        [8, 10, 12],
            "mfi_limit":     [70, 75, 80],
            "stop_loss_pct": [0.02, 0.025, 0.03, 0.035],
        }

    async def start(self):
        self.running = True
        logger.info("✅ Auto-Optimizer started")
        await asyncio.sleep(120)
        while self.running:
            try:
                await self.run_optimization()
                await asyncio.sleep(self.OPTIMIZE_INTERVAL)
            except Exception as e:
                logger.error(f"Optimizer error: {e}")
                await asyncio.sleep(3600)

    async def stop(self):
        self.running = False

    async def run_optimization(self):
        logger.info("🔧 بدء عملية الـ Optimization على 30 يوم...")
        all_results = []

        for symbol in config.SYMBOLS:
            logger.info(f"   تحليل {symbol}...")
            df = await self._fetch_ohlcv(symbol, "1h", 720)
            if df is None or len(df) < 200:
                logger.warning(f"بيانات غير كافية لـ {symbol}")
                continue

            result = await self._optimize_symbol(symbol, df)
            if result:
                all_results.append(result)
                self.history.append(result)

        if all_results:
            await self._send_report(all_results)

    async def _optimize_symbol(self, symbol: str, df: pd.DataFrame) -> Optional[OptimizationResult]:
        n = len(df)
        train_end = int(n * 0.7)
        train_df = df.iloc[:train_end]
        test_df  = df.iloc[train_end:]

        if len(test_df) < 50:
            return None

        # أداء الإعدادات الحالية
        current_p = CURRENT_PARAMS.get(symbol, {})
        current_params = StrategyParams(**{k: v for k, v in current_p.items() if k in self.PARAM_GRID}) if current_p else StrategyParams()
        current_perf = self._backtest(df, current_params)

        # البحث عن أفضل إعدادات
        best_params, best_perf = await self._grid_search(train_df, test_df)

        if best_params is None or best_perf["score"] <= 0:
            best_params = current_params
            best_perf = current_perf

        improvement = 0.0
        if current_perf["score"] > 0:
            improvement = (best_perf["score"] - current_perf["score"]) / current_perf["score"]
        elif best_perf["score"] > 0:
            improvement = 1.0

        applied = False
        if improvement >= self.AUTO_APPLY_THRESHOLD and best_perf["total_signals"] >= 5:
            CURRENT_PARAMS[symbol] = {
                "mfi_period":    best_params.mfi_period,
                "mfi_min_buy":   best_params.mfi_min_buy,
                "bb_period":     best_params.bb_period,
                "bb_mult":       best_params.bb_mult,
                "kc_period":     best_params.kc_period,
                "kc_mult":       best_params.kc_mult,
                "lr_period":     best_params.lr_period,
                "st_mult":       best_params.st_mult,
                "st_len":        best_params.st_len,
                "mfi_limit":     best_params.mfi_limit,
                "stop_loss_pct": best_params.stop_loss_pct,
            }
            applied = True
            logger.info(f"✅ Auto-applied for {symbol} — improvement: {improvement:.1%}")

        return OptimizationResult(
            symbol=symbol,
            old_params={
                "mfi_period": current_params.mfi_period,
                "mfi_min_buy": current_params.mfi_min_buy,
                "bb_period": current_params.bb_period,
                "bb_mult": current_params.bb_mult,
                "kc_period": current_params.kc_period,
                "kc_mult": current_params.kc_mult,
                "lr_period": current_params.lr_period,
                "st_mult": current_params.st_mult,
                "st_len": current_params.st_len,
                "mfi_limit": current_params.mfi_limit,
                "stop_loss_pct": current_params.stop_loss_pct,
            },
            new_params={
                "mfi_period": best_params.mfi_period,
                "mfi_min_buy": best_params.mfi_min_buy,
                "bb_period": best_params.bb_period,
                "bb_mult": best_params.bb_mult,
                "kc_period": best_params.kc_period,
                "kc_mult": best_params.kc_mult,
                "lr_period": best_params.lr_period,
                "st_mult": best_params.st_mult,
                "st_len": best_params.st_len,
                "mfi_limit": best_params.mfi_limit,
                "stop_loss_pct": best_params.stop_loss_pct,
            },
            old_score=current_perf["score"],
            new_score=best_perf["score"],
            old_winrate=current_perf["win_rate"],
            new_winrate=best_perf["win_rate"],
            old_drawdown=current_perf["max_drawdown"],
            new_drawdown=best_perf["max_drawdown"],
            improvement_pct=improvement * 100,
            applied=applied,
            timestamp=time.time(),
        )

    async def _grid_search(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Optional[StrategyParams], Dict]:
        best_train = []

        for _ in range(200):
            params = StrategyParams(
                mfi_period   = random.choice(self.PARAM_GRID["mfi_period"]),
                mfi_min_buy  = random.choice(self.PARAM_GRID["mfi_min_buy"]),
                bb_period    = random.choice(self.PARAM_GRID["bb_period"]),
                bb_mult      = random.choice(self.PARAM_GRID["bb_mult"]),
                kc_period    = random.choice(self.PARAM_GRID["kc_period"]),
                kc_mult      = random.choice(self.PARAM_GRID["kc_mult"]),
                lr_period    = random.choice(self.PARAM_GRID["lr_period"]),
                st_mult      = random.choice(self.PARAM_GRID["st_mult"]),
                st_len       = random.choice(self.PARAM_GRID["st_len"]),
                mfi_limit    = random.choice(self.PARAM_GRID["mfi_limit"]),
                stop_loss_pct= random.choice(self.PARAM_GRID["stop_loss_pct"]),
            )
            perf = self._backtest(train_df, params)
            # فقط نتائج إيجابية وذات معنى
            if perf["total_signals"] >= 3 and perf["score"] > 0 and perf["win_rate"] >= 45:
                best_train.append((params, perf))

            await asyncio.sleep(0)

        if not best_train:
            return None, {}

        best_train.sort(key=lambda x: x[1]["score"], reverse=True)
        top5 = best_train[:5]

        test_results = []
        for params, _ in top5:
            test_perf = self._backtest(test_df, params)
            if test_perf["score"] > 0 and test_perf["win_rate"] >= 45:
                test_results.append((params, test_perf))

        if not test_results:
            return None, {}

        test_results.sort(key=lambda x: x[1]["score"], reverse=True)
        return test_results[0]

    def _backtest(self, df: pd.DataFrame, params: StrategyParams) -> Dict:
        """
        Backtest مع معيار تقييم صحيح:
        Score = Win Rate × Profit Factor
        (بدل Sharpe الذي يعطي نتائج مضللة)
        """
        try:
            strategy = SovereignStrategy(params)
            close  = df['close'].values
            high   = df['high'].values
            low    = df['low'].values
            volume = df['volume'].values

            if len(close) < 60:
                return {"total_signals": 0, "win_rate": 0, "score": 0, "max_drawdown": 0}

            st_vals, st_dir = strategy.calc_supertrend(high, low, close, params.st_len, params.st_mult)
            squeeze = strategy.calc_squeeze(high, low, close, params.bb_period, params.bb_mult, params.kc_period, params.kc_mult)
            mfi     = strategy.calc_mfi(high, low, close, volume, params.mfi_period)
            lr      = strategy.calc_linreg(close, high, low, params.lr_period)

            trades = []
            pos_state = 0
            entry_price = 0.0
            peak_hit = False

            for i in range(50, len(close)):
                is_bullish = st_dir[i] < 0
                is_bearish = st_dir[i] > 0
                mfi_val = mfi[i]
                lr_val  = lr[i]

                bb_m = np.mean(close[max(0, i-params.bb_period):i+1])
                bb_s = np.std(close[max(0, i-params.bb_period):i+1], ddof=0)
                bb_u = bb_m + params.bb_mult * bb_s

                if pos_state == 1:
                    sl   = close[i] < entry_price * (1 - params.stop_loss_pct)
                    peak = not peak_hit and close[i] > bb_u and mfi_val >= params.mfi_limit
                    flip = is_bearish and close[i] < entry_price
                    if sl or peak or flip:
                        pnl = (close[i] - entry_price) / entry_price * 100
                        trades.append(pnl)
                        pos_state = 0
                        peak_hit = peak

                if pos_state == 0:
                    if is_bullish and bool(squeeze[i]) and lr_val > 0 and mfi_val > params.mfi_min_buy:
                        pos_state = 1
                        entry_price = close[i]
                        peak_hit = False

            if len(trades) < 2:
                return {"total_signals": len(trades), "win_rate": 0, "score": 0, "max_drawdown": 0}

            wins   = [p for p in trades if p > 0]
            losses = [abs(p) for p in trades if p <= 0]

            win_rate     = len(wins) / len(trades) * 100
            avg_win      = np.mean(wins) if wins else 0
            avg_loss     = np.mean(losses) if losses else 0.001
            profit_factor = (sum(wins)) / (sum(losses) + 0.001)

            # Score = Win Rate × Profit Factor (كلاهما موجب دائماً)
            score = win_rate * profit_factor

            cum = np.cumsum(trades)
            peak_arr = np.maximum.accumulate(cum)
            max_dd = float(np.max(peak_arr - cum)) if len(cum) > 0 else 0

            return {
                "total_signals": len(trades),
                "win_rate": round(win_rate, 1),
                "score": round(score, 2),
                "profit_factor": round(profit_factor, 2),
                "max_drawdown": round(max_dd, 2),
            }
        except Exception:
            return {"total_signals": 0, "win_rate": 0, "score": 0, "max_drawdown": 0}

    async def _send_report(self, results: List[OptimizationResult]):
        lines = [
            "⚙️ *Auto-Optimizer — Sovereign System*",
            f"📅 {time.strftime('%Y-%m-%d %H:%M', time.localtime())}",
            f"📊 بيانات: آخر 30 يوم\n",
        ]

        for r in results:
            sym = r.symbol.replace("USDT", "")
            emoji = "🟢" if r.improvement_pct > 10 else "🟡"
            applied = "✅ *طُبّقت تلقائياً*" if r.applied else "📋 للمراجعة"

            lines.append("━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"*{sym}* {emoji} تحسين: `+{r.improvement_pct:.1f}%` — {applied}\n")

            lines.append(f"*قبل:* Win Rate: `{r.old_winrate:.1f}%` | DD: `{r.old_drawdown:.1f}%`")
            lines.append(f"*بعد:*  Win Rate: `{r.new_winrate:.1f}%` | DD: `{r.new_drawdown:.1f}%`\n")

            np_val = r.new_params
            op_val = r.old_params

            lines.append("*التعديلات المقترحة على المؤشر:*")

            fields = [
                ("MFI Period",         "mfi_period",    ""),
                ("MFI الحد الأدنى",    "mfi_min_buy",   ""),
                ("MFI تشبع البيع",     "mfi_limit",     ""),
                ("Bollinger Period",   "bb_period",     ""),
                ("Bollinger Mult",     "bb_mult",       ""),
                ("Keltner Period",     "kc_period",     ""),
                ("Keltner Mult",       "kc_mult",       ""),
                ("LinReg Period",      "lr_period",     ""),
                ("ST Factor",          "st_mult",       ""),
                ("ST ATR",             "st_len",        ""),
                ("وقف الخسارة %",      "stop_loss_pct", "%"),
            ]

            for label, key, suffix in fields:
                old_v = op_val.get(key)
                new_v = np_val.get(key)
                if key == "stop_loss_pct":
                    old_v = f"{old_v*100:.1f}%"
                    new_v = f"{new_v*100:.1f}%"
                changed = "🔄" if old_v != new_v else "  "
                lines.append(f"  {changed} *{label}:* `{old_v}` → `{new_v}`")

            lines.append("")

        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        lines.append("_🔄 = تغيّر | الإعدادات تُطبَّق تلقائياً إذا التحسين > 10%_")

        await self._send_telegram("\n".join(lines))

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
            return
        try:
            url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, json=payload, timeout=15)
            )
            if response.status_code == 200:
                logger.info("✅ Optimization report sent")
            else:
                logger.error(f"Telegram error: {response.status_code}")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")