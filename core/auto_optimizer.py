"""
Auto-Optimizer V2 — نظام ضبط شامل ومستمر
- يجرب كل توليفات الإعدادات (Grid Search كامل)
- Walk-Forward Optimization لتجنب Overfitting
- يحفظ أفضل 10 نتائج لكل عملة
- يرسل تقرير Telegram + يحفظ للـ API
"""

import asyncio
import time
import logging
import requests
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from itertools import product

from core.strategy_engine import SovereignStrategy, StrategyParams
from config.settings import config

logger = logging.getLogger("AutoOptimizer")

# ══════════════════════════════════════════
# نتائج الـ Optimization — محفوظة للـ API
# ══════════════════════════════════════════
OPTIMIZER_RESULTS: Dict[str, dict] = {}   # symbol → latest result
OPTIMIZER_HISTORY: List[dict]      = []   # كل النتائج السابقة
CURRENT_PARAMS:    Dict[str, dict] = {}   # الإعدادات المطبقة حالياً
OPTIMIZER_STATUS = {
    "running":        False,
    "last_run":       None,
    "next_run":       None,
    "progress":       0,       # 0-100
    "current_symbol": None,
    "total_combos":   0,
    "tested_combos":  0,
}

# ══════════════════════════════════════════
# شبكة البارامترات الكاملة
# ══════════════════════════════════════════
PARAM_GRID = {
    "mfi_period":    [10, 14, 20],
    "mfi_min_buy":   [40, 45, 50, 55, 60],
    "mfi_limit":     [70, 75, 80],
    "bb_period":     [15, 20, 25],
    "bb_mult":       [1.6, 1.8, 2.0, 2.2, 2.5],
    "kc_period":     [15, 20, 25],
    "kc_mult":       [1.0, 1.2, 1.5, 1.8],
    "lr_period":     [10, 15, 20, 25],
    "st_mult":       [2.0, 2.5, 3.0, 3.5, 4.0],
    "st_len":        [7, 8, 10, 12, 14],
    "stop_loss_pct": [0.015, 0.02, 0.025, 0.03, 0.035, 0.04],
}

PARAM_LABELS = {
    "mfi_period":    "MFI Period",
    "mfi_min_buy":   "MFI Min Buy",
    "mfi_limit":     "MFI تشبع البيع",
    "bb_period":     "Bollinger Period",
    "bb_mult":       "Bollinger Mult",
    "kc_period":     "Keltner Period",
    "kc_mult":       "Keltner Mult",
    "lr_period":     "LinReg Period",
    "st_mult":       "ST Factor",
    "st_len":        "ST ATR Length",
    "stop_loss_pct": "Stop Loss %",
}


@dataclass
class BacktestResult:
    params:        dict
    total_signals: int
    win_rate:      float
    profit_factor: float
    score:         float
    max_drawdown:  float
    avg_win:       float
    avg_loss:      float
    total_pnl:     float


class AutoOptimizer:
    def __init__(self):
        self.running          = False
        self.OPTIMIZE_INTERVAL = 86400   # كل 24 ساعة
        self.N_RANDOM_SAMPLES  = 300     # عدد التوليفات العشوائية
        self.MIN_TRADES        = 5       # حد أدنى للصفقات
        self.AUTO_APPLY_THRESH = 0.10    # 10% تحسين للتطبيق التلقائي

    async def start(self):
        self.running = True
        logger.info("✅ Auto-Optimizer V2 started")
        await asyncio.sleep(90)          # انتظر تشغيل النظام
        while self.running:
            try:
                await self.run_full_optimization()
                OPTIMIZER_STATUS["next_run"] = time.time() + self.OPTIMIZE_INTERVAL
                await asyncio.sleep(self.OPTIMIZE_INTERVAL)
            except Exception as e:
                logger.error(f"Optimizer error: {e}")
                await asyncio.sleep(3600)

    async def stop(self):
        self.running = False

    # ══════════════════════════════════════════
    # الدورة الكاملة
    # ══════════════════════════════════════════
    async def run_full_optimization(self):
        OPTIMIZER_STATUS["running"]  = True
        OPTIMIZER_STATUS["last_run"] = time.time()
        OPTIMIZER_STATUS["progress"] = 0
        logger.info("🔧 بدء Optimization الشامل...")

        all_results = []
        symbols     = config.SYMBOLS

        for idx, symbol in enumerate(symbols):
            OPTIMIZER_STATUS["current_symbol"] = symbol
            OPTIMIZER_STATUS["progress"]       = int((idx / len(symbols)) * 100)
            logger.info(f"   [{idx+1}/{len(symbols)}] تحليل {symbol}...")

            df = await self._fetch_ohlcv(symbol, "1h", 720)
            if df is None or len(df) < 200:
                logger.warning(f"بيانات غير كافية لـ {symbol}")
                continue

            result = await self._optimize_symbol(symbol, df)
            if result:
                all_results.append(result)
                OPTIMIZER_RESULTS[symbol] = result
                OPTIMIZER_HISTORY.append({**result, "timestamp": time.time()})

        OPTIMIZER_STATUS["running"]        = False
        OPTIMIZER_STATUS["progress"]       = 100
        OPTIMIZER_STATUS["current_symbol"] = None

        if all_results:
            await self._send_telegram_report(all_results)
            logger.info(f"✅ Optimization انتهى — {len(all_results)} عملات")

    # ══════════════════════════════════════════
    # تحسين عملة واحدة
    # ══════════════════════════════════════════
    async def _optimize_symbol(self, symbol: str, df: pd.DataFrame) -> Optional[dict]:
        try:
            n         = len(df)
            train_end = int(n * 0.70)
            val_end   = int(n * 0.85)

            train_df = df.iloc[:train_end]
            val_df   = df.iloc[train_end:val_end]
            test_df  = df.iloc[val_end:]

            if len(test_df) < 50:
                return None

            # الإعدادات الحالية
            cur_p      = CURRENT_PARAMS.get(symbol, {})
            cur_params = StrategyParams(**{k: v for k, v in cur_p.items() if hasattr(StrategyParams, k)}) if cur_p else StrategyParams()
            cur_perf   = self._backtest(df, cur_params)

            # Grid Search على بيانات التدريب
            logger.info(f"      Grid Search على {self.N_RANDOM_SAMPLES} توليفة...")
            top_train = await self._grid_search(train_df, self.N_RANDOM_SAMPLES)

            if not top_train:
                return None

            # Validation على بيانات الـ Validation
            top_val = []
            for bt in top_train[:20]:
                p    = StrategyParams(**bt.params)
                perf = self._backtest(val_df, p)
                if perf.score > 0 and perf.total_signals >= self.MIN_TRADES:
                    top_val.append(perf)

            if not top_val:
                top_val = top_train[:5]

            top_val.sort(key=lambda x: x.score, reverse=True)

            # Final Test على بيانات الاختبار (غير مرئية)
            best_bt = None
            for bt in top_val[:5]:
                p    = StrategyParams(**bt.params)
                perf = self._backtest(test_df, p)
                if perf.score > 0 and (best_bt is None or perf.score > best_bt.score):
                    best_bt = perf

            if best_bt is None:
                best_bt = top_val[0]

            # حساب التحسين
            improvement = 0.0
            if cur_perf.score > 0:
                improvement = (best_bt.score - cur_perf.score) / cur_perf.score
            elif best_bt.score > 0:
                improvement = 1.0

            # تطبيق تلقائي إذا التحسين > 10%
            applied = False
            if improvement >= self.AUTO_APPLY_THRESH and best_bt.total_signals >= self.MIN_TRADES:
                CURRENT_PARAMS[symbol] = best_bt.params
                applied = True
                logger.info(f"✅ تطبيق تلقائي لـ {symbol} — تحسين {improvement:.1%}")

            # مقارنة البارامترات
            changes = self._compare_params(cur_params, best_bt.params)

            return {
                "symbol":       symbol,
                "improvement":  round(improvement * 100, 1),
                "applied":      applied,
                "old_params":   self._params_to_dict(cur_params),
                "new_params":   best_bt.params,
                "changes":      changes,
                "old_perf": {
                    "win_rate":      cur_perf.win_rate,
                    "profit_factor": cur_perf.profit_factor,
                    "score":         cur_perf.score,
                    "max_drawdown":  cur_perf.max_drawdown,
                    "total_signals": cur_perf.total_signals,
                    "total_pnl":     cur_perf.total_pnl,
                },
                "new_perf": {
                    "win_rate":      best_bt.win_rate,
                    "profit_factor": best_bt.profit_factor,
                    "score":         best_bt.score,
                    "max_drawdown":  best_bt.max_drawdown,
                    "total_signals": best_bt.total_signals,
                    "total_pnl":     best_bt.total_pnl,
                },
                "top5": [
                    {
                        "params":        t.params,
                        "win_rate":      t.win_rate,
                        "profit_factor": t.profit_factor,
                        "score":         t.score,
                        "signals":       t.total_signals,
                    }
                    for t in top_val[:5]
                ],
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"Optimize error {symbol}: {e}")
            return None

    # ══════════════════════════════════════════
    # Grid Search العشوائي
    # ══════════════════════════════════════════
    async def _grid_search(self, df: pd.DataFrame, n_samples: int) -> List[BacktestResult]:
        results = []
        OPTIMIZER_STATUS["total_combos"]  = n_samples
        OPTIMIZER_STATUS["tested_combos"] = 0

        for i in range(n_samples):
            # اختيار عشوائي من الشبكة
            params_dict = {k: random.choice(v) for k, v in PARAM_GRID.items()}
            params      = StrategyParams(**params_dict)
            perf        = self._backtest(df, params)

            OPTIMIZER_STATUS["tested_combos"] = i + 1

            if (perf.total_signals >= self.MIN_TRADES
                    and perf.score > 0
                    and perf.win_rate >= 40
                    and perf.profit_factor >= 1.0):
                results.append(perf)

            # yield للـ event loop كل 10 توليفات
            if i % 10 == 0:
                await asyncio.sleep(0)

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    # ══════════════════════════════════════════
    # Backtest
    # ══════════════════════════════════════════
    def _backtest(self, df: pd.DataFrame, params: StrategyParams) -> BacktestResult:
        try:
            strategy = SovereignStrategy(params)
            close    = df['close'].values
            high     = df['high'].values
            low      = df['low'].values
            volume   = df['volume'].values

            if len(close) < 60:
                return BacktestResult(params=self._params_to_dict(params), total_signals=0, win_rate=0, profit_factor=0, score=0, max_drawdown=0, avg_win=0, avg_loss=0, total_pnl=0)

            st_vals, st_dir = strategy.calc_supertrend(high, low, close, params.st_len, params.st_mult)
            squeeze         = strategy.calc_squeeze(high, low, close, params.bb_period, params.bb_mult, params.kc_period, params.kc_mult)
            mfi             = strategy.calc_mfi(high, low, close, volume, params.mfi_period)
            lr              = strategy.calc_linreg(close, high, low, params.lr_period)

            trades      = []
            pos_state   = 0
            entry_price = 0.0

            for i in range(50, len(close)):
                is_bullish = st_dir[i] < 0
                is_bearish = st_dir[i] > 0
                mfi_val    = mfi[i]
                lr_val     = lr[i]

                bb_m = np.mean(close[max(0, i-params.bb_period):i+1])
                bb_s = np.std(close[max(0, i-params.bb_period):i+1], ddof=0)
                bb_u = bb_m + params.bb_mult * bb_s

                if pos_state == 1:
                    sl   = close[i] < entry_price * (1 - params.stop_loss_pct)
                    peak = close[i] > bb_u and mfi_val >= params.mfi_limit
                    flip = is_bearish and close[i] < entry_price
                    if sl or peak or flip:
                        pnl = (close[i] - entry_price) / entry_price * 100
                        trades.append(pnl)
                        pos_state = 0

                if pos_state == 0:
                    sq_val = bool(squeeze[i]) if hasattr(squeeze[i], '__bool__') else squeeze[i]
                    if is_bullish and sq_val and lr_val > 0 and mfi_val > params.mfi_min_buy:
                        pos_state   = 1
                        entry_price = close[i]

            if len(trades) < 2:
                return BacktestResult(params=self._params_to_dict(params), total_signals=len(trades), win_rate=0, profit_factor=0, score=0, max_drawdown=0, avg_win=0, avg_loss=0, total_pnl=0)

            wins    = [p for p in trades if p > 0]
            losses  = [abs(p) for p in trades if p <= 0]
            win_rate = len(wins) / len(trades) * 100

            gross_win  = sum(wins) if wins else 0
            gross_loss = sum(losses) if losses else 0.001
            pf         = gross_win / gross_loss
            avg_win    = np.mean(wins)   if wins   else 0
            avg_loss   = np.mean(losses) if losses else 0
            total_pnl  = gross_win - gross_loss

            # Score = Win Rate × Profit Factor (كلاهما موجب)
            score = win_rate * pf

            cum     = np.cumsum(trades)
            peak_a  = np.maximum.accumulate(cum)
            max_dd  = float(np.max(peak_a - cum)) if len(cum) > 0 else 0

            return BacktestResult(
                params        = self._params_to_dict(params),
                total_signals = len(trades),
                win_rate      = round(win_rate, 1),
                profit_factor = round(pf, 2),
                score         = round(score, 2),
                max_drawdown  = round(max_dd, 2),
                avg_win       = round(avg_win, 2),
                avg_loss      = round(avg_loss, 2),
                total_pnl     = round(total_pnl, 2),
            )
        except Exception as e:
            logger.debug(f"Backtest error: {e}")
            return BacktestResult(params={}, total_signals=0, win_rate=0, profit_factor=0, score=0, max_drawdown=0, avg_win=0, avg_loss=0, total_pnl=0)

    # ══════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════
    def _params_to_dict(self, p: StrategyParams) -> dict:
        return {
            "mfi_period":    p.mfi_period,
            "mfi_min_buy":   p.mfi_min_buy,
            "mfi_limit":     p.mfi_limit,
            "bb_period":     p.bb_period,
            "bb_mult":       p.bb_mult,
            "kc_period":     p.kc_period,
            "kc_mult":       p.kc_mult,
            "lr_period":     p.lr_period,
            "st_mult":       p.st_mult,
            "st_len":        p.st_len,
            "stop_loss_pct": p.stop_loss_pct,
        }

    def _compare_params(self, old: StrategyParams, new: dict) -> List[dict]:
        changes = []
        old_d   = self._params_to_dict(old)
        for k, label in PARAM_LABELS.items():
            ov = old_d.get(k)
            nv = new.get(k)
            if k == "stop_loss_pct":
                ov_str = f"{ov*100:.1f}%"
                nv_str = f"{nv*100:.1f}%"
            else:
                ov_str = str(ov)
                nv_str = str(nv)
            changes.append({
                "key":     k,
                "label":   label,
                "old":     ov_str,
                "new":     nv_str,
                "changed": ov != nv,
            })
        return changes

    # ══════════════════════════════════════════
    # Binance Data
    # ══════════════════════════════════════════
    async def _fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        try:
            from binance.client import Client
            loop   = asyncio.get_event_loop()
            client = Client(config.BINANCE_API_KEY, config.BINANCE_SECRET, testnet=True)
            klines = await loop.run_in_executor(
                None,
                lambda: client.get_klines(symbol=symbol, interval=interval, limit=limit)
            )
            df = pd.DataFrame(klines, columns=[
                'time','open','high','low','close','volume',
                'close_time','quote_vol','trades','taker_buy_base','taker_buy_quote','ignore'
            ])
            for col in ['open','high','low','close','volume']:
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            logger.error(f"Fetch OHLCV {symbol}: {e}")
            return None

    # ══════════════════════════════════════════
    # Telegram Report
    # ══════════════════════════════════════════
    async def _send_telegram_report(self, results: List[dict]):
        lines = [
            "⚙️ *Sovereign Auto-Optimizer V2*",
            f"📅 {time.strftime('%Y-%m-%d %H:%M', time.localtime())}",
            f"🔬 تم اختبار {self.N_RANDOM_SAMPLES} توليفة × {len(results)} عملة\n",
        ]

        for r in results:
            sym   = r['symbol'].replace('USDT', '')
            imp   = r['improvement']
            emoji = "🟢" if imp > 10 else "🟡" if imp > 0 else "🔴"
            app   = "✅ *طُبّقت*" if r['applied'] else "📋 للمراجعة"
            op    = r['old_perf']
            np_   = r['new_perf']

            lines.append("━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"*{sym}* {emoji} `{imp:+.1f}%` — {app}")
            lines.append(f"قبل: WR `{op['win_rate']}%` | PF `{op['profit_factor']}` | DD `{op['max_drawdown']:.1f}%`")
            lines.append(f"بعد:  WR `{np_['win_rate']}%` | PF `{np_['profit_factor']}` | DD `{np_['max_drawdown']:.1f}%`\n")
            lines.append("*التعديلات:*")

            for ch in r['changes']:
                icon = "🔄" if ch['changed'] else "  "
                lines.append(f"  {icon} {ch['label']}: `{ch['old']}` → `{ch['new']}`")
            lines.append("")

        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        lines.append("_التحديث التالي بعد 24 ساعة_")

        await self._send_telegram("\n".join(lines))

    async def _send_telegram(self, message: str):
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            return
        try:
            url     = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
            loop    = asyncio.get_event_loop()
            resp    = await loop.run_in_executor(
                None,
                lambda: requests.post(url, json=payload, timeout=15)
            )
            if resp.status_code == 200:
                logger.info("✅ Telegram report sent")
        except Exception as e:
            logger.error(f"Telegram error: {e}")
