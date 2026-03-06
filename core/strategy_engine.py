"""
Sovereign Strategy Engine
مستخرج من Pine Script — مختبر2

يحسب نفس إشارات الاستراتيجية على بيانات Binance مباشرة:
- SuperTrend
- Squeeze (BB inside KC)
- LinReg
- MFI
- HTF Filter
- شروط الدخول والخروج الكاملة
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StrategyParams:
    # MTF
    use_mtf: bool = True
    tf_htf: str = "1h"

    # Engine
    mfi_period: int = 14
    mfi_min_buy: int = 50
    bb_period: int = 20
    bb_mult: float = 2.0
    kc_period: int = 20
    kc_mult: float = 1.5
    lr_period: int = 20

    # SuperTrend
    st_mult: float = 3.0
    st_len: int = 10

    # Peak Exit
    use_peak_sell: bool = True
    mfi_limit: int = 75

    # Protection
    stop_loss_pct: float = 0.03  # 3%

    # Filters
    use_vol_filter: bool = False
    vol_mult: float = 1.2
    use_atr_filter: bool = False
    atr_mult: float = 0.8
    use_ema_filter: bool = False
    use_candle_filter: bool = False
    candle_max_pct: float = 2.0
    use_resist_filter: bool = False
    resist_lookback: int = 20


@dataclass
class SignalResult:
    long_entry: bool = False
    short_entry: bool = False
    exit_sl: bool = False
    exit_peak: bool = False
    exit_flip: bool = False

    # Indicators
    is_bullish: bool = False
    squeeze_fires: bool = False
    lr_val: float = 0.0
    mfi_val: float = 0.0
    htf_ok: bool = False
    st_val: float = 0.0

    # Position
    pos_state: int = 0       # 0=flat, 1=long, -1=short
    entry_price: float = 0.0
    pnl_now: float = 0.0


class SovereignStrategy:
    def __init__(self, params: StrategyParams = None):
        self.params = params or StrategyParams()
        self.pos_state = 0
        self.entry_price = 0.0
        self.high_since_entry = 0.0
        self.low_since_entry = 0.0
        self.peak_hit = False

    # ── Indicators ────────────────────────────────────

    def calc_supertrend(self, high, low, close, period, mult) -> Tuple[np.ndarray, np.ndarray]:
        """SuperTrend — نفس منطق Pine Script"""
        n = len(close)
        atr = self._calc_atr(high, low, close, period)
        st = np.zeros(n)
        direction = np.zeros(n)

        hl2 = (high + low) / 2
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr

        final_upper = np.copy(upper)
        final_lower = np.copy(lower)

        for i in range(1, n):
            final_upper[i] = upper[i] if upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1] else final_upper[i-1]
            final_lower[i] = lower[i] if lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1] else final_lower[i-1]

        for i in range(1, n):
            if close[i] > final_upper[i-1]:
                direction[i] = -1  # bullish
            elif close[i] < final_lower[i-1]:
                direction[i] = 1   # bearish
            else:
                direction[i] = direction[i-1]

            st[i] = final_lower[i] if direction[i] < 0 else final_upper[i]

        return st, direction

    def calc_squeeze(self, high, low, close, bb_period, bb_mult, kc_period, kc_mult) -> np.ndarray:
        """Squeeze Momentum — BB inside KC = no squeeze"""
        n = len(close)
        squeeze = np.zeros(n, dtype=bool)

        for i in range(max(bb_period, kc_period), n):
            # Bollinger Bands
            c = close[max(0, i-bb_period):i+1]
            bb_m = np.mean(c)
            bb_std = np.std(c, ddof=0)
            bb_u = bb_m + bb_mult * bb_std
            bb_l = bb_m - bb_mult * bb_std

            # Keltner Channel
            h = high[max(0, i-kc_period):i+1]
            l = low[max(0, i-kc_period):i+1]
            atr_kc = np.mean(h - l)
            kc_u = bb_m + kc_mult * atr_kc
            kc_l = bb_m - kc_mult * atr_kc

            # Squeeze fires when BB is NOT inside KC
            squeeze[i] = not (bb_u < kc_u and bb_l > kc_l)

        return squeeze

    def calc_mfi(self, high, low, close, volume, period) -> np.ndarray:
        """Money Flow Index"""
        n = len(close)
        mfi = np.full(n, 50.0)
        tp = (high + low + close) / 3

        for i in range(period, n):
            tp_slice = tp[i-period:i]
            vol_slice = volume[i-period:i]
            mf = tp_slice * vol_slice

            pos_mf = np.sum(mf[np.diff(np.append([tp_slice[0]], tp_slice)) > 0])
            neg_mf = np.sum(mf[np.diff(np.append([tp_slice[0]], tp_slice)) <= 0])

            if neg_mf == 0:
                mfi[i] = 100.0
            else:
                mfi[i] = 100 - 100 / (1 + pos_mf / neg_mf)

        return mfi

    def calc_linreg(self, close, high, low, period) -> np.ndarray:
        """LinReg على src = close - avg(highest, lowest, sma)"""
        n = len(close)
        lr = np.zeros(n)

        for i in range(period, n):
            h = np.max(high[i-period:i+1])
            l = np.min(low[i-period:i+1])
            sma = np.mean(close[i-period:i+1])
            src = close[i] - (h + l + sma) / 3

            # Linear regression value
            x = np.arange(period + 1)
            y_slice = close[i-period:i+1] - (np.max(high[i-period:i+1]) + np.min(low[i-period:i+1]) + np.mean(close[i-period:i+1])) / 3
            if len(x) == len(y_slice) and len(x) > 1:
                lr[i] = np.polyval(np.polyfit(x, y_slice, 1), period)
            else:
                lr[i] = src

        return lr

    def _calc_atr(self, high, low, close, period) -> np.ndarray:
        n = len(close)
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros(n)
        if n > period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr

    # ── Main Signal Calculator ─────────────────────────

    def calculate(self, df: pd.DataFrame, htf_df: pd.DataFrame = None) -> SignalResult:
        """
        df: DataFrame مع columns: open, high, low, close, volume
        htf_df: نفس الشيء للفريم الأعلى (اختياري)
        """
        p = self.params
        result = SignalResult()

        if len(df) < 50:
            return result

        high  = df['high'].values
        low   = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        open_ = df['open'].values

        # SuperTrend
        st_vals, st_dir = self.calc_supertrend(high, low, close, p.st_len, p.st_mult)
        is_bullish = st_dir[-1] < 0
        is_bearish = st_dir[-1] > 0
        result.is_bullish = is_bullish
        result.st_val = st_vals[-1]

        # Squeeze
        squeeze = self.calc_squeeze(high, low, close, p.bb_period, p.bb_mult, p.kc_period, p.kc_mult)
        squeeze_fires = bool(squeeze[-1])
        result.squeeze_fires = squeeze_fires

        # MFI
        mfi = self.calc_mfi(high, low, close, volume, p.mfi_period)
        mfi_val = mfi[-1]
        result.mfi_val = mfi_val

        # LinReg
        lr = self.calc_linreg(close, high, low, p.lr_period)
        lr_val = lr[-1]
        result.lr_val = lr_val

        # BB للخروج
        bb_m = np.mean(close[-p.bb_period:])
        bb_std = np.std(close[-p.bb_period:], ddof=0)
        bb_u = bb_m + p.bb_mult * bb_std
        bb_l = bb_m - p.bb_mult * bb_std

        # Filters
        vol_ok = True
        if p.use_vol_filter:
            vol_sma = np.mean(volume[-20:])
            vol_ok = volume[-1] > vol_sma * p.vol_mult

        atr = self._calc_atr(high, low, close, 14)
        atr_ok = True
        if p.use_atr_filter:
            atr_sma = np.mean(atr[-50:])
            atr_ok = atr[-1] > atr_sma * p.atr_mult

        ema200 = self._ema(close, 200)
        ema_ok = True
        ema_short_ok = True
        if p.use_ema_filter:
            ema_ok = close[-1] > ema200[-1]
            ema_short_ok = close[-1] < ema200[-1]

        candle_ok = True
        if p.use_candle_filter:
            candle_pct = abs(close[-1] - open_[-1]) / open_[-1] * 100
            candle_ok = candle_pct < p.candle_max_pct

        resist_ok = True
        if p.use_resist_filter:
            highest = np.max(high[-p.resist_lookback:])
            near_resist = close[-1] > highest * 0.98
            resist_ok = not near_resist

        # HTF Filter
        htf_ok = True
        htf_short_ok = True
        if p.use_mtf and htf_df is not None and len(htf_df) >= 20:
            htf_h = htf_df['high'].values
            htf_l = htf_df['low'].values
            htf_c = htf_df['close'].values
            htf_v = htf_df['volume'].values

            _, htf_st_dir = self.calc_supertrend(htf_h, htf_l, htf_c, 10, 3.0)
            htf_lr = self.calc_linreg(htf_c, htf_h, htf_l, 20)
            htf_ok = htf_lr[-1] > 0 and htf_st_dir[-1] < 0
            htf_short_ok = htf_lr[-1] < 0 and htf_st_dir[-1] > 0

        result.htf_ok = htf_ok

        # ── شروط الدخول ──
        buy_cond = (is_bullish and squeeze_fires and lr_val > 0
                    and mfi_val > p.mfi_min_buy and htf_ok
                    and vol_ok and atr_ok and ema_ok and candle_ok and resist_ok)

        # ── شروط الخروج ──
        sl_hit = False
        tp_peak = False
        st_flip_exit = False

        if self.pos_state == 1:
            sl_hit = close[-1] < self.entry_price * (1 - p.stop_loss_pct)
            tp_peak = (p.use_peak_sell and not self.peak_hit
                       and close[-1] > bb_u and mfi_val >= p.mfi_limit)
            st_flip_exit = is_bearish and close[-1] < self.entry_price

        elif self.pos_state == -1:
            sl_hit = close[-1] > self.entry_price * (1 + p.stop_loss_pct)
            tp_peak = (p.use_peak_sell and not self.peak_hit
                       and close[-1] < bb_l and mfi_val <= (100 - p.mfi_limit))
            st_flip_exit = is_bullish and close[-1] > self.entry_price

        exit_all = (self.pos_state != 0) and (sl_hit or tp_peak or st_flip_exit)

        long_entry  = buy_cond   and self.pos_state == 0 and not exit_all
        short_entry = False      # Spot only in this version

        # ── تحديث الحالة ──
        if exit_all:
            result.exit_sl   = sl_hit
            result.exit_peak = tp_peak and not sl_hit
            result.exit_flip = st_flip_exit and not sl_hit and not tp_peak
            self.peak_hit = tp_peak
            self.pos_state = 0

        if long_entry:
            self.pos_state = 1
            self.entry_price = close[-1]
            self.peak_hit = False
            result.long_entry = True

        # PnL الآن
        if self.pos_state == 1 and self.entry_price > 0:
            result.pnl_now = (close[-1] - self.entry_price) / self.entry_price * 100
        elif self.pos_state == -1 and self.entry_price > 0:
            result.pnl_now = (self.entry_price - close[-1]) / self.entry_price * 100

        result.pos_state = self.pos_state
        result.entry_price = self.entry_price
        result.short_entry = short_entry

        return result

    def _ema(self, data, period):
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema

    def get_dashboard_data(self, df: pd.DataFrame, htf_df=None) -> dict:
        """يرجع كل البيانات للـ Dashboard"""
        result = self.calculate(df, htf_df)
        close = df['close'].values

        return {
            "signal": {
                "long_entry":   result.long_entry,
                "short_entry":  result.short_entry,
                "exit_sl":      result.exit_sl,
                "exit_peak":    result.exit_peak,
                "exit_flip":    result.exit_flip,
            },
            "indicators": {
                "supertrend":     "🟢 صاعد" if result.is_bullish else "🔴 هابط",
                "squeeze":        "🔥 انطلق" if result.squeeze_fires else "🔒 ضغط",
                "mfi":            round(result.mfi_val, 1),
                "linreg":         round(result.lr_val, 4),
                "htf_filter":     "✅ مؤكد" if result.htf_ok else "❌ ضد الاتجاه",
            },
            "position": {
                "state":       result.pos_state,
                "entry_price": result.entry_price,
                "current":     close[-1],
                "pnl_now":     round(result.pnl_now, 2),
                "sl_price":    round(result.entry_price * (1 - self.params.stop_loss_pct), 4) if result.pos_state == 1 else 0,
            },
        }
