"""
BTC 自動売買シミュレーション（入力不要のデフォルト版）
- 3つの戦略（移動平均クロス / RSI 逆張り / ボリンジャーバンド逆張り）を個別にバックテスト
- さらに多数決（Voting）メタ戦略を同時評価
- 手数料・スリッページ、固定割合ポジション、任意のトレーリングストップ
- **引数なしで実行**すると、**直近3年相当のデータ**で自動実行
  - Yahooの制約により 1時間足は直近約2年（730日）まで → 3年は自動で4時間足にフォールバックします

依存: pandas, numpy, matplotlib, yfinance

実行:
  python btc_backtest_default.py

出力:
  - ./plots/price_ma.png / price_rsi.png / price_bb.png
  - ./plots/equity_curves.png
  - コンソール: 各戦略のCAGR, Sharpe, MaxDD, WinRate, Trades, FinalEquity(最終資金)
"""

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:
    yf = None

# ==========================
# インジケータ（自前実装）
# ==========================

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    out = 100 - (100 / (1 + rs))
    return out

def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower

# ==========================
# 成績指標
# ==========================

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0

def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    if returns.dropna().empty or returns.std(ddof=0) == 0:
        return 0.0
    return float((returns.mean() - rf) / (returns.std(ddof=0) + 1e-12))

@dataclass
class Perf:
    name: str
    cagr: float
    sharpe: float
    maxdd: float
    winrate: float
    trades: int
    final_equity: float

# ==========================
# 戦略シグナル
# ==========================

def signal_ma_cross(df: pd.DataFrame, short_ma: int, long_ma: int) -> pd.Series:
    s = sma(df['Close'], short_ma)
    l = sma(df['Close'], long_ma)
    sig = pd.Series(0, index=df.index, dtype=float)
    sig[s > l] = 1.0
    sig[s < l] = -1.0
    return sig

def signal_rsi_reversion(df: pd.DataFrame, period: int = 14, low: float = 30, high: float = 70) -> pd.Series:
    r = rsi(df['Close'], period)
    sig = pd.Series(0, index=df.index, dtype=float)
    sig[r < low] = 1.0
    sig[r > high] = -1.0
    return sig

def signal_bb_reversion(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.Series:
    mid, up, lo = bollinger(df['Close'], n, k)
    sig = pd.Series(0, index=df.index, dtype=float)
    sig[df['Close'] < lo] = 1.0
    sig[df['Close'] > up] = -1.0
    return sig

# ==========================
# バックテスト（共通ロジック）
# ==========================

def backtest_from_signal(
    df: pd.DataFrame,
    signal: pd.Series,
    name: str,
    fee_bps: float = 10,    # 片道手数料(bps)
    slip_bps: float = 1,    # 片道スリッページ(bps)
    risk_per_trade: float = 1.0,  # 現金の%でポジション
    allow_short: bool = False,
    trail_stop_pct: float | None = None,
    initial_equity: float = 10_000.0,  # 初期10,000円
) -> pd.DataFrame:
    df = df.copy()
    signal = signal.reindex(df.index).fillna(0).astype(float)

    cost_rate = (fee_bps + slip_bps) / 1e4

    cash = float(initial_equity)
    pos_qty = 0.0
    pos_side = 0
    entry_price = np.nan
    peak_price = np.nan

    records = []
    sig_next = signal.shift(1).fillna(0)

    for i in range(1, len(df)):
        ts = df.index[i]
        open_px = float(df['Open'].iloc[i])
        close_px = float(df['Close'].iloc[i])
        s = int(sig_next.iloc[i])

        # トレーリングストップ
        exit_by_trail = False
        if pos_side != 0 and trail_stop_pct is not None and np.isfinite(entry_price):
            if pos_side == 1:
                peak_price = max(peak_price, close_px) if np.isfinite(peak_price) else close_px
                stop = peak_price * (1 - trail_stop_pct)
                if close_px <= stop:
                    exit_by_trail = True
            else:
                peak_price = min(peak_price, close_px) if np.isfinite(peak_price) else close_px
                stop = peak_price * (1 + trail_stop_pct)
                if close_px >= stop:
                    exit_by_trail = True

        need_exit = False
        if pos_side != 0:
            if s == 0 or s == -pos_side or exit_by_trail:
                need_exit = True

        if need_exit and pos_qty != 0:
            px = open_px * (1 - cost_rate * np.sign(pos_side))
            cash += pos_qty * px
            pos_qty = 0.0
            pos_side = 0
            entry_price = np.nan
            peak_price = np.nan

        if s != 0 and pos_side == 0:
            if s == -1 and not allow_short:
                pass
            else:
                risk_capital = cash * (risk_per_trade / 100.0)
                if risk_capital > 0:
                    px = open_px * (1 + cost_rate * np.sign(s))
                    qty = risk_capital / px
                    pos_qty = qty * s
                    pos_side = s
                    cash -= qty * px * s
                    entry_price = px
                    peak_price = px

        mtm = cash + pos_qty * close_px
        records.append({
            'Date': ts,
            'Equity': mtm,
            'Cash': cash,
            'Position': pos_qty,
            'Side': pos_side,
            'Close': close_px,
        })

    bt = pd.DataFrame(records).set_index('Date')
    bt['Ret'] = bt['Equity'].pct_change().fillna(0)
    bt.attrs['name'] = name
    return bt

# ==========================
# 可視化
# ==========================

def plot_price_with_signals(df: pd.DataFrame, sigs: dict, out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Close')
    for k, v in sigs.items():
        plt.plot(df.index, v, label=k)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_equity_curves(curves: dict, out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(12,6))
    for name, bt in curves.items():
        plt.plot(bt.index, bt['Equity'], label=name)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ==========================
# 評価ユーティリティ
# ==========================

def evaluate(bt: pd.DataFrame, periods_per_year: int) -> Perf:
    eq = bt['Equity']
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0.0
    sr = sharpe_ratio(bt['Ret']) * np.sqrt(periods_per_year)
    mdd = max_drawdown(eq)

    side = bt['Side'].fillna(0)
    turns = (side.shift(1) != side) & (side != 0)
    trade_idx = bt.index[turns]
    trades = int(len(trade_idx))

    wins = 0
    for idx in trade_idx:
        start_i = bt.index.get_loc(idx)
        end_i = start_i + 1
        while end_i < len(bt) and bt['Side'].iloc[end_i] != 0 and np.sign(bt['Side'].iloc[end_i]) == np.sign(bt['Side'].iloc[start_i]):
            end_i += 1
        pnl = bt['Equity'].iloc[end_i-1] - bt['Equity'].iloc[start_i]
        if pnl > 0:
            wins += 1
    winrate = wins / trades if trades > 0 else 0.0

    return Perf(name=bt.attrs.get('name', 'Strategy'), cagr=cagr, sharpe=sr, maxdd=mdd, winrate=winrate, trades=trades, final_equity=float(eq.iloc[-1]))

# ==========================
# データ取得（直近 ~3年 -> 1h優先 / 4hフォールバック）
# ==========================

def _normalize_ohlcv(df: pd.DataFrame, tz: str = 'Asia/Tokyo') -> pd.DataFrame:
    # 列がMultiIndexのときの平坦化
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            df = df.droplevel(1, axis=1)
        elif 'Close' in df.columns.get_level_values(-1):
            df = df.droplevel(0, axis=1)
        else:
            df.columns = df.columns.get_level_values(-1)
    # インデックスがMultiIndexのときの処理（日時抽出）
    if isinstance(df.index, pd.MultiIndex):
        try:
            df = df.droplevel(0)
        except Exception:
            df = df.reset_index()
            df.index = pd.DatetimeIndex(df.select_dtypes(['datetime']).iloc[:,0])
    # タイムゾーン統一
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert(tz)
    else:
        df.index = df.index.tz_convert(tz)
    cols = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
    return df[cols].dropna()


def load_btc_hourly_last3y(ticker: str = 'BTC-USD') -> pd.DataFrame:
    end_tokyo = pd.Timestamp.now(tz='Asia/Tokyo')
    start_3y_tokyo = end_tokyo - pd.Timedelta(days=365*3 + 5)

    if yf is not None:
        # まずは 1時間足（730日まで）
        try:
            df = yf.Ticker(ticker).history(period='730d', interval='1h', auto_adjust=True)
            if not df.empty:
                return _normalize_ohlcv(df, tz='Asia/Tokyo')
        except Exception:
            pass
        # 4時間足で3年カバー
        try:
            df = yf.Ticker(ticker).history(start=start_3y_tokyo.tz_convert('UTC'), end=end_tokyo.tz_convert('UTC'), interval='4h', auto_adjust=True)
            if not df.empty:
                return _normalize_ohlcv(df, tz='Asia/Tokyo')
        except Exception:
            pass

    # サンプルCSV（任意）
    csv_path = './sample_btc_1h.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
        if df.index.tz is None:
            df.index = df.index.tz_localize('Asia/Tokyo')
        return df[['Open','High','Low','Close','Volume']].dropna()

    # ダミーデータ（デモ用）
    rng = pd.date_range(end=end_tokyo, periods=24*365*3, freq='h', tz='Asia/Tokyo')
    price = 3_000_000 + np.cumsum(np.random.normal(0, 30_000, len(rng)))
    price = np.clip(price, 100_000, None)
    df = pd.DataFrame(index=rng)
    df['Close'] = price
    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
    df['High'] = df[['Open','Close']].max(axis=1) * (1 + np.random.rand(len(df))*0.01)
    df['Low']  = df[['Open','Close']].min(axis=1) * (1 - np.random.rand(len(df))*0.01)
    df['Volume'] = np.random.randint(50, 500, len(df))
    return df

# ==========================
# Buy & Hold（常に保持）
# ==========================

def buy_and_hold(df: pd.DataFrame, initial_equity: float = 10_000.0) -> pd.DataFrame:
    """最初に全額で購入し、その後はずっと保持するベンチマーク。"""
    first_price = float(df['Open'].iloc[0])
    qty = initial_equity / first_price if first_price > 0 else 0.0
    records = []
    for i in range(len(df)):
        ts = df.index[i]
        price = float(df['Close'].iloc[i])
        equity = qty * price
        records.append({
            'Date': ts,
            'Equity': equity,
            'Cash': 0.0,
            'Position': qty,
            'Side': 1,
            'Close': price,
        })
    bt = pd.DataFrame(records).set_index('Date')
    bt['Ret'] = bt['Equity'].pct_change().fillna(0)
    bt.attrs['name'] = 'Buy&Hold'
    return bt

# ==========================
# メイン（引数なし）
# ==========================

def main():
    # 既定パラメータ
    TICKER = 'BTC-USD'
    SHORT_MA = 20
    LONG_MA  = 60
    RSI_PERIOD = 14
    BB_PERIOD = 20
    BB_K = 2.0
    FEE_BPS = 10
    SLIP_BPS = 1
    RISK_PCT = 1.0
    ALLOW_SHORT = False
    TRAIL = None
    INITIAL_EQUITY = 10_000.0  # 元資金 1万円

    df = load_btc_hourly_last3y(TICKER)

    sig_ma = signal_ma_cross(df, SHORT_MA, LONG_MA)
    sig_rsi = signal_rsi_reversion(df, RSI_PERIOD, low=30, high=70)
    sig_bb  = signal_bb_reversion(df, BB_PERIOD, BB_K)

    vote = sig_ma + sig_rsi + sig_bb
    sig_vote = pd.Series(0, index=df.index, dtype=float)
    sig_vote[vote >= 2] = 1.0
    sig_vote[vote <= -2] = -1.0

    ppy = 24*365

    curves = {}
    bt_ma   = backtest_from_signal(df, sig_ma,  'MA Cross',        FEE_BPS, SLIP_BPS, RISK_PCT, ALLOW_SHORT, TRAIL, INITIAL_EQUITY)
    bt_rsi  = backtest_from_signal(df, sig_rsi, 'RSI Reversion',   FEE_BPS, SLIP_BPS, RISK_PCT, ALLOW_SHORT, TRAIL, INITIAL_EQUITY)
    bt_bb   = backtest_from_signal(df, sig_bb,  'BB Reversion',    FEE_BPS, SLIP_BPS, RISK_PCT, ALLOW_SHORT, TRAIL, INITIAL_EQUITY)
    bt_vote = backtest_from_signal(df, sig_vote,'Voting(>=2 agree)',FEE_BPS, SLIP_BPS, RISK_PCT, ALLOW_SHORT, TRAIL, INITIAL_EQUITY)

    curves['MA Cross']          = bt_ma
    curves['RSI Reversion']     = bt_rsi
    curves['BB Reversion']      = bt_bb
    curves['Voting(>=2 agree)'] = bt_vote

    # Buy & Hold を追加
    bt_hold = buy_and_hold(df, INITIAL_EQUITY)
    curves['Buy&Hold'] = bt_hold

    perfs = [evaluate(bt, ppy) for bt in curves.values()]

    table = pd.DataFrame([
        {
            'Strategy': p.name,
            'CAGR': f"{p.cagr*100:.2f}%",
            'Sharpe': f"{p.sharpe:.2f}",
            'MaxDD': f"{p.maxdd*100:.2f}%",
            'WinRate': f"{p.winrate*100:.1f}%",
            'Trades': p.trades,
            'FinalEquity': f"¥{p.final_equity:,.0f}",
        } for p in perfs
    ])

    print("=== Performance Summary (approx last 3y; 1h up to 2y / else 4h) ===")
    print(table.to_string(index=False))

    # 図出力
    os.makedirs('plots', exist_ok=True)
    plot_price_with_signals(
        df,
        {f"SMA{SHORT_MA}": sma(df['Close'], SHORT_MA), f"SMA{LONG_MA}": sma(df['Close'], LONG_MA)},
        out_path='plots/price_ma.png',
        title=f"BTC-USD Price & SMAs",
    )

    r = rsi(df['Close'], RSI_PERIOD)
    plot_price_with_signals(
        df,
        {f"RSI({RSI_PERIOD})": r},
        out_path='plots/price_rsi.png',
        title=f"BTC-USD RSI",
    )

    mid, up, lo = bollinger(df['Close'], BB_PERIOD, BB_K)
    plot_price_with_signals(
        df,
        {"BB mid": mid, "BB up": up, "BB lo": lo},
        out_path='plots/price_bb.png',
        title=f"BTC-USD Bollinger Bands",
    )

    plot_equity_curves(curves, out_path='plots/equity_curves.png', title='Equity Curves')

if __name__ == '__main__':
    main()
