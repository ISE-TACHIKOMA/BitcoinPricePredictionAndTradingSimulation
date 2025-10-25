# 📌 BTC 自動売買バックテスト & シミュレーション（デフォルト自動実行版）

このプログラムは、**ビットコイン（BTC-USD）の過去データを用いて自動売買戦略をバックテストするツール**です。  
ユーザー入力なしで実行でき、**「移動平均クロス」「RSI逆張り」「ボリンジャーバンド逆張り」＋ 多数決戦略** をまとめて評価します。  
さらに **Buy & Hold（ガチホ）**も同時比較され、戦略の優劣が一目で分かる設計です。

---

## 🚀 特徴

| 機能カテゴリ | 内容 |
|------------|------|
| ✅ データ取得 | yfinance により**直近約3年のBTCデータを自動取得**（Yahoo制約で1h→ダメなら4hへ自動フォールバック） |
| ✅ 戦略比較 | MAクロス / RSI / ボリンジャーバンド / Voting(多数決) / Buy&Hold |
| ✅ トレード仕様 | 手数料・スリッページ・ポジション管理・トレーリングストップ対応 |
| ✅ 出力 | 成績指標（CAGR / Sharpe / MaxDD / WinRate など）＋ 各種グラフPNGを自動保存 |
| ✅ 完全自動実行 | `python btc_backtest_default.py` するだけ |

---

## 📌 実行方法

```bash
pip install pandas numpy matplotlib yfinance
python btc_backtest_default.py
```

---

## 📌 出力される内容

### 📄 **コンソール出力（例）**
| 指標 | 説明 |
|-------|------|
| CAGR | 年率換算リターン |
| Sharpe | リスク調整リターン |
| MaxDD | 最大ドローダウン |
| WinRate | 勝率 |
| Trades | 売買回数 |
| FinalEquity | 最終資産 |

```
=== Performance Summary (approx last 3y; 1h up to 2y / else 4h) ===
 Strategy           CAGR Sharpe  MaxDD WinRate Trades FinalEquity
 MA Cross         xx.xx%   x.xx -xx.xx%   xx.x%     xx   ¥xx,xxx
 RSI Reversion    xx.xx%   x.xx -xx.xx%   xx.x%     xx   ¥xx,xxx
 BB Reversion     xx.xx%   x.xx -xx.xx%   xx.x%     xx   ¥xx,xxx
 Voting           xx.xx%   x.xx -xx.xx%   xx.x%     xx   ¥xx,xxx
 Buy&Hold         xx.xx%   x.xx -xx.xx%   xx.x%     xx   ¥xx,xxx
```

---

### 📊 **保存されるグラフ**
出力先：`./plots/`

| ファイル | 内容 |
|----------|-------|
| `price_ma.png` | 指数移動平均シグナル表示 |
| `price_rsi.png` | RSI表示 |
| `price_bb.png` | ボリンジャーバンド表示 |
| `equity_curves.png` | 各戦略の資産曲線比較 |

例：

```
plots/
 ├─ price_ma.png
 ├─ price_rsi.png
 ├─ price_bb.png
 └─ equity_curves.png
```

（必要なら README にグラフを貼れます👇）

```markdown
![Equity Curves](./plots/equity_curves.png)
```

---

## 📌 評価している戦略

| 戦略名 | ロジック概要 |
|---------|--------------|
| MA Cross | 短期SMA > 長期SMAで買い、下回れば売り |
| RSI Reversion | RSI<30で買い / RSI>70で売り |
| Bollinger Reversion | バンド下部で買い / 上部で売り |
| Voting | 3戦略のうち賛成2つ以上で売買 |
| Buy&Hold | ベンチマーク（最初に買って放置） |

---

## 📌 設計方針

- シンプルな **戦略差比較ツール**
- 見やすい **資産曲線**
- 拡張しやすい構造（`signal_xxx()` に戦略を追加するだけ）

トレードの本質比較に集中できる構成にしてあります。

---

## 📌 注意事項（免責）

このコードは**過去データ検証用の学術・研究目的ツール**です。  
実売買の損益について作者は責任を負いません。

---

## 📌 ライセンス

MIT License
