# AnthropicDisruptionTradingStrategy-Backtest2
Trading Strategy and Backtest on Anthropic Disruption

=============================================================================
ANTHROPIC DISRUPTION TRADING STRATEGY
=============================================================================
Strategy Logic:
  1. Scrape Anthropic's latest announcements (website scrapping+ X/Twitter API)
  2. Use Claude Sonnet API to identify which sector(s) each announcement disrupts
  3. Pick Top 2-3 companies by market cap in that sector
  4. SHORT those companies: buy weekly put options OR short-sell the stock
  5. Hold for 3 trading days, then close position and capture profit

Backtest:
  - Replays historical Anthropic announcements (hardcoded catalog)
  - Pulls price data via yfinance
  - Simulates short-sell P&L over 3-day holding windows
  - Reports win rate, avg return, Sharpe, max drawdown
