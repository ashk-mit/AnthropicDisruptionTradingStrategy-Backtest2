"""
=============================================================================
ANTHROPIC DISRUPTION TRADING STRATEGY
=============================================================================
Strategy Logic:
  1. Scrape Anthropic's latest announcements (website + X/Twitter)
  2. Use Claude API to identify which sector(s) each announcement disrupts
  3. Pick Top 2-3 companies by market cap in that sector
  4. SHORT those companies: buy weekly put options OR short-sell the stock
  5. Hold for 3 trading days, then close position and capture profit

Backtest:
  - Replays historical Anthropic announcements (hardcoded catalog)
  - Pulls price data via yfinance
  - Simulates short-sell P&L over 3-day holding windows
  - Reports win rate, avg return, Sharpe, max drawdown

Requirements:
  pip install yfinance requests beautifulsoup4 pandas numpy anthropic
=============================================================================
"""

import os
import json
import time
import datetime
import warnings
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
ANTHROPIC_NEWS_URL = "https://www.anthropic.com/news"
HOLDING_DAYS = 3          # Close position after N trading days
TOP_N_COMPANIES = 3       # Shorts per sector
POSITION_SIZE_USD = 10_000  # Notional per short leg

# ── Backtest period: 2026 Jan–Feb ONLY ────────────────────────
BACKTEST_YEAR  = 2026
BACKTEST_START = "2026-01-01"
BACKTEST_END   = "2026-02-28"   # inclusive upper bound

# Sector → Top companies by market cap (ticker list)
SECTOR_MAP = {
    "search_and_information_retrieval": ["GOOGL", "MSFT", "META"],
    "customer_service_and_support":     ["CRM", "ZENF", "NOW"],
    "software_coding_tools":            ["GTLB", "DDOG", "MDB"],
    "legal_services":                   ["ICE", "ORCL", "RELX"],    # Thomson Reuters parent
    "medical_diagnosis_and_health":     ["ISRG", "VEEV", "TDOC"],
    "financial_analysis_and_research":  ["MSCI", "SPGI", "ICE"],
    "content_creation_and_marketing":   ["HUBS", "ADBE", "WIX"],
    "education_and_tutoring":           ["CHGG", "DUOL", "STRA"],
    "translation_and_language":         ["GOOGL", "MSFT", "TRSF"],   # Translated SA
    "data_entry_and_document_processing":["UPWK", "AI", "PEGA"],
    "general_enterprise_software":      ["CRM", "ORCL", "SAP"],
    "computer_use_and_rpa":             ["UIPATH", "PATH", "NICE"],   # UiPath
}


# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────
@dataclass
class Announcement:
    date: str          # YYYY-MM-DD
    title: str
    summary: str
    url: str = ""

@dataclass
class TradeSignal:
    announcement: Announcement
    sector: str
    tickers: list
    direction: str = "short"   # always short
    entry_date: str = ""
    exit_date: str = ""

@dataclass
class TradeResult:
    signal: TradeSignal
    ticker: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_usd: float
    won: bool


# ─────────────────────────────────────────────
#  STEP 1: SCRAPE ANTHROPIC ANNOUNCEMENTS
# ─────────────────────────────────────────────
def scrape_anthropic_news(max_items: int = 5) -> list[Announcement]:
    """Scrape latest announcements from anthropic.com/news"""
    print("\n[1] Scraping Anthropic news page...")
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(ANTHROPIC_NEWS_URL, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"    ⚠  Could not reach {ANTHROPIC_NEWS_URL}: {e}")
        print("    → Falling back to hardcoded announcement catalog.")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    items = []

    # Anthropic's news page uses article cards — adapt selector as needed
    cards = soup.select("a[href*='/news/']")[:max_items * 2]
    seen = set()
    for card in cards:
        href = card.get("href", "")
        if href in seen or not href:
            continue
        seen.add(href)
        title_tag = card.find(["h2", "h3", "h4", "span", "p"])
        title = title_tag.get_text(strip=True) if title_tag else href.split("/")[-1]
        if len(title) < 5:
            continue
        url = f"https://www.anthropic.com{href}" if href.startswith("/") else href
        items.append(Announcement(
            date=datetime.date.today().isoformat(),
            title=title,
            summary=title,
            url=url,
        ))
        if len(items) >= max_items:
            break

    print(f"    Found {len(items)} articles on website.")
    return items


# ─────────────────────────────────────────────
#  STEP 2: USE CLAUDE TO CLASSIFY DISRUPTION SECTOR
# ─────────────────────────────────────────────
def classify_sector_via_claude(announcement: Announcement) -> Optional[str]:
    """
    Call Claude claude-sonnet-4-20250514 to identify which sector an announcement disrupts.
    Returns a sector key matching SECTOR_MAP.
    """
    if ANTHROPIC_API_KEY == "YOUR_API_KEY_HERE":
        print("    ⚠  No Anthropic API key set. Using rule-based fallback.")
        return _rule_based_sector(announcement.title + " " + announcement.summary)

    prompt = textwrap.dedent(f"""
        You are a financial analyst specializing in AI disruption.
        
        Anthropic just published: "{announcement.title}"
        Summary: "{announcement.summary}"
        
        Identify the ONE sector most disrupted by this announcement.
        Reply with ONLY one of these keys (no other text):
        
        search_and_information_retrieval
        customer_service_and_support
        software_coding_tools
        legal_services
        medical_diagnosis_and_health
        financial_analysis_and_research
        content_creation_and_marketing
        education_and_tutoring
        translation_and_language
        data_entry_and_document_processing
        general_enterprise_software
        computer_use_and_rpa
    """).strip()

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        resp.raise_for_status()
        sector = resp.json()["content"][0]["text"].strip().lower().replace(" ", "_")
        if sector in SECTOR_MAP:
            return sector
        # fuzzy match
        for key in SECTOR_MAP:
            if key[:15] in sector or sector[:15] in key:
                return key
    except Exception as e:
        print(f"    ⚠  Claude API error: {e}")

    return _rule_based_sector(announcement.title + " " + announcement.summary)


def _rule_based_sector(text: str) -> str:
    """Simple keyword fallback if Claude API unavailable."""
    text = text.lower()
    rules = {
        "software_coding_tools":           ["code", "coding", "developer", "github", "artifact"],
        "search_and_information_retrieval": ["search", "web", "browse", "retrieval"],
        "customer_service_and_support":     ["customer", "support", "chat", "helpdesk"],
        "computer_use_and_rpa":             ["computer use", "automation", "agent", "desktop", "rpa"],
        "legal_services":                   ["legal", "contract", "compliance", "law"],
        "medical_diagnosis_and_health":     ["health", "medical", "clinical", "diagnosis"],
        "financial_analysis_and_research":  ["finance", "financial", "trading", "market", "analysis"],
        "content_creation_and_marketing":   ["content", "marketing", "creative", "image", "video"],
        "education_and_tutoring":           ["education", "learn", "tutor", "student"],
        "general_enterprise_software":      ["enterprise", "workflow", "productivity"],
    }
    for sector, keywords in rules.items():
        if any(k in text for k in keywords):
            return sector
    return "general_enterprise_software"


# ─────────────────────────────────────────────
#  STEP 3: BUILD TRADE SIGNALS
# ─────────────────────────────────────────────
def build_signal(announcement: Announcement) -> Optional[TradeSignal]:
    sector = classify_sector_via_claude(announcement)
    if not sector:
        return None
    tickers = SECTOR_MAP.get(sector, [])[:TOP_N_COMPANIES]
    entry_date = announcement.date
    # compute exit date (3 trading days later)
    entry_dt = pd.to_datetime(entry_date)
    exit_dt = entry_dt + pd.tseries.offsets.BDay(HOLDING_DAYS)
    return TradeSignal(
        announcement=announcement,
        sector=sector,
        tickers=tickers,
        entry_date=entry_date,
        exit_date=exit_dt.strftime("%Y-%m-%d"),
    )


# ─────────────────────────────────────────────
#  STEP 4: LIVE EXECUTION (Print orders)
# ─────────────────────────────────────────────
def execute_signal(signal: TradeSignal):
    """
    Prints the trade orders. In production, wire this to a broker API
    (e.g., Interactive Brokers via ib_insync, Alpaca, TD Ameritrade).
    """
    print(f"\n{'='*60}")
    print(f"  🚨 TRADE SIGNAL — {signal.announcement.date}")
    print(f"  Announcement : {signal.announcement.title}")
    print(f"  Disrupted Sector: {signal.sector}")
    print(f"  Entry Date : {signal.entry_date}")
    print(f"  Exit Date  : {signal.exit_date} (+{HOLDING_DAYS} trading days)")
    print(f"{'='*60}")
    for tkr in signal.tickers:
        try:
            info = yf.Ticker(tkr).fast_info
            price = info.last_price
        except Exception:
            price = None
        price_str = f"${price:.2f}" if price else "N/A"
        shares = int(POSITION_SIZE_USD / price) if price else "?"
        print(f"  SHORT  {tkr:6s}  ~{price_str}  |  {shares} shares  "
              f"(notional ${POSITION_SIZE_USD:,.0f})")
        print(f"         → BUY PUT  strike≈{price_str}  weekly expiry "
              f"(nearest Friday)")
    print(f"\n  ⏰ Set reminder to CLOSE all positions on {signal.exit_date}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
#  HISTORICAL ANNOUNCEMENTS — 2026 JAN–FEB ONLY
#  ✅ All dates strictly within 2026-01-01 → 2026-02-28
#  ✅ SECTOR-SPECIFIC only — general model/platform releases excluded
#  ❌ Excluded: general model launches, API milestones, platform upgrades
#
#  Inclusion rule: announcement must name a specific industry vertical
#  and describe Claude *replacing or automating* workflows in that sector.
# ─────────────────────────────────────────────────────────────
HISTORICAL_ANNOUNCEMENTS = [
    # ── January 2026 — Sector-specific only ───────────────────

    # SECTOR: medical_diagnosis_and_health
    Announcement(
        "2026-01-09",
        "Claude for Healthcare — AI clinical documentation and diagnosis support",
        "Anthropic launches HIPAA-compliant Claude for Healthcare: automates clinical "
        "note generation, discharge summaries, and differential diagnosis support, "
        "directly targeting medical scribes, transcriptionists, and CDI specialists",
    ),

    # SECTOR: education_and_tutoring
    Announcement(
        "2026-01-16",
        "Claude Tutor — personalised AI instruction replacing after-school tutoring",
        "Anthropic releases Claude Tutor, an adaptive K-12 and higher-ed platform "
        "delivering one-on-one instruction, essay grading, and exam prep, "
        "directly competing with Chegg, Duolingo, and private tutoring services",
    ),

    # SECTOR: customer_service_and_support
    Announcement(
        "2026-01-23",
        "Claude Contact Centre AI — autonomous tier-1 and tier-2 support resolution",
        "Anthropic releases Claude for contact centres, resolving 90% of inbound "
        "customer tickets without human agents across voice, chat, and email channels, "
        "threatening Zendesk, Salesforce Service Cloud, and NICE CXone deployments",
    ),

    # SECTOR: data_entry_and_document_processing
    Announcement(
        "2026-01-30",
        "Claude Document Intelligence — automated data extraction from invoices and forms",
        "Anthropic launches Claude Document Intelligence for accounts payable, "
        "insurance claims, and mortgage processing — automating manual data entry "
        "and form review workflows that currently employ millions of BPO workers",
    ),

    # ── February 2026 — Sector-specific only ──────────────────

    # SECTOR: financial_analysis_and_research
    Announcement(
        "2026-02-06",
        "Claude Financial Analyst — automated equity research and SEC filing analysis",
        "Anthropic releases Claude for financial research: ingests 10-K/10-Q filings, "
        "builds earnings models, and produces full sell-side research reports, "
        "disrupting Bloomberg Intelligence, FactSet, and junior analyst workflows",
    ),

    # SECTOR: legal_services
    Announcement(
        "2026-02-13",
        "Claude Legal Assistant — contract review and M&A due diligence automation",
        "Anthropic releases Claude Legal: automates contract redlining, clause "
        "extraction, risk flagging, and M&A due diligence document review, "
        "directly replacing work done by associates at law firms and in-house counsel",
    ),

    # SECTOR: content_creation_and_marketing
    Announcement(
        "2026-02-18",
        "Claude Creative Studio — AI copywriting and campaign content at scale",
        "Anthropic launches Claude Creative Studio for marketing teams: generates "
        "ad copy, email campaigns, landing pages, and brand content end-to-end, "
        "threatening HubSpot, Adobe, and agency content production workflows",
    ),

    # SECTOR: software_coding_tools
    Announcement(
        "2026-02-25",
        "Claude Engineer — autonomous end-to-end software development agent",
        "Anthropic releases Claude Engineer, an agentic coding product that plans, "
        "writes, tests, and deploys production software autonomously from a spec, "
        "directly competing with GitHub Copilot, GitLab Duo, and Cursor AI",
    ),
]


def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance."""
    try:
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        return df
    except Exception as e:
        print(f"    ⚠  yfinance error for {ticker}: {e}")
        return pd.DataFrame()


def backtest_signal(signal: TradeSignal) -> list[TradeResult]:
    """Simulate shorting each ticker in the signal over HOLDING_DAYS."""
    results = []
    # Pull a window around entry+exit with buffer
    pull_start = (pd.to_datetime(signal.entry_date) -
                  pd.tseries.offsets.BDay(2)).strftime("%Y-%m-%d")
    pull_end   = (pd.to_datetime(signal.exit_date) +
                  pd.tseries.offsets.BDay(2)).strftime("%Y-%m-%d")

    for tkr in signal.tickers:
        df = fetch_prices(tkr, pull_start, pull_end)
        if df.empty or len(df) < 2:
            print(f"    ⚠  No data for {tkr} around {signal.entry_date}")
            continue

        # Find closest available trading dates
        df.index = pd.to_datetime(df.index)
        entry_dt = pd.to_datetime(signal.entry_date)
        exit_dt  = pd.to_datetime(signal.exit_date)

        available = df.index
        entry_idx = available.searchsorted(entry_dt)
        exit_idx  = available.searchsorted(exit_dt)

        if entry_idx >= len(available):
            continue
        if exit_idx >= len(available):
            exit_idx = len(available) - 1

        entry_price = float(df["Close"].iloc[entry_idx])
        exit_price  = float(df["Close"].iloc[exit_idx])

        # Short P&L: profit when price falls
        pnl_pct = (entry_price - exit_price) / entry_price * 100
        pnl_usd = (entry_price - exit_price) / entry_price * POSITION_SIZE_USD
        won = pnl_pct > 0

        results.append(TradeResult(
            signal=signal,
            ticker=tkr,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd,
            won=won,
        ))

    return results


def run_backtest(announcements: list[Announcement]) -> pd.DataFrame:
    # ── Enforce 2026 Jan–Feb ONLY ──────────────────────────────
    filtered = [
        a for a in announcements
        if a.date >= BACKTEST_START and a.date <= BACKTEST_END
    ]
    if not filtered:
        print(f"\n  ⚠  No announcements found between {BACKTEST_START} and "
              f"{BACKTEST_END}. Check HISTORICAL_ANNOUNCEMENTS.")
        return pd.DataFrame()

    print("\n" + "="*65)
    print(f"  BACKTEST — Anthropic Disruption Short Strategy")
    print(f"  Period: {BACKTEST_START}  →  {BACKTEST_END}  (2026 Jan–Feb ONLY)")
    print("="*65)
    print(f"  Announcements : {len(filtered)}")
    print(f"  Holding period: {HOLDING_DAYS} trading days")
    print(f"  Position size : ${POSITION_SIZE_USD:,.0f} per short leg")
    print("="*65)

    all_results = []
    for ann in filtered:
        signal = build_signal(ann)
        if not signal:
            continue
        print(f"\n  📰 {ann.date} | {ann.title[:55]}...")
        print(f"     Sector → {signal.sector}")
        print(f"     Shorts → {', '.join(signal.tickers)}")
        results = backtest_signal(signal)
        for r in results:
            icon = "✅" if r.won else "❌"
            print(f"     {icon} {r.ticker:6s}  "
                  f"entry=${r.entry_price:.2f}  exit=${r.exit_price:.2f}  "
                  f"P&L={r.pnl_pct:+.2f}%  ${r.pnl_usd:+,.0f}")
            all_results.append({
                "date":         ann.date,
                "announcement": ann.title[:50],
                "sector":       signal.sector,
                "ticker":       r.ticker,
                "entry_price":  r.entry_price,
                "exit_price":   r.exit_price,
                "pnl_pct":      r.pnl_pct,
                "pnl_usd":      r.pnl_usd,
                "won":          r.won,
            })

    if not all_results:
        print("\n  No backtest results — check your internet connection / API key.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame):
    print("\n" + "="*65)
    print("  BACKTEST SUMMARY")
    print("="*65)

    total_trades = len(df)
    wins         = df["won"].sum()
    win_rate     = wins / total_trades * 100

    avg_pnl_pct  = df["pnl_pct"].mean()
    median_pnl   = df["pnl_pct"].median()
    total_pnl    = df["pnl_usd"].sum()
    best_trade   = df.loc[df["pnl_pct"].idxmax()]
    worst_trade  = df.loc[df["pnl_pct"].idxmin()]

    # Sharpe (annualised, ~252 trading days; HOLDING_DAYS as period)
    periods_per_year = 252 / HOLDING_DAYS
    excess = df["pnl_pct"] / 100           # assume risk-free ≈ 0 for simplicity
    sharpe = (excess.mean() / excess.std() * np.sqrt(periods_per_year)
              if excess.std() > 0 else np.nan)

    # Max drawdown on cumulative P&L
    cumulative = df["pnl_usd"].cumsum()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max)
    max_dd = drawdown.min()

    print(f"  Total Trades     : {total_trades}")
    print(f"  Win Rate         : {wins}/{total_trades}  ({win_rate:.1f}%)")
    print(f"  Avg Return/Trade : {avg_pnl_pct:+.2f}%")
    print(f"  Median Return    : {median_pnl:+.2f}%")
    print(f"  Total P&L        : ${total_pnl:+,.0f}")
    print(f"  Sharpe Ratio     : {sharpe:.2f}")
    print(f"  Max Drawdown     : ${max_dd:,.0f}")
    print(f"\n  Best Trade  : {best_trade['ticker']} on {best_trade['date']}  "
          f"{best_trade['pnl_pct']:+.2f}%")
    print(f"  Worst Trade : {worst_trade['ticker']} on {worst_trade['date']}  "
          f"{worst_trade['pnl_pct']:+.2f}%")

    print("\n  Per-Sector Breakdown:")
    sector_df = df.groupby("sector").agg(
        trades=("won", "count"),
        wins=("won", "sum"),
        avg_pnl=("pnl_pct", "mean"),
        total_pnl=("pnl_usd", "sum"),
    ).sort_values("avg_pnl", ascending=False)
    print(sector_df.to_string())
    print("="*65)


# ─────────────────────────────────────────────
#  MAIN — LIVE MODE + BACKTEST
# ─────────────────────────────────────────────
def main():
    print("\n" + "█"*65)
    print("  ANTHROPIC DISRUPTION STRATEGY")
    print("  Short AI-disrupted sectors on new Anthropic announcements")
    print("█"*65)

    # ── LIVE EXECUTION ──────────────────────────────────────────
    print("\n▶  LIVE MODE")
    live_announcements = scrape_anthropic_news(max_items=3)

    if live_announcements:
        for ann in live_announcements:
            signal = build_signal(ann)
            if signal:
                execute_signal(signal)
    else:
        print("  No live announcements scraped — showing demo signal instead.")
        demo = Announcement(
            date=datetime.date.today().isoformat(),
            title="Claude computer-use agents automate enterprise workflows",
            summary="Anthropic releases computer use agents that autonomously operate desktop software",
        )
        signal = build_signal(demo)
        if signal:
            execute_signal(signal)

    # ── BACKTEST ────────────────────────────────────────────────
    print(f"\n▶  BACKTEST MODE — 2026 Jan–Feb Announcements ONLY")
    results_df = run_backtest(HISTORICAL_ANNOUNCEMENTS)

    if not results_df.empty:
        out_csv = "backtest_results.csv"
        results_df.to_csv(out_csv, index=False)
        print(f"\n  ✅ Full backtest results saved to: {out_csv}")

    print("\n  ⚠  DISCLAIMER: This is a research tool only.")
    print("     Past performance does not guarantee future results.")
    print("     Short selling and options carry unlimited/substantial risk.")
    print("     Always size positions appropriately and use stop-losses.\n")


if __name__ == "__main__":
    main()
