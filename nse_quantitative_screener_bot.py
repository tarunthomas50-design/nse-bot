"""
NSE QUANTITATIVE SCREENER v3.0 — FREE CLOUD + TELEGRAM BOT
===========================================================
100% free deployment | No paid services needed
Telegram bot handles: /scan /manage /enter /close /trim /status /help

Environment variables required:
  BOT_TOKEN, CHAT_ID, ANTHROPIC_API_KEY (optional)

Deployment: PythonAnywhere (free tier) — 1 scheduled task + polling
"""

import os
import sys
import json
import re
import warnings
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import requests
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

if not BOT_TOKEN or not CHAT_ID:
    print("ERROR: BOT_TOKEN and CHAT_ID environment variables required")
    sys.exit(1)

if CHAT_ID.isdigit():
    CHAT_ID = int(CHAT_ID)

CAPITAL = 10000
MAX_POSITIONS = 3
RISK_PER_TRADE_PCT = 0.015
TARGET_PCT = 0.10
STOP_LOSS_PCT = 0.05
MIN_VOLUME_RATIO = 1.5
RSI_MIN = 50
RSI_MAX = 70
TIME_STOP_DAYS = 15

MASTER_SCORE_MIN_FOR_NEW_TRADES = 40
VIX_MAX_FOR_NEW_TRADES = 22

MOMENTUM_WEIGHT = 0.35
QUALITY_WEIGHT = 0.35
RISK_WEIGHT = 0.15
SENTIMENT_ALIGNMENT_WEIGHT = 0.15
COMPOSITE_MIN_FOR_RECOMMENDATION = 60

ENABLE_CLAUDE = bool(ANTHROPIC_API_KEY)
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_COST_USD_PER_RUN = 3.0
CLAUDE_CACHE_TTL_DAYS = 30
CLAUDE_ANALYZE_TOP_N = 2

MAX_SECTOR_PCT = 40.0
MAX_CORRELATION = 0.80
MIN_ADV_LAKHS = 50.0
MAX_POSITION_PCT = 5.0
MAX_GROSS_EXPOSURE = 105.0
MAX_NET_BETA = 0.20
EARNINGS_BLACKOUT_DAYS = 2

DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

BASE_DIR = Path(__file__).parent
BASE_DIR.mkdir(parents=True, exist_ok=True)

PAPER_TRADE_FILE = BASE_DIR / "paper_trades.json"
CANDIDATE_FILE = BASE_DIR / "daily_candidates.json"
SENTIMENT_FILE = BASE_DIR / "daily_sentiment.json"
CLAUDE_CACHE_FILE = BASE_DIR / "claude_analysis_cache.json"
CLAUDE_COST_FILE = BASE_DIR / "claude_cost_log.json"
TRADES_BACKUP_DIR = BASE_DIR / "trades_backups"
LOG_FILE = BASE_DIR / "daily_run.log"
os.makedirs(TRADES_BACKUP_DIR, exist_ok=True)

BENCHMARK_TICKER = "^NSEI"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

WATCHLIST = [
    "HINDPETRO.NS", "M&MFIN.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
    "INFY.NS", "SBIN.NS", "LT.NS", "ITC.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "ADANIENT.NS", "POWERGRID.NS", "NTPC.NS"
]

# ============================================================
# LOGGING
# ============================================================
class Color:
    RESET = "\033[0m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    WHITE = "\033[97m"

def log_info(msg): 
    print(f"{Color.CYAN}[INFO]{Color.RESET} {msg}")
    _file_log(msg)

def log_ok(msg): 
    print(f"{Color.GREEN}[OK]{Color.RESET} {msg}")
    _file_log(f"OK: {msg}")

def log_warn(msg): 
    print(f"{Color.YELLOW}[WARN]{Color.RESET} {msg}")
    _file_log(f"WARN: {msg}")

def log_error(msg): 
    print(f"{Color.RED}[ERROR]{Color.RESET} {msg}")
    _file_log(f"ERROR: {msg}")

def log_alpha(msg): 
    print(f"{Color.BOLD}{Color.WHITE}[ALPHA]{Color.RESET} {msg}")
    _file_log(f"ALPHA: {msg}")

def _file_log(msg):
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

# ============================================================
# TELEGRAM
# ============================================================

def send_telegram_message(message: str, chat_id=None):
    if DRY_RUN:
        log_info(f"[DRY RUN] {message[:200]}...")
        return
    cid = chat_id if chat_id else CHAT_ID
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": cid, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=20)
    except Exception:
        try:
            requests.post(url, json=payload, timeout=20)
        except Exception as e2:
            log_error(f"Telegram send failed: {e2}")

def get_telegram_updates(offset=None):
    url = f"{TELEGRAM_API_URL}/getUpdates"
    params = {"offset": offset, "limit": 10} if offset else {"limit": 10}
    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        if data.get("ok"):
            return data.get("result", [])
    except Exception as e:
        log_warn(f"Telegram poll error: {e}")
    return []

# ============================================================
# JSON UTILITIES
# ============================================================
def safe_load_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    except Exception as e:
        log_error(f"JSON load failed ({path.name}): {e}")
        return None

def safe_save_json(path: Path, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

# ============================================================
# PANDAS / YFINANCE HELPERS
# ============================================================
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def safe_scalar(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if hasattr(val, "item"):
        return val.item()
    try:
        return float(val)
    except Exception:
        return val

def fetch_yf_history(ticker: str, period: str = "120d", interval: str = "1d") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = flatten_columns(df)
        df = df.reset_index().rename(columns={"Date": "DATE"})
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                return None
        return df
    except Exception as e:
        log_warn(f"Yahoo fetch failed for {ticker}: {e}")
        return None

def fetch_latest_price(ticker: str) -> Optional[float]:
    df = fetch_yf_history(ticker, period="5d", interval="1d")
    if df is not None and not df.empty:
        return safe_scalar(df["Close"].iloc[-1])
    try:
        symbol = ticker.replace(".NS", "")
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        })
        session.get("https://www.nseindia.com", timeout=10)
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        r = session.get(url, timeout=10)
        return float(r.json()["priceInfo"]["lastPrice"])
    except Exception:
        return None

# ============================================================
# FUNDAMENTAL CACHE
# ============================================================
class FundamentalCache:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache = safe_load_json(cache_path) or {}
        self.today = datetime.now().strftime("%Y-%m-%d")

    def get(self, ticker: str) -> Optional[Dict]:
        entry = self.cache.get(ticker)
        if entry and entry.get("date") == self.today:
            return entry.get("data")
        return None

    def set(self, ticker: str, data: Dict):
        self.cache[ticker] = {"date": self.today, "data": data}
        safe_save_json(self.cache_path, self.cache)

_fund_cache = FundamentalCache(BASE_DIR / "fundamental_cache.json")

def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    cached = _fund_cache.get(ticker)
    if cached:
        return cached

    result = {
        "roe": None, "debt_to_equity": None, "current_ratio": None,
        "gross_margin": None, "operating_margin": None, "ebitda": None,
        "net_income": None, "total_assets": None, "total_liabilities": None,
        "market_cap": None, "shares_outstanding": None, "revenue": None,
        "free_cash_flow": None, "earnings_growth": None, "revenue_growth": None,
        "sector": "Unknown", "industry": "Unknown",
        "piotroski_available": False, "altman_available": False,
    }

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        result["roe"] = safe_scalar(info.get("returnOnEquity"))
        result["debt_to_equity"] = safe_scalar(info.get("debtToEquity"))
        result["current_ratio"] = safe_scalar(info.get("currentRatio"))
        result["gross_margin"] = safe_scalar(info.get("grossMargins"))
        result["operating_margin"] = safe_scalar(info.get("operatingMargins"))
        result["market_cap"] = safe_scalar(info.get("marketCap"))
        result["shares_outstanding"] = safe_scalar(info.get("sharesOutstanding"))
        result["revenue"] = safe_scalar(info.get("totalRevenue"))
        result["earnings_growth"] = safe_scalar(info.get("earningsGrowth"))
        result["revenue_growth"] = safe_scalar(info.get("revenueGrowth"))
        result["sector"] = info.get("sector", "Unknown")
        result["industry"] = info.get("industry", "Unknown")
        result["ebitda"] = safe_scalar(info.get("ebitda"))
        result["total_debt"] = safe_scalar(info.get("totalDebt"))
        result["total_cash"] = safe_scalar(info.get("totalCash"))

        bs = stock.balance_sheet
        fin = stock.financials
        cf = stock.cashflow

        if bs is not None and not bs.empty and fin is not None and not fin.empty:
            try:
                c0 = bs.columns[0]
                c1 = bs.columns[1] if len(bs.columns) > 1 else c0
                f0 = fin.columns[0]
                f1 = fin.columns[1] if len(fin.columns) > 1 else f0

                ta0 = safe_scalar(bs.loc["Total Assets", c0]) if "Total Assets" in bs.index else None
                ta1 = safe_scalar(bs.loc["Total Assets", c1]) if "Total Assets" in bs.index else None
                tl0 = safe_scalar(bs.loc["Total Liabilities Net Minority Interest", c0]) if "Total Liabilities Net Minority Interest" in bs.index else None
                tl1 = safe_scalar(bs.loc["Total Liabilities Net Minority Interest", c1]) if "Total Liabilities Net Minority Interest" in bs.index else None

                ni0 = safe_scalar(fin.loc["Net Income", f0]) if "Net Income" in fin.index else None
                ni1 = safe_scalar(fin.loc["Net Income", f1]) if "Net Income" in fin.index else None
                rev0 = safe_scalar(fin.loc["Total Revenue", f0]) if "Total Revenue" in fin.index else None
                rev1 = safe_scalar(fin.loc["Total Revenue", f1]) if "Total Revenue" in fin.index else None
                gp0 = safe_scalar(fin.loc["Gross Profit", f0]) if "Gross Profit" in fin.index else None
                gp1 = safe_scalar(fin.loc["Gross Profit", f1]) if "Gross Profit" in fin.index else None

                cfo0 = safe_scalar(cf.loc["Operating Cash Flow", c0]) if cf is not None and "Operating Cash Flow" in cf.index else None

                result["total_assets"] = ta0
                result["total_liabilities"] = tl0
                result["net_income"] = ni0
                result["revenue"] = rev0 if rev0 else result["revenue"]
                result["cfo"] = cfo0
                result["gross_profit"] = gp0

                f_score = 0
                if ni0 and ni0 > 0: f_score += 1
                if cfo0 and cfo0 > 0: f_score += 1
                if ni0 and ni1 and ni0 > ni1: f_score += 1
                if cfo0 and ni0 and cfo0 > ni0: f_score += 1
                if tl0 and tl1 and tl0 < tl1: f_score += 1
                if ta0 and ta1 and ta0 > ta1: f_score += 1
                if rev0 and rev1 and ta0 and ta1 and (rev0/ta0) > (rev1/ta1): f_score += 1
                if gp0 and gp1 and rev0 and rev1 and (gp0/rev0) > (gp1/rev1): f_score += 1

                result["piotroski_score"] = f_score
                result["piotroski_available"] = True

                if (ta0 and tl0 and rev0 and ni0 and result["market_cap"] and result["market_cap"] > 0):
                    wc = (safe_scalar(bs.loc["Working Capital", c0]) if "Working Capital" in bs.index else None) or (ta0 * 0.1)
                    re = safe_scalar(bs.loc["Retained Earnings", c0]) if "Retained Earnings" in bs.index else (ni0 * 3)
                    ebit = safe_scalar(fin.loc["EBIT", f0]) if "EBIT" in fin.index else (ni0 * 1.2)
                    if ebit and re is not None:
                        z = (1.2 * (wc / ta0) + 1.4 * (re / ta0) + 3.3 * (ebit / ta0) + 0.6 * (result["market_cap"] / tl0) + 1.0 * (rev0 / ta0))
                        result["altman_z"] = z
                        result["altman_available"] = True
            except Exception as e:
                log_warn(f"Fundamental parse error for {ticker}: {e}")
    except Exception as e:
        log_warn(f"Fundamental fetch error for {ticker}: {e}")

    _fund_cache.set(ticker, result)
    return result


# ============================================================
# L2 SCORING ENGINE
# ============================================================

def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    last_rs = safe_scalar(rs.iloc[-1])
    if last_rs is None or last_rs == 0:
        return 0.0 if last_rs == 0 else 50.0
    return 100.0 - (100.0 / (1.0 + last_rs))

def calculate_momentum_score(ticker: str, df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> Tuple[float, Dict]:
    if df is None or len(df) < 60:
        return 50.0, {}
    df = df.sort_values("DATE").reset_index(drop=True)
    closes = df["Close"].values
    latest_idx = len(closes) - 1
    idx_1m = max(0, latest_idx - 21)
    idx_3m = max(0, latest_idx - 63)
    idx_6m = max(0, latest_idx - 126)
    idx_12m = max(0, latest_idx - 252)

    def pct_return(s, e):
        if s < 0 or e >= len(closes):
            return 0.0
        return (closes[e] - closes[s]) / closes[s] * 100

    ret_12_1 = pct_return(idx_12m, idx_1m)
    ret_6m = pct_return(idx_6m, latest_idx)
    ret_3m = pct_return(idx_3m, latest_idx)
    high_52w = df["High"].max()
    current = closes[-1]
    proximity = (current / high_52w) * 100 if high_52w > 0 else 50
    rsi = calculate_rsi(df["Close"])

    rs_score = 50.0
    if benchmark_df is not None and not benchmark_df.empty:
        try:
            bench = benchmark_df.sort_values("DATE").reset_index(drop=True)
            merged = pd.merge(df[["DATE", "Close"]], bench[["DATE", "Close"]], on="DATE", suffixes=("", "_b"))
            if len(merged) >= 63:
                stock_ret = (merged["Close"].iloc[-1] - merged["Close"].iloc[-63]) / merged["Close"].iloc[-63] * 100
                bench_ret = (merged["Close_b"].iloc[-1] - merged["Close_b"].iloc[-63]) / merged["Close_b"].iloc[-63] * 100
                rs_raw = stock_ret - bench_ret
                rs_score = max(0, min(100, 50 + rs_raw * 1.5))
        except Exception:
            pass

    sub = {"ret_12_1": round(ret_12_1, 2), "ret_6m": round(ret_6m, 2), "ret_3m": round(ret_3m, 2),
           "proximity_52w": round(proximity, 2), "rsi": round(rsi, 1), "rel_strength": round(rs_score, 1)}

    def norm_return(v):
        return max(0, min(100, (v + 20) / 70 * 100))
    s1, s2, s3, s4, s5, s6 = norm_return(ret_12_1), norm_return(ret_6m), norm_return(ret_3m), proximity, max(0, min(100, (rsi - 30) / 40 * 100)), rs_score
    return round(np.mean([s1, s2, s3, s4, s5, s6]), 2), sub

def calculate_quality_score(ticker: str, fundamentals: Dict) -> Tuple[float, Dict]:
    sub = {}
    if fundamentals.get("piotroski_available"):
        sub["piotroski"] = round((fundamentals["piotroski_score"] / 9.0) * 100, 1)
    else:
        sub["piotroski"] = 50.0

    roe = fundamentals.get("roe")
    sub["roe"] = max(0, min(100, roe * 300)) if roe is not None and roe != 0 else 50.0

    de = fundamentals.get("debt_to_equity")
    sub["debt_equity"] = max(0, min(100, 100 - de)) if de is not None and de >= 0 else 50.0

    cfo, ni = fundamentals.get("cfo"), fundamentals.get("net_income")
    sub["cfo_ni"] = max(0, min(100, (cfo / ni) * 50)) if cfo and ni and ni != 0 else 50.0

    gm = fundamentals.get("gross_margin")
    sub["gross_margin"] = max(0, min(100, gm * 200)) if gm is not None and gm > 0 else 50.0

    if fundamentals.get("altman_available"):
        z = fundamentals["altman_z"]
        sub["altman"] = 100.0 if z > 2.99 else (60.0 if z > 1.81 else 20.0)
    else:
        sub["altman"] = 50.0

    return round(np.mean([sub["piotroski"], sub["roe"], sub["debt_equity"], sub["cfo_ni"], sub["gross_margin"], sub["altman"]]), 2), sub

def calculate_risk_score(ticker: str, df: pd.DataFrame, fundamentals: Dict) -> Tuple[float, Dict]:
    if df is None or len(df) < 20:
        return 50.0, {"volatility": 50, "drawdown": 50, "liquidity": 50}
    df = df.sort_values("DATE")
    closes = df["Close"].values
    returns = np.diff(closes) / closes[:-1]
    vol = np.std(returns) * np.sqrt(252) * 100
    vol_score = max(0, min(100, (60 - vol) / 40 * 100))
    cummax = np.maximum.accumulate(closes)
    dd = (closes - cummax) / cummax
    max_dd = np.min(dd) * 100
    dd_score = max(0, min(100, (max_dd + 30) / 30 * 100))
    adv = (df["Volume"].rolling(20).mean().iloc[-1] * closes[-1]) if len(df) >= 20 else 0
    mcap = fundamentals.get("market_cap") or 0
    if mcap > 0 and adv > 0:
        liq_score = max(0, min(100, (adv / mcap) * 100000))
    else:
        liq_score = 50.0 if adv > 0 else 20.0
    sub = {"volatility": round(vol, 2), "vol_score": round(vol_score, 1), "max_drawdown": round(max_dd, 2),
           "drawdown_score": round(dd_score, 1), "adv_inr": round(adv, 0), "liquidity_score": round(liq_score, 1)}
    return round(np.mean([vol_score, dd_score, liq_score]), 2), sub

def calculate_composite_score(ticker: str, momentum: float, quality: float, risk: float, sentiment: Dict) -> Tuple[float, Dict]:
    regime = sentiment.get("market_regime", "NEUTRAL")
    sent_score = 70.0 if regime == "BULLISH" else (40.0 if regime == "BEARISH" else 55.0)
    composite = (MOMENTUM_WEIGHT * momentum + QUALITY_WEIGHT * quality + RISK_WEIGHT * risk + SENTIMENT_ALIGNMENT_WEIGHT * sent_score)
    return round(composite, 2), {"momentum": momentum, "quality": quality, "risk": risk, "sentiment": sent_score, "composite": round(composite, 2)}

# ============================================================
# L3: CLAUDE AI
# ============================================================

class ClaudeAnalyzer:
    def __init__(self):
        self.client = None
        self.cache = safe_load_json(CLAUDE_CACHE_FILE) or {}
        self.cost_log = safe_load_json(CLAUDE_COST_FILE) or {"total_usd": 0.0, "runs": 0, "history": []}
        self._init_client()

    def _init_client(self):
        if not ENABLE_CLAUDE or not ANTHROPIC_API_KEY:
            return
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            log_ok("Claude SDK initialized.")
        except ImportError:
            log_warn("anthropic package not installed. L3 disabled.")
        except Exception as e:
            log_warn(f"Claude init error: {e}")

    def _get_cache_key(self, ticker: str) -> str:
        return f"{ticker}_{datetime.now().strftime('%Y-%m')}"

    def _is_cached(self, ticker: str) -> Optional[Dict]:
        key = self._get_cache_key(ticker)
        entry = self.cache.get(key)
        if not entry:
            return None
        try:
            cached_date = datetime.strptime(entry["date"], "%Y-%m-%d")
            if (datetime.now() - cached_date).days <= CLAUDE_CACHE_TTL_DAYS:
                return entry["data"]
        except Exception:
            pass
        return None

    def _save_cache(self, ticker: str, data: Dict):
        key = self._get_cache_key(ticker)
        self.cache[key] = {"date": datetime.now().strftime("%Y-%m-%d"), "data": data}
        safe_save_json(CLAUDE_CACHE_FILE, self.cache)

    def _track_cost(self, input_tokens: int, output_tokens: int):
        cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000
        self.cost_log["total_usd"] += cost
        self.cost_log["runs"] += 1
        self.cost_log["history"].append({
            "date": datetime.now().isoformat(),
            "cost_usd": round(cost, 4),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })
        safe_save_json(CLAUDE_COST_FILE, self.cost_log)
        return cost

    def analyze(self, candidate: Dict, fundamentals: Dict) -> Optional[Dict]:
        if not self.client:
            return None
        run_cost_so_far = sum(h["cost_usd"] for h in self.cost_log["history"]
                              if h["date"].startswith(datetime.now().strftime("%Y-%m-%d")))
        if run_cost_so_far >= CLAUDE_MAX_COST_USD_PER_RUN:
            log_warn(f"Claude daily cost ${run_cost_so_far:.2f} >= ${CLAUDE_MAX_COST_USD_PER_RUN}. Skipping.")
            return None
        cached = self._is_cached(candidate["ticker"])
        if cached:
            log_info(f"Claude cache hit for {candidate['ticker']}")
            return cached

        prompt = self._build_prompt(candidate, fundamentals)
        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=600,
                system=("You are JARVIS, a senior Indian equity analyst. "
                        "Review breakout candidates. Respond ONLY in valid JSON."),
                messages=[{"role": "user", "content": prompt}],
            )
            cost = self._track_cost(response.usage.input_tokens, response.usage.output_tokens)
            log_ok(f"Claude call for {candidate['ticker']}: ${cost:.4f}")
            text = response.content[0].text
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text)
            result = json.loads(text)
            expected = {"verdict", "confidence", "bull_case", "bear_case", "key_risk", "position_size_note"}
            if not expected.issubset(result.keys()):
                log_warn(f"Claude response missing keys for {candidate['ticker']}")
                return None
            self._save_cache(candidate["ticker"], result)
            return result
        except Exception as e:
            log_warn(f"Claude error for {candidate['ticker']}: {e}")
            return None

    def _build_prompt(self, candidate: Dict, fundamentals: Dict) -> str:
        mom = candidate.get("momentum_sub", {})
        return f"""Analyze this Indian equity breakout candidate.
Stock: {candidate['ticker']} | Sector: {candidate['sector']} | Score: {candidate['composite_score']:.0f}/100
Momentum: {candidate['momentum_score']:.0f} | Quality: {candidate['quality_score']:.0f} | Risk: {candidate['risk_score']:.0f}
12-1M Return: {mom.get('ret_12_1', 'N/A')}% | 6M: {mom.get('ret_6m', 'N/A')}% | 3M: {mom.get('ret_3m', 'N/A')}%
RSI: {candidate.get('rsi', 'N/A')} | 52W Proximity: {mom.get('proximity_52w', 'N/A')}%
Piotroski: {fundamentals.get('piotroski_score', 'N/A')} | ROE: {fundamentals.get('roe', 'N/A')} | D/E: {fundamentals.get('debt_to_equity', 'N/A')}
Entry: Rs {candidate['entry']:.2f} | SL: Rs {candidate['stop_loss']:.2f} | TGT: Rs {candidate['target']:.2f}
Respond in JSON: {{"verdict": "PROCEED/CAUTION/AVOID", "confidence": 1-10, "bull_case": "...", "bear_case": "...", "key_risk": "...", "position_size_note": "Full/Half/Skip"}}"""

_claude = ClaudeAnalyzer()

# ============================================================
# L5 RISK CHECKS
# ============================================================

def get_beta_vs_nifty(ticker: str, df: pd.DataFrame, nifty_df: pd.DataFrame) -> float:
    if df is None or nifty_df is None or len(df) < 30 or len(nifty_df) < 30:
        return 1.0
    try:
        merged = pd.merge(df[["DATE", "Close"]], nifty_df[["DATE", "Close"]], on="DATE", suffixes=("", "_n"))
        if len(merged) < 30:
            return 1.0
        merged["ret"] = merged["Close"].pct_change()
        merged["ret_n"] = merged["Close_n"].pct_change()
        merged = merged.dropna()
        if len(merged) < 20:
            return 1.0
        cov = np.cov(merged["ret"], merged["ret_n"])[0, 1]
        var_n = np.var(merged["ret_n"])
        return 1.0 if var_n == 0 else cov / var_n
    except Exception:
        return 1.0

def calculate_correlation_with_existing(ticker: str, df: pd.DataFrame, existing_trades: List[Dict]) -> Tuple[bool, str]:
    if not existing_trades or df is None or df.empty:
        return True, "No existing positions"
    for t in existing_trades:
        if t["status"] != "OPEN":
            continue
        ex_ticker = t["ticker"] + ".NS" if not t["ticker"].endswith(".NS") else t["ticker"]
        ex_df = fetch_yf_history(ex_ticker, period="90d", interval="1d")
        if ex_df is None or ex_df.empty:
            continue
        try:
            merged = pd.merge(df[["DATE", "Close"]], ex_df[["DATE", "Close"]], on="DATE", suffixes=("", "_ex"))
            if len(merged) < 20:
                continue
            corr = merged["Close"].corr(merged["Close_ex"])
            if corr and abs(corr) > MAX_CORRELATION:
                return False, f"Correlation {corr:.2f} with {t['ticker']} > {MAX_CORRELATION}"
        except Exception:
            continue
    return True, "Correlation OK"

def pre_trade_veto(candidate: Dict, existing_trades: List[Dict],
                     fundamentals: Dict, df: pd.DataFrame,
                     nifty_df: pd.DataFrame, sentiment: Dict) -> Tuple[bool, List[str]]:
    reasons = []
    ticker = candidate["ticker"]
    sector = fundamentals.get("sector", "Unknown")
    open_count = sum(1 for t in existing_trades if t["status"] == "OPEN")
    if open_count >= MAX_POSITIONS:
        reasons.append(f"1. Max positions ({MAX_POSITIONS})")

    try:
        stock = yf.Ticker(ticker)
        cal = stock.earnings_dates
        if cal is not None and not cal.empty:
            now = datetime.now()
            for idx in cal.index[:3]:
                if isinstance(idx, pd.Timestamp):
                    days_to = (idx.date() - now.date()).days
                    if 0 <= days_to <= EARNINGS_BLACKOUT_DAYS:
                        reasons.append(f"2. Earnings in {days_to}d (blackout)")
                        break
    except Exception:
        pass

    adv_inr = candidate.get("adv_inr", 0)
    if adv_inr < MIN_ADV_LAKHS * 100000:
        reasons.append(f"3. Liquidity Rs {adv_inr:,.0f} < min Rs {MIN_ADV_LAKHS}L")

    pos_pct = (candidate["investment"] / CAPITAL) * 100
    if pos_pct > MAX_POSITION_PCT:
        reasons.append(f"4. Position {pos_pct:.1f}% > max {MAX_POSITION_PCT}%")

    sector_exposure = sum(t["investment"] for t in existing_trades if t["status"] == "OPEN" and t.get("sector") == sector)
    sector_pct = ((sector_exposure + candidate["investment"]) / CAPITAL) * 100
    if sector_pct > MAX_SECTOR_PCT:
        reasons.append(f"5. Sector {sector} {sector_pct:.1f}% > max {MAX_SECTOR_PCT}%")

    total_exposure = sum(t["investment"] for t in existing_trades if t["status"] == "OPEN")
    gross_pct = ((total_exposure + candidate["investment"]) / CAPITAL) * 100
    if gross_pct > MAX_GROSS_EXPOSURE:
        reasons.append(f"6. Gross exposure {gross_pct:.1f}% > max {MAX_GROSS_EXPOSURE}%")

    beta = get_beta_vs_nifty(ticker, df, nifty_df)
    port_beta = 0.0
    total_inv = sum(t["investment"] for t in existing_trades if t["status"] == "OPEN")
    for t in existing_trades:
        if t["status"] == "OPEN":
            b = t.get("beta", 1.0)
            port_beta += b * (t["investment"] / max(total_inv, 1))
    port_beta += beta * (candidate["investment"] / max(total_inv + candidate["investment"], 1))
    if abs(port_beta) > MAX_NET_BETA:
        reasons.append(f"7. Net beta {port_beta:.2f} exceeds +/-{MAX_NET_BETA}")

    corr_ok, corr_reason = calculate_correlation_with_existing(ticker, df, existing_trades)
    if not corr_ok:
        reasons.append(f"8. {corr_reason}")

    return len(reasons) == 0, reasons

# ============================================================
# SENTIMENT
# ============================================================

def compute_master_sentiment_score(s: Dict) -> int:
    score = 0
    if s.get("fii_net", 0) > 0: score += 15
    if s.get("dii_net", 0) > 0: score += 10
    if s.get("advances", 0) > s.get("declines", 0): score += 15
    if s.get("india_vix", 20) < 18: score += 20
    if s.get("pcr", 1) >= 1: score += 20
    if s.get("sector_up", 0) > s.get("sector_down", 0): score += 20
    return min(score, 100)

def detect_market_regime(score: int, vix: float) -> str:
    if score >= 70 and vix <= 18: return "BULLISH"
    if score <= 45 and vix >= 20: return "BEARISH"
    return "NEUTRAL"

def get_daily_sentiment() -> Dict:
    s = safe_load_json(SENTIMENT_FILE)
    if s is None:
        s = {"fii_net": 0, "dii_net": 0, "advances": 0, "declines": 0,
             "unchanged": 0, "nifty_change_pct": 0, "india_vix": 20,
             "pcr": 1, "sector_up": 0, "sector_down": 0}
    s["master_sentiment_score"] = compute_master_sentiment_score(s)
    s["market_regime"] = detect_market_regime(s["master_sentiment_score"], s.get("india_vix", 20))
    return s


# ============================================================
# BREAKOUT + SCORING
# ============================================================

def check_breakout(ticker: str, sentiment: Dict, nifty_df: pd.DataFrame) -> Optional[Dict]:
    df = fetch_yf_history(ticker, period="120d", interval="1d")
    if df is None or len(df) < 50:
        return None
    df = df.sort_values("DATE").reset_index(drop=True)
    latest = df.iloc[-1]
    current = safe_scalar(latest["Close"])
    high20 = safe_scalar(df["High"].rolling(20).max().iloc[-2])
    ma50 = safe_scalar(df["Close"].rolling(50).mean().iloc[-1])
    vol10 = safe_scalar(df["Volume"].rolling(10).mean().iloc[-1])
    latest_vol = safe_scalar(latest["Volume"])
    vol_ratio = (latest_vol / vol10) if (vol10 and vol10 > 0) else 0
    rsi = calculate_rsi(df["Close"])

    if not (current > high20 and current > ma50 and vol_ratio >= MIN_VOLUME_RATIO and RSI_MIN <= rsi <= RSI_MAX):
        return None

    fundamentals = fetch_fundamentals(ticker)
    mom_score, mom_sub = calculate_momentum_score(ticker, df, nifty_df)
    qual_score, qual_sub = calculate_quality_score(ticker, fundamentals)
    risk_score, risk_sub = calculate_risk_score(ticker, df, fundamentals)
    comp_score, comp_breakdown = calculate_composite_score(ticker, mom_score, qual_score, risk_score, sentiment)

    if comp_score < COMPOSITE_MIN_FOR_RECOMMENDATION:
        return None

    entry = current
    target = entry * (1 + TARGET_PCT)
    stop = entry * (1 - STOP_LOSS_PCT)
    risk_amt = CAPITAL * RISK_PER_TRADE_PCT
    per_share_risk = entry - stop
    if per_share_risk <= 0:
        return None
    shares = int(risk_amt / per_share_risk)
    if shares <= 0:
        return None
    if shares * entry > CAPITAL:
        shares = int(CAPITAL / entry)
        if shares <= 0:
            return None
    investment = shares * entry

    return {
        "ticker": ticker.replace(".NS", ""),
        "full_ticker": ticker,
        "entry": round(entry, 2),
        "target": round(target, 2),
        "stop_loss": round(stop, 2),
        "shares": shares,
        "investment": round(investment, 2),
        "rsi": round(rsi, 1),
        "volume_ratio": round(vol_ratio, 2),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "status": "RECOMMENDED",
        "sector": fundamentals.get("sector", "Unknown"),
        "beta": get_beta_vs_nifty(ticker, df, nifty_df),
        "adv_inr": risk_sub.get("adv_inr", 0),
        "momentum_score": mom_score,
        "quality_score": qual_score,
        "risk_score": risk_score,
        "composite_score": comp_score,
        "score_breakdown": comp_breakdown,
        "momentum_sub": mom_sub,
        "quality_sub": qual_sub,
        "risk_sub": risk_sub,
        "fundamentals": {k: v for k, v in fundamentals.items() if k not in ["cfo", "net_income", "total_assets"]},
        "claude_analysis": None,
    }

# ============================================================
# TRADES STORAGE
# ============================================================

def load_trades() -> List[Dict]:
    data = safe_load_json(PAPER_TRADE_FILE)
    return data if isinstance(data, list) else []

def save_trades(trades: List[Dict]):
    safe_save_json(PAPER_TRADE_FILE, trades)

def backup_trades():
    if os.path.exists(PAPER_TRADE_FILE):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = TRADES_BACKUP_DIR / f"paper_trades_{ts}.json"
        shutil.copy2(PAPER_TRADE_FILE, dst)
        log_ok(f"Backup: {dst}")

def load_candidates() -> List[Dict]:
    data = safe_load_json(CANDIDATE_FILE)
    return data if isinstance(data, list) else []

def save_candidates(candidates: List[Dict]):
    safe_save_json(CANDIDATE_FILE, candidates)

# ============================================================
# TELEGRAM BOT COMMAND HANDLERS
# ============================================================

def handle_help(chat_id):
    msg = (
        "🤖 <b>JARVIS — NSE Quantitative Screener</b>\n\n"
        "<b>Commands:</b>\n"
        "📊 <code>/scan</code> — Run morning scan now\n"
        "📋 <code>/manage</code> — Check all open positions\n"
        "✅ <code>/enter TICKER PRICE SHARES</code> — Record trade\n"
        "   Example: <code>/enter RELIANCE 2450 10</code>\n"
        "❌ <code>/close TICKER [REASON]</code> — Close trade\n"
        "   Example: <code>/close RELIANCE target hit</code>\n"
        "✂️ <code>/trim TICKER PCT</code> — Partial exit\n"
        "   Example: <code>/trim RELIANCE 0.25</code>\n"
        "📈 <code>/status</code> — Positions + candidates\n"
        "💰 <code>/pnl</code> — Realized + unrealized PnL\n"
        "🎯 <code>/help</code> — This message\n\n"
        "<i>All trades are manual via your broker app. I track and alert.</i>"
    )
    send_telegram_message(msg, chat_id)

def handle_scan(chat_id):
    send_telegram_message("🔍 Running scan... wait 2-3 minutes.", chat_id)
    try:
        sentiment = get_daily_sentiment()
        run_daily_scan(sentiment)
        send_telegram_message("✅ Scan complete. Check messages above.", chat_id)
    except Exception as e:
        send_telegram_message(f"❌ Scan failed: {str(e)}", chat_id)

def handle_manage(chat_id):
    send_telegram_message("📋 Checking positions...", chat_id)
    try:
        sentiment = get_daily_sentiment()
        trades = manage_all_open_trades(sentiment)
        open_count = sum(1 for t in trades if t["status"] == "OPEN")
        send_telegram_message(f"✅ Done. {open_count} open positions.", chat_id)
    except Exception as e:
        send_telegram_message(f"❌ Error: {str(e)}", chat_id)

def handle_enter(chat_id, args):
    if len(args) < 3:
        send_telegram_message("❌ Usage: <code>/enter TICKER PRICE SHARES</code>\nExample: <code>/enter RELIANCE 2450 10</code>", chat_id)
        return
    try:
        ticker = args[0].upper()
        price = float(args[1])
        shares = int(args[2])
        full_ticker = ticker + ".NS" if not ticker.endswith(".NS") else ticker

        trades = load_trades()
        for t in trades:
            if t["ticker"] == ticker and t["status"] == "OPEN":
                send_telegram_message(f"⚠️ {ticker} already OPEN. Close first: <code>/close {ticker}</code>", chat_id)
                return

        fundamentals = fetch_fundamentals(full_ticker)
        nifty_df = fetch_yf_history(BENCHMARK_TICKER, period="90d", interval="1d")
        df = fetch_yf_history(full_ticker, period="60d", interval="1d")
        target = price * (1 + TARGET_PCT)
        stop = price * (1 - STOP_LOSS_PCT)
        beta = 1.0
        if df is not None and nifty_df is not None:
            beta = get_beta_vs_nifty(full_ticker, df, nifty_df)

        trade = {
            "ticker": ticker, "avg_entry": round(price, 2), "entry": round(price, 2),
            "target": round(target, 2), "stop_loss": round(stop, 2), "shares": shares,
            "investment": round(shares * price, 2), "date": datetime.now().strftime("%Y-%m-%d"),
            "status": "OPEN", "sector": fundamentals.get("sector", "Unknown"),
            "beta": round(beta, 2), "days_held": 0, "added_qty": 0, "notes": "",
        }
        trades.append(trade)
        save_trades(trades)
        backup_trades()

        msg = (
            f"✅ <b>ENTRY RECORDED</b>\n"
            f"<b>{ticker}</b> @ Rs {price:.2f} | {shares} shares\n"
            f"Target: Rs {target:.2f} | SL: Rs {stop:.2f} | Beta: {beta:.2f}\n"
            f"Sector: {trade['sector']} | Investment: Rs {trade['investment']:.2f}"
        )
        send_telegram_message(msg, chat_id)
    except Exception as e:
        send_telegram_message(f"❌ Entry failed: {str(e)}", chat_id)

def handle_close(chat_id, args):
    if len(args) < 1:
        send_telegram_message("❌ Usage: <code>/close TICKER [REASON]</code>\nExample: <code>/close RELIANCE target hit</code>", chat_id)
        return
    try:
        ticker = args[0].upper()
        reason = " ".join(args[1:]) if len(args) > 1 else "MANUAL EXIT"

        trades = load_trades()
        today = datetime.now().strftime("%Y-%m-%d")
        found = False
        for t in trades:
            if t["ticker"] == ticker and t["status"] == "OPEN":
                exit_price = fetch_latest_price(ticker)
                if exit_price is None:
                    send_telegram_message(f"❌ Cannot close {ticker}: no price", chat_id)
                    return
                entry = t["avg_entry"]
                pnl = round((exit_price - entry) / entry * 100, 2)
                t["status"] = reason
                t["exit_price"] = round(exit_price, 2)
                t["exit_date"] = today
                t["pnl"] = pnl
                save_trades(trades)

                msg = (
                    f"🔴 <b>CLOSED</b> | {reason}\n"
                    f"<b>{ticker}</b> @ Rs {exit_price:.2f}\n"
                    f"Entry: Rs {entry} | PnL: {pnl}%"
                )
                send_telegram_message(msg, chat_id)
                found = True
                break
        if not found:
            send_telegram_message(f"⚠️ No open trade for {ticker}", chat_id)
    except Exception as e:
        send_telegram_message(f"❌ Close failed: {str(e)}", chat_id)

def handle_trim(chat_id, args):
    if len(args) < 2:
        send_telegram_message("❌ Usage: <code>/trim TICKER PCT</code>\nExample: <code>/trim RELIANCE 0.25</code>", chat_id)
        return
    try:
        ticker = args[0].upper()
        qty_pct = float(args[1])
        price = fetch_latest_price(ticker)
        if price is None:
            send_telegram_message(f"❌ Cannot trim {ticker}: no price", chat_id)
            return

        trades = load_trades()
        found = False
        for t in trades:
            if t["ticker"] == ticker and t["status"] == "OPEN":
                trim_qty = max(1, int(t["shares"] * qty_pct))
                if trim_qty >= t["shares"]:
                    today = datetime.now().strftime("%Y-%m-%d")
                    entry = t["avg_entry"]
                    pnl = round((price - entry) / entry * 100, 2)
                    t["status"] = "TRIM CLOSE"
                    t["exit_price"] = round(price, 2)
                    t["exit_date"] = today
                    t["pnl"] = pnl
                    save_trades(trades)
                    send_telegram_message(f"✂️ <b>TRIM CLOSE</b> {ticker} @ Rs {price:.2f} | PnL: {pnl}%", chat_id)
                else:
                    t["shares"] -= trim_qty
                    t["investment"] = round(t["shares"] * t["avg_entry"], 2)
                    save_trades(trades)
                    send_telegram_message(
                        f"✂️ <b>TRIMMED</b> {trim_qty} shares of {ticker} @ Rs {price:.2f}\n"
                        f"Remaining: {t['shares']} shares | Investment: Rs {t['investment']:.2f}",
                        chat_id
                    )
                found = True
                break
        if not found:
            send_telegram_message(f"⚠️ No open trade for {ticker}", chat_id)
    except Exception as e:
        send_telegram_message(f"❌ Trim failed: {str(e)}", chat_id)

def handle_status(chat_id):
    trades = load_trades()
    candidates = load_candidates()
    open_trades = [t for t in trades if t["status"] == "OPEN"]

    msg = "📊 <b>PORTFOLIO STATUS</b>\n\n"

    if not open_trades:
        msg += "<i>No open positions.</i>\n\n"
    else:
        msg += "<b>Open Positions:</b>\n"
        for t in open_trades:
            price = fetch_latest_price(t["ticker"])
            days = t.get("days_held", 0)
            if price:
                pnl = round((price - t["avg_entry"]) / t["avg_entry"] * 100, 2)
                emoji = "🟢" if pnl >= 0 else "🔴"
                msg += f"{emoji} <b>{t['ticker']}</b> | Entry: Rs {t['avg_entry']} | CMP: Rs {price:.2f} | PnL: {pnl}% | Days: {days}\n"
            else:
                msg += f"⚪ <b>{t['ticker']}</b> | Entry: Rs {t['avg_entry']} | CMP: N/A | Days: {days}\n"
        msg += "\n"

    today = datetime.now().strftime("%Y-%m-%d")
    today_candidates = [c for c in candidates if c.get("date") == today]
    if today_candidates:
        msg += "<b>Today's Candidates:</b>\n"
        for c in sorted(today_candidates, key=lambda x: x["composite_score"], reverse=True)[:5]:
            status = "✅" if c.get("veto_approved") else "❌"
            cl = c.get("claude_analysis", {})
            cl_str = f" | JARVIS: {cl.get('verdict', 'N/A')}" if cl else ""
            msg += f"{status} <b>{c['ticker']}</b> | Score: {c['composite_score']:.0f}{cl_str}\n"
    else:
        msg += "<i>No candidates today.</i>"

    send_telegram_message(msg, chat_id)

def handle_pnl(chat_id):
    trades = load_trades()
    closed = [t for t in trades if t["status"] not in ("OPEN", "RECOMMENDED")]
    open_trades = [t for t in trades if t["status"] == "OPEN"]

    realized_pnl = sum(t.get("pnl", 0) for t in closed)

    unrealized = 0
    for t in open_trades:
        price = fetch_latest_price(t["ticker"])
        if price:
            unrealized += ((price - t["avg_entry"]) / t["avg_entry"] * 100)

    msg = (
        f"💰 <b>P&L SUMMARY</b>\n\n"
        f"Realized PnL: {realized_pnl:.2f}%\n"
        f"Unrealized PnL: {unrealized:.2f}%\n"
        f"Total Trades: {len(closed)} closed, {len(open_trades)} open\n\n"
        f"<i>Realized = closed positions. Unrealized = current open positions.</i>"
    )
    send_telegram_message(msg, chat_id)

def process_bot_command(message_text: str, chat_id, from_id=None):
    if str(chat_id) != str(CHAT_ID):
        log_warn(f"Unauthorized access from chat_id: {chat_id}")
        return

    text = message_text.strip()
    if not text.startswith("/"):
        return

    parts = text.split()
    command = parts[0].lower()
    args = parts[1:]

    log_info(f"Command: {command} from {chat_id}")

    if command == "/help":
        handle_help(chat_id)
    elif command == "/scan":
        handle_scan(chat_id)
    elif command == "/manage":
        handle_manage(chat_id)
    elif command == "/enter":
        handle_enter(chat_id, args)
    elif command == "/close":
        handle_close(chat_id, args)
    elif command == "/trim":
        handle_trim(chat_id, args)
    elif command == "/status":
        handle_status(chat_id)
    elif command == "/pnl":
        handle_pnl(chat_id)
    elif command == "/start":
        handle_help(chat_id)
    else:
        send_telegram_message(f"❓ Unknown: {command}\nTry <code>/help</code>", chat_id)


# ============================================================
# BOT POLLING LOOP
# ============================================================

def run_bot_polling():
    log_alpha("Starting Telegram bot polling...")
    offset = None
    while True:
        try:
            updates = get_telegram_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1
                if "message" in update and "text" in update["message"]:
                    msg_text = update["message"]["text"]
                    chat_id = update["message"]["chat"]["id"]
                    process_bot_command(msg_text, chat_id)
        except Exception as e:
            log_error(f"Polling error: {e}")
        time.sleep(10)

# ============================================================
# DAILY SCAN
# ============================================================

def run_daily_scan(sentiment: Dict):
    log_alpha("Starting daily scan...")
    nifty_df = fetch_yf_history(BENCHMARK_TICKER, period="120d", interval="1d")
    existing_trades = load_trades()

    if sentiment["master_sentiment_score"] < MASTER_SCORE_MIN_FOR_NEW_TRADES:
        log_warn(f"Sentiment {sentiment['master_sentiment_score']} < {MASTER_SCORE_MIN_FOR_NEW_TRADES}. Aborted.")
        send_telegram_message(f"⚠️ Scan aborted. Sentiment too weak: {sentiment['master_sentiment_score']}/100")
        return []
    if sentiment.get("india_vix", 20) > VIX_MAX_FOR_NEW_TRADES:
        log_warn(f"VIX {sentiment['india_vix']} > {VIX_MAX_FOR_NEW_TRADES}. Aborted.")
        send_telegram_message(f"⚠️ Scan aborted. VIX too high: {sentiment['india_vix']}")
        return []

    candidates = []
    open_tickers = {t["ticker"] for t in existing_trades if t["status"] == "OPEN"}

    for ticker in WATCHLIST:
        base = ticker.replace(".NS", "")
        if base in open_tickers:
            continue
        log_info(f"Analyzing {ticker}...")
        c = check_breakout(ticker, sentiment, nifty_df)
        if c:
            fundamentals = fetch_fundamentals(ticker)
            df = fetch_yf_history(ticker, period="90d", interval="1d")
            approved, reasons = pre_trade_veto(c, existing_trades, fundamentals, df, nifty_df, sentiment)
            c["veto_approved"] = approved
            c["veto_reasons"] = reasons
            candidates.append(c)

    candidates.sort(key=lambda x: x["composite_score"], reverse=True)

    if ENABLE_CLAUDE and _claude.client:
        approved_top = [c for c in candidates if c["veto_approved"]][:CLAUDE_ANALYZE_TOP_N]
        for c in approved_top:
            log_info(f"Claude analyzing {c['ticker']}...")
            full_ticker = c["full_ticker"]
            fundamentals = fetch_fundamentals(full_ticker)
            claude_result = _claude.analyze(c, fundamentals)
            if claude_result:
                c["claude_analysis"] = claude_result
                log_ok(f"Claude: {claude_result.get('verdict', 'N/A')} (conf {claude_result.get('confidence', 'N/A')}/10)")

    save_candidates(candidates)

    approved = [c for c in candidates if c["veto_approved"]]
    rejected = [c for c in candidates if not c["veto_approved"]]

    if approved:
        msg = f"✅ <b>{len(approved)} APPROVED CANDIDATES</b> | {datetime.now().strftime('%d %b %Y')}\n"
        for c in approved[:MAX_POSITIONS]:
            cl = c.get("claude_analysis", {})
            claude_block = ""
            if cl:
                verdict_emoji = "🟢" if cl.get("verdict") == "PROCEED" else ("🟡" if cl.get("verdict") == "CAUTION" else "🔴")
                claude_block = (
                    f"\n   🤖 <b>JARVIS:</b> {verdict_emoji} {cl.get('verdict', 'N/A')} (conf: {cl.get('confidence', 'N/A')}/10)\n"
                    f"   📈 Bull: {cl.get('bull_case', 'N/A')}\n"
                    f"   📉 Bear: {cl.get('bear_case', 'N/A')}\n"
                    f"   ⚠️ Risk: {cl.get('key_risk', 'N/A')}\n"
                    f"   💡 Size: {cl.get('position_size_note', 'N/A')}"
                )
            msg += (
                f"\n📈 <b>{c['ticker']}</b> | Score: <b>{c['composite_score']:.0f}</b>/100{claude_block}\n"
                f"   Mom: {c['momentum_score']:.0f} | Qual: {c['quality_score']:.0f} | Risk: {c['risk_score']:.0f}\n"
                f"   Entry: Rs {c['entry']:.2f} | SL: Rs {c['stop_loss']:.2f} | TGT: Rs {c['target']:.2f}\n"
                f"   Sector: {c['sector']} | Beta: {c['beta']:.2f}\n"
                f"   ➡️ <i>Manual execution via broker app</i>"
            )
        send_telegram_message(msg)
    else:
        send_telegram_message(f"📭 <b>No approved candidates today.</b>\nRegime: {sentiment['market_regime']} | Score: {sentiment['master_sentiment_score']}/100")

    if rejected:
        msg = f"⚠️ <b>{len(rejected)} REJECTED</b> (Risk Veto)\n"
        for c in rejected[:3]:
            msg += f"\n❌ {c['ticker']} | Score: {c['composite_score']:.0f} | {', '.join(c['veto_reasons'][:2])}"
        send_telegram_message(msg)

    log_alpha(f"Scan done. {len(approved)} approved, {len(rejected)} rejected.")
    return candidates

# ============================================================
# POSITION MANAGEMENT
# ============================================================

def manage_open_trade(trade: Dict, sentiment: Dict, open_positions: int) -> Dict:
    today = datetime.now()
    price = fetch_latest_price(trade["ticker"])
    if price is None:
        return trade
    entry = trade["avg_entry"]
    sl = trade["stop_loss"]
    target = trade["target"]
    pnl = round((price - entry) / entry * 100, 2)
    entry_date = datetime.strptime(trade["date"], "%Y-%m-%d")
    days_held = (today - entry_date).days
    trade["days_held"] = days_held

    if price >= target:
        action, reason, next_step = "TARGET HIT", "Profit target reached.", f"Close full @ Rs {price:.2f}"
    elif price <= sl:
        action, reason, next_step = "STOP LOSS HIT", "Stop-loss triggered.", f"Exit now @ Rs {price:.2f}"
    elif days_held >= TIME_STOP_DAYS:
        action, reason, next_step = "TIME STOP", f"Max hold ({days_held}d).", f"Close @ Rs {price:.2f}"
    elif pnl <= -5:
        action, reason, next_step = "RISK ALERT", "Drawdown >5%.", f"Consider trim @ Rs {price:.2f}"
    elif sl < price < entry * 0.98:
        action, reason, next_step = "REDUCE", "Below comfort zone.", f"Watch closely @ Rs {price:.2f}"
    else:
        action, reason, next_step = "HOLD", "Trend intact.", "Maintain position"

    total_deployed = sum(t['investment'] for t in load_trades() if t['status'] == 'OPEN')
    total_pct = round((total_deployed / CAPITAL) * 100, 1)
    cash_pct = round(100 - total_pct, 1)

    msg = (
        f"📊 <b>{trade['ticker']}</b> | {action}\n"
        f"CMP Rs {price:.2f} | Entry Rs {entry} | SL Rs {sl} | TGT Rs {target} | PnL {pnl}%\n"
        f"Days: {days_held} | Sector: {trade.get('sector', 'N/A')} | Beta: {trade.get('beta', 'N/A')}\n"
        f"💼 Allocation: {round((trade['investment']/CAPITAL)*100,1)}% of Rs {CAPITAL} | Total: {total_pct}% | Cash: {cash_pct}%\n"
        f"🧠 {reason}\n➡️ {next_step}"
    )
    send_telegram_message(msg)

    if action in ("TARGET HIT", "STOP LOSS HIT", "TIME STOP"):
        trade["status"] = action
        trade["exit_price"] = round(price, 2)
        trade["exit_date"] = today.strftime("%Y-%m-%d")
        trade["pnl"] = pnl
    return trade

def manage_all_open_trades(sentiment: Dict):
    trades = load_trades()
    open_trades = [t for t in trades if t["status"] == "OPEN"]
    open_positions = len(open_trades)
    for i, t in enumerate(trades):
        if t["status"] == "OPEN":
            trades[i] = manage_open_trade(t, sentiment, open_positions)
    backup_trades()
    save_trades(trades)
    return trades

# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="NSE Bot — Free Cloud Edition")
    parser.add_argument("--scan", action="store_true", help="Run morning scan")
    parser.add_argument("--manage", action="store_true", help="Run evening manage")
    parser.add_argument("--bot", action="store_true", help="Start Telegram bot polling")
    parser.add_argument("--once", action="store_true", help="Run scan+manage once then exit")
    args = parser.parse_args()

    if args.bot:
        log_alpha("Starting bot mode...")
        run_bot_polling()
    elif args.scan:
        sentiment = get_daily_sentiment()
        run_daily_scan(sentiment)
    elif args.manage:
        sentiment = get_daily_sentiment()
        manage_all_open_trades(sentiment)
    elif args.once:
        print("\n" + "="*70)
        print("NSE BOT — Running once")
        print("="*70)
        sentiment = get_daily_sentiment()
        run_daily_scan(sentiment)
        manage_all_open_trades(sentiment)
        print("Done.")
    else:
        # Default: run once for scheduled task
        sentiment = get_daily_sentiment()
        run_daily_scan(sentiment)
        manage_all_open_trades(sentiment)

if __name__ == "__main__":
    main()
